#!/usr/bin/env python3
"""
10_select_parent_sample_dr8.py — Phase 4b step 2.

Filter the DR8 sweep files in data/dr8_sweep/{south,north}/ to the Huang+2021
parent sample (§4): galaxies with

  - TYPE in the selected set (default {'DEV','COMP'}; --type-set rex adds 'REX')
  - NOBS_G >= 3 AND NOBS_R >= 3 AND NOBS_Z >= 3
  - z-band mag <= 20.0  (mag = 22.5 - 2.5 log10(FLUX_Z); FLUX_Z >= 10)

Each surviving row is tagged with a `footprint` column ('south' = DECaLS/DECam,
'north' = BASS+MzLS) derived from the sweep file's directory. 11b uses this to
route brick-image downloads to dr8/{south,north}/coadd/. The RELEASE column is
also kept and cross-checked against the footprint (south expects 8000-8004,
north expects 9010).

Huang+2021 reports ~15.4M DEV/COMP cutouts (primary) over ~14,000 deg², plus a
~6.7M REX secondary pass. Clone of huang-2020/10_select_parent_sample.py
generalised to two footprints + a configurable TYPE set.

Usage:
  ./10_select_parent_sample_dr8.py                       # DEV/COMP primary
  ./10_select_parent_sample_dr8.py --type-set rex --out data/parent_dr8_rex.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / "data" / "dr8_sweep"
OUT_PATH = HERE / "data" / "parent_dr8.parquet"

KEEP_COLS = [
    "RELEASE", "BRICKID", "BRICKNAME", "OBJID", "TYPE",
    "RA", "DEC",
    "FLUX_G", "FLUX_R", "FLUX_Z",
    "NOBS_G", "NOBS_R", "NOBS_Z",
]

TYPE_SETS = {
    "primary": {"DEV", "COMP"},
    "rex": {"REX"},
    "all": {"DEV", "COMP", "REX"},
}
# DR8 RELEASE codes per footprint (verified from the built parent sample):
# south (DECam/DECaLS) = 8000, north (BASS+MzLS) = 8001.
RELEASE_RANGE = {"south": (8000, 8000), "north": (8001, 8001)}


def filter_sweep(path: Path, footprint: str, type_ok_set: set[str]) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        t = hdul[1].data
        types = t["TYPE"]
        if types.dtype.kind == "S":
            tdec = np.char.strip(types).astype(str)
        else:
            tdec = np.char.strip(types.astype(str))
        type_ok = np.isin(tdec, list(type_ok_set))
        nobs_ok = (t["NOBS_G"] >= 3) & (t["NOBS_R"] >= 3) & (t["NOBS_Z"] >= 3)
        fz = t["FLUX_Z"]
        # z-mag <= 20.0  <=>  FLUX_Z >= 10
        m = type_ok & nobs_ok & (fz >= 10.0)
        if int(m.sum()) == 0:
            cols = KEEP_COLS + ["footprint"]
            return pd.DataFrame(columns=cols)
        out = {}
        for col in KEEP_COLS:
            arr = t[col][m]
            if arr.dtype.kind in ("S", "U"):
                if arr.dtype.kind == "S":
                    arr = arr.astype(str)
                arr = np.array([s.strip() for s in arr], dtype=object)
            elif arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="))
            out[col] = arr
        df = pd.DataFrame(out)
        df["footprint"] = footprint
        return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type-set", choices=tuple(TYPE_SETS), default="primary")
    ap.add_argument("--limit", type=int, default=None, help="for testing")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    type_ok_set = TYPE_SETS[args.type_set]

    work: list[tuple[Path, str]] = []
    for fp in ("south", "north"):
        sweeps = sorted((SWEEP_DIR / fp).glob("sweep-*.fits"))
        for p in sweeps:
            work.append((p, fp))
    if not work:
        raise SystemExit(f"no sweeps found under {SWEEP_DIR}/{{south,north}}")
    if args.limit:
        work = work[: args.limit]
    print(f"[init] processing {len(work)} sweep files  TYPE in {sorted(type_ok_set)}")

    writer = None
    n_out = 0
    type_counts: dict[str, int] = {}
    fp_counts: dict[str, int] = {}
    release_warn = 0
    z_mags = []
    try:
        for path, fp in tqdm(work, desc="filtering"):
            df = filter_sweep(path, fp, type_ok_set)
            if len(df) == 0:
                continue
            n_out += len(df)
            for tv, c in df["TYPE"].value_counts().items():
                type_counts[tv] = type_counts.get(tv, 0) + int(c)
            fp_counts[fp] = fp_counts.get(fp, 0) + len(df)
            lo, hi = RELEASE_RANGE[fp]
            rel = df["RELEASE"].values
            release_warn += int(((rel < lo) | (rel > hi)).sum())
            with np.errstate(divide="ignore"):
                z_mags.append(22.5 - 2.5 * np.log10(df["FLUX_Z"].values))

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.out, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    z_all = np.concatenate(z_mags) if z_mags else np.array([])
    print(f"[done] wrote {n_out:,} rows to {args.out}")
    print(f"[stat] footprint: " + "  ".join(
        f"{k}={v:,} ({v/max(n_out,1):.1%})" for k, v in sorted(fp_counts.items())))
    print(f"[stat] TYPE: " + "  ".join(
        f"{k}={v:,}" for k, v in sorted(type_counts.items())))
    if len(z_all) > 0:
        print(f"[stat] z-mag: median={np.median(z_all):.3f}  "
              f"range=({z_all.min():.3f}, {z_all.max():.3f})")
    if release_warn:
        print(f"[warn] {release_warn:,} rows have RELEASE outside the expected "
              f"per-footprint range — footprint routing may be off")
    if not args.limit and args.type_set == "primary":
        lo, hi = 14_000_000, 17_000_000
        if n_out < lo or n_out > hi:
            print(f"[warn] DEV/COMP row count {n_out:,} outside expected "
                  f"[{lo:,}, {hi:,}] — re-check cuts")
        else:
            print(f"[ok] DEV/COMP row count within expected ~14-17M range")


if __name__ == "__main__":
    main()
