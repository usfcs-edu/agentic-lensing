#!/usr/bin/env python3
"""
10_select_parent_sample.py — Phase 3b M1 step 2.

Filter the 292 DR7 sweep files in data/dr7_sweep/ to the Huang+2020
parent sample (§4.1):

  - TYPE in {'DEV ', 'COMP'}
  - NOBS_G >= 3 AND NOBS_R >= 3 AND NOBS_Z >= 3
  - z-band mag <= 20.0  (mag = 22.5 - 2.5 log10(FLUX_Z); FLUX_Z > 0)

Concatenate surviving rows into data/parent_dr7.parquet.
Expected size: 5.5-6.0M rows × ~70 bytes = ~400 MB parquet.

Usage:
  ./10_select_parent_sample.py [--limit N]   # for testing
  ./10_select_parent_sample.py               # full run
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
SWEEP_DIR = HERE / "data" / "dr7_sweep"
OUT_PATH = HERE / "data" / "parent_dr7.parquet"

# Columns we need; drop the rest to keep memory low.
KEEP_COLS = [
    "RELEASE", "BRICKID", "BRICKNAME", "OBJID", "TYPE",
    "RA", "DEC",
    "FLUX_G", "FLUX_R", "FLUX_Z",
    "NOBS_G", "NOBS_R", "NOBS_Z",
]


def filter_sweep(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        t = hdul[1].data
        # Build mask in NumPy directly (fast, no Table overhead)
        types = t["TYPE"]
        # TYPE column is bytes type; .strip() works for both 'DEV ' and b'DEV '.
        if types.dtype.kind == "S":
            tdec = np.char.strip(types).astype(str)
        else:
            tdec = np.char.strip(types.astype(str))
        type_ok = (tdec == "DEV") | (tdec == "COMP")
        nobs_ok = (t["NOBS_G"] >= 3) & (t["NOBS_R"] >= 3) & (t["NOBS_Z"] >= 3)
        fz = t["FLUX_Z"]
        fz_pos = fz > 0
        # z-mag <= 20.0  <=>  FLUX_Z >= 10**((22.5 - 20.0) / 2.5) = 10
        zmag_ok = fz >= 10.0
        m = type_ok & nobs_ok & fz_pos & zmag_ok
        n_kept = int(m.sum())
        if n_kept == 0:
            return pd.DataFrame(columns=KEEP_COLS)
        # Materialize only kept rows. FITS arrays are big-endian; pyarrow needs
        # native byte order so we cast numeric columns. String columns also
        # arrive padded to their fixed width — strip and store as python objects
        # so downstream equality comparisons (e.g. == "DEV") match cleanly.
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
        return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="for testing")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    sweeps = sorted(SWEEP_DIR.glob("sweep-*.fits"))
    if not sweeps:
        raise SystemExit(f"no sweeps found in {SWEEP_DIR}")
    if args.limit:
        sweeps = sweeps[: args.limit]
    print(f"[init] processing {len(sweeps)} sweep files")

    # Stream rows into parquet via an incremental writer to bound memory.
    writer = None
    n_total_in = n_total_out = 0
    n_dev = n_comp = 0
    z_mags = []
    try:
        for path in tqdm(sweeps, desc="filtering"):
            df = filter_sweep(path)
            if len(df) == 0:
                continue
            n_total_in += len(df)
            n_total_out += len(df)
            # Update counters
            t_vals = df["TYPE"].values
            n_dev += int((t_vals == "DEV").sum())
            n_comp += int((t_vals == "COMP").sum())
            with np.errstate(divide="ignore"):
                zm = 22.5 - 2.5 * np.log10(df["FLUX_Z"].values)
            z_mags.append(zm)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.out, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    z_all = np.concatenate(z_mags) if z_mags else np.array([])
    print(f"[done] wrote {n_total_out:,} rows to {args.out}")
    print(f"[stat] type breakdown: DEV={n_dev:,} ({n_dev/max(n_total_out,1):.1%})  "
          f"COMP={n_comp:,} ({n_comp/max(n_total_out,1):.1%})")
    if len(z_all) > 0:
        print(f"[stat] z-mag: median={np.median(z_all):.3f}  "
              f"mean={z_all.mean():.3f}  range=({z_all.min():.3f}, {z_all.max():.3f})")
    target_lo, target_hi = 5_500_000, 6_000_000
    if not args.limit:
        if n_total_out < target_lo or n_total_out > target_hi:
            print(f"[warn] row count {n_total_out:,} outside expected "
                  f"[{target_lo:,}, {target_hi:,}] — re-check cuts")
        else:
            print(f"[ok] row count within expected 5.5-6.0M range")


if __name__ == "__main__":
    main()
