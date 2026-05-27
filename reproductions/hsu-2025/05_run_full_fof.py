#!/usr/bin/env python3
"""
05_run_full_fof.py

Run Hsu+2025 §3 algorithm at full DR1 scale on zall-pix-iron.fits.

Authoritative target (Hsu+2025 §3.2):
  13,218 groups (13,044 pairs + 165 triplets + 7 quartets + 2 quintets)
  containing 26,621 spectra.

Pre-filter targets (§3.1):
  28M -> 15.8M useful spectra after ZWARN==0, !STAR, ZCAT_PRIMARY.

Outputs:
  data/dr1_pairs.parquet  -- one row per spectrum in a kept group
  data/dr1_stats.json     -- count breakdowns at each stage
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from spherimatch import fof


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ZCAT = DATA / "zall-pix-iron.fits"
OUT_PAIRS = DATA / "dr1_pairs.parquet"
OUT_STATS = DATA / "dr1_stats.json"

LINK_DEG = 3.0 / 3600.0
Z_RATIO_MIN = 1.3


def _native(arr, dtype):
    return np.ascontiguousarray(arr).astype(dtype, copy=False)


def load_zcat(path: Path) -> pd.DataFrame:
    t0 = time.time()
    print(f"[load] {path}")
    with fits.open(path, memmap=True) as hdul:
        zhdu = next(h for h in hdul if h.name == "ZCATALOG")
        # First inspect: how big is this catalog
        nrows = zhdu.header["NAXIS2"]
        print(f"[load] {nrows:,} rows in ZCATALOG HDU")
        data = zhdu.data
        cols = {c.name for c in zhdu.columns}
        out: dict[str, np.ndarray] = {
            "TARGETID":     _native(data["TARGETID"], np.int64),
            "RA":           _native(data["TARGET_RA"], np.float64),
            "DEC":          _native(data["TARGET_DEC"], np.float64),
            "Z":            _native(data["Z"], np.float64),
            "ZERR":         _native(data["ZERR"], np.float32),
            "SPECTYPE":     np.char.strip(np.asarray(data["SPECTYPE"], dtype="U20")),
            "ZWARN":        _native(data["ZWARN"], np.int64),
            "ZCAT_PRIMARY": _native(data["ZCAT_PRIMARY"], np.bool_),
        }
        if "SURVEY" in cols:
            out["SURVEY"] = np.char.strip(np.asarray(data["SURVEY"], dtype="U6"))
        if "PROGRAM" in cols:
            out["PROGRAM"] = np.char.strip(np.asarray(data["PROGRAM"], dtype="U20"))
    df = pd.DataFrame(out)
    print(f"[load] {len(df):,} rows loaded in {time.time()-t0:.1f} s")
    return df


def prefilter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"raw": int(len(df))}
    df = df[df["ZCAT_PRIMARY"]]
    stats["after_zcat_primary"] = int(len(df))
    df = df[df["ZWARN"] == 0]
    stats["after_zwarn_zero"] = int(len(df))
    df = df[df["SPECTYPE"] != "STAR"]
    stats["after_spectype_not_star"] = int(len(df))
    df = df[df["Z"] > 0]
    stats["after_z_positive"] = int(len(df))
    df = df.reset_index(drop=True)
    return df, stats


def run_fof_and_zcut(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    radec = df[["RA", "DEC"]].to_numpy(dtype=np.float64)
    t0 = time.time()
    print(f"[fof ] running spherimatch.fof on N={len(radec):,} at link=3.00″")
    res = fof(radec, LINK_DEG)
    dt = time.time() - t0
    print(f"[fof ] done in {dt:.1f} s")

    group_df = res.get_group_dataframe().reset_index().rename(
        columns={"Group": "group_id", "Object": "obj_idx"}
    )
    print(f"[fof ] {group_df['group_id'].nunique():,} total groups")

    # Filter to groups with size >= 2
    sizes = group_df.groupby("group_id").size()
    big_ids = sizes[sizes >= 2].index
    big = group_df[group_df["group_id"].isin(big_ids)]
    print(f"[fof ] {len(big_ids):,} groups of size >= 2 covering {len(big):,} spectra")

    # Join back to df to get redshift info
    df_idx = df.reset_index().rename(columns={"index": "obj_idx"})
    big = big.merge(df_idx, on="obj_idx", suffixes=("", "_df"))

    # z_max/z_min >= 1.3 group cut
    z_stats = big.groupby("group_id")["Z"].agg(zmin="min", zmax="max").reset_index()
    z_stats["z_ratio"] = z_stats["zmax"] / z_stats["zmin"]
    keep_ids = z_stats.loc[z_stats["z_ratio"] >= Z_RATIO_MIN, "group_id"]
    kept = big[big["group_id"].isin(keep_ids)].copy()

    mult = kept.groupby("group_id").size().value_counts().sort_index().to_dict()
    mult = {int(k): int(v) for k, v in mult.items()}
    stats = {
        "fof_groups_total": int(group_df["group_id"].nunique()),
        "fof_groups_size>=2": int(len(big_ids)),
        "fof_spectra_size>=2": int(len(big)),
        "after_z_ratio_groups": int(len(keep_ids)),
        "after_z_ratio_spectra": int(len(kept)),
        "multiplicity_counts": mult,
        "fof_wallclock_s": float(dt),
    }
    return kept, stats


def main() -> None:
    if not ZCAT.exists():
        raise SystemExit(f"missing {ZCAT}; run 01_download_dr1_zcatalog.py first")
    df = load_zcat(ZCAT)
    df, pre_stats = prefilter(df)
    print(f"[pre ] {pre_stats}")

    kept, fof_stats = run_fof_and_zcut(df)
    print(f"[fof ] {fof_stats}")

    out_cols = ["group_id", "TARGETID", "RA", "DEC", "Z", "ZERR", "SPECTYPE"]
    keep_df = kept[out_cols].copy()
    table = pa.Table.from_pandas(keep_df, preserve_index=False)
    pq.write_table(table, OUT_PAIRS)
    print(f"[save] wrote {OUT_PAIRS}  ({len(keep_df):,} rows)")

    pub = {
        "filtered_pub": 15_800_000,
        "groups_pub": 13_218,
        "spectra_pub": 26_621,
        "multiplicity_pub": {2: 13_044, 3: 165, 4: 7, 5: 2},
    }
    print("\n[verify]  PUBLISHED  vs  OURS")
    print(f"  filtered     {pub['filtered_pub']:>10,}    {pre_stats['after_z_positive']:>10,}")
    print(f"  groups       {pub['groups_pub']:>10,}    {fof_stats['after_z_ratio_groups']:>10,}")
    print(f"  spectra      {pub['spectra_pub']:>10,}    {fof_stats['after_z_ratio_spectra']:>10,}")
    print(f"  multiplicity {pub['multiplicity_pub']}    {fof_stats['multiplicity_counts']}")

    OUT_STATS.write_text(
        json.dumps(
            {
                "prefilter": pre_stats,
                "fof_zcut": fof_stats,
                "published": pub,
            },
            indent=2,
        )
    )
    print(f"[save] wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
