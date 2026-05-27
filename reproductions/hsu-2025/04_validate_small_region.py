#!/usr/bin/env python3
"""
04_validate_small_region.py

Validation pass for Hsu+2025 §3 on a single small-region sample (SV3 dark).
End-to-end:

  pre-filter  -> spherimatch FoF (3.0″)  -> z_max/z_min >= 1.3
              -> count groups by multiplicity
              -> save data/sv3dark_pairs.parquet

The paper's authoritative algorithm (verified from the PDF, §3.1–3.2):

  §3.1 Pre-filtering:
    1. drop ZWARN != 0   (pipeline-flagged unreliable redshift)
    2. drop SPECTYPE == 'STAR'   (paper says "target type TGT classified as star";
       we use Redrock's SPECTYPE which is the natural extragalactic filter)
    3. retain longest-exposure coadd per object  ->  use ZCAT_PRIMARY == True
       (the DESI flag that marks the primary, best-exposure coadd per TARGETID)
    Result on full DR1: 28M -> ~15.8M useful spectra.

  §3.2 Grouping:
    spherimatch.fof at linking length = 3.0″, then keep groups with
    z_max / z_min >= 1.3.
    Result on full DR1: 13,218 groups (13,044 pairs + 165 triplets + 7
    quartets + 2 quintets) containing 26,621 spectra.

This small-region run should scale to those numbers by
  N_groups(sv3-dark) ≈ (N_filtered_sv3-dark / 15.8e6) × 13218.
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
DATA.mkdir(exist_ok=True)

ZCAT = DATA / "zpix-sv3-dark.fits"
OUT_PAIRS = DATA / "sv3dark_pairs.parquet"
OUT_STATS = DATA / "sv3dark_stats.json"

LINK_DEG = 3.0 / 3600.0
Z_RATIO_MIN = 1.3
NEEDED_COLS = [
    "TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "ZERR",
    "SPECTYPE", "ZWARN", "ZCAT_PRIMARY", "SURVEY", "PROGRAM",
]


def _native(arr, dtype):
    """Convert FITS big-endian column to native-byte-order array of `dtype`."""
    return np.ascontiguousarray(arr).astype(dtype, copy=False)


def load_zcat(path: Path) -> pd.DataFrame:
    t0 = time.time()
    print(f"[load] {path}")
    with fits.open(path, memmap=True) as hdul:
        zhdu = next(h for h in hdul if h.name == "ZCATALOG")
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
        # zall-pix-iron.fits carries SURVEY/PROGRAM; per-program zpix files do not
        if "SURVEY" in cols:
            out["SURVEY"] = np.char.strip(np.asarray(data["SURVEY"], dtype="U6"))
        if "PROGRAM" in cols:
            out["PROGRAM"] = np.char.strip(np.asarray(data["PROGRAM"], dtype="U20"))
    df = pd.DataFrame(out)
    print(f"[load] {len(df):,} rows in {time.time()-t0:.1f} s")
    return df


def prefilter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"raw": len(df)}
    df = df[df["ZCAT_PRIMARY"]]
    stats["after_zcat_primary"] = len(df)
    df = df[df["ZWARN"] == 0]
    stats["after_zwarn_zero"] = len(df)
    df = df[df["SPECTYPE"] != "STAR"]
    stats["after_spectype_not_star"] = len(df)
    # The paper also drops rows where redshift is unphysical; not stated explicitly
    # but Hsu's z-ratio cut implicitly requires z>0. Be conservative.
    df = df[df["Z"] > 0]
    stats["after_z_positive"] = len(df)
    df = df.reset_index(drop=True)
    return df, stats


def run_fof_and_zcut(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    radec = df[["RA", "DEC"]].to_numpy(dtype=np.float64)
    t0 = time.time()
    print(f"[fof ] running spherimatch.fof on N={len(radec):,} at link={3.0:.2f}″")
    res = fof(radec, LINK_DEG)
    print(f"[fof ] done in {time.time()-t0:.1f} s")

    sizes = res.get_group_sizes()
    group_df = res.get_group_dataframe()
    # group_df has MultiIndex (Group, Object). The Object index is the row position
    # in the input array, i.e. df.index already (since we reset_index above).
    group_df = group_df.reset_index()
    group_df = group_df.rename(columns={"Group": "group_id", "Object": "obj_idx"})

    # Keep only groups of size >= 2
    big = group_df.groupby("group_id").filter(lambda g: len(g) >= 2)
    big = big.merge(
        df.reset_index().rename(columns={"index": "obj_idx"}),
        on="obj_idx", suffixes=("", "_df"),
    )

    # Apply z_max/z_min >= 1.3 group filter
    z_stats = big.groupby("group_id")["Z"].agg(zmin="min", zmax="max").reset_index()
    z_stats["z_ratio"] = z_stats["zmax"] / z_stats["zmin"]
    keep_ids = z_stats.loc[z_stats["z_ratio"] >= Z_RATIO_MIN, "group_id"]
    kept = big[big["group_id"].isin(keep_ids)].copy()

    # Multiplicity breakdown
    mult = kept.groupby("group_id").size()
    mult_counts = mult.value_counts().sort_index().to_dict()
    mult_counts = {int(k): int(v) for k, v in mult_counts.items()}
    stats = {
        "fof_total_objects_in_groups>=2": int(len(big)),
        "fof_groups_size>=2": int(big["group_id"].nunique()),
        "after_z_ratio_groups": int(len(keep_ids)),
        "after_z_ratio_spectra": int(len(kept)),
        "multiplicity_counts": mult_counts,
        "fof_wallclock_s": float(time.time() - t0),
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

    # Save kept pairs/groups as parquet keyed on (group_id, TARGETID)
    out_cols = ["group_id", "TARGETID", "RA", "DEC", "Z", "ZERR", "SPECTYPE"]
    keep_df = kept[out_cols].copy()
    table = pa.Table.from_pandas(keep_df, preserve_index=False)
    pq.write_table(table, OUT_PAIRS)
    print(f"[save] wrote {OUT_PAIRS}  ({len(keep_df):,} rows)")

    # Predicted scaling to full DR1
    pre_filt_full = 15_800_000
    pre_filt_here = pre_stats["after_z_positive"]
    pub_groups = 13_218
    pub_spectra = 26_621
    scale = pre_filt_here / pre_filt_full
    expected_groups = scale * pub_groups
    expected_spectra = scale * pub_spectra
    print(
        f"\n[scale] sv3-dark filtered {pre_filt_here:,} / "
        f"published full-DR1 filtered ≈ {pre_filt_full:,}\n"
        f"[scale]   expected groups  ≈ {expected_groups:.0f}, "
        f"actual {fof_stats['after_z_ratio_groups']:,}\n"
        f"[scale]   expected spectra ≈ {expected_spectra:.0f}, "
        f"actual {fof_stats['after_z_ratio_spectra']:,}"
    )

    OUT_STATS.write_text(
        json.dumps(
            {
                "prefilter": pre_stats,
                "fof_zcut": fof_stats,
                "scaling": {
                    "filtered_local": pre_filt_here,
                    "filtered_full_dr1_pub": pre_filt_full,
                    "expected_groups_local": expected_groups,
                    "expected_spectra_local": expected_spectra,
                },
            },
            indent=2,
        )
    )
    print(f"[save] wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
