#!/usr/bin/env python3
"""
04_build_negatives.py

Build a non-lens training set by sampling random galaxies from the DESI DR1
fiber catalog (reused from the Phase-2 Hsu reproduction).

Why DR1 fibers as non-lenses:
  - All are real galaxies in the DECaLS/MzLS/BASS footprint (the same
    footprint Huang+2020 searched).
  - All have spec-confirmed extragalactic SPECTYPE (we drop STARs).
  - All have ZWARN=0 (reliable Redrock fit).
  - The expected strong-lens rate is ~1 in O(10^4-10^5) galaxies (paper §3.1),
    so the false-positive contamination from accidentally including a real
    lens is negligible compared to the 5000-row size.

We do exclude rows that match any of our Huang+2020 positives within 10″
(safety against label leakage).

Output: data/negatives.parquet — columns: row_id, RA, DEC, Z, SPECTYPE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from astropy.io import fits


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HSU_DATA = HERE.parent / "hsu-2025" / "data"
ZCAT = HSU_DATA / "zall-pix-iron.fits"
POSITIVES = DATA / "positives_huang2020.parquet"
OUT = DATA / "negatives.parquet"

# DECaLS footprint dec limits (Dey+2019). Huang 2020 focused on DECaLS.
DECALS_DEC_LO = -68.0
DECALS_DEC_HI = +32.5
EXCLUDE_ARCSEC = 10.0


def _native(arr, dtype):
    return np.ascontiguousarray(arr).astype(dtype, copy=False)


def load_zcat_galaxies() -> pd.DataFrame:
    print(f"[load] {ZCAT}")
    with fits.open(ZCAT, memmap=True) as hdul:
        zhdu = next(h for h in hdul if h.name == "ZCATALOG")
        data = zhdu.data
        out = {
            "TARGETID": _native(data["TARGETID"], np.int64),
            "RA":       _native(data["TARGET_RA"], np.float64),
            "DEC":      _native(data["TARGET_DEC"], np.float64),
            "Z":        _native(data["Z"], np.float64),
            "SPECTYPE": np.char.strip(np.asarray(data["SPECTYPE"], dtype="U20")),
            "ZWARN":    _native(data["ZWARN"], np.int64),
            "ZCAT_PRIMARY": _native(data["ZCAT_PRIMARY"], np.bool_),
        }
    df = pd.DataFrame(out)
    print(f"[load] {len(df):,} raw rows")
    df = df[
        df["ZCAT_PRIMARY"]
        & (df["ZWARN"] == 0)
        & (df["SPECTYPE"] != "STAR")
        & (df["Z"] > 0.05)            # galaxy-scale lens-relevant redshift floor
        & (df["DEC"] >= DECALS_DEC_LO)
        & (df["DEC"] <= DECALS_DEC_HI)
    ].reset_index(drop=True)
    print(f"[load] {len(df):,} after DECaLS-footprint galaxy filter")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if not ZCAT.exists():
        raise SystemExit(f"missing {ZCAT}; need Phase-2 zall-pix-iron.fits")
    if not POSITIVES.exists():
        raise SystemExit(f"missing {POSITIVES}; run 02_filter_catalog.py")

    df = load_zcat_galaxies()
    rng = np.random.default_rng(args.seed)
    over = int(args.n * 1.2 + 200)  # oversample for exclusion budget
    pick_idx = rng.choice(len(df), size=over, replace=False)
    cand = df.iloc[pick_idx].reset_index(drop=True)

    # Exclusion: drop any candidate within 10″ of a Huang+2020 positive
    pos = pd.read_parquet(POSITIVES)
    print(f"[mask] excluding within {EXCLUDE_ARCSEC}″ of {len(pos):,} positives")
    sc_cand = SkyCoord(ra=cand["RA"].to_numpy() * u.deg,
                        dec=cand["DEC"].to_numpy() * u.deg)
    sc_pos = SkyCoord(ra=pos["RA"].to_numpy() * u.deg,
                       dec=pos["DEC"].to_numpy() * u.deg)
    idx_cand, _, sep, _ = search_around_sky(sc_cand, sc_pos, EXCLUDE_ARCSEC * u.arcsec)
    drop_mask = np.zeros(len(cand), dtype=bool)
    drop_mask[np.unique(idx_cand)] = True
    n_dropped = int(drop_mask.sum())
    cand = cand[~drop_mask].reset_index(drop=True)
    print(f"[mask] dropped {n_dropped} candidates near positives")

    cand = cand.head(args.n).copy()
    cand.insert(0, "row_id", [f"neg_{i:08d}" for i in range(len(cand))])
    cand[["row_id", "TARGETID", "RA", "DEC", "Z", "SPECTYPE"]].to_parquet(OUT, index=False)
    print(f"[save] {OUT}  ({len(cand):,} rows)")


if __name__ == "__main__":
    main()
