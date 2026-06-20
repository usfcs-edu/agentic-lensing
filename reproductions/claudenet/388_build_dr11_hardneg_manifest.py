#!/usr/bin/env python3
"""388_build_dr11_hardneg_manifest.py — DR11-native HARD negatives for the fine-tune (task D):
the CNN-high mimics from the mean-ranked survivor pool. These are the contaminant-aware signal
(lrg+companion etc.) that teaches lens-vs-mimic discrimination — the hard residual the fine-tune
targets. Runs ON PERLMUTTER (survivors_dr11s_mean.parquet already carries RA/DEC/brick).

Selection (PU-safe): take the mean-ranked survivors, drop any within --pu-arcsec of a known
lens (positives + Storfer/Inchausti), SKIM the top --skim by mean (the genuine candidate region
incl. the vetted top-500 + NEW shortlist, which may contain real unconfirmed lenses), then take
the next --n by mean as hard negatives (still very high CNN score = hard, but below the candidate
region = overwhelmingly mimics per the campaign's resolution-limited finding).

  module load python
  python 388_build_dr11_hardneg_manifest.py --skim 2000 --n 20000 --out dr11s_hardneg_manifest_brick.parquet
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = "/pscratch/sd/g/gdbenson/claudenet/sweep_dr11/south"
CATDIR = "/global/homes/g/gdbenson/claudenet/data"


def xyz(ra, dec):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skim", type=int, default=2000, help="top-by-mean candidate rows to exclude")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--pu-arcsec", type=float, default=10.0)
    ap.add_argument("--survivors", default=f"{BASE}/survivors_dr11s_mean.parquet")
    ap.add_argument("--pos-manifest", default="dr11s_positive_manifest_brick.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s = pd.read_parquet(a.survivors).sort_values("mean5", ascending=False).reset_index(drop=True)
    print(f"[388] survivors: {len(s):,}  mean5 range [{s.mean5.min():.3f}, {s.mean5.max():.3f}]")

    pu = [pd.read_parquet(a.pos_manifest)[["RA", "DEC"]]]
    for f in ("storfer2024_published_catalog.csv", "inchausti2025_published_catalog.csv"):
        pu.append(pd.read_csv(f"{CATDIR}/{f}")[["RA", "DEC"]])
    pucat = pd.concat(pu, ignore_index=True).dropna()
    tree = cKDTree(xyz(pucat.RA, pucat.DEC))
    d, _ = tree.query(xyz(s.RA, s.DEC), k=1)
    near = (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600) < a.pu_arcsec
    print(f"[388] PU-guard ({a.pu_arcsec}\"): {int(near.sum()):,} survivors near known lenses (dropped)")

    safe = s[~near].reset_index(drop=True)              # PU-safe, still mean-sorted
    hard = safe.iloc[a.skim:a.skim + a.n].copy()        # skim candidate region, take next n
    hard["label"] = 0
    cols = ["row_id", "RA", "DEC", "footprint", "brick", "label"]
    hard[cols].to_parquet(a.out, index=False)
    print(f"[388] skimmed top {a.skim:,} (candidate region); hard negatives: {len(hard):,} "
          f"(mean5 {hard.mean5.min():.3f}-{hard.mean5.max():.3f}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
