#!/usr/bin/env python3
"""386_sample_dr11_negatives.py — sample random DR11-south parent galaxies as DR11-native
training NEGATIVES (PU-guarded), producing a 111 input manifest. Runs ON PERLMUTTER.

The parent sweep manifests already carry (row_id, RA, DEC, footprint, brick), so no brick
assignment is needed. We drop any parent row within --pu-arcsec of a known lens (the positive
manifest + Storfer/Inchausti) so unlabeled real lenses don't leak into the negative class, then
random-sample --n. Training on DR11-native randoms is what corrects the deeper-survey
fatter-high-score-tail at the feature level (the genuine DR9->DR11 shift).

  module load python
  python 386_sample_dr11_negatives.py --n 50000 --out dr11s_negative_manifest_brick.parquet
"""
from __future__ import annotations

import argparse
import glob
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
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--pu-arcsec", type=float, default=10.0)
    ap.add_argument("--pos-manifest", default="dr11s_positive_manifest_brick.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    man = pd.concat([pd.read_parquet(p, columns=["row_id", "RA", "DEC", "footprint", "brick"])
                     for p in sorted(glob.glob(f"{BASE}/sweep_manifest_part*of32.parquet"))],
                    ignore_index=True)
    print(f"[386] parent rows: {len(man):,}")

    pu = [pd.read_parquet(a.pos_manifest)[["RA", "DEC"]]]
    for f in ("storfer2024_published_catalog.csv", "inchausti2025_published_catalog.csv"):
        pu.append(pd.read_csv(f"{CATDIR}/{f}")[["RA", "DEC"]])
    pucat = pd.concat(pu, ignore_index=True).dropna()
    tree = cKDTree(xyz(pucat.RA, pucat.DEC))
    d, _ = tree.query(xyz(man.RA, man.DEC), k=1)
    near = (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600) < a.pu_arcsec
    man = man[~near].reset_index(drop=True)
    print(f"[386] PU-guard ({a.pu_arcsec}\"): dropped {int(near.sum()):,} near known lenses -> {len(man):,}")

    rng = np.random.default_rng(a.seed)
    sel = man.iloc[rng.choice(len(man), size=min(a.n, len(man)), replace=False)].copy()
    sel["label"] = 0
    sel[["row_id", "RA", "DEC", "footprint", "brick", "label"]].to_parquet(a.out, index=False)
    print(f"[386] sampled {len(sel):,} negatives -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
