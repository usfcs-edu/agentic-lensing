#!/usr/bin/env python3
"""314_build_blends.py — ClaudeNet v3 A2: build the blended mined manifests for 121.

v2-lean's `hard` members displaced n_mine=10,000 bootstrap negatives with 10,000 CNN-high
RANDOM-pool negatives (mined_hard_fits_manifest.parquet). v3 keeps the recipe byte-identical
(same n_mine, same displaced bootstrap rows — the displacement seed is per-member, not per-
variant) and changes ONLY the negative *content*: a fraction f of the 10,000 are typed
lens-MIMICS (313 train-pool, hardest-by-p_final), the rest stay v2 hard negatives. This
isolates the contaminant-awareness effect (the staf327 thesis) from everything else.

  blend_b{30,50,70}.parquet = R.head(f*10k) (mimics) + H.head((1-f)*10k) (v2 hard),
  with R the collision-filtered train-pool (mimic row_ids that collide with any v1 member
  train table are dropped — 121 asserts mined∩train==∅; DR9/DR10 row_ids can coincide).

    python 314_build_blends.py            # -> data/v2/blend_b{30,50,70}.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

V2 = C.DATA / "v2"
MEMBERS = ("effnet_S2", "effnet_B3", "resnet46_C")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mimic-manifest", default=str(V2 / "mimic_bank_fits_manifest.parquet"))
    ap.add_argument("--hard-manifest", default=str(V2 / "mined_hard_fits_manifest.parquet"))
    ap.add_argument("--n-mine", type=int, default=10000)
    ap.add_argument("--fracs", default="0.30,0.50,0.70")
    args = ap.parse_args()

    mimic = pd.read_parquet(args.mimic_manifest)
    mimic["row_id"] = mimic["row_id"].astype(str)
    hard = pd.read_parquet(args.hard_manifest)
    hard["row_id"] = hard["row_id"].astype(str)

    # collision guard: drop mimic rows whose row_id appears in ANY v1 member train table
    train_ids = set()
    for m in MEMBERS:
        t = pd.read_parquet(C.DATA / f"member_{m}_train.parquet", columns=["row_id"])
        train_ids |= set(t["row_id"].astype(str))
    n0 = len(mimic)
    mimic = mimic[~mimic["row_id"].isin(train_ids)].reset_index(drop=True)
    print(f"[314] mimic train-pool: {len(mimic):,} rows "
          f"({n0 - len(mimic)} dropped as v1-train collisions); hard pool {len(hard):,}")

    cols = ["row_id", "label", "fits_dir"]
    hard = hard[cols]
    mimic_m = mimic[cols]
    for fs in args.fracs.split(","):
        f = float(fs)
        n_mim = int(round(args.n_mine * f))
        n_hard = args.n_mine - n_mim
        assert len(mimic_m) >= n_mim, f"train-pool has {len(mimic_m)} < {n_mim} for f={f}"
        assert len(hard) >= n_hard, f"hard pool has {len(hard)} < {n_hard} for f={f}"
        blend = pd.concat([mimic_m.head(n_mim), hard.head(n_hard)], ignore_index=True)
        assert len(blend) == args.n_mine and blend["row_id"].is_unique
        tag = f"b{int(round(f * 100)):02d}"
        out = V2 / f"blend_{tag}.parquet"
        blend.to_parquet(out, index=False)
        print(f"[314] {out.name}: {n_mim:,} mimic + {n_hard:,} hard = {len(blend):,} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
