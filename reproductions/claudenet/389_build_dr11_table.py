#!/usr/bin/env python3
"""389_build_dr11_table.py — assemble the shared DR11-native member training table for the
fine-tune (task D): tier-subsampled positives + DR11 random negatives + DR11 hard negatives,
with a train/val split and per-row fits_dir (the trainer honors a per-row fits_dir column).

Confidence weighting is realized by per-tier subsampling (gold 100% / silver 70% / bronze 40% /
candidate 20%) since _train has no per-sample weight. Only cutouts that extracted ok are kept.

  /home2/benson/.venvs/claudenet/bin/python 389_build_dr11_table.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _clib as C

D = C.DATA
WEIGHT = {"gold": 1.0, "silver": 0.7, "bronze": 0.4, "candidate": 0.2}
SEED = 2026


def ok_ids(sub):
    idx = pd.read_parquet(D / "_dr11shards" / sub / "index.parquet")
    return set(idx[idx.ok].row_id.astype(str))


def main():
    # positives: tier from 384 manifest, keep only extracted-ok, subsample by tier weight
    pman = pd.read_parquet(D / "v3" / "dr11s_positive_manifest.parquet").astype({"row_id": str})
    pman = pman[pman.row_id.isin(ok_ids("positives"))].copy()
    parts = []
    for tier, w in WEIGHT.items():
        t = pman[pman.tier == tier]
        parts.append(t if w >= 1.0 else t.sample(frac=w, random_state=SEED))
    pos = pd.concat(parts, ignore_index=True)
    pos = pd.DataFrame({"row_id": pos.row_id, "label": 1, "RA": pos.RA, "DEC": pos.DEC,
                        "fits_dir": str(D / "cutouts_fits_dr11_pos"), "tier": pos.tier})

    def negs(sub, fdir, tag):
        ids = sorted(ok_ids(sub))
        return pd.DataFrame({"row_id": ids, "label": 0, "RA": np.nan, "DEC": np.nan,
                             "fits_dir": str(D / fdir), "tier": tag})

    neg = negs("negatives", "cutouts_fits_dr11_neg", "random")
    hard = negs("hardneg", "cutouts_fits_dr11_hardneg", "hard")

    df = pd.concat([pos, neg, hard], ignore_index=True)
    # stratified 85/15 train/val by label
    df["split"] = "train"
    val = df.groupby("label", group_keys=False).sample(frac=0.15, random_state=7).index
    df.loc[val, "split"] = "val"

    out = D / "member_dr11_train.parquet"
    df[["row_id", "label", "RA", "DEC", "fits_dir", "split", "tier"]].to_parquet(out, index=False)
    npos = int((df.label == 1).sum()); nneg = int((df.label == 0).sum())
    print(f"[389] DR11 table: {len(df):,} rows  pos={npos:,} neg={nneg:,} (1:{nneg/npos:.1f})")
    print(f"[389] positives by tier: {pos.tier.value_counts().to_dict()}")
    print(f"[389] negatives: random={int((df.tier=='random').sum()):,} hard={int((df.tier=='hard').sum()):,}")
    print(f"[389] split: train={int((df.split=='train').sum()):,} val={int((df.split=='val').sum()):,}")
    print(f"[389] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
