#!/usr/bin/env python3
"""384_build_dr11_positive_manifest.py — turn the expanded DR11-south positive pool (383's
expanded_positives_dr11s.parquet) into the weighted positive manifest for the DR11-native
fine-tune (task D). Confidence-weighted full pool per the locked scope:

  gold 1.0 | silver 0.7 | bronze 0.4 | candidate 0.2   (image + nonlens + held-out EXCLUDED)

plus the 24 HSC-tier-2 DR11-confirmed campaign systems (dr11s_confirmed_AB.csv) as gold (1.0),
deduped at 5" against the pool. Output is the brick-less manifest (row_id, RA, DEC, label,
weight, tier, source) that 130b/111 turn into DR11-south cutouts; positions without real DR11
coverage drop out at extraction (true-footprint enforcement).

  /home2/benson/.venvs/claudenet/bin/python 384_build_dr11_positive_manifest.py
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import _clib as C

WEIGHT = {"gold": 1.0, "silver": 0.7, "bronze": 0.4, "candidate": 0.2}
CAMP = C.ROOT.parent / "dr11-campaign" / "data" / "dr11s_confirmed_AB.csv"
OUT = C.DATA / "v3" / "dr11s_positive_manifest.parquet"


def xyz(ra, dec):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def slug(s, i):
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_")[:40]
    return f"p{i:06d}_{s}" if s else f"p{i:06d}"


def main():
    pool = pd.read_parquet(C.DATA / "harvest" / "expanded_positives_dr11s.parquet")
    keep = pool[pool.tier.isin(WEIGHT) & (~pool.heldout)].copy()
    keep["weight"] = keep.tier.map(WEIGHT)

    rows = [dict(RA=float(r.ra), DEC=float(r.dec), weight=float(r.weight),
                 tier=r.tier, source=str(r.source)) for r in keep.itertuples()]

    # fold in the 24 DR11 HSC-tier-2 confirmed (gold), deduped at 5" vs the pool
    if CAMP.exists():
        conf = pd.read_csv(CAMP)
        rc = {c.lower(): c for c in conf.columns}
        ra_c, dec_c = rc.get("ra"), rc.get("dec")
        if ra_c and dec_c:
            tree = cKDTree(xyz(keep.ra, keep.dec))
            d, _ = tree.query(xyz(conf[ra_c], conf[dec_c]), k=1)
            sep = np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600
            n_add = 0
            for r, s in zip(conf.itertuples(), sep):
                if s >= 5.0:                       # genuinely new vs the pool
                    rows.append(dict(RA=float(getattr(r, ra_c)), DEC=float(getattr(r, dec_c)),
                                     weight=1.0, tier="gold", source="dr11_confirmed"))
                    n_add += 1
            print(f"[384] DR11-confirmed folded in: {n_add} new (of {len(conf)}; rest already in pool)")

    man = pd.DataFrame(rows)
    man = man[man.DEC <= 32.375].reset_index(drop=True)
    man.insert(0, "row_id", [slug(s, i) for i, s in enumerate(man.source)])
    man["label"] = 1
    man = man[["row_id", "RA", "DEC", "label", "weight", "tier", "source"]]
    man.to_parquet(OUT, index=False)

    print(f"[384] positive manifest: {len(man):,} rows -> {OUT}")
    print(f"[384] by tier: {man.tier.value_counts().to_dict()}")
    print(f"[384] weighted positive mass: {man.weight.sum():,.0f} "
          f"(gold-equiv); raw count {len(man):,}")
    print(f"[384] DEC range [{man.DEC.min():.2f}, {man.DEC.max():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
