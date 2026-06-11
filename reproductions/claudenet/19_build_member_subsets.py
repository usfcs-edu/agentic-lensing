#!/usr/bin/env python3
"""19_build_member_subsets.py — build the Phase-1 member training sets and the
shared evaluation manifests.

Diversity levers: each supervised member trains on ALL staged-train positives plus
a per-member BOOTSTRAP resample of the staged-train negatives (different seed) at
the baseline 1:33 ratio, with a per-member augmentation seed and a different
architecture. PU-learning guard: drop any training negative within 10" of a
published lens (so unlabeled real lenses are never used as negatives).

Shared eval manifests (held out from every member's gradient training):
  eval_val      staged val pos+neg     -> calibration + combiner fitting
  eval_testneg  staged test negatives  -> matched-FPR threshold
  eval_storfer  / eval_inchausti       -> held-out published positives

    /home2/benson/.venvs/claudenet/bin/python 19_build_member_subsets.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _clib as C

MEMBERS = [  # (name, arch, boot_seed, aug_seed, gpu, variant)
    ("shielded_A", "shielded", 1, 101, 0, None),
    ("effnet_B", "efficientnet", 2, 202, 2, "tf_efficientnetv2_s"),
    ("resnet46_C", "l18", 3, 303, 3, None),
    ("dihedral_D", "dihedral", 4, 404, 4, None),
    # strong, diverse EfficientNet-family members (DES recipe: combine diverse-but-
    # strong finders) — decorrelated via backbone + negative subset + aug seed.
    ("effnet_S2", "efficientnet", 5, 505, 2, "tf_efficientnetv2_s"),
    ("effnet_B3", "efficientnet", 6, 606, 5, "tf_efficientnet_b3"),
]


def remap(df):
    df = df.copy()
    df["fits_dir"] = df["fits_dir"].apply(lambda p: str(C.DATA / __import__("pathlib").Path(str(p)).name))
    return df


def pu_drop(neg, radius_arcsec=10.0):
    """Drop negatives within radius of any known-lens catalog position."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cats = []
    for p in C.known_lens_catalogs():
        c = pd.read_csv(p)
        cols = {x.lower(): x for x in c.columns}
        ra = cols.get("ra"); dec = cols.get("dec")
        if ra and dec:
            cats.append(c[[ra, dec]].rename(columns={ra: "RA", dec: "DEC"}))
    if not cats:
        print("[pu] no lens catalogs with RA/DEC; skipping guard")
        return neg, 0
    known = pd.concat(cats, ignore_index=True).dropna()
    kc = SkyCoord(known.RA.values * u.deg, known.DEC.values * u.deg)
    nc = SkyCoord(neg.RA.values * u.deg, neg.DEC.values * u.deg)
    idx, sep, _ = nc.match_to_catalog_sky(kc)
    keep = sep.arcsec > radius_arcsec
    return neg[keep].reset_index(drop=True), int((~keep).sum())


def main():
    split = remap(pd.read_parquet(C.DATA / "training_split_staged.parquet"))
    tr = split[split.split == "train"]
    tr_pos = tr[tr.label == 1].reset_index(drop=True)
    tr_neg = tr[tr.label == 0].reset_index(drop=True)
    tr_neg, ndrop = pu_drop(tr_neg)
    print(f"[pu] dropped {ndrop} train negatives within 10\" of a known lens "
          f"-> {len(tr_neg)} clean negs")

    val = split[split.split == "val"]
    testneg = split[(split.split == "test") & (split.label == 0)]

    # per-member bootstrap-negative training sets (+ shared val)
    for name, arch, bseed, aseed, gpu, variant in MEMBERS:
        rng = np.random.default_rng(C.SEED + bseed)
        boot = tr_neg.iloc[rng.integers(0, len(tr_neg), size=len(tr_neg))]
        dfm = pd.concat([tr_pos.assign(split="train"), boot.assign(split="train"),
                         val.assign(split="val")], ignore_index=True)
        dfm[["row_id", "label", "RA", "DEC", "fits_dir", "split"]].to_parquet(
            C.DATA / f"member_{name}_train.parquet", index=False)
        print(f"[member] {name:12s} arch={arch:12s} train={len(tr_pos)}pos+{len(boot)}neg "
              f"val={len(val)} (boot_seed={bseed} aug_seed={aseed} gpu={gpu})")

    # shared eval manifests
    val[["row_id", "label", "RA", "DEC", "fits_dir"]].to_parquet(C.DATA / "eval_val.parquet", index=False)
    testneg[["row_id", "label", "RA", "DEC", "fits_dir"]].to_parquet(C.DATA / "eval_testneg.parquet", index=False)
    for name, cut, csv in (("storfer", "cutouts_fits_candidates_storfer", "storfer2024_published_catalog.csv"),
                           ("inchausti", "cutouts_fits_candidates_inchausti", "inchausti2025_published_catalog.csv")):
        cat = pd.read_csv(C.DATA / csv)
        d = pd.DataFrame({"row_id": cat["name"], "label": 1})
        d["fits_dir"] = str(C.DATA / cut)
        d.to_parquet(C.DATA / f"eval_{name}.parquet", index=False)

    (C.DATA / "members.json").write_text(json.dumps(
        [{"name": n, "arch": a, "boot_seed": b, "aug_seed": g, "gpu": gp, "variant": v}
         for n, a, b, g, gp, v in MEMBERS], indent=2))
    print(f"[19] members + eval manifests written (val={len(val)}, testneg={len(testneg)})")


if __name__ == "__main__":
    main()
