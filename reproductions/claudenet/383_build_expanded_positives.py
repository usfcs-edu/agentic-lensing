#!/usr/bin/env python3
"""383_build_expanded_positives.py — merge the DR11-south confirmed-lens harvest (data/harvest/*.parquet
from the two harvest workflows) with the v1 curated positives, deduplicate at 5", classify each system
into a confidence tier, flag the held-out recall catalogs, and write the expanded DR11-south positive
pool + a provenance summary.

Tiers (per system; a cluster takes its best member's tier):
  gold      spectroscopic / confirmed system-level lens (grade A, *_spec, "confirmed", AGEL)
  silver    probable (grade B, "high")
  bronze    grade C
  candidate ungraded / "candidate" / "?" / KiDS-LinKS CNN / SIMBAD candidate otypes
  image     SIMBAD LeG/LeI = individual lensed ARC/image, not the deflector (low value for a
            galaxy-galaxy DECam finder; kept + flagged, NOT counted as a clean positive)
  nonlens   SLACS/S4TM "*-X" = graded NON-lens -> EXCLUDED from the training pool
Training-eligible = tier not in {nonlens}; clean high-confidence pool = gold|silver, non-image,
not held-out. Held-out = within 5" of a Storfer2024/Inchausti2025 recall-metric lens.

  /home2/benson/.venvs/claudenet/bin/python 383_build_expanded_positives.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import _clib as C

HARVEST = C.DATA / "harvest"
SOUTH_DEC = 32.375
MATCH_ARCSEC = 5.0
COLS = ["name", "ra", "dec", "grade", "source", "ref"]
TIER_ORDER = ["gold", "silver", "bronze", "candidate", "image", "nonlens"]
TIER_RANK = {t: len(TIER_ORDER) - i for i, t in enumerate(TIER_ORDER)}  # gold highest


def xyz(ra, dec):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def chord(arcsec):
    return 2.0 * np.sin(np.radians(arcsec / 3600.0) / 2.0)


_LETTER = re.compile(r"(?<![A-Za-z])([ABC])(?![A-Za-z])")


def classify(grade, source):
    """-> tier (see module docstring). Operates on the harvest's heterogeneous grade strings."""
    g = str(grade).strip(); gl = g.lower()
    if source == "SLACS_BELLS" and re.search(r"[:\-]x$", g, re.I):
        return "nonlens"
    if gl.startswith("simbad:leg") or gl.startswith("simbad:lei"):
        return "image"
    letters = set(_LETTER.findall(g))
    if "spec" in gl or "confirmed" in gl or source == "AGEL" or "A" in letters:
        return "gold"
    if "B" in letters or "high" in gl:
        return "silver"
    if "C" in letters:
        return "bronze"
    return "candidate"


def load_harvest():
    frames = []
    for p in sorted(glob.glob(str(HARVEST / "*.parquet"))):
        if Path(p).name.startswith("expanded_positives"):
            continue
        d = pd.read_parquet(p)
        miss = [c for c in COLS if c not in d.columns]
        assert not miss, f"{p} missing {miss} (cols={list(d.columns)})"
        frames.append(d[COLS].copy())
        print(f"[383] {Path(p).name:24s} {len(d):7,} rows")
    return pd.concat(frames, ignore_index=True)


def load_curated():
    cur = pd.read_parquet(C.DATA / "positives_curated.parquet").rename(columns={"RA": "ra", "DEC": "dec"})
    cur = cur[cur.dec <= SOUTH_DEC]
    return pd.DataFrame(dict(name=cur.name.astype(str), ra=cur.ra, dec=cur.dec,
                             grade="train:" + cur.get("grade", "?").astype(str),
                             source="curated_v1", ref="v1 curated training positive"))


def heldout_mask(df):
    pts = [pd.read_csv(C.DATA / f)[["RA", "DEC"]].rename(columns={"RA": "ra", "DEC": "dec"})
           for f in ("storfer2024_published_catalog.csv", "inchausti2025_published_catalog.csv")]
    held = pd.concat(pts, ignore_index=True).dropna()
    tree = cKDTree(xyz(held.ra, held.dec))
    d, _ = tree.query(xyz(df.ra, df.dec), k=1)
    return (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600) < MATCH_ARCSEC


def components(df):
    tree = cKDTree(xyz(df.ra, df.dec))
    pairs = tree.query_pairs(chord(MATCH_ARCSEC), output_type="ndarray")
    parent = np.arange(len(df))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(len(df))])


def main():
    df = pd.concat([load_harvest(), load_curated()], ignore_index=True)
    df = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    df = df[df.dec <= SOUTH_DEC].reset_index(drop=True)
    df["tier"] = [classify(g, s) for g, s in zip(df.grade, df.source)]
    df["tier_rank"] = df.tier.map(TIER_RANK)
    df["heldout"] = heldout_mask(df)
    df["is_curated"] = df.source == "curated_v1"
    print(f"[383] merged south rows: {len(df):,}  raw tier counts: {df.tier.value_counts().to_dict()}")

    df["cluster"] = components(df)
    # representative = best tier, then prefer a curated row
    df["_pri"] = df.tier_rank + df.is_curated.astype(int) * 0.5
    df = df.sort_values("_pri", ascending=False)
    g = df.groupby("cluster", sort=False)
    reps = g.head(1).copy()
    agg = g.agg(n_dups=("name", "size"),
                sources_all=("source", lambda s: ",".join(sorted(set(s)))),
                in_curated=("is_curated", "any"),
                heldout=("heldout", "any"),
                best_tier_rank=("tier_rank", "max")).reset_index()
    out = reps.drop(columns=["heldout", "is_curated", "tier_rank", "_pri"]).merge(agg, on="cluster")
    out["tier"] = out.best_tier_rank.map({v: k for k, v in TIER_RANK.items()})
    out = out[["name", "ra", "dec", "grade", "tier", "source", "ref", "n_dups",
               "sources_all", "in_curated", "heldout"]].reset_index(drop=True)

    elig = out[(out.tier != "nonlens") & (~out.heldout)]          # training-eligible
    netnew = elig[~elig.in_curated]
    clean = elig[(elig.tier.isin(["gold", "silver"])) & (elig.tier != "image")]  # high-confidence
    clean_new = clean[~clean.in_curated]
    rep = {
        "match_arcsec": MATCH_ARCSEC, "south_dec": SOUTH_DEC,
        "merged_rows": int(len(df)), "unique_systems": int(len(out)),
        "tier_counts": out.tier.value_counts().to_dict(),
        "nonlens_excluded": int((out.tier == "nonlens").sum()),
        "heldout_systems": int(out.heldout.sum()),
        "training_eligible": int(len(elig)),
        "training_eligible_by_tier": elig.tier.value_counts().to_dict(),
        "net_new_vs_curated": int(len(netnew)),
        "net_new_by_tier": netnew.tier.value_counts().to_dict(),
        "high_conf_pool_gold_silver": int(len(clean)),
        "high_conf_net_new": int(len(clean_new)),
        "high_conf_net_new_by_source": clean_new.source.value_counts().to_dict(),
        "curated_v1_south": int(out.in_curated.sum()),
    }
    out.to_parquet(HARVEST / "expanded_positives_dr11s.parquet", index=False)
    json.dump(rep, open(HARVEST / "expanded_positives_summary.json", "w"), indent=2, default=int)

    print(f"\n[383] unique systems (5\" dedup): {len(out):,}")
    print(f"[383] tier counts: {rep['tier_counts']}")
    print(f"[383] non-lens excluded: {rep['nonlens_excluded']}  | held-out (recall cats): {rep['heldout_systems']}")
    print(f"[383] training-eligible: {rep['training_eligible']:,} by tier {rep['training_eligible_by_tier']}")
    print(f"[383] NET-NEW vs v1 curated: {rep['net_new_vs_curated']:,} by tier {rep['net_new_by_tier']}")
    print(f"[383] HIGH-CONFIDENCE (gold+silver, non-image) pool: {rep['high_conf_pool_gold_silver']:,}  "
          f"net-new: {rep['high_conf_net_new']:,}")
    print(f"[383] high-conf net-new by source: {rep['high_conf_net_new_by_source']}")
    print(f"[383] wrote expanded_positives_dr11s.parquet + expanded_positives_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
