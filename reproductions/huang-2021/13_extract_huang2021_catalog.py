#!/usr/bin/env python3
"""
13_extract_huang2021_catalog.py — Phase 4c step 1.

Build the published Huang+2021 candidate catalog from the NeuraLens release
table (data/neuralens_catalog.csv), which already carries everything we need:
  Name (DESI-RA±DEC), Score (averaged visual grade 2.0-4.0), ResNet Model
  (L18 | shielded — which of the two deployed nets flagged it), Region
  (DECaLS | MzLS), Probability, Tractor Type, g/r/z mag, Spec, Photo-z.

Letter grades follow Huang+2021 §3.5:
  A: averaged score >= 3.5
  B: averaged score == 3.0
  C: averaged score in {2.0, 2.5}

Target (paper Table 3): 1,312 total = 216 A + 199 B + 897 C, split across the
two models (~948 L18 + ~364 shielded). We report the breakdown and warn on
mismatch.

Output:
  data/huang2021_published_catalog.csv
    columns: name, RA, DEC, grade, score, resnet_model, region, probability,
             tractor_type, spec_z, photo_z
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NAME_RE = re.compile(r"DESI-(\d{3}\.\d{4})([+\-]\d{2}\.\d{4})")


def grade_of(score: float) -> str:
    if score >= 3.5:
        return "A"
    if score >= 3.0:   # exactly 3.0
        return "B"
    return "C"         # 2.0 / 2.5


def main() -> None:
    cat = pd.read_csv(DATA / "neuralens_catalog.csv")
    cat.columns = [c.strip() for c in cat.columns]

    def col(key):
        return [c for c in cat.columns if c.lower() == key][0]

    rows = []
    for _, r in cat.iterrows():
        m = NAME_RE.search(str(r[col("name")]))
        if not m:
            continue
        ra, dec = float(m.group(1)), float(m.group(2))
        try:
            score = float(r[col("score")])
        except (ValueError, TypeError):
            continue
        rows.append({
            "name": str(r[col("name")]).strip(),
            "RA": ra, "DEC": dec,
            "grade": grade_of(score),
            "score": score,
            "resnet_model": str(r[col("resnet model")]).strip().lower(),
            "region": str(r[col("region")]).strip(),
            "probability": pd.to_numeric(r[col("probability")], errors="coerce"),
            "tractor_type": str(r[col("tractor type")]).strip(),
            "spec_z": pd.to_numeric(r[col("spec")], errors="coerce"),
            "photo_z": str(r[col("photo-z")]).strip(),
        })
    df = pd.DataFrame(rows)
    out = DATA / "huang2021_published_catalog.csv"
    df.to_csv(out, index=False)

    print(f"[done] wrote {out}  ({len(df)} candidates)")
    print("\n[grades]")
    print(df["grade"].value_counts().reindex(["A", "B", "C"]).to_string())
    print("\n[model]")
    print(df["resnet_model"].value_counts().to_string())
    print("\n[region]")
    print(df["region"].value_counts().to_string())
    print("\n[grade x model]")
    print(pd.crosstab(df["grade"], df["resnet_model"]).reindex(["A", "B", "C"]).to_string())

    # Sanity vs paper Table 3.
    target = {"A": 216, "B": 199, "C": 897, "total": 1312}
    got = {g: int((df["grade"] == g).sum()) for g in "ABC"}
    got["total"] = len(df)
    print("\n[check] grade counts vs paper Table 3:")
    for k in ("A", "B", "C", "total"):
        flag = "" if abs(got[k] - target[k]) <= max(3, 0.05 * target[k]) else "  <-- MISMATCH"
        print(f"   {k:>5s}: got {got[k]:>4d}  paper {target[k]:>4d}{flag}")


if __name__ == "__main__":
    main()
