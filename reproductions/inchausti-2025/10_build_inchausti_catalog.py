#!/usr/bin/env python3
"""
10_build_inchausti_catalog.py — Phase-5 step.

Build the published Inchausti+2025 (DR10, arXiv:2508.20087) candidate catalog
from the NeuraLens project's public per-grade Google Sheets. Crucially, this
catalog carries the THREE published per-model probabilities — ResNet,
EfficientNet, and the meta-learner — so it is both a recovery target AND a direct
validation reference for our reproduced scores. RA/Dec are parsed at full
precision from the legacysurvey viewer URL embedded in each row (the sheets have
no numeric RA/Dec column; the DESI name is only 4-decimal).

Target (paper Table 3): 811 new candidates = 90 A + 104 B + 617 C (the sheets
also list ~904 grade-D rejects, dropped here). Idempotent download with a
fall-back to previously-downloaded raw sheets.

Output:
  data/inchausti2025_published_catalog.csv
    columns: name, RA, DEC, grade, p_resnet, p_effnet, p_meta, tractor_type
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SHEETS = {  # grade -> Google Sheets CSV-export URL (NeuraLens Inchausti-2025 page)
    "A": "https://docs.google.com/spreadsheets/d/1sf1M5s_dckm84mH8GbaEA9dSAeP7EEyeOlMd1KCXvzg/export?format=csv",
    "B": "https://docs.google.com/spreadsheets/d/1FCpqMqQwKqvVUAnbrkDB9nLBBHYwQkm8bP6zPGmRa1U/export?format=csv",
    "C": "https://docs.google.com/spreadsheets/d/1pggjJvt4Icxi9OHiO-25UbWLo2GifX_ica8CjkgwrW4/export?format=csv",
}
RAW = {g: DATA / f"inchausti2025_grade{g}_raw.csv" for g in SHEETS}
OUT = DATA / "inchausti2025_published_catalog.csv"
TARGET = {"A": 90, "B": 104, "C": 617, "total": 811}
RADEC_RE = re.compile(r"ra=([\d.]+)&dec=(-?[\d.]+)")


def fetch_grade(grade: str) -> pd.DataFrame:
    raw = RAW[grade]
    try:
        import requests
        r = requests.get(SHEETS[grade], timeout=60)
        r.raise_for_status()
        raw.write_bytes(r.content)
        print(f"[fetch] grade {grade}: wrote {raw.name} ({len(r.content)} bytes)")
    except Exception as e:
        if raw.exists():
            print(f"[fetch] grade {grade}: download failed ({e}); using cached {raw.name}")
        else:
            raise
    return pd.read_csv(raw)


def main() -> None:
    frames = []
    for grade in ("A", "B", "C"):
        df = fetch_grade(grade)
        norm = {str(c).split("\n")[0].strip().lower(): c for c in df.columns}

        def col(key):
            return norm[key]

        ra, dec = [], []
        for url in df[col("name url")].astype(str):
            m = RADEC_RE.search(url)
            ra.append(float(m.group(1)) if m else float("nan"))
            dec.append(float(m.group(2)) if m else float("nan"))
        frames.append(pd.DataFrame({
            "name": df[col("name")].astype(str).str.strip(),
            "RA": ra, "DEC": dec,
            "grade": grade,
            "p_resnet": pd.to_numeric(df[col("resnet probability")], errors="coerce"),
            "p_effnet": pd.to_numeric(df[col("efficientnet probability")], errors="coerce"),
            "p_meta": pd.to_numeric(df[col("meta-learner probability")], errors="coerce"),
            "tractor_type": df[col("tractor type")].astype(str).str.strip(),
        }))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["RA", "DEC"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print(f"\n[done] wrote {OUT}  ({len(out)} candidates)")
    print("[grades]\n" + out["grade"].value_counts().reindex(["A", "B", "C"]).to_string())
    print("\n[published per-model prob medians]")
    for c in ("p_resnet", "p_effnet", "p_meta"):
        print(f"   {c}: median {out[c].median():.4f}  min {out[c].min():.4f}")
    got = {g: int((out["grade"] == g).sum()) for g in "ABC"}
    got["total"] = len(out)
    print("\n[check] vs paper Table 3:")
    for k in ("A", "B", "C", "total"):
        flag = "" if abs(got[k] - TARGET[k]) <= max(3, 0.02 * TARGET[k]) else "  <-- MISMATCH"
        print(f"   {k:>5s}: got {got[k]:>4d}  paper {TARGET[k]:>4d}{flag}")


if __name__ == "__main__":
    main()
