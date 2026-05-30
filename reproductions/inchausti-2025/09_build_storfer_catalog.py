#!/usr/bin/env python3
"""
09_build_storfer_catalog.py — Phase-5 step.

Build the published Storfer+2024 (DR9, ApJS 274:16) candidate catalog. The paper
released no VizieR table; the authoritative machine-readable source is the
NeuraLens project's public Google Sheet (combined Drive CSV). We download it
(idempotent; falls back to a previously-downloaded copy) and normalise to the
same schema as huang-2021/13's huang2021_published_catalog.csv.

Target (paper Table 2): 1,895 candidates = 115 A + 526 B + 1254 C (the catalog
also lists ~1,865 grade-D rejects, which we drop). Storfer used the two Paper-II
models (L18 + shielded) but publishes a single `probability` column.

Output:
  data/storfer2024_published_catalog.csv
    columns: name, RA, DEC, grade, probability, region, tractor_type, ref
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Combined Drive CSV (all grades incl. D) — NeuraLens Storfer-et-al-2022 page.
DRIVE_CSV = "https://drive.google.com/uc?export=download&id=1Lgx_bRXnVHLMeTF86sRJzMbzKpTxpnCN"
RAW = DATA / "storfer2024_drive_all.csv"
OUT = DATA / "storfer2024_published_catalog.csv"
TARGET = {"A": 115, "B": 526, "C": 1254, "total": 1895}


def fetch() -> pd.DataFrame:
    try:
        import requests
        print(f"[fetch] downloading Storfer Drive CSV ...")
        r = requests.get(DRIVE_CSV, timeout=60)
        r.raise_for_status()
        RAW.write_bytes(r.content)
        print(f"[fetch] wrote {RAW} ({len(r.content)} bytes)")
    except Exception as e:
        if RAW.exists():
            print(f"[fetch] download failed ({e}); using cached {RAW.name}")
        else:
            raise
    return pd.read_csv(RAW)


def main() -> None:
    df = fetch()
    df.columns = [str(c).split("\n")[0].strip().lower() for c in df.columns]

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        raise KeyError(names)

    out = pd.DataFrame({
        "name": df[col("name")].astype(str).str.strip(),
        "RA": pd.to_numeric(df[col("ra")], errors="coerce"),
        "DEC": pd.to_numeric(df[col("dec")], errors="coerce"),
        "grade": df[col("grade")].astype(str).str.strip().str.upper(),
        "probability": pd.to_numeric(df[col("probability")], errors="coerce"),
        "region": df[col("region")].astype(str).str.strip(),
        "tractor_type": df[col("type")].astype(str).str.strip(),
        "ref": df[col("ref")].astype(str).str.strip() if "ref" in df.columns else "",
    })
    # Keep only the published A/B/C candidates (drop grade-D rejects + bad coords).
    out = out[out["grade"].isin(["A", "B", "C"])].dropna(subset=["RA", "DEC"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print(f"\n[done] wrote {OUT}  ({len(out)} candidates)")
    print("[grades]\n" + out["grade"].value_counts().reindex(["A", "B", "C"]).to_string())
    print("[region]\n" + out["region"].value_counts().to_string())
    got = {g: int((out["grade"] == g).sum()) for g in "ABC"}
    got["total"] = len(out)
    print("\n[check] vs paper Table 2:")
    for k in ("A", "B", "C", "total"):
        flag = "" if abs(got[k] - TARGET[k]) <= max(3, 0.02 * TARGET[k]) else "  <-- MISMATCH"
        print(f"   {k:>5s}: got {got[k]:>4d}  paper {TARGET[k]:>4d}{flag}")


if __name__ == "__main__":
    main()
