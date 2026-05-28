#!/usr/bin/env python3
"""
13_extract_huang2020_catalog.py — Phase 3b M3.

Parse the 342 published lens candidates from Huang+2020 (60 A + 106 B + 176 C)
out of the local PDF at:

  /raid/benson/git/agentic-lensing/papers/Huang_2020_DECaLS_lenses.pdf

The candidate IDs are encoded in the form DESI-RRR.RRRR±DD.DDDD, listed
under Figures 4 (Grade A), 5 (Grade B), and 6 (Grade C) on PDF pages
9, 10, 11. We page-scope the regex so a name appearing in two figures
(e.g., a duplicate between text and tables) is counted only once per grade.

Outputs:
  data/huang2020_published_catalog.csv  (name, RA, DEC, grade)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pypdf


HERE = Path(__file__).resolve().parent
PDF = Path("/raid/benson/git/agentic-lensing/papers/Huang_2020_DECaLS_lenses.pdf")
OUT = HERE / "data" / "huang2020_published_catalog.csv"

# Pages (1-indexed in paper, 0-indexed for pypdf) where Figure 4/5/6 list candidates.
GRADE_PAGES = {
    "A": [8],   # PDF page 9 in human counting
    "B": [9],   # page 10
    "C": [10],  # page 11
}

# DESI-RRR.RRRR-DD.DDDD or DESI-RRR.RRRR+DD.DDDD
NAME_RE = re.compile(r"DESI-(\d{3}\.\d{4})([+\-]\d{2}\.\d{4})")


def parse_grade_page(text: str) -> list[tuple[str, float, float]]:
    seen = set()
    out: list[tuple[str, float, float]] = []
    for m in NAME_RE.finditer(text):
        ra_s, dec_s = m.group(1), m.group(2)
        # Reconstruct the canonical DESI name (preserve sign on dec).
        name = f"DESI-{ra_s}{dec_s}"
        if name in seen:
            continue
        seen.add(name)
        out.append((name, float(ra_s), float(dec_s)))
    return out


def main() -> None:
    print(f"[init] reading {PDF}")
    pdf = pypdf.PdfReader(str(PDF))
    print(f"[init] {len(pdf.pages)} pages")

    rows: list[tuple[str, float, float, str]] = []
    for grade, page_idxs in GRADE_PAGES.items():
        for p_idx in page_idxs:
            text = pdf.pages[p_idx].extract_text()
            candidates = parse_grade_page(text)
            print(f"[grade {grade}] page {p_idx + 1}: extracted {len(candidates)} names")
            for name, ra, dec in candidates:
                rows.append((name, ra, dec, grade))

    df = pd.DataFrame(rows, columns=["name", "RA", "DEC", "grade"])
    # Drop any duplicates that span grades (shouldn't happen but be defensive).
    n_before = len(df)
    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    if len(df) != n_before:
        print(f"[warn] dropped {n_before - len(df)} cross-grade duplicates")

    print()
    print(f"[summary] total candidates: {len(df)}")
    print(df["grade"].value_counts().sort_index().to_string())
    print()
    print(f"[summary] RA range: [{df['RA'].min():.3f}, {df['RA'].max():.3f}]")
    print(f"[summary] DEC range: [{df['DEC'].min():.3f}, {df['DEC'].max():.3f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[done] wrote {OUT}")

    # Sanity check against the paper text
    expected = {"A": 60, "B": 106, "C": 176}
    counts = df["grade"].value_counts().to_dict()
    for grade, exp in expected.items():
        got = counts.get(grade, 0)
        flag = "OK" if got == exp else "MISMATCH"
        print(f"  [{flag}] grade {grade}: got {got}, expected {exp}")


if __name__ == "__main__":
    main()
