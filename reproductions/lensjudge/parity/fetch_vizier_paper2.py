#!/usr/bin/env python3
"""Fetch the Huang et al. 2021 (Paper II) VizieR candidate table WITH per-grader score spread.

Why this table matters: it is the ONLY machine-readable per-grader signal in the whole
Huang/Storfer program. Paper II's grading protocol (Sec. 3) had two graders (XH, AD) each
assign an integer score 1-4; the published catalog records the AVERAGE (`Score`, 2.0-4.0 in
0.5 steps) and the ABSOLUTE DIFFERENCE (`delSc`, 0-2). The unordered grader pair is exactly
recoverable as {Score - delSc/2, Score + delSc/2}. No other paper in the series (I, III/DR9,
IV/DR10) publishes any per-grader quantity, so this is the basis for the first inter-grader
reliability statistic of the program (see intergrader_stats.py).

Truncation caveat (carried through all downstream analysis): the catalog contains ACCEPTED
candidates only (Score >= 2.0 => grades A/B/C). Pairs that averaged below 2.0 are absent,
so every agreement statistic computed from it is an UPPER bound on full-pool agreement.

  python lensjudge/parity/fetch_vizier_paper2.py

Writes parity/data/vizier_huang2021_cand.csv and verifies row/grade counts against the
published paper (1,312 candidates) and the local reproduction catalog
(reproductions/huang-2021/data/huang2021_published_catalog.csv, which lacks delSc).
"""
from __future__ import annotations

import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = HERE.parents[2]
LOCAL_CATALOG = REPO / "reproductions" / "huang-2021" / "data" / "huang2021_published_catalog.csv"

ASU_URL = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
SOURCE = "J/ApJ/909/27/cand"  # Huang+ 2021, ApJ 909, 27 -- "Gravitational lens candidates"
EXPECTED_N = 1312  # published candidate count (Paper II)


def fetch_tsv() -> str:
    params = urllib.parse.urlencode(
        {"-source": SOURCE, "-out.max": "unlimited", "-out.all": "1"}
    )
    with urllib.request.urlopen(f"{ASU_URL}?{params}", timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_tsv(text: str) -> pd.DataFrame:
    """VizieR ASU-TSV: '#' comment header, then name/unit/dash-rule rows, then data."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    header = lines[0].split("\t")
    body = [ln for ln in lines[1:] if not set(ln.replace("\t", "")) <= {"-", " "}]
    # drop the units row (first body row has no recno integer)
    rows = [ln.split("\t") for ln in body]
    rows = [r for r in rows if r[0].strip().isdigit()]
    df = pd.DataFrame(rows, columns=header).apply(lambda s: s.str.strip())
    for col in ("Score", "Prob", "zsp", "zph", "e_zph", "gmag", "rmag", "zmag", "_RA", "_DE"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delSc"] = pd.to_numeric(df["delSc"], errors="coerce").astype("Int64")
    return df


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    df = parse_tsv(fetch_tsv())

    # --- verification 1: row count and grade counts vs the published paper ---
    assert len(df) == EXPECTED_N, f"expected {EXPECTED_N} rows, got {len(df)}"
    grade_counts = df["Q"].value_counts().to_dict()
    print(f"rows: {len(df)}  grades: {grade_counts}")
    # Paper II: 1,312 candidates. The VizieR column note says A has 216 occurrences.
    assert grade_counts.get("A") == 216, f"grade-A count {grade_counts.get('A')} != 216 (VizieR note)"

    # --- verification 2: Score/grade agree with the local reproduction catalog ---
    if LOCAL_CATALOG.exists():
        local = pd.read_csv(LOCAL_CATALOG)
        merged = df.merge(local, left_on="Name", right_on="name", how="inner", suffixes=("_viz", "_loc"))
        n_match = len(merged)
        score_ok = np.isclose(merged["Score"], pd.to_numeric(merged["score"], errors="coerce")).mean()
        grade_ok = (merged["Q"] == merged["grade"].astype(str).str.strip().str.upper()).mean()
        print(f"local-catalog join: {n_match}/{len(df)} matched; Score agree {score_ok:.3f}; grade agree {grade_ok:.3f}")
        assert n_match >= EXPECTED_N * 0.99, "poor name join vs local catalog"
        assert score_ok > 0.999 and grade_ok > 0.999, "Score/grade mismatch vs local catalog"
    else:
        print(f"WARNING: local catalog not found at {LOCAL_CATALOG}; skipped cross-check")

    # --- verification 3: grader-pair recovery is well-formed ---
    # Score = mean of two integer scores 1-4  =>  2*Score and delSc must have equal parity.
    # Known exceptions (data-entry glitches in the published table, impossible combinations):
    # DESI-094.5639+50.3059 (Score 3.5, delSc 0) and DESI-227.9362+06.5090 (Score 2.0, delSc 1).
    df["pair_ok"] = (2 * df["Score"]).round().astype(int) % 2 == df["delSc"].astype(int) % 2
    n_bad = int((~df["pair_ok"]).sum())
    assert n_bad <= 2, f"{n_bad} parity violations (expected <=2 known glitches): " + ", ".join(
        df.loc[~df["pair_ok"], "Name"]
    )
    ok = df[df["pair_ok"]]
    s1 = ok["Score"] - ok["delSc"].astype(float) / 2
    s2 = ok["Score"] + ok["delSc"].astype(float) / 2
    assert ((s1 >= 1) & (s2 <= 4)).all(), "recovered grader scores outside 1-4"
    print(
        f"grader pairs recover cleanly for {len(ok)}/{len(df)} rows "
        f"({n_bad} flagged pair_ok=False); delSc dist = {ok['delSc'].value_counts().sort_index().to_dict()}"
    )

    out = DATA / "vizier_huang2021_cand.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
