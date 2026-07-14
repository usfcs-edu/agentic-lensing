#!/usr/bin/env python3
"""Assemble the human-baseline results table for DESI-resolution lens grading.

Folds four evidence classes into one summary (the parity target the machine must meet):
  1. SAME-TEAM, SAME-PROTOCOL: Paper II two-grader statistics
     (outputs/parity_intergrader_paper2.csv, from parity/intergrader_stats.py)
  2. CROSS-TEAM: QWK between independent expert teams on the same candidates
     (outputs/human_ceiling.csv, from eval/human_ceiling.py)
  3. LITERATURE: published human-reliability numbers at ground-based resolution
     (constants below, with references)
  4. TRUTH-REFERENCED: grade-stratified confirmation purity
     (outputs/parity_grade_purity.csv, from parity/grade_purity_table.py)

Provenance note (do NOT use as a human anchor): claudenet campaign
consensus_full_737.csv columns my_grade/first_grade are BOTH machine grades (the visual
workflow's first pass and its skeptic-gated final; claudenet/campaign/230_collect_visual.py)
— an earlier internal note mislabeled them as two human passes.

  python lensjudge/parity/human_baseline_summary.py

Output: outputs/human_baseline_summary.csv + a markdown block printed for the site report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"

# (statistic, value, ci, n, scope, source)
LITERATURE = [
    ("intra-rater repeat consistency", "73% identical", "", "3,797 repeats / 55 experts",
     "DES-quality sims+real", "Rojas et al. 2023, arXiv:2301.03670"),
    ("team score error vs truth (marginal systems)", "0.17 (2 graders) / 0.12 (4) / 0.07 (8)", "",
     "1,489 images", "DES-quality", "Rojas et al. 2023"),
    ("zero score dispersion among 5 graders", "30% of systems", "", "blind set",
     "HSC ground-based", "Canameras et al. 2021 (HOLISMOKES VI), arXiv:2107.07829"),
    ("panel decisive fraction (10 experts/object)", "~38% of candidates", "", "Q1 engine",
     "Euclid 0.1\" space", "Euclid Collab. 2025, arXiv:2503.15324"),
    ("cross-team A-candidate acceptance", "25.8% of Stein+22 grade-A got A/B", "", "DR9 same imaging",
     "DESI Legacy", "Storfer et al. 2024, ApJS 274:16"),
    ("confirmed lenses missed collectively", "11/11 SLACS in DES judged non-lens", "", "",
     "DES-quality", "Rojas et al. 2023"),
]


def main() -> None:
    rows = []

    p2 = OUT / "parity_intergrader_paper2.csv"
    if p2.exists():
        d = pd.read_csv(p2).set_index("statistic")
        def g(k):
            r = d.loc[k]
            return f"{r.value}", f"[{r.ci_lo}, {r.ci_hi}]"
        for stat, label in [
            ("exact_score_agree", "P(two graders give identical 1-4 score)"),
            ("sym_qwk_scores", "two-grader QWK (symmetrized, scores 1-4)"),
            ("qwk_grader_vs_consensus", "individual grader vs published consensus QWK (UPPER bound)"),
            ("exact_agree_within_A", "both graders at 4 among published grade-A"),
            ("p_one_grader_reject", "one grader at reject level among accepted"),
        ]:
            v, ci = g(stat)
            rows.append({"class": "same-team (Paper II, n=1,310)", "statistic": label,
                         "value": v, "ci": ci,
                         "source": "Huang+2021 VizieR Score/delSc; parity/intergrader_stats.py"})

    hc = OUT / "human_ceiling.csv"
    if hc.exists():
        d = pd.read_csv(hc)
        for _, r in d[d.comparison.str.startswith("HUMAN")].iterrows():
            rows.append({"class": "cross-team (same candidates)", "statistic": r.comparison,
                         "value": f"QWK {r.qwk}", "ci": f"[{r.qwk_lo}, {r.qwk_hi}] n={r.n}",
                         "source": "eval/human_ceiling.py"})

    for stat, val, ci, n, scope, src in LITERATURE:
        rows.append({"class": f"literature ({scope})", "statistic": stat, "value": val,
                     "ci": (f"{ci} {n}".strip()), "source": src})

    gp = OUT / "parity_grade_purity.csv"
    if gp.exists():
        d = pd.read_csv(gp)
        for _, r in d.iterrows():
            if r.decided and r.decided > 0:
                rows.append({"class": "truth-referenced (curated follow-up)",
                             "statistic": f"grade-{r.grade} purity among decided",
                             "value": f"{r.purity_decided} ({r.confirmed}/{r.decided})",
                             "ci": f"[{r.purity_ci_lo}, {r.purity_ci_hi}] Jeffreys",
                             "source": "parity/grade_purity_table.py (selection-biased, see caveats)"})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "human_baseline_summary.csv", index=False)
    print("=== HUMAN BASELINE for A/B/C lens grading at DESI resolution ===\n")
    print(df.to_string(index=False))
    print(f"\nsaved {OUT / 'human_baseline_summary.csv'}\n")
    print("markdown block for the site report:\n")
    cols = list(df.columns)
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
