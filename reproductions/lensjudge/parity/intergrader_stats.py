#!/usr/bin/env python3
"""First inter-grader reliability statistics of the Huang/Storfer grading program.

Input: parity/data/vizier_huang2021_cand.csv (fetch_vizier_paper2.py). Paper II is the only
paper in the series that publishes a per-grader signal: two graders each scored 1-4, the
catalog records the average (`Score`) and absolute difference (`delSc`). The unordered pair
{Score - delSc/2, Score + delSc/2} recovers both integer scores exactly (grader IDENTITY is
lost, so all statistics below treat raters as interchangeable).

Statistics (2000-sample bootstrap CIs over candidate pairs, seed 2026, matching
eval/human_ceiling.py conventions):
  - exact score agreement P(delSc=0), one-step P(delSc=1), two-step P(delSc=2)
  - P(one grader at reject level 1 among accepted candidates)
  - symmetrized quadratic-weighted kappa on scores 1-4 (Cohen QWK on the order-doubled
    data, which equalizes marginals -- appropriate for interchangeable raters)
  - Krippendorff's alpha, interval metric, 2 coders/unit
  - letter-grade agreement (individual score -> letter: 4=A, 3=B, 2=C, 1=reject) and
    symmetrized QWK on letters
  - QWK(individual grader letter, published consensus grade) -- the agreement-endpoint
    (E1) target: how well ONE trained Huang-group grader matches the published consensus.
    NOTE: not independent (each grader is half the consensus) => upper bound.
  - per-published-grade dispersion (exact-agreement within A, B, C)

TRUNCATION CAVEAT (applies to every number): the catalog contains ACCEPTED candidates only
(Score >= 2.0). Pairs averaging below 2.0 -- including all cases where both graders rejected
and most cases where one did -- are absent. Published-catalog agreement is therefore an
UPPER bound on agreement over the full inspection pool. Letter-grade construction also
constrains per-grade dispersion mechanically (e.g. grade B can only arise from {3,3} or
{2,4}), so score-level statistics are primary.

  python lensjudge/parity/intergrader_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "vizier_huang2021_cand.csv"
OUT = HERE.parent / "outputs"
_RNG = np.random.default_rng(2026)

# individual integer score -> letter (ordinal for QWK); 1 = would-reject
SCORE_TO_ORD = {4: 3, 3: 2, 2: 1, 1: 0}  # A=3 B=2 C=1 R=0, matches GMAP ordering
GRADE_TO_ORD = {"A": 3, "B": 2, "C": 1, "D": 0}


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["pair_ok"]].copy()
    df["s_lo"] = (df["Score"] - df["delSc"] / 2).round().astype(int)
    df["s_hi"] = (df["Score"] + df["delSc"] / 2).round().astype(int)
    assert df["s_lo"].between(1, 4).all() and df["s_hi"].between(1, 4).all()
    return df


def _sym_qwk(lo: np.ndarray, hi: np.ndarray) -> float:
    """Cohen QWK on order-doubled data: raters are interchangeable, so score both
    assignments (lo,hi) and (hi,lo); doubling equalizes the marginals."""
    a = np.concatenate([lo, hi])
    b = np.concatenate([hi, lo])
    if len(np.unique(a)) < 2:
        return np.nan
    return cohen_kappa_score(a, b, weights="quadratic")


def _kripp_alpha_interval(lo: np.ndarray, hi: np.ndarray) -> float:
    """Krippendorff's alpha, interval metric, exactly 2 coders per unit."""
    d_obs = float(np.mean((lo - hi) ** 2))
    pooled = np.concatenate([lo, hi]).astype(float)
    n = len(pooled)
    # expected squared difference over all pairs of pooled values
    d_exp = (2 * n * np.var(pooled)) / (n - 1)  # == mean_{i!=j}(vi-vj)^2
    return 1.0 - d_obs / d_exp if d_exp > 0 else np.nan


def _stats(df: pd.DataFrame) -> dict:
    lo, hi = df["s_lo"].to_numpy(), df["s_hi"].to_numpy()
    d = hi - lo
    cons = df["Q"].map(GRADE_TO_ORD).to_numpy()
    llo = np.vectorize(SCORE_TO_ORD.get)(lo)
    lhi = np.vectorize(SCORE_TO_ORD.get)(hi)
    a_mask = df["Q"].to_numpy() == "A"
    b_mask = df["Q"].to_numpy() == "B"
    c_mask = df["Q"].to_numpy() == "C"
    return {
        "exact_score_agree": (d == 0).mean(),
        "one_step": (d == 1).mean(),
        "two_step": (d == 2).mean(),
        "p_one_grader_reject": (lo == 1).mean(),
        "sym_qwk_scores": _sym_qwk(lo, hi),
        "kripp_alpha_interval": _kripp_alpha_interval(lo, hi),
        "letter_exact_agree": (llo == lhi).mean(),
        "sym_qwk_letters": _sym_qwk(llo, lhi),
        "qwk_grader_vs_consensus": cohen_kappa_score(
            np.concatenate([llo, lhi]), np.concatenate([cons, cons]), weights="quadratic"
        ),
        "exact_agree_within_A": (d[a_mask] == 0).mean(),
        "exact_agree_within_B": (d[b_mask] == 0).mean(),
        "exact_agree_within_C": (d[c_mask] == 0).mean(),
    }


def main() -> None:
    df = load_pairs()
    point = _stats(df)

    boots: dict[str, list[float]] = {k: [] for k in point}
    n = len(df)
    for _ in range(2000):
        s = df.iloc[_RNG.integers(0, n, n)]
        for k, v in _stats(s).items():
            if not np.isnan(v):
                boots[k].append(v)

    rows = []
    for k, v in point.items():
        lo_ci, hi_ci = np.percentile(boots[k], [2.5, 97.5])
        rows.append({"statistic": k, "value": round(float(v), 3),
                     "ci_lo": round(float(lo_ci), 3), "ci_hi": round(float(hi_ci), 3)})
    res = pd.DataFrame(rows)

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "parity_intergrader_paper2.csv", index=False)

    print(f"\n=== Huang+2021 (Paper II) inter-grader reliability, n={n} accepted candidates ===")
    print("(two graders, integer scores 1-4, identity lost; TRUNCATED at acceptance =>")
    print(" every agreement number is an UPPER bound on full-pool agreement)\n")
    print(res.to_string(index=False))
    print(f"\ndelSc distribution: {df['delSc'].value_counts().sort_index().to_dict()}")
    print(f"score-pair table:\n{pd.crosstab(df['s_lo'], df['s_hi'])}")
    print(f"\nsaved {OUT / 'parity_intergrader_paper2.csv'}")
    print("\nContext anchors: Rojas+23 intra-rater repeat consistency 73% (55 experts);")
    print("inter-team QWK 0.29 (DESI vs SuGOHI) / 0.17 (DESI vs Euclid) [eval/human_ceiling.py].")


if __name__ == "__main__":
    main()
