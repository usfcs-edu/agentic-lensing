#!/usr/bin/env python3
"""Q2 arithmetic: what parity margin can the truth-anchored bench actually decide?

Reads outputs/parity_bench_arm1.csv (build_parity_bench.py) and reports:

  1. AUC of the HUMAN grade (ordinal A=3,B=2,C=1) against confirmed-vs-refuted truth
     on the decided arm-1 rows, with a stratified bootstrap CI. This is the human
     benchmark the machine must be non-inferior to (E2).
  2. The paired-DeltaAUC noise floor: Monte-Carlo SE of DeltaAUC = AUC_m - AUC_h for
     a machine of EQUAL true AUC whose scores correlate with the human grade at rho,
     using a truth-conditional binormal model calibrated to the observed human AUC.
     The one-sided non-inferiority margin decidable at alpha=0.05 with 80% power is
     delta_min ~= (z_0.95 + z_0.80) * SE = 2.49 * SE.
  3. The same for projected scenarios (more refuted negatives from the grade-agnostic
     DESI Secondary Target Program and Euclid DR1-era follow-up), so the report can
     say when a small-margin verdict becomes affordable.

Caveats printed with the table: the 8 refuted negatives dominate everything; the
binormal/rho=0.6 model is a planning device, not an analysis promise (the real Phase D
analysis is the paired bootstrap on actual machine scores).

  python lensjudge/parity/power_check.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
RNG = np.random.default_rng(2026)
Z_ALPHA, Z_BETA = norm.ppf(0.95), norm.ppf(0.80)  # one-sided 0.05, power 0.80


def human_auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 4000) -> tuple[float, float, float]:
    auc = roc_auc_score(y, s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        bi = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        if len(np.unique(y[bi])) < 2:
            continue
        vals.append(roc_auc_score(y[bi], s[bi]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, float(lo), float(hi)


def paired_dauc_se(n_pos: int, n_neg: int, auc: float, rho: float,
                   n_sim: int = 3000) -> float:
    """MC standard error of DeltaAUC for two equal-AUC predictors correlated at rho.

    Truth-conditional binormal: within each class, (h, m) ~ N(0, [[1,rho],[rho,1]]);
    positives are shifted by d' = sqrt(2) * Phi^-1(AUC) on both coordinates.
    """
    dprime = np.sqrt(2) * norm.ppf(auc)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    deltas = np.empty(n_sim)
    for i in range(n_sim):
        zp = RNG.standard_normal((n_pos, 2)) @ L.T + dprime
        zn = RNG.standard_normal((n_neg, 2)) @ L.T
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        h = np.r_[zp[:, 0], zn[:, 0]]
        m = np.r_[zp[:, 1], zn[:, 1]]
        deltas[i] = roc_auc_score(y, m) - roc_auc_score(y, h)
    return float(deltas.std(ddof=1))


def main() -> None:
    man = pd.read_csv(OUT / "parity_bench_arm1.csv")
    dec = man[(man.arm == "arm1") & (man.truth_label != "") & man.truth_label.notna()]
    y = (dec.truth_label == "lens").to_numpy(int)
    s = dec.human_ordinal.to_numpy(float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    auc, lo, hi = human_auc_ci(y, s)
    print("=== E2 human benchmark (arm1 decided rows) ===")
    print(f"  n = {len(dec)} ({n_pos} confirmed lens, {n_neg} refuted)")
    print(f"  AUC(human ordinal grade vs truth) = {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("  (only 3 score levels and 8 refutations: the human 'ROC' is 3 operating")
    print("   points, and the negative count dominates the uncertainty)")

    print("\n=== paired-DeltaAUC noise floor (equal-AUC machine, binormal, rho=0.6) ===")
    scenarios = [
        ("today: arm1 decided", n_pos, n_neg),
        ("+ STP-scale refuted (hypothetical)", n_pos, 30),
        ("+ Euclid-DR1-era refuted (hypothetical)", n_pos + 100, 60),
        ("reader-study-scale positives+negatives", 250, 100),
    ]
    rows = []
    for label, npos, nneg in scenarios:
        se = paired_dauc_se(npos, nneg, auc=min(auc, 0.95), rho=0.6)
        rows.append({"scenario": label, "n_pos": npos, "n_neg": nneg,
                     "SE(dAUC)": round(se, 4),
                     "min NI margin (80% power)": round((Z_ALPHA + Z_BETA) * se, 3)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(OUT / "parity_power_check.csv", index=False)

    print(f"\nsaved {OUT / 'parity_power_check.csv'}")
    print("\nReading: a margin of ~0.03-0.05 AUC is the reader-study standard for a")
    print("parity claim. Today's bench decides only coarse margins; it supports")
    print("ESTIMATION (CI on dAUC) and gross-difference detection. The verdict-grade")
    print("test needs the grade-agnostic STP outcomes and/or Euclid-era refutations.")
    print("Planning model caveats: binormal, rho=0.6 assumed; Phase D uses the real")
    print("paired bootstrap on actual machine scores (residual_ab_full_metrics.py).")


if __name__ == "__main__":
    main()
