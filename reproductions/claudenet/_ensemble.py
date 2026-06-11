#!/usr/bin/env python3
"""_ensemble.py — ClaudeNet ensemble & evaluation utilities (pure numpy/sklearn).

The matched-FPR recovery arithmetic is copied VERBATIM (same quantile/threshold
logic) from inchausti-2025/22_fpr_operating_point.py so ClaudeNet numbers are
directly comparable to the reproduced baseline:

    thr = np.quantile(neg_scores, 1 - fpr)      # threshold set on held-out NEGATIVES
    recovery = (cand_scores >= thr).mean()      # TPR of held-out positives at that thr

Adds: per-member calibration (Platt/isotonic), ensemble combiners (naive average /
logistic / random forest), diversity diagnostics (score correlation + per-member
error correlation + Yule's Q), and imbalance/calibration metrics (AUPRC, ECE).
"""
from __future__ import annotations

import numpy as np


# ===== matched-FPR recovery (verbatim arithmetic from 22_fpr_operating_point.py) =====

def fpr_threshold(neg_scores, fpr: float) -> float:
    ns = np.asarray(neg_scores, dtype=np.float64)
    ns = ns[np.isfinite(ns)]
    return float(np.quantile(ns, 1.0 - fpr))


def recovery_at_fpr(neg_scores, cand_scores, fprs=(0.01, 0.001)) -> dict:
    """Recovery (TPR) of `cand_scores` at thresholds set by the (1-fpr) quantile of
    the held-out `neg_scores`. Returns {fpr: {"threshold", "recovery", "n_cand"}}.
    Identical to the inchausti Stage-C operating-point analysis."""
    cs = np.asarray(cand_scores, dtype=np.float64)
    cs = cs[np.isfinite(cs)]
    out = {}
    for fpr in fprs:
        thr = fpr_threshold(neg_scores, fpr)
        out[fpr] = {"threshold": thr,
                    "recovery": float((cs >= thr).mean()) if len(cs) else float("nan"),
                    "n_cand": int(len(cs))}
    return out


# ===== calibration ===================================================================

def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic (Platt) scaling on the logit of the raw probability."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, p, y):
        self.lr.fit(_logit(p).reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def transform(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone isotonic calibration (non-parametric)."""

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p, y):
        self.ir.fit(np.asarray(p, dtype=np.float64), np.asarray(y).astype(float))
        return self

    def transform(self, p):
        return self.ir.transform(np.asarray(p, dtype=np.float64))


def make_calibrator(kind: str):
    return {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator}[kind]()


# ===== combiners =====================================================================

def fit_combiner(kind: str, P, y):
    """Fit an N-member combiner on calibrated member probs P:(n, n_members), labels y.
    Returns a callable predict(Q:(m, n_members)) -> prob:(m,).
    `average` is the parameterless baseline; `logistic`/`rf` are the trainable combiners
    (DES 2510.23782 found tree combiners beat averaging for diverse finders)."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if kind == "average":
        return lambda Q: np.asarray(Q, dtype=np.float64).mean(axis=1)
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, C=1.0).fit(P, y)
        return lambda Q: m.predict_proba(np.asarray(Q, dtype=np.float64))[:, 1]
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=25,
                                   random_state=2026, n_jobs=-1).fit(P, y)
        return lambda Q: m.predict_proba(np.asarray(Q, dtype=np.float64))[:, 1]
    raise ValueError(f"unknown combiner {kind!r}")


# ===== diversity diagnostics =========================================================

def score_correlation(P, method: str = "pearson"):
    """Continuous correlation matrix between member SCORE vectors. P:(n, n_members)."""
    P = np.asarray(P, dtype=np.float64)
    if method == "spearman":
        from scipy.stats import spearmanr
        rho, _ = spearmanr(P)
        return np.atleast_2d(rho)
    return np.corrcoef(P.T)


def error_correlation(P, y, thr=0.5):
    """Pearson correlation of per-member 0/1 ERROR vectors at threshold `thr`.
    Lower off-diagonal => more diverse errors (the active ensemble ingredient)."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y).astype(int).reshape(-1, 1)
    err = ((P >= thr).astype(int) != y).astype(float)
    return np.corrcoef(err.T)


def q_statistic(P, y, thr=0.5):
    """Yule's Q between member correct/incorrect vectors. Q in [-1,1]; lower=more
    diverse (independent classifiers -> Q~0). P:(n, n_members)."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y).astype(int).reshape(-1, 1)
    correct = ((P >= thr).astype(int) == y)
    M = correct.shape[1]
    Q = np.eye(M)
    for i in range(M):
        for j in range(i + 1, M):
            a, b = correct[:, i], correct[:, j]
            n11 = int((a & b).sum()); n00 = int((~a & ~b).sum())
            n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
            denom = n11 * n00 + n01 * n10
            q = (n11 * n00 - n01 * n10) / denom if denom > 0 else 0.0
            Q[i, j] = Q[j, i] = q
    return Q


# ===== imbalance / calibration metrics ===============================================

def auprc(y, p) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(average_precision_score(y[ok], p[ok]))


def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error (equal-width bins)."""
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p); y, p = y[ok], p[ok]
    if len(p) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e, N = 0.0, len(p)
    for i in range(n_bins):
        hi = p <= bins[i + 1] if i == n_bins - 1 else p < bins[i + 1]
        m = (p >= bins[i]) & hi
        if m.sum() == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * m.sum() / N
    return float(e)
