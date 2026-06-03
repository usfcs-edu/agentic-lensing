"""
Evaluation metrics for the AION-1 downstream tasks.

All paper experiments reduce to one of: R^2 (property regression), accuracy
(morphology), residual standard deviation (APOGEE), IoU (segmentation), or
nDCG@10 (retrieval). Kept dependency-light (numpy + sklearn) so the same
definitions are reused across every probe/retrieval script.
"""

from __future__ import annotations

import numpy as np


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, 1 - SS_res/SS_tot (per-column then mean
    is done by the caller; this is single-target)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    y_true = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    y_pred = np.atleast_2d(np.asarray(y_pred, dtype=np.float64))
    if y_true.shape[0] != y_pred.shape[0]:
        y_true, y_pred = y_true.T, y_pred.T
    return [r2(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def residual_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard deviation of (pred - truth); the APOGEE comparison metric."""
    return float(np.std(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)))


def iou(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Binary intersection-over-union for one (or a batch of) mask(s)."""
    p = np.asarray(pred_mask) >= threshold
    t = np.asarray(true_mask) >= 0.5
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / union) if union > 0 else float("nan")


def mean_iou(pred_masks: np.ndarray, true_masks: np.ndarray, threshold: float = 0.5) -> float:
    """Per-image IoU averaged over the batch (the convention we document)."""
    vals = [iou(p, t, threshold) for p, t in zip(pred_masks, true_masks)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def ndcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    """nDCG@k for a single ranked list. `relevances` is the binary/graded
    relevance of the top results in *ranked order* (most similar first).
    Uses the standard (2^rel - 1) gain; for binary gain this reduces to rel."""
    relevances = np.asarray(relevances, dtype=np.float64)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    dcg = np.sum((2 ** relevances - 1) * discounts)
    ideal = np.sort(relevances)[::-1]
    idcg = np.sum((2 ** ideal - 1) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_ndcg_at_k(ranked_relevances: list[np.ndarray], k: int = 10) -> float:
    """Average nDCG@k over a set of queries (each a ranked relevance list)."""
    vals = [ndcg_at_k(r, k) for r in ranked_relevances]
    return float(np.mean(vals)) if vals else 0.0
