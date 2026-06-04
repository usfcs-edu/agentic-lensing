#!/usr/bin/env python3
"""Score a LensJudge predictions parquet -> LensBench-VI metrics.

All agreement numbers are CONSENSUS-REFERENCED, NO HUMAN CEILING: the labels are a
single 2-senior-author consensus grade (A/B/C) plus Grade-D human-rejects (D), with
NO per-rater data — so we report agreement, not "accuracy", and cannot place it
against a human-vs-human ceiling (see the plan's §1 gap).

Tasks reported:
  (i)   binary lens(A/B/C) vs non-lens(D): ROC-AUC + recovery @1%/0.1% FPR (vs the
        group's meta 90.8%/96.8%), by grade and region.
  (ii)  ordinal A/B/C/D vs consensus: quadratic-weighted kappa + confusion + adjacent.
  (iii) calibration of p_lens: ECE + Brier.
  (iv)  agent-vs-CNN (p_meta) agreement.

  python lensjudge/eval/score.py --preds outputs/preds.parquet [--report outputs/lensbench.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = ["A", "B", "C", "D"]
_ORD = {g: i for i, g in enumerate(LABELS)}


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    from sklearn.metrics import cohen_kappa_score
    t = [_ORD.get(x) for x in y_true]
    p = [_ORD.get(x) for x in y_pred]
    keep = [(a, b) for a, b in zip(t, p) if a is not None and b is not None]
    if len(keep) < 2:
        return float("nan")
    a, b = zip(*keep)
    return float(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2, 3]))


def recovery_at_fpr(y_islens, score, fpr_target=0.01):
    """TPR (lens recovery) at a fixed false-positive rate on the non-lens class."""
    from sklearn.metrics import roc_curve
    m = ~(pd.isna(score))
    y, s = np.asarray(y_islens)[m], np.asarray(score)[m]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx]), float(thr[idx])


def ece(y, p, bins=10):
    m = ~pd.isna(p)
    y, p = np.asarray(y)[m].astype(float), np.asarray(p)[m].astype(float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if sel.sum():
            e += sel.sum() / len(y) * abs(y[sel].mean() - p[sel].mean())
    return float(e)


def score(preds: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score, cohen_kappa_score
    out = {}
    n = len(preds)
    ok = preds[preds["parse_ok"] == True].copy()  # noqa: E712
    out["n"] = n
    out["parse_rate"] = round(len(ok) / max(1, n), 4)
    out["mean_cost_usd"] = round(float(preds["cost_usd"].mean()), 4)
    out["mean_wall_s"] = round(float(preds["wall_s"].mean()), 2)

    ok["is_lens"] = ok["grade_truth"].isin(["A", "B", "C"]).astype(int)
    ok["pred_is_lens"] = ok["grade_pred"].isin(["A", "B", "C"]).astype(int)

    # (i) binary
    if ok["is_lens"].nunique() > 1 and ok["p_lens"].notna().any():
        try:
            out["binary_auc"] = round(float(roc_auc_score(ok["is_lens"], ok["p_lens"].fillna(0))), 4)
        except Exception:
            out["binary_auc"] = None
        for fpr in (0.01, 0.001):
            tpr, thr = recovery_at_fpr(ok["is_lens"], ok["p_lens"], fpr)
            out[f"recovery@{fpr:g}FPR"] = None if np.isnan(tpr) else round(tpr, 4)
        # per-grade recovery (fraction graded A/B/C by the agent), per region
        out["recovery_by_grade"] = {g: round(float((ok[ok.grade_truth == g]["pred_is_lens"]).mean()), 3)
                                    for g in ["A", "B", "C"] if (ok.grade_truth == g).any()}
        if "region" in ok:
            out["mean_p_lens_by_region"] = {r: round(float(d["p_lens"].mean()), 3)
                                            for r, d in ok.groupby("region") if len(d)}

    # (ii) ordinal
    out["qwk_vs_consensus"] = round(quadratic_weighted_kappa(ok["grade_truth"], ok["grade_pred"]), 4)
    valid = ok[ok["grade_truth"].isin(LABELS) & ok["grade_pred"].isin(LABELS)]
    if len(valid):
        ct = pd.crosstab(valid["grade_truth"], valid["grade_pred"]).reindex(
            index=LABELS, columns=LABELS, fill_value=0)
        out["confusion"] = ct.to_dict()
        d = np.array([abs(_ORD[a] - _ORD[b]) for a, b in zip(valid.grade_truth, valid.grade_pred)])
        out["exact_acc"] = round(float((d == 0).mean()), 4)
        out["adjacent_acc"] = round(float((d <= 1).mean()), 4)

    # (iii) calibration
    if ok["p_lens"].notna().any() and ok["is_lens"].nunique() > 1:
        out["ece_p_lens"] = round(ece(ok["is_lens"], ok["p_lens"]), 4)
        out["brier_p_lens"] = round(float(((ok["p_lens"].fillna(0) - ok["is_lens"]) ** 2).mean()), 4)

    # (iv) agent vs CNN p_meta
    if "p_meta" in ok and ok["p_meta"].notna().any():
        cnn_lens = (ok["p_meta"] >= 0.5).astype(int)  # nominal; real op-point set elsewhere
        try:
            out["agent_vs_cnn_kappa"] = round(float(cohen_kappa_score(ok["pred_is_lens"], cnn_lens)), 4)
        except Exception:
            pass
        out["cnn_mean_p_meta_by_truth"] = {g: round(float(ok[ok.grade_truth == g]["p_meta"].mean()), 3)
                                           for g in LABELS if (ok.grade_truth == g).any()}

    # escalation
    if "escalate" in ok:
        out["escalation_rate"] = round(float(ok["escalate"].mean()), 3)
        out["escalation_by_grade"] = {g: round(float(ok[ok.grade_truth == g]["escalate"].mean()), 3)
                                      for g in LABELS if (ok.grade_truth == g).any()}
    return out


def format_report(out: dict) -> str:
    import json
    lines = ["# LensBench-VI results", "",
             "> **CONSENSUS-REFERENCED, NO HUMAN CEILING** — labels are a single "
             "2-author consensus (A/B/C) + Grade-D human-rejects; no per-rater data, "
             "so these are agreement metrics, not accuracy, and have no human-vs-human "
             "ceiling for comparison.", "",
             f"- n={out.get('n')}  parse_rate={out.get('parse_rate')}  "
             f"mean_cost=${out.get('mean_cost_usd')}  mean_wall={out.get('mean_wall_s')}s", ""]
    for k in ("binary_auc", "recovery@0.01FPR", "recovery@0.001FPR", "qwk_vs_consensus",
              "exact_acc", "adjacent_acc", "ece_p_lens", "brier_p_lens",
              "agent_vs_cnn_kappa", "escalation_rate"):
        if k in out:
            lines.append(f"- **{k}** = {out[k]}")
    for k in ("recovery_by_grade", "escalation_by_grade", "cnn_mean_p_meta_by_truth",
              "mean_p_lens_by_region", "confusion"):
        if k in out:
            lines.append(f"\n**{k}**\n```\n{json.dumps(out[k], indent=2)}\n```")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    preds = pd.read_parquet(args.preds)
    out = score(preds)
    rep = format_report(out)
    print(rep)
    if args.report:
        Path(args.report).write_text(rep)
        print(f"\n[written] {args.report}")


if __name__ == "__main__":
    main()
