#!/usr/bin/env python3
"""residual_ab_analysis.py — deterministic pre-step for the residual A/B (old vs new residual).

Reads the K-replicate prediction parquets for each arm (legacy vs new residual), merges with the
frozen manifest, and computes the paired, stratified comparison the analysis workflow interprets:
  - per-arm metrics: ROC-AUC + recovery@FPR (benchmark A lens-vs-random, B lens-vs-graded-D),
    grade-based FP-rate, mean p_lens by truth grade -- overall and per stratum (source, grade,
    tractor_type);
  - paired deltas: McNemar on the lens/non-lens decision flips (positives = recall flips,
    negatives = FP flips), p_lens shift;
  - replicate noise: within-arm std of p_lens across replicates vs the between-arm |delta|;
  - a flips table (candidates whose decision or modal grade changed), with each arm's p_lens and
    a representative rationale, for qualitative classification.

Usage:
  python lensjudge/eval/residual_ab_analysis.py \
     --old "lensjudge/outputs/preds_pilot_legacy_r*.parquet" \
     --new "lensjudge/outputs/preds_pilot_new_r*.parquet" \
     --manifest lensjudge/outputs/manifest_pilot_ok.csv \
     --out-json lensjudge/outputs/residual_ab_report.json \
     --out-flips lensjudge/outputs/residual_ab_flips.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

LENS_GRADES = ("A", "B", "C")


def _is_lens(g):
    return str(g).strip().upper() in LENS_GRADES


def _load_arm(pattern):
    """Concat replicate parquets; return long df + per-candidate aggregate (mean p_lens, modal grade)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquets match {pattern}")
    long = pd.concat([pd.read_parquet(f).assign(_rep=i) for i, f in enumerate(files)], ignore_index=True)
    long = long[long["parse_ok"] == True].copy()  # noqa: E712
    agg = (long.groupby("name")
           .agg(p_lens=("p_lens", "mean"),
                p_lens_std=("p_lens", "std"),
                grade_pred=("grade_pred", lambda s: Counter(s.dropna()).most_common(1)[0][0] if len(s.dropna()) else None),
                n_rep=("p_lens", "size"),
                rationale=("rationale", "first"))
           .reset_index())
    agg["p_lens_std"] = agg["p_lens_std"].fillna(0.0)
    return long, agg, len(files)


def _recovery_at_fpr(y_islens, score, fpr_target):
    if len(np.unique(y_islens)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_islens, score)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(idx, 0)])


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 and pd.notna(s).all() else None


def _arm_metrics(df):
    """df: per-candidate, merged with manifest (cols: source, grade_truth, tractor_type, p_lens, grade_pred)."""
    df = df.copy()
    df["is_lens"] = df["grade_truth"].map(_is_lens)
    df["pred_lens"] = df["grade_pred"].map(_is_lens)
    pos = df[df["source"] == "graded"]
    out = {"n": int(len(df))}
    for bench, neg_src in (("A_lens_vs_random", "random_neg"), ("B_lens_vs_gradeD", "graded_D")):
        neg = df[df["source"] == neg_src]
        sub = pd.concat([pos, neg])
        if len(neg):
            out[bench] = {
                "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                "auc": _auc(sub["is_lens"].values, sub["p_lens"].values),
                "recovery@0.1FPR": _recovery_at_fpr(sub["is_lens"].values, sub["p_lens"].values, 0.1),
                "fp_rate_grade": float(neg["pred_lens"].mean()),   # negatives predicted lens
            }
    # FP-rate per negative source + per tractor_type (the morphology stratification)
    negs = df[df["source"].isin(["random_neg", "graded_D"])]
    out["fp_rate_by_source"] = {s: float(g["pred_lens"].mean()) for s, g in negs.groupby("source")}
    out["fp_rate_by_tractor"] = {str(t): {"n": int(len(g)), "fp_rate": float(g["pred_lens"].mean())}
                                 for t, g in negs.groupby("tractor_type") if len(g) >= 3}
    # recall + mean p_lens by truth grade
    out["by_truth_grade"] = {g: {"n": int(len(s)),
                                 "mean_p_lens": float(s["p_lens"].mean()),
                                 "recall_lens": float(s["pred_lens"].mean())}
                             for g, s in df.groupby("grade_truth")}
    # gold (spectroscopic anchor)
    gold = df[df["source"] == "gold"]
    if len(gold):
        out["gold"] = {"n": int(len(gold)), "mean_p_lens": float(gold["p_lens"].mean()),
                       "recall_lens": float(gold[gold["is_lens"]]["pred_lens"].mean()) if gold["is_lens"].any() else None}
    return out


def _mcnemar(b, c):
    """Discordant-pair McNemar chi2 (b, c = the two off-diagonal counts)."""
    if b + c == 0:
        return {"b": int(b), "c": int(c), "chi2": 0.0, "p_approx": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)   # continuity-corrected
    # 1-dof chi2 survival ~ exp(-chi2/2) is a rough p; report chi2 + the simple approx
    from math import exp
    return {"b": int(b), "c": int(c), "chi2": float(chi2), "p_approx": float(exp(-chi2 / 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="glob for legacy-arm replicate parquets")
    ap.add_argument("--new", required=True, help="glob for new-arm replicate parquets")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-flips", required=True)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest).astype({"name": str})
    keep = ["name", "source", "grade_truth", "tractor_type", "binary_label"]
    man = man[[c for c in keep if c in man.columns]]

    long_o, agg_o, ko = _load_arm(args.old)
    long_n, agg_n, kn = _load_arm(args.new)
    for a in (agg_o, agg_n):
        a["name"] = a["name"].astype(str)
    df_o = agg_o.merge(man, on="name", how="inner")
    df_n = agg_n.merge(man, on="name", how="inner")

    rep = {"k_replicates": {"legacy": ko, "new": kn},
           "n_candidates": int(len(df_o)),
           "legacy": _arm_metrics(df_o),
           "new": _arm_metrics(df_n)}

    # paired comparison (same candidates)
    j = df_o[["name", "source", "grade_truth", "tractor_type", "p_lens", "p_lens_std", "grade_pred"]].merge(
        df_n[["name", "p_lens", "p_lens_std", "grade_pred"]], on="name", suffixes=("_old", "_new"))
    j["lens_old"] = j["grade_pred_old"].map(_is_lens)
    j["lens_new"] = j["grade_pred_new"].map(_is_lens)
    j["truth_lens"] = j["grade_truth"].map(_is_lens)
    j["dp_lens"] = j["p_lens_new"] - j["p_lens_old"]

    # McNemar on the lens/non-lens decision, split by truth (positives=recall, negatives=FP)
    neg = j[~j["truth_lens"]]
    pos = j[j["truth_lens"]]
    # negatives: old-said-lens & new-said-not (FP fixed) vs old-not & new-lens (FP introduced)
    rep["paired"] = {
        "negatives_FP_flips": _mcnemar(b=int(((neg.lens_old) & (~neg.lens_new)).sum()),    # FP fixed by new
                                       c=int(((~neg.lens_old) & (neg.lens_new)).sum())),    # FP introduced by new
        "positives_recall_flips": _mcnemar(b=int(((~pos.lens_old) & (pos.lens_new)).sum()),  # recall gained
                                           c=int(((pos.lens_old) & (~pos.lens_new)).sum())),  # recall lost
        "mean_dp_lens_overall": float(j["dp_lens"].mean()),
        "mean_dp_lens_negatives": float(neg["dp_lens"].mean()),
        "mean_dp_lens_positives": float(pos["dp_lens"].mean()),
    }
    # replicate-noise floor vs between-arm signal
    within = float(pd.concat([j["p_lens_std_old"], j["p_lens_std_new"]]).mean())
    rep["replicate_noise"] = {"mean_within_arm_p_lens_std": within,
                              "mean_abs_between_arm_dp_lens": float(j["dp_lens"].abs().mean()),
                              "signal_exceeds_noise": bool(j["dp_lens"].abs().mean() > within)}

    # flips table: decision changed OR modal grade changed
    flips = j[(j["lens_old"] != j["lens_new"]) | (j["grade_pred_old"] != j["grade_pred_new"])].copy()
    flips = flips.merge(agg_o[["name", "rationale"]].rename(columns={"rationale": "rationale_old"}), on="name")
    flips = flips.merge(agg_n[["name", "rationale"]].rename(columns={"rationale": "rationale_new"}), on="name")
    flips = flips.sort_values("dp_lens")
    flips_cols = ["name", "source", "grade_truth", "tractor_type", "grade_pred_old", "grade_pred_new",
                  "p_lens_old", "p_lens_new", "dp_lens", "rationale_old", "rationale_new"]
    flips[flips_cols].to_csv(args.out_flips, index=False)
    rep["n_flips"] = int(len(flips))

    with open(args.out_json, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in ("k_replicates", "n_candidates", "paired", "replicate_noise", "n_flips")}, indent=2))
    print(f"\nwrote {args.out_json} and {args.out_flips} ({rep['n_flips']} flips)")


if __name__ == "__main__":
    main()
