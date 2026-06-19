#!/usr/bin/env python3
"""residual_ab_full_metrics.py — AUC-primary, calibrated comparison for the full residual A/B.

PRIMARY endpoint: ROC-AUC (threshold-free) per arm + paired bootstrap CI on delta-AUC (new-legacy),
for benchmark A (lens vs random) and B (lens vs grade-D). Because the residual *description* mainly
moves the grader's OPERATING POINT (pilot finding), we also report recall at a MATCHED false-positive
rate (calibrated), so recall/FP differences reflect discrimination, not threshold. Secondary
(non-inferiority): grade-A and gold recall at the matched operating point.

  python lensjudge/eval/residual_ab_full_metrics.py \
     --old "lensjudge/outputs/preds_full_legacy_r*.parquet" \
     --new "lensjudge/outputs/preds_full_new_r*.parquet" \
     --manifest lensjudge/outputs/manifest_full_ok.csv \
     --out-json lensjudge/outputs/residual_ab_full_report.json
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
RNG = np.random.default_rng(2026)


def _is_lens(g):
    return str(g).strip().upper() in LENS_GRADES


def _load_arm(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquets match {pattern}")
    long = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    long = long[long["parse_ok"] == True]  # noqa: E712
    agg = (long.groupby("name").agg(
        p_lens=("p_lens", "mean"),
        grade_pred=("grade_pred", lambda s: Counter(s.dropna()).most_common(1)[0][0] if len(s.dropna()) else None),
    ).reset_index())
    agg["name"] = agg["name"].astype(str)
    return agg, len(files)


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _thr_at_fpr(y_islens, score, fpr_target):
    """p_lens threshold whose FPR on the negatives ~= fpr_target."""
    fpr, tpr, thr = roc_curve(y_islens, score)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(thr[max(idx, 0)])


def _bench(df_o, df_n, pos_src, neg_src, n_boot=2000):
    """Paired AUC + bootstrap CIs + recall at matched FPR, for one benchmark."""
    sub_o = df_o[df_o.source.isin([pos_src, neg_src])].set_index("name")
    sub_n = df_n[df_n.source.isin([pos_src, neg_src])].set_index("name")
    names = sub_o.index.intersection(sub_n.index)
    y = sub_o.loc[names, "is_lens"].values.astype(int)
    so = sub_o.loc[names, "p_lens"].values
    sn = sub_n.loc[names, "p_lens"].values
    auc_o, auc_n = _auc(y, so), _auc(y, sn)
    # paired bootstrap over candidates
    boot_o, boot_n, boot_d = [], [], []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        ao, an = _auc(y[b], so[b]), _auc(y[b], sn[b])
        boot_o.append(ao); boot_n.append(an); boot_d.append(an - ao)
    def ci(a): return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "auc_legacy": auc_o, "auc_legacy_ci": ci(boot_o),
           "auc_new": auc_n, "auc_new_ci": ci(boot_n),
           "delta_auc_new_minus_legacy": auc_n - auc_o, "delta_auc_ci": ci(boot_d),
           "delta_auc_excludes_0": bool(ci(boot_d)[0] > 0 or ci(boot_d)[1] < 0)}
    # recall at MATCHED FPR (calibrated operating point)
    for fpr_t in (0.1, 0.2):
        to = _thr_at_fpr(y, so, fpr_t); tn = _thr_at_fpr(y, sn, fpr_t)
        rec_o = float(((so >= to) & (y == 1)).sum() / max((y == 1).sum(), 1))
        rec_n = float(((sn >= tn) & (y == 1)).sum() / max((y == 1).sum(), 1))
        out[f"recall@FPR{fpr_t}"] = {"legacy": rec_o, "new": rec_n, "delta": rec_n - rec_o}
    return out


def _stratum_recall_matched(df_o, df_n, neg_src, fpr_t=0.2):
    """grade-A and gold recall at the threshold matching FPR=fpr_t on neg_src (per arm)."""
    res = {}
    for arm, df in (("legacy", df_o), ("new", df_n)):
        neg = df[df.source == neg_src]
        pos_all = df[df.is_lens]
        thr = _thr_at_fpr(pd.concat([pos_all, neg]).is_lens.values.astype(int),
                          pd.concat([pos_all, neg]).p_lens.values, fpr_t)
        for strat, mask in (("gradeA", df.grade_truth == "A"), ("gold", df.source == "gold")):
            s = df[mask & df.is_lens]
            if len(s):
                res.setdefault(strat, {})[arm] = {"n": int(len(s)),
                                                  "recall_matched": float((s.p_lens >= thr).mean()),
                                                  "mean_p_lens": float(s.p_lens.mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    man = pd.read_csv(args.manifest).astype({"name": str})[["name", "source", "grade_truth"]]
    agg_o, ko = _load_arm(args.old); agg_n, kn = _load_arm(args.new)
    df_o = agg_o.merge(man, on="name"); df_n = agg_n.merge(man, on="name")
    for d in (df_o, df_n):
        d["is_lens"] = d["grade_truth"].map(_is_lens)

    rep = {"k_replicates": {"legacy": ko, "new": kn}, "n": int(len(df_o)),
           "primary_benchmark_A_lens_vs_random": _bench(df_o, df_n, "graded", "random_neg"),
           "benchmark_B_lens_vs_gradeD": _bench(df_o, df_n, "graded", "graded_D"),
           "non_inferiority_recall_matched_FPR0.2": _stratum_recall_matched(df_o, df_n, "random_neg", 0.2)}
    with open(args.out_json, "w") as f:
        json.dump(rep, f, indent=2)
    A = rep["primary_benchmark_A_lens_vs_random"]
    print(f"PRIMARY (lens vs random, n_pos={A['n_pos']} n_neg={A['n_neg']}):")
    print(f"  AUC legacy={A['auc_legacy']:.3f} {A['auc_legacy_ci']}  new={A['auc_new']:.3f} {A['auc_new_ci']}")
    print(f"  dAUC(new-legacy)={A['delta_auc_new_minus_legacy']:+.3f} CI={A['delta_auc_ci']}  excludes_0={A['delta_auc_excludes_0']}")
    print(f"  recall@FPR0.2: legacy={A['recall@FPR0.2']['legacy']:.3f} new={A['recall@FPR0.2']['new']:.3f} d={A['recall@FPR0.2']['delta']:+.3f}")
    ni = rep["non_inferiority_recall_matched_FPR0.2"]
    for k, v in ni.items():
        if "legacy" in v and "new" in v:
            print(f"  {k} recall@matchedFPR: legacy={v['legacy']['recall_matched']:.3f}(n{v['legacy']['n']}) new={v['new']['recall_matched']:.3f}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
