#!/usr/bin/env python3
"""make_residual_ab_figure.py — results figure for papers/residual.tex (the grading A/B).

Three panels from the committed A/B artifacts:
  (a) per-arm ROC-AUC with bootstrap 95% CIs (lens-vs-random and lens-vs-grade-D) -> overlapping
      CIs = discrimination-neutral;
  (b) negative false-positive rate, RAW (hot grade) vs CALIBRATED (FPR-matched threshold), per arm
      -> the new arm's flood is removed by calibration;
  (c) grade-A recall, raw vs calibrated -> non-inferiority at the matched operating point.

Reads outputs/residual_ab_full_report.json (AUC + CIs) and recomputes the raw/calibrated decisions
from the prediction parquets + eval/residual_op_point_{new,legacy}.json (reusing eval/calibrate.py).

  ~/.venvs/lensjudge/bin/python lensjudge/eval/make_residual_ab_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPRO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPRO))
from lensjudge.eval import calibrate as C  # noqa: E402

OUT = REPRO / "lensjudge" / "outputs"
EVAL = REPRO / "lensjudge" / "eval"
FIG = REPRO / "lensjudge" / "papers" / "figures"
MAN = OUT / "manifest_full_ok.csv"
LEG_BLUE, NEW_RED = "#2c6fbb", "#b2182b"


def _fp_recall(arm):
    df = C.load_preds(str(OUT / f"preds_full_{arm}_r*.parquet"), str(MAN))
    calib = json.load(open(EVAL / f"residual_op_point_{arm}.json"))
    d = C.apply(df, calib)
    raw = d["grade_pred"].map(C._is_lens)
    fp = {src: (float(raw[d.source == src].mean()), float(d.loc[d.source == src, "lens_decision_cal"].mean()))
          for src in ("random_neg", "graded_D")}
    a = d[d.grade_truth == "A"]
    recA = (float(a["grade_pred"].map(C._is_lens).mean()), float(a["lens_decision_cal"].mean()))
    return fp, recA


def main():
    rep = json.load(open(OUT / "residual_ab_full_report.json"))
    A, B = rep["primary_benchmark_A_lens_vs_random"], rep["benchmark_B_lens_vs_gradeD"]
    fp_l, recA_l = _fp_recall("legacy")
    fp_n, recA_n = _fp_recall("new")

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.0))

    # (a) AUC +/- CI
    labels = ["lens vs\nrandom", "lens vs\ngrade-D"]
    x = np.arange(2)
    for off, arm, col, b in [(-0.12, "legacy", LEG_BLUE, [A, B]), (0.12, "new", NEW_RED, [A, B])]:
        aucs = [bb["auc_legacy" if arm == "legacy" else "auc_new"] for bb in b]
        cis = [bb["auc_legacy_ci" if arm == "legacy" else "auc_new_ci"] for bb in b]
        err = [[a - c[0] for a, c in zip(aucs, cis)], [c[1] - a for a, c in zip(aucs, cis)]]
        ax[0].errorbar(x + off, aucs, yerr=err, fmt="o", color=col, capsize=3, label=arm, ms=5)
    ax[0].axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8); ax[0].set_ylabel("ROC-AUC", fontsize=9)
    ax[0].set_ylim(0.32, 0.80); ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title(f"(a) discrimination: neutral\n$\\Delta$AUC={A['delta_auc_new_minus_legacy']:+.3f} "
                    f"CI[{A['delta_auc_ci'][0]:+.2f},{A['delta_auc_ci'][1]:+.2f}]", fontsize=8)

    # (b) negative FP-rate raw vs calibrated
    srcs = ["random_neg", "graded_D"]; xb = np.arange(2)
    bars = [("legacy raw", LEG_BLUE, 0.35, [fp_l[s][0] for s in srcs]),
            ("new raw", NEW_RED, 0.35, [fp_n[s][0] for s in srcs]),
            ("new calibrated", NEW_RED, 1.0, [fp_n[s][1] for s in srcs])]
    w = 0.26
    for i, (lab, col, alpha, vals) in enumerate(bars):
        ax[1].bar(xb + (i - 1) * w, vals, w, color=col, alpha=alpha,
                  hatch=("//" if "calibrated" in lab else None), label=lab, edgecolor="k", lw=0.4)
    ax[1].set_xticks(xb); ax[1].set_xticklabels(["random", "grade-D"], fontsize=8)
    ax[1].set_ylabel("false-positive rate", fontsize=9); ax[1].legend(fontsize=6.5)
    ax[1].set_title("(b) hot scores $\\to$ calibrate:\nthe FP flood is removed", fontsize=8)

    # (c) grade-A recall raw vs calibrated
    cats = ["legacy\n(raw=cal)", "new\nraw", "new\ncalibrated"]
    vals = [recA_l[0], recA_n[0], recA_n[1]]
    cols = [LEG_BLUE, NEW_RED, NEW_RED]; alph = [1, 0.35, 1]
    for i, (v, c2, a2) in enumerate(zip(vals, cols, alph)):
        ax[2].bar(i, v, 0.6, color=c2, alpha=a2, hatch=("//" if i == 2 else None), edgecolor="k", lw=0.4)
    ax[2].axhline(recA_l[0], ls=":", c=LEG_BLUE, lw=1)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(cats, fontsize=7.5)
    ax[2].set_ylabel("grade-A recall", fontsize=9)
    ax[2].set_title("(c) non-inferiority:\ncalibrated new = legacy", fontsize=8)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "resid_ab_results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "resid_ab_results.png")
    print(f"  AUC A: legacy {A['auc_legacy']:.3f} new {A['auc_new']:.3f} dAUC {A['delta_auc_new_minus_legacy']:+.3f}")
    print(f"  FP random raw->cal: legacy {fp_l['random_neg'][0]:.2f}->{fp_l['random_neg'][1]:.2f}  "
          f"new {fp_n['random_neg'][0]:.2f}->{fp_n['random_neg'][1]:.2f}")
    print(f"  grade-A recall: legacy {recA_l[0]:.2f}  new raw {recA_n[0]:.2f} cal {recA_n[1]:.2f}")


if __name__ == "__main__":
    main()
