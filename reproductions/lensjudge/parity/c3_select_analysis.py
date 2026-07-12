#!/usr/bin/env python3
"""C3 selection analysis: matched-inputs vs plain-direct on the valsel slice.

Consumes outputs/preds_c3val_{matched,direct}.parquet (run_batch over
outputs/c3_valsel_slice.csv, same model both arms) and reports:
  (a) does the matched information set help? — per-mode AUC on the HARD contrast
      (graded A/B vs graded_D) and ALL contrast (A/B/C vs D+random), plus PAIRED
      ΔAUC (matched − direct) with a 2000-sample bootstrap CI over common rows,
      and QWK vs the consensus grade per mode;
  (b) the matched-mode operating point — decision threshold at target FPR on the
      slice's graded_D negatives, plus an isotonic map p_lens -> soft target,
      saved to outputs/c3_matched_op_point.json (calibrate.py conventions).

This is SELECTION (valsel only). The winning mode + operating point get frozen
and then touched to the gate exactly once, later.

  python lensjudge/parity/c3_select_analysis.py [--fpr 0.2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parents[1] / "outputs"
RNG = np.random.default_rng(2026)
GMAP = {"A": 3, "B": 2, "C": 1, "D": 0}
SOFT = {"A": 0.95, "B": 0.80, "C": 0.45, "D": 0.05, "random": 0.02}


def _boot_auc(y, s, n=2000):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    a = roc_auc_score(y, s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        bi = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        if len(np.unique(y[bi])) == 2:
            v.append(roc_auc_score(y[bi], s[bi]))
    return a, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _paired_dauc(y, s1, s2, n=2000):
    ok = ~np.isnan(s1) & ~np.isnan(s2)
    y, s1, s2 = y[ok], s1[ok], s2[ok]
    d = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        bi = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        if len(np.unique(y[bi])) == 2:
            v.append(roc_auc_score(y[bi], s1[bi]) - roc_auc_score(y[bi], s2[bi]))
    return d, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.2, help="target FPR for the op point")
    args = ap.parse_args()

    man = pd.read_csv(OUT / "c3_valsel_slice.csv")
    preds = {}
    for mode in ("matched", "direct"):
        p = pd.read_parquet(OUT / f"preds_c3val_{mode}.parquet")
        cols = {c.lower(): c for c in p.columns}
        preds[mode] = p[[cols["name"], cols["p_lens"], cols.get("grade_pred", cols.get("grade"))]] \
            .set_axis(["name", f"p_{mode}", f"g_{mode}"], axis=1)
    df = man.merge(preds["matched"], on="name", how="left").merge(preds["direct"], on="name", how="left")
    print(f"rows: {len(df)}; matched preds {df.p_matched.notna().sum()}, "
          f"direct preds {df.p_direct.notna().sum()}")

    hard = df[df.source.isin(["graded", "graded_D"]) & df.grade_truth.isin(["A", "B", "D"])]
    y_h = (hard.binary_label == "lens").to_numpy(int)
    all_ = df.copy()
    y_a = (all_.binary_label == "lens").to_numpy(int)

    print("\n=== (a) matched vs direct — selection contrasts (valsel slice) ===")
    for label, sub, y in (("HARD A/B vs D", hard, y_h), ("ALL A/B/C vs D+random", all_, y_a)):
        for mode in ("matched", "direct"):
            a, lo, hi = _boot_auc(y, sub[f"p_{mode}"].to_numpy(float))
            print(f"  {label:22s} {mode:8s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")
        d, lo, hi = _paired_dauc(y, sub.p_matched.to_numpy(float), sub.p_direct.to_numpy(float))
        print(f"  {label:22s} paired dAUC (matched-direct) {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n  QWK vs consensus grade (graded+D rows):")
    gsub = df[df.source.isin(["graded", "graded_D"])]
    truth_ord = gsub.grade_truth.map(GMAP)
    for mode in ("matched", "direct"):
        pred_ord = gsub[f"g_{mode}"].astype(str).str.strip().str.upper().str[0].map(GMAP)
        ok = truth_ord.notna() & pred_ord.notna()
        k = cohen_kappa_score(truth_ord[ok], pred_ord[ok], weights="quadratic")
        print(f"    {mode:8s} QWK {k:.3f}  (n={int(ok.sum())})")

    # --- (b) operating point for the matched mode ---
    neg = df[df.source == "graded_D"].p_matched.dropna().to_numpy(float)
    thr = float(np.quantile(neg, 1 - args.fpr))
    y_iso = df.grade_truth.map(SOFT).where(df.source != "random_neg", SOFT["random"])
    ok = df.p_matched.notna() & y_iso.notna()
    iso = IsotonicRegression(out_of_bounds="clip").fit(df.p_matched[ok], y_iso[ok])
    grid = np.linspace(0, 1, 101)
    op = {"mode": "matched", "model": "claude-sonnet-5", "fpr_target": args.fpr,
          "threshold": thr, "n_neg": int(len(neg)),
          "isotonic_grid_x": grid.tolist(),
          "isotonic_grid_y": iso.predict(grid).tolist(),
          "slice_sha": (OUT / "c3_valsel_slice.csv.sha").read_text().strip(),
          "fit_on": "valsel slice (selection artifact; gate untouched)"}
    with open(OUT / "c3_matched_op_point.json", "w") as f:
        json.dump(op, f, indent=1)
    tpr_at = float((df[df.binary_label == "lens"].p_matched > thr).mean())
    print(f"\n=== (b) matched operating point ===")
    print(f"  threshold @ FPR={args.fpr:.2f} on graded_D: p_lens > {thr:.3f} "
          f"(recall on slice lens rows: {tpr_at:.3f})")
    print(f"  isotonic calibration fit on {int(ok.sum())} rows; saved c3_matched_op_point.json")


if __name__ == "__main__":
    main()
