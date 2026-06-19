#!/usr/bin/env python3
"""calibrate.py — operating-point calibration for LensJudge grader scores.

The residual A/B showed the new signed-chi residual makes the grader run ~2x HOTTER (a uniform
p_lens inflation, not better discrimination): uncalibrated, the lens/non-lens decision floods false
positives (random-neg FP 0.12->0.30, grade-D 0.27->0.49) while ROC-AUC is unchanged. The fix is to
make the DECISION at a controlled operating point instead of trusting the raw (hot) grade. This
module provides two calibrations, fit on a labelled set and persisted as JSON:

  - threshold @ target FPR : the p_lens cut whose false-positive rate on a labelled NEGATIVE
    calibration set equals the target -- the operational core; pins the FP rate regardless of
    how hot a given residual version runs.
  - isotonic recalibration  : a monotone map raw p_lens -> calibrated probability matching the
    empirical lens fraction, so p_lens_cal is a real probability comparable across residual
    versions (nice-to-have for ranking/reporting).

Because hotness differs by residual version, fit ONE calibration per LENSJUDGE_RESIDUAL_VERSION.
Note: fitting and evaluating on the same set is optimistic; in production fit the threshold on a
dedicated held-out negative sample.

  # fit (per arm) and persist
  python lensjudge/eval/calibrate.py fit --preds "outputs/preds_full_new_r*.parquet" \
      --manifest outputs/manifest_full_ok.csv --target-fpr 0.2 --out eval/residual_op_point_new.json
  # apply to new predictions
  python lensjudge/eval/calibrate.py apply --preds outputs/preds_X.parquet \
      --manifest outputs/m.csv --calib eval/residual_op_point_new.json --out outputs/preds_X_cal.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

LENS_GRADES = ("A", "B", "C")
DEFAULT_NEG = ("random_neg", "graded_D")


def _is_lens(g):
    return str(g).strip().upper() in LENS_GRADES


def load_preds(pattern, manifest):
    """Concat replicate parquets -> per-candidate (mean p_lens, modal grade), merged with manifest."""
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
    man = pd.read_csv(manifest).astype({"name": str})[["name", "source", "grade_truth"]]
    df = agg.merge(man, on="name", how="inner")
    df["is_lens"] = df["grade_truth"].map(_is_lens)
    return df


def fit(df, target_fpr=0.2, neg_sources=DEFAULT_NEG):
    """Threshold @ target FPR on the negatives + isotonic p_lens -> P(lens)."""
    neg = df[df["source"].isin(neg_sources)]
    if len(neg):
        negp = np.sort(neg["p_lens"].values)[::-1]
        k = int(np.floor(target_fpr * len(negp)))
        thr = float(negp[min(k, len(negp) - 1)])
        achieved_fpr = float((neg["p_lens"] >= thr).mean())
    else:
        thr, achieved_fpr = 0.5, None
    y, p = df["is_lens"].values.astype(int), df["p_lens"].values
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    xs = np.round(np.linspace(0.0, 1.0, 101), 4)
    ys = np.round(iso.predict(xs), 5)
    pos = df[df["is_lens"]]
    return {
        "target_fpr": target_fpr,
        "threshold": round(thr, 5),
        "achieved_fpr": achieved_fpr,
        "recall_at_op": round(float((pos["p_lens"] >= thr).mean()), 4) if len(pos) else None,
        "auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
        "neg_sources": list(neg_sources),
        "n_fit": int(len(df)), "n_neg": int(len(neg)),
        "isotonic": {"x": xs.tolist(), "y": ys.tolist()},
    }


def apply(df, calib):
    """Add p_lens_cal (isotonic) and lens_decision_cal (p_lens >= threshold)."""
    xs = np.asarray(calib["isotonic"]["x"]); ys = np.asarray(calib["isotonic"]["y"])
    out = df.copy()
    out["p_lens_cal"] = np.interp(out["p_lens"].values, xs, ys)
    out["lens_decision_cal"] = out["p_lens"].values >= calib["threshold"]
    return out


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--preds", required=True); f.add_argument("--manifest", required=True)
    f.add_argument("--target-fpr", type=float, default=0.2); f.add_argument("--out", required=True)
    a = sub.add_parser("apply"); a.add_argument("--preds", required=True); a.add_argument("--manifest", required=True)
    a.add_argument("--calib", required=True); a.add_argument("--out", required=True)
    args = ap.parse_args()
    df = load_preds(args.preds, args.manifest)
    if args.cmd == "fit":
        calib = fit(df, args.target_fpr)
        json.dump(calib, open(args.out, "w"), indent=2)
        print(f"fit @ FPR={calib['target_fpr']}: threshold={calib['threshold']} "
              f"(achieved FPR {calib['achieved_fpr']}, recall {calib['recall_at_op']}, AUC {calib['auc']}) "
              f"-> {args.out}")
    else:
        calib = json.load(open(args.calib))
        out = apply(df, calib)
        out.to_parquet(args.out, index=False)
        print(f"applied -> {args.out}: {int(out['lens_decision_cal'].sum())} lens decisions "
              f"of {len(out)} (threshold {calib['threshold']})")


if __name__ == "__main__":
    _cli()
