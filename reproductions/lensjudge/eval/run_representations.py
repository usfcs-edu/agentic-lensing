#!/usr/bin/env python3
"""Standalone representation-feature AUC gate (the cheap gate; ~no model $).

Computes every Tier-1 scalar lensing-feature (common/representations.compute_features)
for each candidate in a manifest, then reports per-feature AND logistic-combo ROC-AUC
with bootstrap CIs on three contrasts:
  HARD  graded A/B/C   vs Grade-D human-rejects   (the wall; what vision/GIGA-Lens fail on)
  EASY  graded A       vs random-galaxy negatives  (the deployable lens-vs-galaxy regime)
  GOLD  Foundry-II confirmed vs confirmed non-lens (clean spectroscopic labels)
plus a SILVER block on the noiseless lensed/unlensed sims (band-agnostic features only).

  python lensjudge/eval/run_representations.py --manifest lensjudge/outputs/lensbench_large.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import fetch, representations as R  # noqa: E402

SEED = 2026


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), (float("nan"), float("nan"))
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(1000):
        ip = rng.integers(0, len(pos), len(pos)); ineg = rng.integers(0, len(neg), len(neg))
        yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        ss = np.r_[pos[ip], neg[ineg]]
        try: boot.append(roc_auc_score(yy, ss))
        except Exception: pass
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
    # orient so AUC>=0.5 (a feature can discriminate in either direction)
    if a < 0.5:
        a, lo, hi = 1 - a, 1 - hi, 1 - lo
    return float(a), (float(lo), float(hi))


def _materialize(man: pd.DataFrame) -> pd.DataFrame:
    def one(r):
        cube = fetch.get_cube(name=str(r["name"]), ra=r.get("ra"), dec=r.get("dec"),
                              survey=r.get("survey_key", "storfer"))
        if cube is None:
            return None
        return {"name": str(r["name"]), "grade": str(r.get("grade_truth", r.get("grade"))),
                "source": r.get("source", "?"), **R.compute_features(cube)}
    rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(lambda kv: one(kv[1]), man.iterrows()):
            if res:
                rows.append(res)
    return pd.DataFrame(rows)


def _silver_block(feat_cols):
    """Band-agnostic feature AUC on noiseless lensed/unlensed sims (sanity check)."""
    imgs_p = config.REPRO / "silver-2025" / "data" / "model1_images.npy"
    lbl_p = config.REPRO / "silver-2025" / "data" / "model1_labels.npy"
    if not imgs_p.exists() or not lbl_p.exists():
        return None
    imgs = np.load(imgs_p, mmap_mode="r"); lbl = np.load(lbl_p)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(lbl), size=min(400, len(lbl)), replace=False)
    band_agnostic = ["tangential_extent_deg", "tangentiality", "asymmetry_index",
                     "arcness_score", "arc_spread_px", "blue_tangential_extent_deg",
                     "blue_arcness_score"]
    rows = []
    for i in idx:
        im = np.asarray(imgs[i, 0], dtype=float)
        cube = np.stack([im, im, im], 0)              # pseudo-grz (color features meaningless)
        rows.append({"y": int(lbl[i]), **R.compute_features(cube)})
    df = pd.DataFrame(rows)
    out = {}
    for c in band_agnostic:
        if c in df:
            a, ci = _auc(df[df.y == 1][c], df[df.y == 0][c])
            out[c] = (a, ci)
    return out, int((df.y == 1).sum()), int((df.y == 0).sum())


def _combo_auc(df, pos_mask, neg_mask, feat_cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    sub = df[pos_mask | neg_mask]
    y = pos_mask[pos_mask | neg_mask].astype(int).values
    X = sub[feat_cols].fillna(0).values
    if y.sum() < 5 or (1 - y).sum() < 5:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.OUT / "lensbench_large.csv"))
    ap.add_argument("--out", default=str(config.OUT / "representation_features.parquet"))
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    print(f"[gate] computing features for {len(man)} candidates ...")
    df = _materialize(man)
    df.to_parquet(args.out, index=False)
    print(f"[gate] features OK {len(df)}/{len(man)} -> {args.out}")

    feat_cols = [c for c in df.columns if c not in ("name", "grade", "source")]
    is_graded = df.source == "graded"
    hard_pos = is_graded & df.grade.isin(["A", "B", "C"])
    hard_neg = df.source == "graded_D"
    easy_pos = is_graded & (df.grade == "A")
    easy_neg = df.source == "random_neg"
    gold = df.source == "gold"
    gold_pos = gold & (df.grade == "A"); gold_neg = gold & (df.grade == "D")

    def tbl(name, pos_mask, neg_mask):
        print(f"\n=== {name}: {int(pos_mask.sum())} pos vs {int(neg_mask.sum())} neg ===")
        res = []
        for c in feat_cols:
            a, (lo, hi) = _auc(df[pos_mask][c], df[neg_mask][c])
            res.append((c, a, lo, hi))
        for c, a, lo, hi in sorted(res, key=lambda t: -(t[1] if not np.isnan(t[1]) else 0)):
            star = " *" if (lo > 0.5 or hi < 0.5) else ""
            print(f"   {c:30s} AUC={a:.3f}  [{lo:.2f},{hi:.2f}]{star}")
        combo = _combo_auc(df, pos_mask, neg_mask, feat_cols)
        print(f"   {'LOGISTIC COMBO (5-fold CV)':30s} AUC={combo:.3f}")

    tbl("HARD (graded A/B/C vs Grade-D rejects)", hard_pos, hard_neg)
    tbl("EASY (graded A vs random galaxy)", easy_pos, easy_neg)
    tbl("GOLD (Foundry-II confirmed vs non-lens)", gold_pos, gold_neg)

    sb = _silver_block(feat_cols)
    if sb:
        out, npos, nneg = sb
        print(f"\n=== SILVER sims (band-agnostic only): {npos} lensed vs {nneg} unlensed ===")
        for c, (a, (lo, hi)) in sorted(out.items(), key=lambda t: -t[1][0]):
            print(f"   {c:30s} AUC={a:.3f}  [{lo:.2f},{hi:.2f}]")
    print("\n[note] * = 95% bootstrap CI excludes 0.5. CONSENSUS-REFERENCED, NO HUMAN CEILING.")


if __name__ == "__main__":
    main()
