#!/usr/bin/env python3
"""
22_fpr_operating_point.py — Phase-5 Stage C, operating-point analysis.

Stage C retrained at a realistic ~1:25 negative:positive ratio, which
recalibrates the output probabilities toward the low positive base rate: a fixed
p>=0.5 threshold is no longer meaningful (both random galaxies AND real lenses
score low). The honest comparison is recovery at a MATCHED false-positive rate.

For the Stage-B (~2.5:1) and Stage-C (~1:25) models, this scores the same
held-out random-galaxy negatives to set each model's operating threshold at a
target FPR (1% and 0.1%), then measures recovery (TPR) of the published Storfer
/ Inchausti candidates at those thresholds. This is the apples-to-apples
"recovery at fixed FPR" that a fixed-0.5 cut cannot give.

Output: data/operating_point.csv + printed table.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _scorelib as SL

_spec = importlib.util.spec_from_file_location(
    "meta_learner", str(Path(__file__).resolve().parent / "03_meta_learner.py"))
_mm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mm)
MetaLearner = _mm.MetaLearner

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STAGES = {
    "B": dict(sh="checkpoint_best_shielded194k_stageb.pt", ef="checkpoint_best_efficientnet_stageb.pt",
              meta="checkpoint_best_meta_stageb.pt"),
    "C": dict(sh="checkpoint_best_shielded194k_stagec.pt", ef="checkpoint_best_efficientnet_stagec.pt",
              meta="checkpoint_best_meta_stagec.pt"),
}
TARGET_FPR = (0.01, 0.001)


def score_set(paths, ck, device):
    """Return dict p_resnet/p_effnet/p_meta for a list of cutout paths."""
    sh, a1, m1, s1, _ = SL.load_checkpoint_model(DATA / ck["sh"], device)
    pr = SL.score_paths(paths, sh, "shielded", m1, s1, device); del sh; torch.cuda.empty_cache()
    ef, a2, m2, s2, _ = SL.load_checkpoint_model(DATA / ck["ef"], device)
    pe = SL.score_paths(paths, ef, "efficientnet", m2, s2, device); del ef; torch.cuda.empty_cache()
    meta = MetaLearner().to(device)
    meta.load_state_dict(torch.load(str(DATA / ck["meta"]), map_location="cpu", weights_only=False)["state_dict"])
    meta.eval()
    P = np.stack([pr, pe], 1).astype(np.float32); ok = np.isfinite(P).all(1)
    pm = np.full(len(P), np.nan, np.float32)
    with torch.no_grad():
        pm[ok] = torch.sigmoid(meta(torch.from_numpy(P[ok]).to(device))).cpu().numpy()
    return {"resnet": pr, "effnet": pe, "meta": pm}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # held-out negatives = Stage-C test-split negatives
    split = pd.read_parquet(DATA / "training_split_stagec.parquet")
    neg = split[(split.split == "test") & (split.label == 0)]
    neg_paths = [Path(r.fits_dir) / f"{r.row_id}.fits" for r in neg.itertuples()]
    print(f"[init] {len(neg_paths)} held-out negatives")
    cats = {}
    for key, cut, csv in (("storfer", "cutouts_fits_candidates_storfer", "storfer2024_published_catalog.csv"),
                          ("inchausti", "cutouts_fits_candidates_inchausti", "inchausti2025_published_catalog.csv")):
        c = pd.read_csv(DATA / csv)
        cats[key] = (c, [DATA / cut / f"{n}.fits" for n in c["name"]])

    rows = []
    for stage, ck in STAGES.items():
        negs = score_set(neg_paths, ck, device)
        cand = {k: score_set(p, ck, device) for k, (c, p) in cats.items()}
        for model in ("resnet", "effnet", "meta"):
            ns = negs[model][np.isfinite(negs[model])]
            for fpr in TARGET_FPR:
                thr = float(np.quantile(ns, 1 - fpr))
                row = {"stage": stage, "model": model, "target_fpr": fpr, "threshold": thr}
                for k in cats:
                    cs = cand[k][model]; cs = cs[np.isfinite(cs)]
                    row[f"recovery_{k}"] = float((cs >= thr).mean())
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "operating_point.csv", index=False)

    print("\n=== Recovery at matched false-positive rate (the honest operating point) ===")
    print(f"{'stage':>5} {'model':>7} {'FPR':>6} {'thresh':>8} {'storfer':>9} {'inchausti':>10}")
    for _, r in out.iterrows():
        print(f"{r.stage:>5} {r.model:>7} {r.target_fpr:>6.3f} {r.threshold:>8.3f} "
              f"{r.recovery_storfer:>9.3f} {r.recovery_inchausti:>10.3f}")
    print("\n[done] wrote operating_point.csv")


if __name__ == "__main__":
    main()
