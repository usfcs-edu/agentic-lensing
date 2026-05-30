#!/usr/bin/env python3
"""
13_score_candidates_direct.py — Phase-5 targeted-recovery, track (ii).

Score the freshly-downloaded cutouts of the published Storfer-2024 (DR9) and
Inchausti-2025 (DR10) candidates (12_) with our three reproduced models
(shielded-194K ResNet, EfficientNetV2, meta-learner) + the simple-average
baseline. This is the honest "would our reproduction have flagged this published
lens" signal — we score the exact published positions, no parent-sample sweep.

For the Inchausti catalog this also lets us compare OUR per-model probabilities
against the PUBLISHED ResNet/EfficientNet/meta probabilities row-for-row.

Outputs (per catalog):
  data/candidate_scores_<catalog>.csv
    published columns + our_p_resnet, our_p_effnet, our_p_meta, our_p_avg, cutout_ok
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _scorelib as SL

_spec = importlib.util.spec_from_file_location(
    "meta_learner", str(Path(__file__).resolve().parent / "03_meta_learner.py"))
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
MetaLearner = _mm.MetaLearner
simple_average = _mm.simple_average

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CATALOGS = {
    "storfer":   dict(csv="storfer2024_published_catalog.csv",
                      cut="cutouts_fits_candidates_storfer"),
    "inchausti": dict(csv="inchausti2025_published_catalog.csv",
                      cut="cutouts_fits_candidates_inchausti"),
}


def apply_meta(p_resnet, p_effnet, device) -> np.ndarray:
    ck = torch.load(str(DATA / "checkpoint_best_meta.pt"), map_location="cpu", weights_only=False)
    meta = MetaLearner().to(device)
    meta.load_state_dict(ck["state_dict"])
    meta.eval()
    P = np.stack([p_resnet, p_effnet], axis=1).astype(np.float32)  # FEATURE_ORDER
    out = np.full(len(P), np.nan, np.float32)
    ok = np.isfinite(P).all(axis=1)
    with torch.no_grad():
        out[ok] = torch.sigmoid(meta(torch.from_numpy(P[ok]).to(device))).cpu().numpy()
    return out


def run_catalog(key: str, device, args) -> None:
    cfg = CATALOGS[key]
    cat = pd.read_csv(DATA / cfg["csv"])
    cut_dir = DATA / cfg["cut"]
    paths = [cut_dir / f"{n}.fits" for n in cat["name"]]
    cat["cutout_ok"] = [p.exists() and p.stat().st_size > 0 for p in paths]
    print(f"[{key}] {len(cat)} candidates; cutouts present: {int(cat['cutout_ok'].sum())}")

    scores = {}
    for col, ckpt in (("our_p_resnet", "checkpoint_best_shielded194k.pt"),
                      ("our_p_effnet", "checkpoint_best_efficientnet.pt")):
        model, arch, mean, std, va = SL.load_checkpoint_model(DATA / ckpt, device)
        print(f"[{key}] scoring with {ckpt} (arch={arch}, val_auc={va:.4f})")
        scores[col] = SL.score_paths(paths, model, arch, mean, std, device, batch=args.batch)
        del model
        torch.cuda.empty_cache()

    cat["our_p_resnet"] = scores["our_p_resnet"]
    cat["our_p_effnet"] = scores["our_p_effnet"]
    cat["our_p_meta"] = apply_meta(scores["our_p_resnet"], scores["our_p_effnet"], device)
    cat["our_p_avg"] = simple_average(
        np.stack([scores["our_p_resnet"], scores["our_p_effnet"]], axis=1))

    out = DATA / f"candidate_scores_{key}.csv"
    cat.to_csv(out, index=False)
    nok = int(cat["cutout_ok"].sum())
    print(f"[{key}] wrote {out.name}")
    for col in ("our_p_resnet", "our_p_effnet", "our_p_meta", "our_p_avg"):
        v = cat.loc[cat["cutout_ok"], col]
        print(f"    {col}: median {v.median():.4f}  frac>=0.5 {(v>=0.5).mean():.3f}  "
              f"frac>=0.9 {(v>=0.9).mean():.3f}")
    # Inchausti: quick agreement check vs published meta probability.
    if key == "inchausti" and "p_meta" in cat.columns:
        m = cat["cutout_ok"] & cat["p_meta"].notna() & cat["our_p_meta"].notna()
        if m.sum() > 5:
            corr = np.corrcoef(cat.loc[m, "p_meta"], cat.loc[m, "our_p_meta"])[0, 1]
            print(f"    [validation] corr(our_p_meta, published p_meta) = {corr:.3f} over {int(m.sum())} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", choices=("storfer", "inchausti", "both"), default="both")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = ("storfer", "inchausti") if args.catalog == "both" else (args.catalog,)
    for k in keys:
        run_catalog(k, device, args)


if __name__ == "__main__":
    main()
