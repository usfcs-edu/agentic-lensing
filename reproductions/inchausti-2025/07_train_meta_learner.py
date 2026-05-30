#!/usr/bin/env python3
"""
07_train_meta_learner.py — Phase-5 ensemble headline.

Train the feature-weighted-stacking meta-learner (03_meta_learner.py) over the
two base models' probabilities, on the SHARED train/val/test split, and report
the meta-learner vs simple-average AUCs — reproducing Inchausti+2025 Fig. 6
(ResNet 0.9984, EfficientNet 0.9987, meta-learner 0.9989 == average 0.9989).

Procedure:
  1. Read the shared split (training_split_shielded194k.parquet) + the matching
     DR9 cutouts.
  2. Score every row with the two base checkpoints (shielded194k, efficientnet)
     -> meta-features [p_resnet, p_effnet].
  3. Train the 300-node MLP on the TRAIN rows (BCE), early-stop on VAL AUC,
     evaluate on TEST. (As in the paper, the meta-learner sees the base models'
     in-sample train probabilities; we report the held-out TEST AUC and also a
     simple-average baseline. The correlated bases make stacking ~ averaging.)

Outputs:
  data/meta_features_split.parquet   row_id,label,split,p_resnet,p_effnet,p_meta,p_avg
  data/checkpoint_best_meta.pt        meta state_dict + feature order + base ckpts
  data/meta_metrics.json              per-split AUCs (resnet, effnet, meta, avg)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import _trainlib as TL
import _scorelib as SL

_spec = importlib.util.spec_from_file_location(
    "meta_learner", str(Path(__file__).resolve().parent / "03_meta_learner.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MetaLearner = _mod.MetaLearner
simple_average = _mod.simple_average
FEATURE_ORDER = _mod.FEATURE_ORDER

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def build_features(args, device) -> pd.DataFrame:
    split = pd.read_parquet(DATA / args.split)
    fits_dir = DATA / TL.DR_TO_FITS[args.dr]
    paths = [fits_dir / f"{r}.fits" for r in split["row_id"]]
    feats = {}
    for name, ckpt in (("shielded194k", args.ckpt_resnet),
                       ("efficientnet", args.ckpt_effnet)):
        model, arch, mean, std, va = SL.load_checkpoint_model(DATA / ckpt, device)
        print(f"[score] {name}: arch={arch} val_auc={va:.4f} scoring {len(paths)} rows")
        feats[name] = SL.score_paths(paths, model, arch, mean, std, device, batch=args.batch)
        del model
        torch.cuda.empty_cache()
    df = split[["row_id", "label", "split"]].copy()
    df["p_resnet"] = feats["shielded194k"]
    df["p_effnet"] = feats["efficientnet"]
    df = df.dropna(subset=["p_resnet", "p_effnet"]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="training_split_shielded194k.parquet")
    ap.add_argument("--dr", choices=("dr9", "dr7"), default="dr9")
    ap.add_argument("--ckpt-resnet", default="checkpoint_best_shielded194k.pt", dest="ckpt_resnet")
    ap.add_argument("--ckpt-effnet", default="checkpoint_best_efficientnet.pt", dest="ckpt_effnet")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()
    torch.manual_seed(TL.SEED)
    np.random.seed(TL.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = build_features(args, device)
    P = df[["p_resnet", "p_effnet"]].to_numpy(np.float32)   # column order == FEATURE_ORDER
    y = df["label"].to_numpy(np.float32)
    tr, va, te = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy(), \
                 (df["split"] == "test").to_numpy()

    # Train the meta-learner on TRAIN rows; early-stop on VAL AUC.
    meta = MetaLearner().to(device)
    opt = torch.optim.Adam(meta.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.from_numpy(P[tr]).to(device); ytr = torch.from_numpy(y[tr]).to(device)
    Xva = torch.from_numpy(P[va]).to(device)
    best_val, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        meta.train(); opt.zero_grad()
        lossf(meta(Xtr), ytr).backward(); opt.step()
        meta.eval()
        with torch.no_grad():
            pv = torch.sigmoid(meta(Xva)).cpu().numpy()
        va_auc = roc_auc_score(y[va], pv) if len(np.unique(y[va])) > 1 else float("nan")
        if va_auc > best_val:
            best_val = va_auc
            best_state = {k: v.cpu().clone() for k, v in meta.state_dict().items()}
    meta.load_state_dict(best_state)

    # Final per-split probabilities.
    meta.eval()
    with torch.no_grad():
        p_meta = torch.sigmoid(meta(torch.from_numpy(P).to(device))).cpu().numpy()
    p_avg = simple_average(P)
    df["p_meta"] = p_meta
    df["p_avg"] = p_avg
    df.to_parquet(DATA / "meta_features_split.parquet", index=False)

    def auc(mask, col):
        return float(roc_auc_score(y[mask], df[col].to_numpy()[mask])) \
            if len(np.unique(y[mask])) > 1 else float("nan")

    metrics = {"feature_order": list(FEATURE_ORDER),
               "ckpt_resnet": args.ckpt_resnet, "ckpt_effnet": args.ckpt_effnet,
               "n": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}}
    for split_name, mask in (("val", va), ("test", te)):
        metrics[split_name] = {c: auc(mask, c) for c in
                               ("p_resnet", "p_effnet", "p_meta", "p_avg")}
    (DATA / "meta_metrics.json").write_text(json.dumps(metrics, indent=2))

    torch.save({"state_dict": meta.state_dict(), "feature_order": list(FEATURE_ORDER),
                "arch": "meta", "ckpt_resnet": args.ckpt_resnet,
                "ckpt_effnet": args.ckpt_effnet, "best_val_auc": best_val},
               DATA / "checkpoint_best_meta.pt")

    print("\n[meta] held-out AUCs (paper Fig.6: resnet .9984 / effnet .9987 / meta .9989 = avg .9989)")
    for split_name in ("val", "test"):
        m = metrics[split_name]
        print(f"  {split_name:>4s}: resnet={m['p_resnet']:.4f}  effnet={m['p_effnet']:.4f}  "
              f"meta={m['p_meta']:.4f}  avg={m['p_avg']:.4f}")
    print(f"\n[done] wrote meta_features_split.parquet + checkpoint_best_meta.pt + meta_metrics.json")


if __name__ == "__main__":
    main()
