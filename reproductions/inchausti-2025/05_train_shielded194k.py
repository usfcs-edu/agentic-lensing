#!/usr/bin/env python3
"""
05_train_shielded194k.py

Train the shielded-ResNet base model of the Inchausti+2025 ensemble. This is the
SAME `ShieldedDeepLens` architecture from the Huang+2021 reproduction
(01b_shielded_resnet.py, symlinked), instantiated at the Inchausti parameter
count via constructor args — no model-file edit needed.

Configs (`--config`):
  194k : stage_out=52, stage_mid=32, shield_ch=12, final_out=24 -> 194,501 params
         (Inchausti+2025 reports 194,433; +68 / 0.035% — closest 4-shield,
         15-block fit, the channel widths being unpublished).
  60k  : the Huang+2021 default (final_out=32) -> 59,905 params, for the baseline
         row in the controlled comparison.

Recipes (`--recipe`, hyperparameters only — architecture is fixed by --config):
  inchausti : batch 256 (capped from the paper's 4xA100 batch 2048), Adam lr 1e-3
              /10 @ epoch 40, 130 epochs (paper's best epoch was 126).
  storfer   : batch 128, Adam lr 5e-4 /5 @ epoch 80, 145 epochs (Storfer+2024 DR9
              recipe; pair with --config 60k for the Storfer single-model baseline).
  huang     : batch 128, Adam lr 1e-3 /10 @ epoch 40, 120 epochs (Phase-4a recipe).

Trains on the SAME DR9 cutouts / positives / negatives / SEED / split as the
huang-2020 L18 and huang-2021 shielded runs (via `_trainlib.build_split`), so the
controlled comparison isolates architecture + recipe. BCEWithLogitsLoss, per-band
mean/std normalise + clamp +/-250, rotation/flip/zoom augmentation.

Outputs (suffix = --tag, default `shielded194k`):
  data/training_split_<tag>.parquet     row_id -> {train,val,test} + label
  data/checkpoint_best_<tag>.pt         state_dict + mean/std + shielded_cfg + arch
  data/training_history_<tag>.json      per-epoch (loss, val_auc, lr)
  data/test_result_<tag>.json           held-out test AUC
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import _trainlib as TL

# Local import — the shielded model (filename starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "shielded_resnet", str(Path(__file__).resolve().parent / "01b_shielded_resnet.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ShieldedDeepLens = _mod.ShieldedDeepLens

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
POSITIVES = DATA / "positives_huang2020.parquet"
NEGATIVES = DATA / "negatives.parquet"

CONFIGS = {
    "194k": dict(stage_out=52, stage_mid=32, shield_ch=12, final_out=24),
    "60k": dict(final_out=32),
}
RECIPES = {
    # batch 128 (not the paper's 4xA100 batch 2048) to fit the wide 194K net's
    # full-101x101 stage-1 activations on a 15 GB A16; BN/LR effect only.
    "inchausti": dict(batch=128, lr=1e-3, decay_epoch=40, decay_factor=10.0, epochs=130),
    "storfer":   dict(batch=128, lr=5e-4, decay_epoch=80, decay_factor=5.0, epochs=145),
    "huang":     dict(batch=128, lr=1e-3, decay_epoch=40, decay_factor=10.0, epochs=120),
}


def train(args) -> None:
    cfg = CONFIGS[args.config]
    rec = dict(RECIPES[args.recipe])
    if args.epochs is not None:
        rec["epochs"] = args.epochs
    tag = args.tag or f"shielded{args.config}"
    fits_dir = DATA / TL.DR_TO_FITS[args.dr]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] tag={tag} config={args.config}{cfg} recipe={args.recipe}{rec} "
          f"dr={args.dr} device={device}")

    import pandas as pd
    pos = pd.read_parquet(POSITIVES)
    neg = pd.read_parquet(NEGATIVES)
    df = TL.build_split(pos, neg, fits_dir, seed=TL.SEED)
    df.to_parquet(DATA / f"training_split_{tag}.parquet", index=False)

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"[stat] computing per-band mean/std from {min(500, len(df_train))} cutouts")
    mean, std = TL.compute_band_stats(df_train, fits_dir, n_sample=500)
    print(f"[stat] mean={mean.tolist()}  std={std.tolist()}")

    pin = device.type == "cuda"
    train_dl = DataLoader(TL.LensDataset(df_train, fits_dir, mean, std, True),
                          batch_size=rec["batch"], shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(TL.LensDataset(df_val, fits_dir, mean, std, False),
                        batch_size=rec["batch"], shuffle=False,
                        num_workers=args.workers, pin_memory=pin)
    test_dl = DataLoader(TL.LensDataset(df_test, fits_dir, mean, std, False),
                         batch_size=rec["batch"], shuffle=False,
                         num_workers=args.workers, pin_memory=pin)

    model = ShieldedDeepLens(in_channels=3, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] shielded params: {n_params:,}")
    optim = torch.optim.Adam(model.parameters(), lr=rec["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=rec["decay_epoch"], gamma=1.0 / rec["decay_factor"])
    loss_fn = nn.BCEWithLogitsLoss()

    history, best_val_auc = [], -1.0
    best_path = DATA / f"checkpoint_best_{tag}.pt"
    t0 = time.time()
    for epoch in range(1, rec["epochs"] + 1):
        model.train()
        tr_loss = tr_n = 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(y)
            tr_n += len(y)

        model.eval()
        vl, vy, val_loss, val_n = [], [], 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logit = model(x)
                val_loss += loss_fn(logit, y).item() * len(y)
                val_n += len(y)
                vl.append(logit.cpu().numpy()); vy.append(y.cpu().numpy())
        vl, vy = np.concatenate(vl), np.concatenate(vy)
        val_auc = float(roc_auc_score(vy, vl)) if len(np.unique(vy)) > 1 else float("nan")
        cur_lr = optim.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "lr": cur_lr,
                        "train_loss": tr_loss / max(1, tr_n),
                        "val_loss": val_loss / max(1, val_n),
                        "val_auc": val_auc, "elapsed_s": elapsed})
        print(f"[e{epoch:>3d}] lr={cur_lr:.1e} train={history[-1]['train_loss']:.4f} "
              f"val={history[-1]['val_loss']:.4f} val_auc={val_auc:.4f} t={elapsed/60:.1f}m")
        scheduler.step()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_auc": val_auc, "mean": mean.tolist(), "std": std.tolist(),
                        "arch": "shielded", "shielded_cfg": cfg,
                        "config": args.config, "recipe": args.recipe}, best_path)
        (DATA / f"training_history_{tag}.json").write_text(json.dumps(history, indent=2))

    # Test pass with the best checkpoint.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    test_model = ShieldedDeepLens(in_channels=3, **ckpt["shielded_cfg"]).to(device)
    test_model.load_state_dict(ckpt["state_dict"]); test_model.eval()
    tl, ty = [], []
    with torch.no_grad():
        for x, y in test_dl:
            tl.append(test_model(x.to(device)).cpu().numpy()); ty.append(y.numpy())
    tl, ty = np.concatenate(tl), np.concatenate(ty)
    test_auc = float(roc_auc_score(ty, tl))
    print(f"\n[test] best_val_auc={best_val_auc:.4f} test_auc={test_auc:.4f} "
          f"(epoch {ckpt['epoch']})")
    (DATA / f"test_result_{tag}.json").write_text(json.dumps({
        "arch": "shielded", "tag": tag, "config": args.config, "recipe": args.recipe,
        "dr": args.dr, "n_params": int(n_params), "best_val_auc": best_val_auc,
        "test_auc": test_auc, "best_epoch": int(ckpt["epoch"]), "n_test": int(len(ty)),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=tuple(CONFIGS), default="194k")
    ap.add_argument("--recipe", choices=tuple(RECIPES), default="inchausti")
    ap.add_argument("--dr", choices=("dr9", "dr7"), default="dr9")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe's epoch count")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(TL.SEED)
    np.random.seed(TL.SEED)
    train(args)


if __name__ == "__main__":
    main()
