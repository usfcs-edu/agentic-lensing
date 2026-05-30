#!/usr/bin/env python3
"""
06_train_efficientnet.py

Fine-tune the EfficientNetV2-S base model (02_efficientnet.py) of the
Inchausti+2025 ensemble on the SAME DR9 cutouts / positives / negatives / SEED /
split as 05_train_shielded194k.py — so the two base models are trained on a
byte-identical partition (a hard requirement for the meta-learner, which stacks
their probabilities).

Per the paper (§3.2.2): pretrained then fine-tuned with cross-entropy loss
(2-class). val AUC 0.9987 at epoch 50. The paper used batch 512 on 4xA100; on a
15 GB A16 we use a smaller batch + gradient accumulation to recover the effective
batch (BN/LR are mildly batch-sensitive; the architecture being reproduced is
not). Normalisation = the shared per-band mean/std + clamp +/-250 from _trainlib
(NOT ImageNet stats — see 02_efficientnet.py docstring).

Outputs (suffix = --tag, default `efficientnet`):
  data/training_split_<tag>.parquet   (identical to the shielded split by construction)
  data/checkpoint_best_<tag>.pt       state_dict + mean/std + arch='efficientnet' + variant/head_dim
  data/training_history_<tag>.json
  data/test_result_<tag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import _trainlib as TL

_spec = importlib.util.spec_from_file_location(
    "efficientnet", str(Path(__file__).resolve().parent / "02_efficientnet.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EfficientNetV2Lens = _mod.EfficientNetV2Lens

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
POSITIVES = DATA / "positives_huang2020.parquet"
NEGATIVES = DATA / "negatives.parquet"


@torch.no_grad()
def eval_auc(model, dl, device) -> tuple[float, float]:
    model.eval()
    probs, ys, loss_sum, n = [], [], 0.0, 0
    lossf = nn.CrossEntropyLoss()
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        yl = y.long().to(device, non_blocking=True)
        logits = model(x)
        loss_sum += lossf(logits, yl).item() * len(y)
        n += len(y)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else float("nan")
    return auc, loss_sum / max(1, n)


def train(args) -> None:
    tag = args.tag
    fits_dir = DATA / TL.DR_TO_FITS[args.dr]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] tag={tag} dr={args.dr} batch={args.batch} accum={args.accum} "
          f"(eff batch {args.batch * args.accum}) lr={args.lr} epochs={args.epochs} device={device}")

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
                          batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(TL.LensDataset(df_val, fits_dir, mean, std, False),
                        batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=pin)
    test_dl = DataLoader(TL.LensDataset(df_test, fits_dir, mean, std, False),
                         batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=pin)

    model = EfficientNetV2Lens(pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] EfficientNetV2 params: {n_params:,}  pretrained_loaded={model.pretrained_loaded}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_epoch,
                                                gamma=1.0 / args.decay_factor)
    lossf = nn.CrossEntropyLoss()

    history, best_val_auc = [], -1.0
    best_path = DATA / f"checkpoint_best_{tag}.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_n = 0
        optim.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            yl = y.long().to(device, non_blocking=True)
            loss = lossf(model(x), yl) / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
            tr_loss += loss.item() * args.accum * len(y)
            tr_n += len(y)
        val_auc, val_loss = eval_auc(model, val_dl, device)
        cur_lr = optim.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "lr": cur_lr,
                        "train_loss": tr_loss / max(1, tr_n), "val_loss": val_loss,
                        "val_auc": val_auc, "elapsed_s": elapsed})
        print(f"[e{epoch:>3d}] lr={cur_lr:.1e} train={history[-1]['train_loss']:.4f} "
              f"val={val_loss:.4f} val_auc={val_auc:.4f} t={elapsed/60:.1f}m")
        scheduler.step()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_auc": val_auc, "mean": mean.tolist(), "std": std.tolist(),
                        "arch": "efficientnet", "variant": model.variant,
                        "head_dim": model.head_dim, "num_classes": model.num_classes},
                       best_path)
        (DATA / f"training_history_{tag}.json").write_text(json.dumps(history, indent=2))

    # Test pass with the best checkpoint.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    test_model = EfficientNetV2Lens(pretrained=False, variant=ckpt["variant"],
                                    head_dim=ckpt["head_dim"],
                                    num_classes=ckpt["num_classes"]).to(device)
    test_model.load_state_dict(ckpt["state_dict"])
    test_auc, _ = eval_auc(test_model, test_dl, device)
    print(f"\n[test] best_val_auc={best_val_auc:.4f} test_auc={test_auc:.4f} (epoch {ckpt['epoch']})")
    (DATA / f"test_result_{tag}.json").write_text(json.dumps({
        "arch": "efficientnet", "tag": tag, "variant": ckpt["variant"], "dr": args.dr,
        "n_params": int(n_params), "best_val_auc": best_val_auc, "test_auc": test_auc,
        "best_epoch": int(ckpt["epoch"]),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr", choices=("dr9", "dr7"), default="dr9")
    ap.add_argument("--tag", default="efficientnet")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--accum", type=int, default=2, help="gradient-accumulation steps")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--decay-epoch", type=int, default=30, dest="decay_epoch")
    ap.add_argument("--decay-factor", type=float, default=10.0, dest="decay_factor")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(TL.SEED)
    np.random.seed(TL.SEED)
    train(args)


if __name__ == "__main__":
    main()
