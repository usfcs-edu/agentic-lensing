#!/usr/bin/env python3
"""
07_plot_training_curves.py — Phase 4a figures.

  papers/figures/arch_training_curves.png   val-AUC vs epoch, shielded (DR9+DR7)
                                             overlaid with L18 (DR7) from huang-2020
  papers/figures/shielded_roc.png           ROC of the shielded model on its test split
  papers/figures/shielded_arch.png          schematic of the shielded ResNet
                                             (stem -> [stage×3 -> shield]×4 -> stage5 -> head)

Inputs:
  data/training_history_shielded_{dr9,dr7}.json   (05_train_shielded.py)
  data/checkpoint_best_shielded_dr7.pt + training_split_shielded_dr7.parquet
  data/cutouts_fits_dr7_train/<row_id>.fits
  ../huang-2020/data/training_history_dr7.json    (L18 baseline)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
H2020 = HERE.parent.parent / "huang-2020" / "data"
FIG_DIR = HERE / "papers" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load(spec_name, fname):
    spec = importlib.util.spec_from_file_location(spec_name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plot_curves() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    series = [
        ("shielded / DR9", DATA / "training_history_shielded_dr9.json", "#2ca02c", "-"),
        ("shielded / DR7", DATA / "training_history_shielded_dr7.json", "#1f77b4", "-"),
        ("L18 / DR7 (huang-2020)", H2020 / "training_history_dr7.json", "#d62728", "--"),
    ]
    for label, path, color, ls in series:
        if not path.exists():
            print(f"[warn] missing {path.name}")
            continue
        df = pd.DataFrame(json.loads(path.read_text()))
        ax.plot(df["epoch"], df["val_auc"], color=color, ls=ls, lw=1.5, label=label)
        best = df.loc[df["val_auc"].idxmax()]
        ax.scatter([best["epoch"]], [best["val_auc"]], color=color, s=20, zorder=5)
    ax.axvline(40, color="gray", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation AUC")
    ax.set_ylim(0.80, 1.005)
    ax.set_title("Shielded vs L18 — validation AUC")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "arch_training_curves.png", dpi=180)
    plt.close(fig)
    print("[done] arch_training_curves.png")


def plot_roc() -> None:
    ckpt_path = DATA / "checkpoint_best_shielded_dr7.pt"
    split_path = DATA / "training_split_shielded_dr7.parquet"
    if not (ckpt_path.exists() and split_path.exists()):
        print("[skip] ROC — missing shielded DR7 ckpt/split")
        return
    arch = _load("shielded_resnet", "01b_shielded_resnet.py")
    train = _load("train_shielded", "05_train_shielded.py")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean, std = np.array(ckpt["mean"]), np.array(ckpt["std"])
    split = pd.read_parquet(split_path)
    test_df = split[split["split"] == "test"].reset_index(drop=True).copy()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = arch.ShieldedDeepLens(in_channels=3, final_out=int(ckpt.get("final_out", 32))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = train.LensDataset(test_df, DATA / "cutouts_fits_dr7_train", mean, std, train=False)
    # num_workers=0: LensDataset comes from the dynamically-imported 05_train_shielded
    # module, which macOS 'spawn' DataLoader workers cannot re-import (Linux 'fork' could).
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            logits.append(model(x.to(device)).cpu().numpy())
            labels.append(y.cpu().numpy())
    logits = np.concatenate(logits); labels = np.concatenate(labels)
    fpr, tpr, _ = roc_curve(labels, logits)
    auc = roc_auc_score(labels, logits)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(fpr, tpr, color="#1f77b4", lw=1.5,
            label=f"shielded ResNet (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", alpha=0.6, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "shielded_roc.png", dpi=180)
    plt.close(fig)
    print(f"[done] shielded_roc.png  (test AUC={auc:.4f})")


def plot_arch() -> None:
    boxes = [
        ("Input\n3×101²", "#eeeeee"),
        ("Conv 7×7\n32×101²", "#ffd9b3"),
        ("Stage 1 ×3\n32×101²", "#cce5ff"),
        ("Shield 1×1\n→16", "#ffe0e6"),
        ("Stage 2 ×3 /2\n32×51²", "#cce5ff"),
        ("Shield", "#ffe0e6"),
        ("Stage 3 ×3 /2\n32×26²", "#cce5ff"),
        ("Shield", "#ffe0e6"),
        ("Stage 4 ×3 /2\n32×13²", "#cce5ff"),
        ("Shield", "#ffe0e6"),
        ("Stage 5 ×3 /2\n32×7²", "#cce5ff"),
        ("AvgPool\nFC(32→1)", "#d1f0d1"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 1.7))
    x0, w, h, gap = 0.0, 1.15, 1.1, 0.12
    for i, (label, color) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x0, 0), w, h, facecolor=color, edgecolor="black", lw=0.8))
        ax.text(x0 + w / 2, h / 2, label, ha="center", va="center", fontsize=7.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x0 + w + gap, h / 2), xytext=(x0 + w, h / 2),
                        arrowprops=dict(arrowstyle="->", lw=0.8))
        x0 += w + gap
    ax.set_xlim(-0.1, x0 + 0.1); ax.set_ylim(-0.05, h + 0.05); ax.axis("off")
    ax.set_title("Huang+2021 shielded ResNet — 4 1×1 shields, ~60K params", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "shielded_arch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[done] shielded_arch.png")


def main() -> None:
    plot_curves()
    plot_roc()
    plot_arch()


if __name__ == "__main__":
    main()
