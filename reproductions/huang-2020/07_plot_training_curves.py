#!/usr/bin/env python3
"""
07_plot_training_curves.py

Produce the two figures embedded in the Phase 3a tech-report:

  papers/figures/training_curves.png   train_loss + val_auc vs epoch, twin axes
  papers/figures/roc_curve.png         ROC on the held-out test set

Inputs:
  data/training_history.json   (written by 05_train_resnet.py)
  data/test_result.json
  data/checkpoint_best.pt
  data/training_split.parquet  (to identify test rows)
  data/cutouts_fits_dr9/<row_id>.fits

ROC is computed by a single forward pass over the test split with the saved
checkpoint, using the dataset preprocessing from 05_train_resnet.py (same
mean/std stored in the checkpoint).
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
FITS_DIR = DATA / "cutouts_fits_dr9"
FIG_DIR = HERE / "papers" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CKPT = DATA / "checkpoint_best.pt"
HIST = DATA / "training_history.json"
TEST = DATA / "test_result.json"
SPLIT = DATA / "training_split.parquet"


def load_model_module():
    """Load 01_lanusse_resnet.py + 05_train_resnet.py (numeric filenames need
    importlib gymnastics)."""
    arch_spec = importlib.util.spec_from_file_location(
        "lanusse_resnet", HERE / "01_lanusse_resnet.py"
    )
    arch = importlib.util.module_from_spec(arch_spec)
    arch_spec.loader.exec_module(arch)
    train_spec = importlib.util.spec_from_file_location(
        "train_resnet", HERE / "05_train_resnet.py"
    )
    train = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train)
    return arch.CMUDeepLens, train.LensDataset


def plot_training_curves() -> None:
    hist = json.loads(HIST.read_text())
    df = pd.DataFrame(hist)

    fig, ax_loss = plt.subplots(figsize=(6.5, 3.6))
    ax_auc = ax_loss.twinx()

    # Train + val losses on left axis
    ax_loss.plot(df["epoch"], df["train_loss"], color="#1f77b4",
                 lw=1.4, label="train loss")
    ax_loss.plot(df["epoch"], df["val_loss"], color="#ff7f0e",
                 lw=1.4, alpha=0.8, label="val loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("BCE-with-logits loss")
    ax_loss.set_ylim(0, max(df["val_loss"].max(), df["train_loss"].max()) * 1.1)

    # Val AUC on right axis
    ax_auc.plot(df["epoch"], df["val_auc"], color="#2ca02c",
                lw=1.6, label="val AUC")
    ax_auc.set_ylabel("validation AUC", color="#2ca02c")
    ax_auc.tick_params(axis="y", colors="#2ca02c")
    ax_auc.set_ylim(0.0, 1.02)

    # LR decay markers
    for e in (40, 80):
        ax_loss.axvline(e, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax_loss.text(e + 0.5, 0.05 * ax_loss.get_ylim()[1],
                     f"LR /10", fontsize=8, color="gray", rotation=90,
                     va="bottom")

    # Best-epoch marker
    best = max(hist, key=lambda r: r["val_auc"])
    ax_auc.scatter([best["epoch"]], [best["val_auc"]], color="black",
                   s=22, zorder=10)
    ax_auc.annotate(f"best @ e{best['epoch']}: {best['val_auc']:.4f}",
                    xy=(best["epoch"], best["val_auc"]),
                    xytext=(best["epoch"] + 5, best["val_auc"] - 0.06),
                    fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))

    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_auc.get_legend_handles_labels()
    ax_loss.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9, frameon=False)

    fig.tight_layout()
    out = FIG_DIR / "training_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out}")


def plot_roc() -> None:
    """Single forward pass on the held-out test split, plot the ROC."""
    CMUDeepLens, LensDataset = load_model_module()
    ckpt = torch.load(CKPT, map_location="cpu")
    mean = np.array(ckpt["mean"])
    std = np.array(ckpt["std"])

    split = pd.read_parquet(SPLIT)
    test_df = split[split["split"] == "test"].reset_index(drop=True).copy()
    print(f"[roc] {len(test_df)} test rows  "
          f"({test_df['label'].sum()} positives)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMUDeepLens(in_channels=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = LensDataset(test_df, FITS_DIR, mean, std, train=False)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)

    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in dl:
            logits_all.append(model(x.to(device)).cpu().numpy())
            labels_all.append(y.cpu().numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    fpr, tpr, _ = roc_curve(labels, logits)
    auc = roc_auc_score(labels, logits)
    print(f"[roc] test AUC = {auc:.4f}")

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(fpr, tpr, color="#1f77b4", lw=1.5,
            label=f"Lanusse ResNet-46 (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", alpha=0.6,
            label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()
    out = FIG_DIR / "roc_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out}")


def plot_arch_schematic() -> None:
    """Simple 5-stage box diagram of the Lanusse-2018 ResNet-46."""
    stages = [
        ("Input\n3 × 101 × 101", "#eeeeee"),
        ("Conv 7×7 + ELU + BN\n32 × 101 × 101", "#ffd9b3"),
        ("Stage 1 ×3\nResNet-16-32\n32 × 101 × 101", "#cce5ff"),
        ("Stage 2 ×3\nResNet-32-64 (/2)\n64 × 51 × 51", "#cce5ff"),
        ("Stage 3 ×3\nResNet-64-128 (/2)\n128 × 26 × 26", "#cce5ff"),
        ("Stage 4 ×3\nResNet-128-256 (/2)\n256 × 13 × 13", "#cce5ff"),
        ("Stage 5 ×3\nResNet-256-512 (/2)\n512 × 7 × 7", "#cce5ff"),
        ("AvgPool + FC(512→1)\n+ Sigmoid", "#d1f0d1"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 1.6))
    x0 = 0.0
    box_w = 1.35
    box_h = 1.1
    gap = 0.13
    for i, (label, color) in enumerate(stages):
        ax.add_patch(plt.Rectangle((x0, 0), box_w, box_h, facecolor=color,
                                    edgecolor="black", lw=0.8))
        ax.text(x0 + box_w / 2, box_h / 2, label, ha="center", va="center",
                fontsize=8)
        # Arrow between boxes
        if i < len(stages) - 1:
            ax.annotate("", xy=(x0 + box_w + gap, box_h / 2),
                         xytext=(x0 + box_w, box_h / 2),
                         arrowprops=dict(arrowstyle="->", lw=0.8))
        x0 += box_w + gap
    ax.set_xlim(-0.1, x0 + 0.1)
    ax.set_ylim(-0.05, box_h + 0.05)
    ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "lanusse_resnet_arch.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out}")


def main() -> None:
    if not HIST.exists():
        raise SystemExit(f"missing {HIST}; run 05_train_resnet.py first")
    plot_training_curves()
    if CKPT.exists() and SPLIT.exists():
        plot_roc()
    else:
        print(f"[skip] roc curve (missing {CKPT} or {SPLIT})")
    plot_arch_schematic()


if __name__ == "__main__":
    main()
