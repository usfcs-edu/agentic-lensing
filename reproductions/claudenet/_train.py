#!/usr/bin/env python3
"""_train.py — recipe-faithful supervised trainer for ClaudeNet ensemble members.

Mirrors inchausti-2025/19_train_stageb.train_base (Adam, StepLR gamma=0.1,
BCEWithLogits for single-logit nets / CrossEntropy for EfficientNet, best-val-AUC
checkpoint) but is parameterised over an arbitrary model factory, negative subset,
and augmentation seed so we can train DECORRELATED members. Reuses the inchausti
LensDataset (per-row fits_dir) and SL.score_paths verbatim, so members are
directly comparable to the reproduced baseline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import _trainlib as TL
import _scorelib as SL


def band_stats(df_train, n_sample=500):
    """Per-band mean/std honouring per-row fits_dir (positives and negatives live
    in different cutout dirs)."""
    rng = np.random.default_rng(TL.SEED)
    s = df_train.iloc[rng.choice(len(df_train), size=min(n_sample, len(df_train)), replace=False)]
    cubes = []
    for _, r in s.iterrows():
        p = Path(r["fits_dir"]) / f"{r['row_id']}.fits"
        if p.exists():
            cubes.append(TL.load_fits_cube(p))
    c = np.stack(cubes)
    return c.mean((0, 2, 3)), c.std((0, 2, 3)) + 1e-8


class DihedralPool(nn.Module):
    """Make a single-logit base net invariant to the dihedral group D4 by pooling
    its logit over the 8 (rotation x flip) symmetries — a cheap, escnn-free
    equivariant member with a genuinely different inductive bias."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        outs = []
        for k in range(4):
            xr = torch.rot90(x, k, dims=(2, 3))
            outs.append(self.base(xr))
            outs.append(self.base(torch.flip(xr, dims=(3,))))
        return torch.stack(outs, 0).mean(0)


def _loaders(df, mean, std, batch, workers):
    out = {}
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        if len(sub) == 0:
            out[s] = None
            continue
        out[s] = DataLoader(TL.LensDataset(sub, sub["fits_dir"].iloc[0], mean, std, s == "train"),
                            batch_size=batch, shuffle=(s == "train"), num_workers=workers,
                            drop_last=(s == "train"), pin_memory=True)
    return out


def train_supervised(model, arch, df, device, *, epochs, batch=128, lr=1e-3,
                     decay_ep=15, accum=1, aug_seed=2026, workers=4, verbose=False):
    """Train `model`; arch in {shielded,l18,dihedral,efficientnet}. Returns
    (model, best_val_auc, mean, std). score_arch = 'efficientnet' or 'shielded'."""
    torch.manual_seed(aug_seed)
    np.random.seed(aug_seed)
    score_arch = "efficientnet" if arch == "efficientnet" else "shielded"
    mean, std = band_stats(df[df.split == "train"])
    dls = _loaders(df, mean, std, batch, workers)
    model = model.to(device)
    lossf = nn.CrossEntropyLoss() if arch == "efficientnet" else nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_ep, gamma=0.1)

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        for i, (x, y) in enumerate(dls["train"]):
            x = x.to(device, non_blocking=True)
            if arch == "efficientnet":
                loss = lossf(model(x), y.long().to(device)) / accum
            else:
                loss = lossf(model(x), y.to(device)) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad()
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in dls["val"]:
                ps.append(TL.model_prob(model, x.to(device), score_arch).cpu().numpy())
                ys.append(y.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
        if verbose:
            print(f"  ep{ep:03d} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
    model.load_state_dict(best_state); model.eval()
    return model, float(best_auc), mean, std


def score_df(model, score_arch, df, mean, std, device, batch=256):
    """Score the rows of df (per-row fits_dir) -> probs aligned to df order."""
    paths = [Path(r.fits_dir) / f"{r.row_id}.fits" for r in df.itertuples()]
    return SL.score_paths(paths, model, score_arch, mean, std, device, batch=batch)
