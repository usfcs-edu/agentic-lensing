#!/usr/bin/env python3
"""_minelib.py — in-RAM cutout cache + fast cached shielded trainer/scorer for the
Phase-2 hard-negative mining rounds (many retrains -> the per-epoch FITS I/O of
_train.py would dominate; here cutouts are loaded once into RAM).

Augmentation is flips + 90-degree rotations only (torch.rot90, no bilinear), which
is both the standard astronomical augmentation and cheap enough for num_workers=0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import _clib as C
import _trainlib as TL


def _key(fits_dir, row_id):
    return (Path(str(fits_dir)).name, str(row_id))


def load_cache(*frames):
    """Load every (fits_dir,row_id) referenced by the given dataframes into a dict
    key->(3,101,101) float32, once."""
    cache = {}
    for df in frames:
        for r in df.itertuples():
            k = _key(r.fits_dir, r.row_id)
            if k in cache:
                continue
            p = Path(str(r.fits_dir)) / f"{r.row_id}.fits"
            try:
                a = TL.load_fits_cube(p)
            except Exception:
                continue
            if a.shape == (3, 101, 101):
                cache[k] = a
    return cache


class CachedDS(Dataset):
    def __init__(self, rows, cache, mean, std, train):
        self.rows = rows.reset_index(drop=True)
        self.cache = cache
        self.train = train
        self.mean = torch.from_numpy(mean.reshape(3, 1, 1).astype(np.float32))
        self.std = torch.from_numpy(std.reshape(3, 1, 1).astype(np.float32))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = torch.from_numpy(self.cache[_key(r.fits_dir, r.row_id)])
        x = torch.clamp((x - self.mean) / self.std, -250.0, 250.0)
        if self.train:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, [1])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, [2])
            x = torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), dims=[1, 2])
        return x.contiguous(), torch.tensor(float(r.label))


def band_stats(rows, cache, n=500):
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    cubes = [cache[_key(r.fits_dir, r.row_id)] for r in rows.iloc[idx].itertuples()
             if _key(r.fits_dir, r.row_id) in cache]
    c = np.stack(cubes)
    return c.mean((0, 2, 3)), c.std((0, 2, 3)) + 1e-8


def train_shielded(train_df, val_df, cache, device, *, epochs=30, batch=256,
                   lr=1e-3, decay_ep=10, aug_seed=2026):
    torch.manual_seed(aug_seed); np.random.seed(aug_seed)
    mean, std = band_stats(train_df, cache)
    model = C.models()["ShieldedDeepLens"](in_channels=3, **C.CFG194).to(device)
    tr = DataLoader(CachedDS(train_df, cache, mean, std, True), batch_size=batch,
                    shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(CachedDS(val_df, cache, mean, std, False), batch_size=512, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_ep, gamma=0.1)
    lossf = nn.BCEWithLogitsLoss()
    yv = val_df["label"].to_numpy()
    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            lossf(model(x.to(device)), y.to(device)).backward()
            opt.step()
        if ep % 3 == 0 or ep == epochs - 1:
            model.eval(); ps = []
            with torch.no_grad():
                for x, _ in va:
                    ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            auc = roc_auc_score(yv, np.concatenate(ps))
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
    model.load_state_dict(best_state); model.eval()
    return model, mean, std, float(best_auc)


@torch.no_grad()
def score(model, rows, cache, mean, std, device, batch=512):
    mt = torch.from_numpy(mean.reshape(3, 1, 1).astype(np.float32))
    st = torch.from_numpy(std.reshape(3, 1, 1).astype(np.float32))
    out = np.full(len(rows), np.nan, np.float32)
    buf, idx = [], []

    def flush():
        if not buf:
            return
        x = torch.clamp((torch.from_numpy(np.stack(buf)) - mt) / st, -250, 250).to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()
        for j, ii in enumerate(idx):
            out[ii] = p[j]
        buf.clear(); idx.clear()

    for i, r in enumerate(rows.itertuples()):
        k = _key(r.fits_dir, r.row_id)
        if k not in cache:
            continue
        buf.append(cache[k]); idx.append(i)
        if len(buf) >= batch:
            flush()
    flush()
    return out
