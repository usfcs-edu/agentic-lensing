#!/usr/bin/env python3
"""
_trainlib.py  (shared helper module, not a numbered pipeline step)

Common dataset / split / preprocessing / scoring code for the Phase-5 base
models so the shielded ResNet, EfficientNetV2, and the meta-learner all see a
byte-identical train/val/test partition and identical per-band normalisation.
That identity is REQUIRED for the meta-learner: it stacks the two base models'
probabilities, so the two bases must be trained and scored on exactly the same
rows with the same preprocessing.

Lifted verbatim (Dataset / build_split / compute_band_stats / SEED) from
huang-2021/05_train_shielded.py, which itself matches the huang-2020 L18 split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.io import fits
from torch.utils.data import Dataset
from torchvision import transforms as T

SEED = 2026
DR_TO_FITS = {"dr9": "cutouts_fits_dr9", "dr7": "cutouts_fits_dr7_train"}


# ----- FITS I/O -------------------------------------------------------------

def load_fits_cube(path: Path) -> np.ndarray:
    """Load grz FITS cube -> (3, H, W) float32."""
    with fits.open(path, memmap=False) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=0)
    elif data.shape[0] != 3:
        raise ValueError(f"{path}: unexpected shape {data.shape}")
    return data


# ----- Dataset --------------------------------------------------------------

class LensDataset(Dataset):
    """Loads FITS cubes lazily; per-band mean/std normalise + clamp +/-250;
    training augmentation = random rot[-90,90], h/v flip, zoom 0.9-1.0x.

    The same preprocessing is used for the shielded ResNet AND EfficientNetV2
    (we deliberately do NOT use ImageNet normalisation for the pretrained net —
    fine-tuning adapts its stem to the astronomical-flux domain, and identical
    inputs keep the meta-learner's two base features comparable)."""

    def __init__(self, df: pd.DataFrame, fits_dir: Path,
                 mean: np.ndarray, std: np.ndarray, train: bool):
        self.df = df.reset_index(drop=True)
        self.fits_dir = Path(fits_dir)
        self.mean = torch.from_numpy(mean.reshape(3, 1, 1).astype(np.float32))
        self.std = torch.from_numpy(std.reshape(3, 1, 1).astype(np.float32))
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        # honor a per-row fits_dir column if present (Stage-B mixes cutout dirs)
        base = Path(row["fits_dir"]) if "fits_dir" in row.index and isinstance(
            row["fits_dir"], str) else self.fits_dir
        path = base / f"{row['row_id']}.fits"
        arr = load_fits_cube(path)
        x = torch.from_numpy(arr)
        x = (x - self.mean) / self.std
        x = torch.clamp(x, -250.0, 250.0)
        if self.train:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            angle = (torch.rand(1).item() * 180.0) - 90.0
            x = T.functional.rotate(x, angle, interpolation=T.InterpolationMode.BILINEAR)
            if torch.rand(1).item() < 0.7:
                scale = 0.9 + 0.1 * torch.rand(1).item()
                new_size = max(64, int(round(x.shape[-1] * scale)))
                x = T.functional.resize(x, [new_size, new_size],
                                        interpolation=T.InterpolationMode.BILINEAR,
                                        antialias=True)
                x = T.functional.center_crop(x, [101, 101])
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y


# ----- Preprocessing stats --------------------------------------------------

def compute_band_stats(df_train: pd.DataFrame, fits_dir: Path,
                       n_sample: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    sample = df_train.iloc[rng.choice(len(df_train),
                                      size=min(n_sample, len(df_train)),
                                      replace=False)]
    stack = []
    for _, r in sample.iterrows():
        p = Path(fits_dir) / f"{r['row_id']}.fits"
        if not p.exists():
            continue
        stack.append(load_fits_cube(p))
    cube = np.stack(stack, axis=0)
    mean = cube.mean(axis=(0, 2, 3))
    std = cube.std(axis=(0, 2, 3)) + 1e-8
    return mean, std


# ----- Split (identical to the huang-2020/2021 L18 + shielded runs) ---------

def build_split(pos: pd.DataFrame, neg: pd.DataFrame, fits_dir: Path,
                seed: int = SEED) -> pd.DataFrame:
    rows = []
    for _, r in pos.iterrows():
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"]})
    for _, r in neg.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"]})
    df = pd.DataFrame(rows)
    fits_dir = Path(fits_dir)
    df["fits_exists"] = df["row_id"].apply(lambda r: (fits_dir / f"{r}.fits").exists())
    n_pre = len(df)
    df = df[df["fits_exists"]].drop(columns=["fits_exists"]).reset_index(drop=True)
    print(f"[split] kept {len(df)}/{n_pre} rows with FITS on disk "
          f"({df['label'].sum()} positives)")
    rng = np.random.default_rng(seed)
    assign = []
    for lab in (0, 1):
        idx = df.index[df["label"] == lab].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(0.7 * n))
        n_val = int(round(0.2 * n))
        bucket = np.array(["test"] * n, dtype=object)
        bucket[:n_train] = "train"
        bucket[n_train:n_train + n_val] = "val"
        assign.append(pd.DataFrame({"index": idx, "split": bucket}))
    a = pd.concat(assign).sort_values("index").reset_index(drop=True)
    df["split"] = a["split"].values
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"[split] split counts:\n{counts}")
    return df


# ----- Scoring helper (used by 07 / 11 / 13) --------------------------------

@torch.no_grad()
def model_prob(model: nn.Module, x: torch.Tensor, arch: str) -> torch.Tensor:
    """Return the lens-class probability for a batch, per architecture.

    shielded / l18 -> single logit -> sigmoid.
    efficientnet   -> 2 logits     -> softmax(...)[:, 1].
    """
    out = model(x)
    if arch == "efficientnet":
        return torch.softmax(out, dim=1)[:, 1]
    return torch.sigmoid(out)
