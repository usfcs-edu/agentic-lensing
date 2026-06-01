#!/usr/bin/env python3
"""
05_train_resnet.py

Train the Lanusse+2018 ResNet-46 on Huang+2020-style (DECaLS grz, 101x101)
cutouts following Huang's §3.2 hyperparameter recipe.

Inputs (built by scripts 02-04 of this directory):
  data/positives_huang2020.parquet      # 949 L18-model lens candidates
  data/negatives.parquet                # 5,000 DR1 galaxies in DECaLS footprint
  data/cutouts_fits_dr9/<row_id>.fits   # grz FITS cubes (script 03 pulls these)

Hyperparameters (Lanusse §3.4, repeated unchanged by Huang+2020 §3.2):
  optimizer       Adam, default β
  lr_init         1e-3
  lr_decay        / 10 every 40 epochs
  batch_size      128
  epochs          120
  loss            BCE with logits (binary cross-entropy)
  preprocessing   per-band mean subtraction, std-normalisation
  augmentation    random rotation [-90, 90°], mirroring, zoom [0.9, 1.0]
  split           70 / 20 / 10  train / val / test  (per-row_id, seeded)

DR7 variant: reads cutouts_fits_dr7_train/ and writes checkpoint_best_dr7.pt
(plus training_history_dr7.json + test_result_dr7.json + training_split_dr7.parquet)
so the original DR9-trained Phase 3a checkpoint stays available for comparison.

Outputs:
  data/training_split_dr7.parquet      row_id -> {train, val, test} + label
  data/checkpoint_best_dr7.pt          state_dict at best val AUC
  data/training_history_dr7.json       per-epoch (train_loss, val_loss, val_auc, lr)
  data/test_result_dr7.json            held-out test set predictions
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Local import — the model
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "lanusse_resnet", str(Path(__file__).resolve().parent / "01_lanusse_resnet.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CMUDeepLens = _mod.CMUDeepLens

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import pick_device, pin_ok  # MPS/CUDA/CPU device selection


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FITS_DIR = DATA / "cutouts_fits_dr7_train"

POSITIVES = DATA / "positives_huang2020.parquet"
NEGATIVES = DATA / "negatives.parquet"

DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 120
DEFAULT_LR = 1e-3
LR_DECAY_EPOCH = 40
LR_DECAY_FACTOR = 10.0
SEED = 2026


# ----- Dataset --------------------------------------------------------------

def load_fits_cube(path: Path) -> np.ndarray:
    """Load grz FITS cube → (3, H, W) float32."""
    with fits.open(path, memmap=False) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=0)
    elif data.shape[0] != 3:
        raise ValueError(f"{path}: unexpected shape {data.shape}")
    return data


class LensDataset(Dataset):
    """Loads FITS cubes lazily; applies preprocessing + augmentation.

    Preprocessing (Lanusse §3.3): per-band mean subtraction + std-normalisation
    using global statistics computed from the training set. Clipped to ±250σ.

    Augmentation (training only): random rotation [-90, 90°], horizontal +
    vertical flips, zoom 0.9-1.0× via random resized crop back to 101.
    """

    def __init__(self, df: pd.DataFrame, fits_dir: Path,
                 mean: np.ndarray, std: np.ndarray, train: bool):
        self.df = df.reset_index(drop=True)
        self.fits_dir = fits_dir
        self.mean = torch.from_numpy(mean.reshape(3, 1, 1).astype(np.float32))
        self.std = torch.from_numpy(std.reshape(3, 1, 1).astype(np.float32))
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        path = self.fits_dir / f"{row['row_id']}.fits"
        arr = load_fits_cube(path)               # (3, H, W) float32
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
    """Sample up to n_sample training FITS files; compute per-band mean+std."""
    rng = np.random.default_rng(SEED)
    sample = df_train.iloc[rng.choice(len(df_train),
                                        size=min(n_sample, len(df_train)),
                                        replace=False)]
    stack = []
    for _, r in sample.iterrows():
        p = fits_dir / f"{r['row_id']}.fits"
        if not p.exists():
            continue
        stack.append(load_fits_cube(p))
    cube = np.stack(stack, axis=0)  # (N, 3, H, W)
    mean = cube.mean(axis=(0, 2, 3))
    std = cube.std(axis=(0, 2, 3)) + 1e-8
    return mean, std


# ----- Main training loop ---------------------------------------------------

def build_split(pos: pd.DataFrame, neg: pd.DataFrame, fits_dir: Path,
                seed: int = SEED) -> pd.DataFrame:
    """Concatenate pos+neg with row_id + label, drop rows whose FITS is missing,
    and assign a stable train/val/test bucket per row using a seeded hash."""
    rows = []
    for _, r in pos.iterrows():
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"]})
    for _, r in neg.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"]})
    df = pd.DataFrame(rows)
    df["fits_exists"] = df["row_id"].apply(
        lambda r: (fits_dir / f"{r}.fits").exists()
    )
    n_pre = len(df)
    df = df[df["fits_exists"]].drop(columns=["fits_exists"]).reset_index(drop=True)
    print(f"[split] kept {len(df)}/{n_pre} rows with FITS on disk "
          f"({df['label'].sum()} positives)")
    # Stratified seeded shuffle per label
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


def train(args) -> None:
    device = pick_device()
    print(f"[train] device: {device}")
    pos = pd.read_parquet(POSITIVES)
    neg = pd.read_parquet(NEGATIVES)
    df = build_split(pos, neg, FITS_DIR, seed=SEED)
    df.to_parquet(DATA / "training_split_dr7.parquet", index=False)

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"[stat] computing per-band mean/std from {min(500, len(df_train))} training cutouts")
    mean, std = compute_band_stats(df_train, FITS_DIR, n_sample=500)
    print(f"[stat] mean={mean.tolist()},  std={std.tolist()}")

    train_ds = LensDataset(df_train, FITS_DIR, mean, std, train=True)
    val_ds = LensDataset(df_val, FITS_DIR, mean, std, train=False)
    test_ds = LensDataset(df_test, FITS_DIR, mean, std, train=False)

    pin = pin_ok(device)
    persist = args.workers > 0
    dl_kw = dict(num_workers=args.workers, pin_memory=pin,
                 persistent_workers=persist)
    if persist:
        dl_kw["prefetch_factor"] = 4
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **dl_kw)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, **dl_kw)

    model = CMUDeepLens(in_channels=3).to(device)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1 and args.dp:
        model = nn.DataParallel(model)
        print(f"[train] wrapped model in DataParallel across {n_gpus} GPUs")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=LR_DECAY_EPOCH, gamma=1.0 / LR_DECAY_FACTOR
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_val_auc = -1.0
    best_path = DATA / "checkpoint_best_dr7.pt"
    hist_path = DATA / "training_history_dr7.json"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logit = model(x)
            loss = loss_fn(logit, y)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(y)
            tr_n += len(y)

        # Val
        model.eval()
        val_logits = []
        val_labels = []
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logit = model(x)
                val_loss += loss_fn(logit, y).item() * len(y)
                val_n += len(y)
                val_logits.append(logit.cpu().numpy())
                val_labels.append(y.cpu().numpy())
        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels)
        try:
            val_auc = float(roc_auc_score(val_labels, val_logits))
        except ValueError:
            val_auc = math.nan
        cur_lr = optim.param_groups[0]["lr"]
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": tr_loss / max(1, tr_n),
            "val_loss": val_loss / max(1, val_n),
            "val_auc": val_auc,
            "elapsed_s": elapsed,
        }
        history.append(row)
        print(f"[e{epoch:>3d}] lr={cur_lr:.1e}  train={row['train_loss']:.4f}  "
              f"val={row['val_loss']:.4f}  val_auc={val_auc:.4f}  "
              f"t={elapsed/60:.1f} min")
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict())
            torch.save({"epoch": epoch, "state_dict": state,
                        "val_auc": val_auc, "mean": mean.tolist(),
                        "std": std.tolist()}, best_path)
        hist_path.write_text(json.dumps(history, indent=2))

    # Test pass with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    test_model = CMUDeepLens(in_channels=3).to(device)
    test_model.load_state_dict(ckpt["state_dict"])
    test_model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device); y = y.to(device)
            logits_all.append(test_model(x).cpu().numpy())
            labels_all.append(y.cpu().numpy())
    logits_all = np.concatenate(logits_all)
    labels_all = np.concatenate(labels_all)
    test_auc = float(roc_auc_score(labels_all, logits_all))
    print(f"\n[test] best_val_auc={best_val_auc:.4f}  test_auc={test_auc:.4f}  "
          f"(at epoch {ckpt['epoch']})")
    (DATA / "test_result_dr7.json").write_text(json.dumps({
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "best_epoch": int(ckpt["epoch"]),
        "n_test": int(len(labels_all)),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dp", action="store_true",
                    help="enable DataParallel across all visible GPUs")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train(args)


if __name__ == "__main__":
    main()
