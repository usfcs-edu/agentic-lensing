#!/usr/bin/env python3
"""
05_train_shielded.py

Train the Huang+2021 SHIELDED ResNet (01b_shielded_resnet.py) on the SAME
DECaLS grz 101×101 cutouts, positives, negatives, seed, and train/val/test
split used by the Huang+2020 (L18) reproduction. This isolates the architecture
as the only variable — a controlled reproduction of the paper's own
L18-vs-shielded comparison (validation AUC 0.992 → 0.997, §3.3).

The L18 baseline checkpoints already exist from the huang-2020 reproduction and
are symlinked into this directory's data/:
  data/checkpoint_best.pt       (DR9-trained L18)   test_auc 0.9991
  data/checkpoint_best_dr7.pt   (DR7-trained L18)   test_auc 0.9943
`06_compare_architectures.py` builds the head-to-head table.

--dr {dr9,dr7} selects the cutout set and output suffix:
  dr9 -> data/cutouts_fits_dr9          -> *_shielded_dr9.*
  dr7 -> data/cutouts_fits_dr7_train    -> *_shielded_dr7.*   (paper-exact DR)

Hyperparameters match Huang's recipe (Lanusse §3.4) unchanged from script 05:
Adam lr 1e-3 /10 every 40 epochs, BCE-with-logits, batch 128, 120 epochs,
per-band mean/std normalisation + clamp ±250, rotation/flip/zoom augmentation,
70/20/10 stratified seeded split (SEED=2026, identical to the L18 runs so the
partition matches row-for-row).

Outputs (suffix = --dr):
  data/training_split_shielded_<dr>.parquet   row_id -> {train,val,test} + label
  data/checkpoint_best_shielded_<dr>.pt       state_dict + mean/std + final_out
  data/training_history_shielded_<dr>.json    per-epoch (loss, val_auc, lr)
  data/test_result_shielded_<dr>.json         held-out test AUC
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.io import fits
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Local import — the shielded model (filename starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "shielded_resnet", str(Path(__file__).resolve().parent / "01b_shielded_resnet.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ShieldedDeepLens = _mod.ShieldedDeepLens

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

DR_TO_FITS = {"dr9": "cutouts_fits_dr9", "dr7": "cutouts_fits_dr7_train"}

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
    """Loads FITS cubes lazily; per-band mean/std normalise + clamp ±250σ;
    training augmentation = random rot[-90,90], h/v flip, zoom 0.9-1.0×."""

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
        p = fits_dir / f"{r['row_id']}.fits"
        if not p.exists():
            continue
        stack.append(load_fits_cube(p))
    cube = np.stack(stack, axis=0)
    mean = cube.mean(axis=(0, 2, 3))
    std = cube.std(axis=(0, 2, 3)) + 1e-8
    return mean, std


# ----- Split (identical to the L18 runs) ------------------------------------

def build_split(pos: pd.DataFrame, neg: pd.DataFrame, fits_dir: Path,
                seed: int = SEED) -> pd.DataFrame:
    rows = []
    for _, r in pos.iterrows():
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"]})
    for _, r in neg.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"]})
    df = pd.DataFrame(rows)
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


# ----- Training -------------------------------------------------------------

def train(args) -> None:
    dr = args.dr
    fits_dir = DATA / DR_TO_FITS[dr]
    sfx = f"shielded_{dr}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] dr={dr}  fits_dir={fits_dir.name}  final_out={args.final_out}  device={device}")

    pos = pd.read_parquet(POSITIVES)
    neg = pd.read_parquet(NEGATIVES)
    df = build_split(pos, neg, fits_dir, seed=SEED)
    df.to_parquet(DATA / f"training_split_{sfx}.parquet", index=False)

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"[stat] computing per-band mean/std from {min(500, len(df_train))} training cutouts")
    mean, std = compute_band_stats(df_train, fits_dir, n_sample=500)
    print(f"[stat] mean={mean.tolist()},  std={std.tolist()}")

    train_ds = LensDataset(df_train, fits_dir, mean, std, train=True)
    val_ds = LensDataset(df_val, fits_dir, mean, std, train=False)
    test_ds = LensDataset(df_test, fits_dir, mean, std, train=False)

    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=pin)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=pin)

    model = ShieldedDeepLens(in_channels=3, final_out=args.final_out).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] shielded params: {n_params:,}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=LR_DECAY_EPOCH, gamma=1.0 / LR_DECAY_FACTOR
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_val_auc = -1.0
    best_path = DATA / f"checkpoint_best_{sfx}.pt"
    hist_path = DATA / f"training_history_{sfx}.json"
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

        model.eval()
        val_logits, val_labels = [], []
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
        row = {"epoch": epoch, "lr": cur_lr,
               "train_loss": tr_loss / max(1, tr_n),
               "val_loss": val_loss / max(1, val_n),
               "val_auc": val_auc, "elapsed_s": elapsed}
        history.append(row)
        print(f"[e{epoch:>3d}] lr={cur_lr:.1e}  train={row['train_loss']:.4f}  "
              f"val={row['val_loss']:.4f}  val_auc={val_auc:.4f}  t={elapsed/60:.1f} min")
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_auc": val_auc, "mean": mean.tolist(),
                        "std": std.tolist(), "final_out": args.final_out,
                        "arch": "shielded"}, best_path)
        hist_path.write_text(json.dumps(history, indent=2))

    # Test pass with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    test_model = ShieldedDeepLens(in_channels=3,
                                  final_out=ckpt.get("final_out", args.final_out)).to(device)
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
    (DATA / f"test_result_{sfx}.json").write_text(json.dumps({
        "arch": "shielded", "dr": dr, "final_out": args.final_out,
        "n_params": int(n_params),
        "best_val_auc": best_val_auc, "test_auc": test_auc,
        "best_epoch": int(ckpt["epoch"]), "n_test": int(len(labels_all)),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr", choices=("dr9", "dr7"), default="dr9")
    ap.add_argument("--final-out", type=int, default=32, dest="final_out")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train(args)


if __name__ == "__main__":
    main()
