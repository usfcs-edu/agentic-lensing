#!/usr/bin/env python3
"""
05c_train_northaug.py — Phase 4b north-calibration retrain.

Retrains a model (L18 or shielded) on a NORTH-AUGMENTED training set so it is
calibrated on BASS/MzLS imaging, fixing the L18 over-firing on the northern DR8
footprint (see 18_build_north_train_cutouts.py for the diagnosis).

Training set = south (DR9 cutouts, as in Phase 4a) + north (DR8 cutouts, new):
  positives: 787 DECaLS rows of positives_huang2020 (cutouts_fits_dr9)
           + 162 MzLS rows                          (cutouts_fits_north, DR8)
  negatives: 5000 south DR1 galaxies                (cutouts_fits_dr9)
           + 3000 north galaxies                    (cutouts_fits_north, DR8)

Each row carries its own `fits_dir`; the split, band-stats, and dataset all
respect it. Same Lanusse recipe and SEED as 05_train_shielded.py.

  --arch {l18,shielded}   which network to train
Outputs:
  data/checkpoint_best_<arch>_northaug.pt   (+ mean/std/arch/final_out)
  data/training_history_<arch>_northaug.json
  data/test_result_<arch>_northaug.json
  data/training_split_northaug.parquet
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

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

CMUDeepLens = _load("lanusse_resnet", "01_lanusse_resnet.py").CMUDeepLens
ShieldedDeepLens = _load("shielded_resnet", "01b_shielded_resnet.py").ShieldedDeepLens

import sys
sys.path.insert(0, str(HERE))
from device import pick_device, pin_ok  # MPS/CUDA/CPU device selection

DR9 = DATA / "cutouts_fits_dr9"
NORTH = DATA / "cutouts_fits_north"
SEED = 2026
DEFAULT_BATCH, DEFAULT_EPOCHS, DEFAULT_LR = 128, 120, 1e-3
LR_DECAY_EPOCH, LR_DECAY_FACTOR = 40, 10.0


def load_fits_cube(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=0)
    elif data.shape[0] != 3:
        raise ValueError(f"{path}: shape {data.shape}")
    return data


class LensDataset(Dataset):
    def __init__(self, df, mean, std, train):
        self.df = df.reset_index(drop=True)
        self.mean = torch.from_numpy(mean.reshape(3, 1, 1).astype(np.float32))
        self.std = torch.from_numpy(std.reshape(3, 1, 1).astype(np.float32))
        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = torch.from_numpy(load_fits_cube(Path(row["fits_dir"]) / f"{row['row_id']}.fits"))
        x = torch.clamp((x - self.mean) / self.std, -250.0, 250.0)
        if self.train:
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[1])
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
            x = T.functional.rotate(x, (torch.rand(1).item() * 180.0) - 90.0,
                                    interpolation=T.InterpolationMode.BILINEAR)
            if torch.rand(1).item() < 0.7:
                scale = 0.9 + 0.1 * torch.rand(1).item()
                ns = max(64, int(round(x.shape[-1] * scale)))
                x = T.functional.resize(x, [ns, ns],
                                        interpolation=T.InterpolationMode.BILINEAR, antialias=True)
                x = T.functional.center_crop(x, [101, 101])
        return x, torch.tensor(row["label"], dtype=torch.float32)


def compute_band_stats(df_train, n_sample=600):
    rng = np.random.default_rng(SEED)
    sub = df_train.iloc[rng.choice(len(df_train), size=min(n_sample, len(df_train)), replace=False)]
    stack = [load_fits_cube(Path(r["fits_dir"]) / f"{r['row_id']}.fits")
             for _, r in sub.iterrows()
             if (Path(r["fits_dir"]) / f"{r['row_id']}.fits").exists()]
    cube = np.stack(stack, axis=0)
    return cube.mean(axis=(0, 2, 3)), cube.std(axis=(0, 2, 3)) + 1e-8


def build_rows() -> pd.DataFrame:
    """Combine south (DR9) + north (DR8) positives & negatives into one frame
    with per-row (row_id, label, RA, DEC, fits_dir)."""
    rows = []
    pos = pd.read_parquet(DATA / "positives_huang2020.parquet")
    for _, r in pos[pos["Region"] == "DECaLS"].iterrows():     # south positives @ DR9
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"], "fits_dir": str(DR9)})
    npos = pd.read_parquet(DATA / "positives_north.parquet")   # north positives @ DR8
    for _, r in npos.iterrows():
        rows.append({"row_id": r["Name"], "label": 1, "RA": r["RA"], "DEC": r["DEC"], "fits_dir": str(NORTH)})
    neg = pd.read_parquet(DATA / "negatives.parquet")          # south negatives @ DR9
    for _, r in neg.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"], "fits_dir": str(DR9)})
    nneg = pd.read_parquet(DATA / "negatives_north.parquet")   # north negatives @ DR8
    for _, r in nneg.iterrows():
        rows.append({"row_id": r["row_id"], "label": 0, "RA": r["RA"], "DEC": r["DEC"], "fits_dir": str(NORTH)})
    df = pd.DataFrame(rows)
    df["exists"] = df.apply(lambda r: (Path(r["fits_dir"]) / f"{r['row_id']}.fits").exists(), axis=1)
    n0 = len(df); df = df[df["exists"]].drop(columns="exists").reset_index(drop=True)
    print(f"[rows] {len(df)}/{n0} have cutouts "
          f"({int(df['label'].sum())} pos, {int((df['label']==0).sum())} neg; "
          f"north {int((df['fits_dir']==str(NORTH)).sum())})")
    return df


def split(df, seed=SEED):
    rng = np.random.default_rng(seed)
    parts = []
    for lab in (0, 1):
        idx = df.index[df["label"] == lab].to_numpy().copy(); rng.shuffle(idx)
        n = len(idx); ntr = int(round(0.7 * n)); nva = int(round(0.2 * n))
        b = np.array(["test"] * n, dtype=object); b[:ntr] = "train"; b[ntr:ntr + nva] = "val"
        parts.append(pd.DataFrame({"index": idx, "split": b}))
    a = pd.concat(parts).sort_values("index").reset_index(drop=True)
    df = df.copy(); df["split"] = a["split"].values
    print("[split]\n", df.groupby(["split", "label"]).size().unstack(fill_value=0))
    return df


def make_model(arch):
    return ShieldedDeepLens(in_channels=3, final_out=32) if arch == "shielded" else CMUDeepLens(in_channels=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("l18", "shielded"), required=True)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = pick_device()
    sfx = f"{args.arch}_northaug"
    print(f"[train] arch={args.arch} device={dev}")

    df = build_rows()
    df = split(df)
    df.to_parquet(DATA / "training_split_northaug.parquet", index=False)
    dtr, dva, dte = (df[df.split == s].copy() for s in ("train", "val", "test"))
    mean, std = compute_band_stats(dtr)
    print(f"[stat] mean={mean.tolist()} std={std.tolist()}")

    pin = pin_ok(dev)
    persist = args.workers > 0
    dl_kw = dict(num_workers=args.workers, pin_memory=pin, persistent_workers=persist)
    if persist:
        dl_kw["prefetch_factor"] = 4
    tdl = DataLoader(LensDataset(dtr, mean, std, True), batch_size=args.batch, shuffle=True,
                     drop_last=True, **dl_kw)
    vdl = DataLoader(LensDataset(dva, mean, std, False), batch_size=args.batch, **dl_kw)
    edl = DataLoader(LensDataset(dte, mean, std, False), batch_size=args.batch, **dl_kw)

    model = make_model(args.arch).to(dev)
    nP = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {args.arch} params: {nP:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_EPOCH, gamma=1.0 / LR_DECAY_FACTOR)
    lossf = nn.BCEWithLogitsLoss()
    hist = []; best = -1.0
    bp = DATA / f"checkpoint_best_{sfx}.pt"; hp = DATA / f"training_history_{sfx}.json"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl = tn = 0
        for x, y in tdl:
            x, y = x.to(dev, non_blocking=pin), y.to(dev, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * len(y); tn += len(y)
        model.eval(); vlog = []; vlab = []
        with torch.no_grad():
            for x, y in vdl:
                vlog.append(model(x.to(dev)).cpu().numpy()); vlab.append(y.numpy())
        vlog = np.concatenate(vlog); vlab = np.concatenate(vlab)
        try: vauc = float(roc_auc_score(vlab, vlog))
        except ValueError: vauc = math.nan
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"],
                     "train_loss": tl / max(1, tn), "val_auc": vauc, "elapsed_s": time.time() - t0})
        print(f"[e{ep:>3d}] train={tl/max(1,tn):.4f} val_auc={vauc:.4f} t={(time.time()-t0)/60:.1f}m")
        sch.step()
        if vauc > best:
            best = vauc
            torch.save({"epoch": ep, "state_dict": model.state_dict(), "val_auc": vauc,
                        "mean": mean.tolist(), "std": std.tolist(),
                        "arch": args.arch, "final_out": 32}, bp)
        hp.write_text(json.dumps(hist, indent=2))

    ck = torch.load(bp, map_location=dev); tm = make_model(args.arch).to(dev)
    tm.load_state_dict(ck["state_dict"]); tm.eval()
    el = []; ela = []
    with torch.no_grad():
        for x, y in edl:
            el.append(tm(x.to(dev)).cpu().numpy()); ela.append(y.numpy())
    el = np.concatenate(el); ela = np.concatenate(ela); tauc = float(roc_auc_score(ela, el))
    print(f"[test] best_val_auc={best:.4f} test_auc={tauc:.4f} (epoch {ck['epoch']})")
    (DATA / f"test_result_{sfx}.json").write_text(json.dumps(
        {"arch": args.arch, "northaug": True, "n_params": int(nP),
         "best_val_auc": best, "test_auc": tauc, "best_epoch": int(ck["epoch"]),
         "n_test": int(len(ela))}, indent=2))


if __name__ == "__main__":
    main()
