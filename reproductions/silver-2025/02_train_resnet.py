#!/usr/bin/env python3
"""
02_train_resnet.py  --  Train the Silver+2025 "Model 1a" (HST-long) shielded ResNet.

Reproduces the classification training of Silver+2025 §3.1.6 / §3.2.1.1 for the
conventional lens regime (0.5" < theta_E < 1.5"), whose published validation AUC
is 0.9978 (reached by ~epoch 150).

Architecture: the Huang+2021 "shielded" ResNet (reproductions/huang-2021/
01b_shielded_resnet.py) — exactly the model Silver cites ("based on work in H21",
"32 filters in each shielding layer" for Model 1). We instantiate it with
in_channels=1 (single F606W-like band) and the default 32-channel shields/final.

Noise as an in-training augmentation layer (paper §3.1.5): the sims on disk are
NOISELESS and already mean/std-normalized + 99th-pct clipped. During training,
for EACH image EACH iteration we add:
  - Poisson noise with texp ~ 10^U(2,6) s  (exposure-time scaled)
  - Gaussian background noise sigma_BKG ~ U(0, 0.2)
Then (paper, Model 1) the image is NOT renormalized after noise (Model 1 normalizes
BEFORE noise so faint/bright systems see comparable relative noise). We follow that:
normalize at sim time, add noise live, feed to the net.

Training hyperparameters (paper §3.1.6, Model 1):
  - lr0 = 1.25e-3, decay 5x every 80 epochs
  - 360 epochs, batch size 64
  - 80/20 train/val split
  - Adam, BCEWithLogits
  - rotation/flip augmentation (standard; paper uses image augmentation)
Reports validation AUC each epoch; saves best checkpoint + history JSON.

Run pinned to ONE L4 (index 8 or 9):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
    /home/benson/.venvs/lens/bin/python 02_train_resnet.py \
      --images data/model1_images.npy --labels data/model1_labels.npy \
      --epochs 360 --batch 64 --tag model1a
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Import the shielded ResNet from the huang-2021 reproduction (filename starts w/ digit).
_H21 = HERE.parent / "huang-2021" / "01b_shielded_resnet.py"
_spec = importlib.util.spec_from_file_location("shielded_resnet", str(_H21))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ShieldedDeepLens = _mod.ShieldedDeepLens
count_params = _mod.count_params


# --------------------------- Dataset ----------------------------------------
class SimLensDataset(Dataset):
    """Holds noiseless normalized sims in memory; applies rotation/flip aug on the
    fly when training. Noise augmentation is applied on the GPU in the train loop
    (vectorized per batch) so it varies every iteration as the paper specifies."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool):
        self.images = torch.from_numpy(images).float()   # (N,1,H,W)
        self.labels = torch.from_numpy(labels).float()   # (N,)
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i]
        if self.train:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=(2,))
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=(1,))
        return x, self.labels[i]


def add_noise_layer(x: torch.Tensor, sigma_bkg_hi: float = 0.2) -> torch.Tensor:
    """Paper §3.1.5 noise augmentation for Model 1a (HST-long), applied per-batch.

    x: (B,1,H,W) normalized noiseless images on device.
    sigma_BKG ~ U(0, sigma_bkg_hi); texp ~ 10^U(2,6); Poisson + Gaussian background.

    The sims are mean/std-normalized, so they carry both signs. We treat the
    positive part as a photon-rate-like signal: Poisson variance ~ signal/texp
    (texp large => low Poisson noise), implemented via a Gaussian approximation so
    it stays differentiable-friendly and fast. We then add Gaussian background.
    Each image in the batch gets its OWN sigma_BKG and texp (varies every iter)."""
    B = x.shape[0]
    dev = x.device
    log_texp = torch.empty(B, 1, 1, 1, device=dev).uniform_(2.0, 6.0)
    texp = torch.pow(10.0, log_texp)
    sigma_bkg = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.0, sigma_bkg_hi)
    # Poisson (Gaussian approx): variance = |signal| / texp
    poisson_std = torch.sqrt(torch.clamp(x.abs(), min=0.0) / texp)
    x = x + poisson_std * torch.randn_like(x)
    # Gaussian background
    x = x + sigma_bkg * torch.randn_like(x)
    return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    tot_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = add_noise_layer(x)            # validation also sees noise (paper augments at eval-time too)
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        tot_loss += loss.item() * len(y); n += len(y)
        all_logits.append(torch.sigmoid(logit).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_logits); labs = np.concatenate(all_labels)
    auc = roc_auc_score(labs, probs) if len(np.unique(labs)) > 1 else float("nan")
    return tot_loss / n, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/model1_images.npy")
    ap.add_argument("--labels", default="data/model1_labels.npy")
    ap.add_argument("--epochs", type=int, default=360)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr0", type=float, default=1.25e-3)
    ap.add_argument("--lr_decay_epochs", type=int, default=80)
    ap.add_argument("--lr_decay_factor", type=float, default=5.0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--tag", default="model1a")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  tag={args.tag}", flush=True)

    images = np.load(HERE / args.images if not Path(args.images).is_absolute() else args.images)
    labels = np.load(HERE / args.labels if not Path(args.labels).is_absolute() else args.labels)
    print(f"[train] data {images.shape} labels: {int(labels.sum())} pos / "
          f"{len(labels)-int(labels.sum())} neg", flush=True)

    # stratified 80/20 split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(labels))
    val_idx = []
    for c in (0, 1):
        ci = idx[labels == c]
        rng.shuffle(ci)
        nval = int(round(args.val_frac * len(ci)))
        val_idx.append(ci[:nval])
    val_idx = np.concatenate(val_idx)
    train_mask = np.ones(len(labels), bool); train_mask[val_idx] = False
    tr_i, va_i = idx[train_mask], val_idx
    print(f"[train] split: {len(tr_i)} train / {len(va_i)} val", flush=True)

    train_ds = SimLensDataset(images[tr_i], labels[tr_i], train=True)
    val_ds = SimLensDataset(images[va_i], labels[va_i], train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    in_ch = images.shape[1]
    model = ShieldedDeepLens(in_channels=in_ch, final_out=32).to(device)
    print(f"[train] ShieldedDeepLens(in_ch={in_ch}) params={count_params(model):,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr0)

    def lr_at(epoch):
        n_dec = epoch // args.lr_decay_epochs
        return args.lr0 / (args.lr_decay_factor ** n_dec)

    history = []
    best_auc, best_loss = 0.0, math.inf
    ckpt_path = DATA / f"checkpoint_best_{args.tag}.pt"
    hist_path = DATA / f"training_history_{args.tag}.json"

    t0 = time.time()
    for epoch in range(args.epochs):
        lr = lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = add_noise_layer(x)
            opt.zero_grad(set_to_none=True)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        tr_loss = tot / n
        val_loss, val_auc = evaluate(model, val_dl, device)
        history.append(dict(epoch=epoch, lr=lr, train_loss=tr_loss,
                            val_loss=val_loss, val_auc=val_auc))
        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc; improved += "*AUC"
            torch.save(dict(state_dict=model.state_dict(), in_ch=in_ch,
                            final_out=32, epoch=epoch, val_auc=val_auc), ckpt_path)
        if val_loss < best_loss:
            best_loss = val_loss; improved += "*loss"
        dt = time.time() - t0
        print(f"[ep {epoch:3d}] lr={lr:.2e} tr_loss={tr_loss:.4f} "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
              f"best_auc={best_auc:.4f} {improved}  ({dt:.0f}s)", flush=True)
        with open(hist_path, "w") as f:
            json.dump(dict(args=vars(args), history=history,
                           best_auc=best_auc, best_loss=best_loss), f, indent=1)

    print(f"[train] DONE. best val AUC={best_auc:.4f}  best val loss={best_loss:.4f}")
    print(f"  checkpoint -> {ckpt_path}")
    print(f"  history    -> {hist_path}")


if __name__ == "__main__":
    main()
