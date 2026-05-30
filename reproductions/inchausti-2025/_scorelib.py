#!/usr/bin/env python3
"""
_scorelib.py  (shared helper module, not a numbered pipeline step)

Load a trained checkpoint (shielded ResNet / EfficientNetV2 / L18) and score
FITS cutouts to lens-class probabilities, reusing the exact per-band
normalisation from _trainlib. Used by 07 (meta-learner features), 11 (DR8
ensemble re-score) and 13 (direct candidate scoring).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

import _trainlib as TL

HERE = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(HERE / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SR = _load_module("shielded_resnet", "01b_shielded_resnet.py")
_L18 = _load_module("lanusse_resnet", "01_lanusse_resnet.py")
ShieldedDeepLens = _SR.ShieldedDeepLens
CMUDeepLens = _L18.CMUDeepLens


def load_checkpoint_model(ckpt_path, device):
    """Reconstruct a model from a checkpoint dict.

    Returns (model, arch, mean(3,1,1), std(3,1,1), val_auc). `arch` in
    {'shielded','efficientnet','l18'}.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "shielded")
    mean = np.array(ckpt["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(ckpt["std"], dtype=np.float32).reshape(3, 1, 1)
    if arch == "efficientnet":
        eff = _load_module("efficientnet", "02_efficientnet.py")
        model = eff.EfficientNetV2Lens(pretrained=False, variant=ckpt["variant"],
                                       head_dim=ckpt["head_dim"],
                                       num_classes=ckpt["num_classes"])
    elif arch == "shielded":
        cfg = ckpt.get("shielded_cfg", {"final_out": int(ckpt.get("final_out", 32))})
        model = ShieldedDeepLens(in_channels=3, **cfg)
    else:  # l18
        model = CMUDeepLens(in_channels=3)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, arch, mean, std, float(ckpt.get("val_auc", 0.0))


@torch.no_grad()
def score_paths(paths, model, arch, mean, std, device, batch=256):
    """Score a list of FITS paths -> np.float32 probabilities aligned with `paths`.

    NaN for files that are missing or unreadable. Normalisation matches the
    training Dataset: (x-mean)/std clamped to +/-250.
    """
    paths = [Path(p) for p in paths]
    probs = np.full(len(paths), np.nan, dtype=np.float32)
    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)
    buf_idx, buf_x = [], []

    def flush():
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x))
        x = torch.clamp((x - mean_t) / std_t, -250.0, 250.0).to(device)
        p = TL.model_prob(model, x, arch).cpu().numpy()
        for j, i in enumerate(buf_idx):
            probs[i] = p[j]
        buf_idx.clear()
        buf_x.clear()

    for i, p in enumerate(paths):
        try:
            arr = TL.load_fits_cube(p)
        except Exception:
            continue
        if arr.shape != (3, 101, 101):
            continue
        buf_idx.append(i)
        buf_x.append(arr)
        if len(buf_x) >= batch:
            flush()
    flush()
    return probs
