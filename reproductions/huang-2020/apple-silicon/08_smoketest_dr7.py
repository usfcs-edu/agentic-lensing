#!/usr/bin/env python3
"""
08_smoketest_dr7.py — Phase 3b M0 decision gate.

Goal: verify that the Phase 3a checkpoint (trained on DECaLS DR9) still
scores known Huang+2020 Grade-A lenses highly when applied to DECaLS DR7
cutouts. If most picks score >= 0.5, proceed with the full DR7 sweep (M1).
If they consistently underperform, plan a DR7 fine-tune step first.

Inputs:
  - data/checkpoint_best.pt   (Phase 3a trained weights + mean/std)
  - hardcoded list of 6 published Grade-A candidates (Huang+2020 Tables/Figure 4)

Outputs (printed only — no files created in M0):
  per-target: ra, dec, dr7-cutout-size-bytes, sigmoid-score
  summary:    fraction passing >= 0.5
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from astropy.io import fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402
_mod = import_module("01_lanusse_resnet")
CMUDeepLens = _mod.CMUDeepLens
from device import pick_device  # noqa: E402  (MPS/CUDA/CPU device selection)


# Six published Grade-A candidates from Huang+2020 (chosen for spread in RA + Dec).
# (name, RA_deg, Dec_deg) — parsed from "DESI-RRR.RRRR±DD.DDDD" tokens in paper Table 1.
TARGETS = [
    ("DESI-011.5084-01.9412", 11.5084, -1.9412),
    ("DESI-015.8164+00.0823", 15.8164,  0.0823),
    ("DESI-122.0852+10.5284", 122.0852, 10.5284),
    ("DESI-186.3033-00.4390", 186.3033, -0.4390),
    ("DESI-204.0002-03.5250", 204.0002, -3.5250),  # the 40.3" showpiece, with redshift zd
    ("DESI-318.0376-01.7568", 318.0376, -1.7568),
]


def fetch_dr7_cutout(ra: float, dec: float, size: int = 101, pixscale: float = 0.262) -> bytes:
    """Download a 101x101 grz FITS cube from the legacysurvey viewer (DR7 layer)."""
    url = (f"https://www.legacysurvey.org/viewer/fits-cutout"
           f"?ra={ra:.6f}&dec={dec:.6f}&size={size}&layer=decals-dr7"
           f"&pixscale={pixscale}&bands=grz")
    for attempt in range(1, 6):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            if len(r.content) < 256:
                raise RuntimeError(f"cutout too small ({len(r.content)} bytes)")
            return r.content
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(4 * attempt)
    raise RuntimeError("unreachable")


def fits_bytes_to_cube(fits_bytes: bytes) -> np.ndarray:
    """Returns shape (3, H, W) float32 array."""
    with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
        data = hdul[0].data
    if data is None or data.ndim != 3 or data.shape[0] != 3:
        raise ValueError(f"unexpected FITS shape: {None if data is None else data.shape}")
    return data.astype(np.float32)


def main() -> None:
    ckpt_path = HERE / "data" / "checkpoint_best.pt"
    print(f"[init] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(ckpt["std"], dtype=np.float32).reshape(3, 1, 1)
    print(f"[init] checkpoint epoch={ckpt['epoch']} val_auc={ckpt['val_auc']:.4f}")
    print(f"[init] mean={mean.flatten().tolist()}  std={std.flatten().tolist()}")

    device = pick_device()
    model = CMUDeepLens(in_channels=3).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[init] model on {device}")

    print()
    print(f"{'name':32s}  {'RA':>10s} {'Dec':>9s}  {'bytes':>8s}  {'sigmoid':>8s}")
    print("-" * 72)
    scores = []
    for (name, ra, dec) in TARGETS:
        try:
            raw = fetch_dr7_cutout(ra, dec)
            cube = fits_bytes_to_cube(raw)  # (3,101,101)
            x = (cube - mean) / std
            x = np.clip(x, -250.0, 250.0)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(xt).item()
            prob = 1.0 / (1.0 + np.exp(-logit))
            scores.append(prob)
            print(f"{name:32s}  {ra:10.4f} {dec:9.4f}  {len(raw):8d}  {prob:8.4f}")
        except Exception as e:
            print(f"{name:32s}  {ra:10.4f} {dec:9.4f}  FAILED: {e}")
            scores.append(float("nan"))

    arr = np.array(scores)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print("\n[M0] all downloads failed — investigate before proceeding")
        sys.exit(1)
    pass_rate = float((valid >= 0.5).sum()) / len(valid)
    print()
    print(f"[M0] n_valid={len(valid)}/{len(TARGETS)}  mean_score={valid.mean():.3f}  "
          f"median={np.median(valid):.3f}  pass(>=0.5)={pass_rate:.0%}")
    if pass_rate >= 0.5:
        print("[M0] PASS — proceed to M1 (DR7 sweep download).")
    else:
        print("[M0] FAIL — most Grade-A's score < 0.5. Consider DR7 fine-tune before M2.")


if __name__ == "__main__":
    main()
