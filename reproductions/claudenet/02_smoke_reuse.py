#!/usr/bin/env python3
"""02_smoke_reuse.py — confirm the reused inchausti surface works under the
claudenet venv on synced data: model param counts, FITS loading, and score_paths.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      /home2/benson/.venvs/claudenet/bin/python 02_smoke_reuse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import _clib as C
import _trainlib as TL
import _scorelib as SL


def main():
    M = C.models()
    sh = M["ShieldedDeepLens"](in_channels=3, **C.CFG194)
    n_sh = sum(p.numel() for p in sh.parameters())
    ef = M["EfficientNetV2Lens"](pretrained=False)
    n_ef = sum(p.numel() for p in ef.parameters())
    assert 190_000 <= n_sh <= 200_000 and 20_400_000 <= n_ef <= 20_650_000
    print(f"[smoke] params shielded={n_sh:,} effnet={n_ef:,}")

    # find a few synced FITS
    pos_dir = C.DATA / "cutouts_fits_curated_dr9"
    fits = sorted(pos_dir.glob("*.fits"))[:8]
    assert len(fits) >= 4, f"need >=4 FITS in {pos_dir} (run 01_sync first)"
    for p in fits[:3]:
        arr = TL.load_fits_cube(p)
        assert arr.shape == (3, 101, 101), f"{p}: {arr.shape}"
    print(f"[smoke] loaded {len(fits)} FITS, shape (3,101,101) OK")

    # score them with the staged shielded checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = C.DATA / "checkpoint_best_shielded194k_staged.pt"
    model, arch, mean, std, val_auc = SL.load_checkpoint_model(ck, device)
    probs = SL.score_paths([str(p) for p in fits], model, arch, mean, std, device)
    assert np.isfinite(probs).all() and ((0 <= probs) & (probs <= 1)).all(), probs
    print(f"[smoke] {arch} (val_auc {val_auc:.4f}) scored {len(fits)} cutouts: "
          f"p in [{probs.min():.3f},{probs.max():.3f}] — these are known lenses, "
          f"expect high-ish")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
