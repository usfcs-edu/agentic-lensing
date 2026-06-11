#!/usr/bin/env python3
"""northaug_fp_check.py — quantify the north false-positive collapse on MPS.

The headline Phase-4b result of Huang-2021: the south-trained L18 over-fires on
BASS/MzLS (north) imaging until it sees north negatives in training (paper: north
non-lens score>=0.1 rate 91% -> 0.8%).

This scores the HELD-OUT north negatives (the test split written by
05c_train_northaug.py) with BOTH:
  - PRE-northaug L18   (data/checkpoint_best.pt — south-only, from huang-2020)
  - POST-northaug L18  (data/checkpoint_best_l18_northaug.pt — from 05c --arch l18)
on the same set, and reports the fraction scoring >= 0.1. Pure torch on MPS.

Writes data/northaug_fp_check.json {n, thresh, pre_rate, post_rate}.
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from device import pick_device  # noqa: E402
CMUDeepLens = import_module("01_lanusse_resnet").CMUDeepLens

DATA = HERE / "data"
THRESH = 0.1


def load_cube(p: Path) -> np.ndarray:
    with fits.open(p, memmap=False) as h:
        d = np.asarray(h[0].data, dtype=np.float32)
    if d.ndim == 2:
        d = np.stack([d, d, d], axis=0)
    return d


def score_l18(ckpt_path: Path, df: pd.DataFrame, device) -> np.ndarray:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mean = np.array(ck["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(ck["std"], dtype=np.float32).reshape(3, 1, 1)
    model = CMUDeepLens(in_channels=3).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out = []
    with torch.no_grad():
        for _, r in df.iterrows():
            p = Path(r["fits_dir"]) / f"{r['row_id']}.fits"
            if not p.exists():
                continue
            x = np.clip((load_cube(p) - mean) / std, -250.0, 250.0)[None]
            lo = model(torch.from_numpy(x.astype(np.float32)).to(device)).cpu().numpy()
            out.append(1.0 / (1.0 + np.exp(-lo.ravel()[0])))
    return np.asarray(out, dtype=float)


def main() -> None:
    device = pick_device()
    split_path = DATA / "training_split_northaug.parquet"
    if not split_path.exists():
        raise SystemExit("missing training_split_northaug.parquet — run 05c first")
    split = pd.read_parquet(split_path)
    north_neg = split[(split["split"] == "test") & (split["label"] == 0)
                      & (split["fits_dir"].astype(str).str.contains("north"))]
    if len(north_neg) == 0:
        raise SystemExit("no held-out north negatives in the northaug split")

    pre_ck = DATA / "checkpoint_best.pt"             # south-only L18 (pre)
    post_ck = DATA / "checkpoint_best_l18_northaug.pt"  # north-calibrated L18 (post)
    for c in (pre_ck, post_ck):
        if not c.exists():
            raise SystemExit(f"missing {c.name}")

    print(f"[northaug-fp] scoring {len(north_neg)} held-out north negatives "
          f"on {device}")
    pre = score_l18(pre_ck, north_neg, device)
    post = score_l18(post_ck, north_neg, device)
    res = {"n": int(len(post)), "thresh": THRESH,
           "pre_rate": float((pre >= THRESH).mean()),
           "post_rate": float((post >= THRESH).mean())}
    (DATA / "northaug_fp_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[northaug-fp] north non-lens >= {THRESH}:  "
          f"pre={res['pre_rate']:.1%}  ->  post={res['post_rate']:.1%}")


if __name__ == "__main__":
    main()
