#!/usr/bin/env python3
"""387_shards_to_fits.py — convert 111 cutout shards (cutouts_<k>.npy + index.parquet,
shape (n,bands,size,size) float32) into the per-row FITS the trainer expects
(_train.LensDataset reads <fits_dir>/<row_id>.fits as a single PrimaryHDU (bands,H,W),
identical to inchausti-2025/20_build_negatives_brick_dr9). Bridges the sweep-scoring cutout
format to the training cutout format so the DR11-native positives/negatives can fine-tune the
members. Runs LOCALLY after the shards are rsynced back from Perlmutter.

  /home2/benson/.venvs/claudenet/bin/python 387_shards_to_fits.py \
      --in-root <out-root with cutouts_*.npy + index.parquet> --out-dir data/cutouts_fits_dr11_pos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, help="111 out-root (cutouts_<k>.npy + index.parquet)")
    ap.add_argument("--out-dir", required=True, help="per-row FITS dir for the trainer")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    root = Path(a.in_root); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(root / "index.parquet")
    idx = idx[idx.ok].reset_index(drop=True)
    n_written = n_skip = 0
    for k, g in idx.groupby("shard", sort=True):
        arr = np.load(root / f"cutouts_{k}.npy", mmap_mode="r")
        for r in g.itertuples():
            fp = out / f"{r.row_id}.fits"
            if fp.exists() and not a.overwrite:
                n_skip += 1; continue
            cube = np.asarray(arr[r.idx_in_shard], dtype=np.float32)
            fits.PrimaryHDU(data=cube).writeto(fp, overwrite=True)
            n_written += 1
        print(f"[387] shard {k}: {len(g)} rows ({n_written} written, {n_skip} skipped)", flush=True)
    print(f"[387] done: {n_written} FITS written ({n_skip} pre-existing) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
