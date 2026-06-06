#!/usr/bin/env python3
"""11_embed_aion.py — frozen AION-1 embeddings for the Phase-0 gate inputs.

RUN WITH THE AION VENV (only it has the `aion` package); writes plain .npy that
the claudenet venv reads downstream (the on-disk embedding is the venv boundary):

    HF_HOME=/home2/benson/.cache/huggingface CUDA_DEVICE_ORDER=PCI_BUS_ID \
      /home2/benson/.venvs/aion/bin/python 11_embed_aion.py --variant base

Writes data/emb/aion_emb_<split>_<variant>.npy  (M, D) mean-pooled, via the
aion-1 multi_gpu_extract harness over GPUs {0,2,3,4,5,6}.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "aion-1"))   # for _aion_embed + _config
import _aion_embed as E  # noqa: E402

EMB = ROOT / "data" / "emb"
SPLITS = ["trainpool", "testneg", "storfer", "inchausti"]
BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "large", "xlarge"])
    ap.add_argument("--gpus", default="0,2,3,4,5,6")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    for sp in SPLITS:
        flux = EMB / f"aion_in_{sp}.npy"
        if not flux.exists():
            print(f"[skip] missing {flux.name}; run 10_build_aion_inputs.py first")
            continue
        out = EMB / f"aion_emb_{sp}_{args.variant}.npy"
        if out.exists():
            print(f"[skip] {out.name} exists {np.load(out, mmap_mode='r').shape}")
            continue
        specs = [E.image_spec("LegacySurveyImage", str(flux), BANDS)]
        t = time.time()
        arr = E.multi_gpu_extract(specs, args.variant, out, pool="mean", gpus=gpus)
        print(f"[{sp}] {out.name} {arr.shape} in {time.time()-t:.1f}s")
    print("EMBED_AION_OK")


if __name__ == "__main__":
    main()
