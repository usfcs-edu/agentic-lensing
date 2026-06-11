#!/usr/bin/env python3
"""_clib.py — ClaudeNet config hub + reuse loaders.

ClaudeNet improves ML strong-lens FINDING on DESI Legacy grz imaging by attacking
the levers the repo's reproductions proved actually matter (ensemble diversity,
negative quality / operating-point calibration, label efficiency) rather than the
backbone, which Inchausti-2025 showed is inert at the deployment operating point.

This module centralises paths, the fixed seed, the GPU allowlist, and loaders for
the reused code so every numbered script shares the SAME Dataset / preprocessing /
models / matched-FPR arithmetic as the reproduced baseline.

Reuse strategy:
  * inchausti-2025 libs (`_trainlib`, `_scorelib`) and the digit-named model files
    are reached via symlinks in this directory (so `import _trainlib` / `import
    _scorelib` just work) plus the `models()` loader below for the digit-named files.
  * aion-1 harness (`_aion_embed`, `_probe`, `_ls_cutout`, `_config`) is reached by
    `_aion_lens.py`, which puts the aion-1 dir on sys.path.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# --- reproducibility / hardware ---------------------------------------------
SEED = 2026                       # matches inchausti _trainlib.SEED and aion _config.SEED
GPUS = [0, 2, 3, 4, 5, 6]         # usable TITAN RTX; exclude GPU 1 (thermal throttler).
                                  # No NVLink (PCIe PHB/PIX) -> independent per-GPU jobs.
TARGET_FPR = (0.01, 0.001)        # the honest operating points (matched-FPR recovery)

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
REPRO = ROOT.parent
INCH = REPRO / "inchausti-2025"
HUANG20 = REPRO / "huang-2020"
HUANG21 = REPRO / "huang-2021"
SILVER = REPRO / "silver-2025"
AION = REPRO / "aion-1"

DATA = ROOT / "data"
EMB = DATA / "emb"
CKPT = DATA / "ckpt"
for _p in (DATA, EMB, CKPT):
    _p.mkdir(parents=True, exist_ok=True)

# --- phoenix (the data node holding the FITS cutouts on /raid) ---------------
PHOENIX_HOST = os.environ.get("CLAUDENET_PHOENIX_HOST", "phoenix.cs.usfca.edu")
PHOENIX_USER = os.environ.get("CLAUDENET_PHOENIX_USER", "benson")
PHOENIX_RAID = "/raid/benson/git/agentic-lensing/reproductions"

# --- venvs ------------------------------------------------------------------
CLAUDENET_PY = "/home2/benson/.venvs/claudenet/bin/python"   # torch+timm+lenstronomy
AION_PY = "/home2/benson/.venvs/aion/bin/python"             # has the `aion` package
HF_HOME = os.environ.setdefault("HF_HOME", "/home2/benson/.cache/huggingface")

# inchausti shielded-ResNet config that lands at 194,433 params (Inchausti 2025).
CFG194 = dict(stage_out=52, stage_mid=32, shield_ch=12, final_out=24)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def models():
    """Load the reused model classes from inchausti-2025 (digit-named files).

    Returns dict: ShieldedDeepLens, EfficientNetV2Lens, MetaLearner,
    simple_average, CMUDeepLens.
    """
    SR = _load("cn_shielded", INCH / "01b_shielded_resnet.py")
    EF = _load("cn_effnet", INCH / "02_efficientnet.py")
    ME = _load("cn_meta", INCH / "03_meta_learner.py")
    L18 = _load("cn_l18", INCH / "01_lanusse_resnet.py")
    return {
        "ShieldedDeepLens": SR.ShieldedDeepLens,
        "EfficientNetV2Lens": EF.EfficientNetV2Lens,
        "MetaLearner": ME.MetaLearner,
        "simple_average": ME.simple_average,
        "CMUDeepLens": L18.CMUDeepLens,
    }


def gpu_env(gpu_id: int) -> dict:
    """Environment dict pinning a child process to one card by PCI-bus-id order."""
    e = dict(os.environ)
    e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return e


def known_lens_catalogs() -> list[Path]:
    """CSV catalogs of known/published lenses, for the PU-learning guard
    (exclude negatives within a few arcsec of any of these)."""
    d = DATA
    return [p for p in (
        d / "storfer2024_published_catalog.csv",
        d / "inchausti2025_published_catalog.csv",
        d / "huang2021_published_catalog.csv",
    ) if p.exists()]
