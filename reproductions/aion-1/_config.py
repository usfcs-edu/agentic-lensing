"""
Shared configuration for the AION-1 reproduction (Parker et al. 2025,
arXiv:2510.17960).

Central place for: the released checkpoint ids, encoder hidden dims, the
fixed RNG seed, on-disk paths, and small environment helpers. Imported by
every numbered script and helper lib so paths/seeds stay consistent.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- reproducibility ---------------------------------------------------------
SEED = 2026  # paper does not publish seeds; we fix one and document it.

# --- released checkpoints (all three are public on the HF Hub) ---------------
MODELS = {
    "base": "polymathic-ai/aion-base",
    "large": "polymathic-ai/aion-large",
    "xlarge": "polymathic-ai/aion-xlarge",
}
DIMS = {"base": 768, "large": 1024, "xlarge": 2048}
PARAMS_M = {"base": 314, "large": 830, "xlarge": 3130}

# xlarge is 3B params; image+spectrum configs (849 tokens) need a smaller batch
# and fp16 autocast to stay within the 24 GB TITAN RTX.
DEFAULT_BATCH = {"base": 128, "large": 96, "xlarge": 24}
DEFAULT_AMP = {"base": False, "large": False, "xlarge": True}

VARIANTS = ["base", "large", "xlarge"]

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
RESULTS = DATA / "results"
EMB = DATA / "emb"
RAW = DATA / "raw"
for _p in (DATA, FIGS, RESULTS, EMB, RAW):
    _p.mkdir(parents=True, exist_ok=True)

# --- environment -------------------------------------------------------------
# Keep the (large) HF cache on the big /home2 volume, never the small root.
HF_HOME = os.environ.setdefault("HF_HOME", "/home2/benson/.cache/huggingface")

# Redshift token grid: the Z codec spans [0, 6] in 1024 bins (+1 sentinel),
# confirmed by Tutorial.ipynb (Z_HP/6.0*1024) and a (B,1,1025) logit shape.
Z_RANGE = (0.0, 6.0)
Z_NBINS = 1024


def seed_everything(seed: int = SEED) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def paper_targets() -> dict:
    """AION-1 reported numbers (mostly AION-Base column) for side-by-side
    reporting. Absolute matches are *targets*, not pass/fail gates."""
    return {
        # task 1: PROVABGS galaxy property R^2 (Photometry+Image+Spectrum)
        "galaxy_props_R2": {"z": 1.00, "logmass": 0.96, "age": 0.53, "logZ": 0.61, "sSFR": 0.72},
        # task 2: DESI x Gaia stellar property R^2 (DESI+Parallax)
        "stellar_props_R2": {"teff": 0.99, "logg": 0.98, "feh": 0.94, "vmicro": 0.89},
        # task 3: APOGEE x Gaia XP residual std
        "apogee_resid_std": {"teff_K": 94.6, "logg_dex": 0.206, "feh_dex": 0.115},
        # task 4: GZ10 morphology accuracy
        "morphology_acc": 0.840,
        # task 5: GZ3D segmentation IoU
        "segmentation_iou": {"spiral_arms": 0.60, "bar": 0.31},
        # tasks 7-9: retrieval nDCG@10
        "retrieval_ndcg10": {"gz_spirals": 0.938, "gz_mergers": 0.892, "hsc_lenses": 0.968},
    }
