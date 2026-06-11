#!/usr/bin/env bash
# One-time ClaudeNet v2 environment setup on Perlmutter (login node):
#     bash nersc/setup_env.sh
#
# Layers project deps onto NERSC's pytorch module via `pip install --user`
# (PYTHONUSERBASE is per-module, so this never breaks the module), and creates
# the $SCRATCH staging tree. Pattern from redshifty/nersc/setup_env.sh.
# Re-run if the pytorch module version changes.

set -euo pipefail

PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch/2.8.0}"

echo "[1/3] module load $PYTORCH_MODULE"
module load "$PYTORCH_MODULE"

echo "[2/3] pip install project deps (--user, layered onto the module)"
python -m pip install --user --no-cache-dir \
    "timm>=1.0" \
    "astropy>=6.0" \
    "fitsio>=1.2" \
    "pandas>=2.0" \
    "pyarrow>=15.0" \
    "scikit-learn>=1.4" \
    "tqdm>=4.66"

echo "[3/3] create scratch tree"
mkdir -p "$SCRATCH"/claudenet/{smoke,cutouts,scores,ckpt,logs,hf}

echo
echo "Done.  PyTorch module: $PYTORCH_MODULE   Scratch: $SCRATCH/claudenet"
