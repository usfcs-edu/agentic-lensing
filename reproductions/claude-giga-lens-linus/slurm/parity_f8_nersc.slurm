#!/bin/bash
# parity_f8_nersc.slurm — gate F8 (REPORT-ONLY): the cross-stack parity
# harness under a NERSC jax-0.10-nightly-like env, one shared-QOS cell.
# TEMPLATE ONLY — not run in P0. Budget <= 1 A100-h (PLAN §3); append the
# ledger row in CAMPAIGN.md BEFORE reading results (house rule).
#
# Prep (once, on Perlmutter):
#   1. rsync the campaign tree + data/parity_refs.npz to $SCRATCH/cgl2-f8/
#      and md5-audit every campaign .py (DEPLOY LESSON, PLAN §3).
#   2. Build the F8 env: python -m venv $SCRATCH/venvs/cgl2-f8 &&
#      pip install -c constraints-f8.txt "jax[cuda12]==0.10.*" \
#          tensorflow-probability[jax]==0.25.0 lenstronomy objax numpy scipy
#      pip install -e vendor/gigalens-linus --no-deps && pip install -e . --no-deps
#      (record the exact resolved versions in the ledger row)
#
#SBATCH --account=deepsrch_g          # cosmo_g allocation per PLAN §2 -- EDIT
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=cgl2-parity-f8
#SBATCH --output=%x-%j.out

set -euo pipefail
WORK=${WORK:-$SCRATCH/cgl2-f8}
VENV=${VENV:-$SCRATCH/venvs/cgl2-f8}
cd "$WORK"

# md5 deploy audit (canary before burning GPU time)
md5sum -c deploy.md5 || { echo "DEPLOY AUDIT FAILED"; exit 2; }

export GIGALENS_X64=1 JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# NOTE: the aarch64 phoenix workaround flags (autotune 0 / priority-fusion)
# are NOT assumed here; start clean and record any needed flags in the report.

# F8 is REPORT-ONLY: the jax pin guard is the one sanctioned exception
# (ledgered decision, PLAN §5 gate F8). CGL2_F8=1 asks 01_parity_scene to
# skip require_jax_pin and tag the report file.
CGL2_F8=1 "$VENV/bin/python" 01_parity_scene.py \
  || echo "F8: gate battery returned nonzero (REPORT-ONLY — file the numbers)"
cp data/parity_report_scene.json "data/parity_report_scene_f8_${SLURM_JOB_ID}.json"
echo "F8 report: data/parity_report_scene_f8_${SLURM_JOB_ID}.json"
