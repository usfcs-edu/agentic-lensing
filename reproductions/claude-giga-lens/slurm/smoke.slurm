#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J cgl-smoke
#SBATCH -o slurm-%j.out
# claude-giga-lens smoke template (Perlmutter). QOS debug, 1 GPU, 30 min.
# ALL campaign jobs charge deepsrch_g (NOT cosmo_g) — CAMPAIGN.md decision 2026-07-06.
# Ledger discipline: append the job row to CAMPAIGN.md BEFORE reading results.
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS=--xla_gpu_autotune_level=0
cd "$HOME/claude-giga-lens/repo"   # rsync'd campaign dir
srun -n 1 python "$@"
