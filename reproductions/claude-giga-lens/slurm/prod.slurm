#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J cgl-prod
#SBATCH -o slurm-%j.out
# claude-giga-lens production template (Perlmutter). Override walltime/gpus per job:
#   sbatch -t 02:00:00 --gpus-per-node=4 slurm/prod.slurm <script.py> [args...]
# ALL campaign jobs charge deepsrch_g (NOT cosmo_g) — CAMPAIGN.md decision 2026-07-06.
# Budget: P1c<=30 / P2c<=45 / P3<=15 A100-h, HARD STOP 100 total.
# Ledger discipline: append the job row to CAMPAIGN.md BEFORE reading results.
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS=--xla_gpu_autotune_level=0
cd "$HOME/claude-giga-lens/repo"
srun -n 1 python "$@"
