#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:35:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl-e2-smoke
#SBATCH -o slurm-%j.out
# E2 (P1c) pre-flight on SHARED QOS, single A100 (bills fractionally, not the
# full 4-GPU node; schedules faster than the contended debug queue). Runs the
# three products SEQUENTIALLY in one process (smoke() loops --tags): builds each
# target, validates both basin starts (logp-finite + rendered-model sanity),
# times ~50 HMC steps each -> A100 per-step numbers to size production.
# ALL jobs charge deepsrch_g. Ledger the row BEFORE reading results.
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
mkdir -p data/results
python 10_run_e2.py --mode smoke --tags v3b,v2d,v3 \
    --hmc-steps "${HMC_STEPS:-50}" --smoke-chains "${SMOKE_CHAINS:-8}" \
    --out data/results/e2_smoke_shared.json
