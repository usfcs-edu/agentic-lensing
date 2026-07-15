#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:45:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl-e2-canary-v3blow
#SBATCH -o slurm-%j.out
# P1c METRIC-FIX CANARY (2026-07-10). Re-run v3b LOW (the MONEY basin) with the
# regularized-PD stage-1 metric (--metric diagraw). SAME seed (2) as the frozen
# production run (55713240) so the metric is the SINGLE changed variable.
# GATE: R-hat_max drops 5.94 -> < ~1.1 AND gamma lands near the sane MAP
# gamma_map=1.2695 (NOT the frozen stuck-chain artifact 1.098).
# Modest budget: 24ch, stage1 400+400, stage2 400+600 (1800 steps ~= 1 A100-h at
# the measured 1.95 s/step; -t caps ~1.75 A100-h). Shared QOS bills fractionally.
# ALL P1c jobs charge cosmo_g (deepsrch_g congested by LensJudge).
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
mkdir -p data/results
echo "[canary] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1)"
python 10_run_e2.py --mode prod --tag v3b --basins low \
    --metric diagraw \
    --chains 24 --num-leapfrog 16 --step-size 0.1 \
    --stage1-burn 400 --stage1-keep 400 --burn 400 --keep 600 \
    --seed 2 \
    --out data/results/e2_v3b_low_canary_diagraw.json \
    > data/results/e2_v3b_low_canary_diagraw_run.log 2>&1
echo "=== CANARY SUMMARY (v3b low, diagraw metric) ==="
grep -E "MAP:|Laplace:|METRIC|SVI:|posterior gamma" \
    data/results/e2_v3b_low_canary_diagraw_run.log || true
