#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J cgl-e2-smoke
#SBATCH -o slurm-%j.out
# E2 (P1c) debug-QOS pre-flight: build all three targets, validate every basin
# start (logp finite + rendered-model sanity), time ~50 HMC steps each. Runs
# the three products IN PARALLEL, one per GPU (debug bills the full 4-GPU node),
# so it fits the 30-min debug window even with the fine 260^2 f64 compile.
# ALL jobs charge deepsrch_g. Ledger the row in CAMPAIGN.md BEFORE reading.
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
# autotune off (P2a stack defect #3) + priority-fusion off (P0 sampler livelock
# policy: DEFAULT for every Perlmutter sampler process drawing f64 normals).
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"

HMC_STEPS="${HMC_STEPS:-50}"
CH="${SMOKE_CHAINS:-8}"
mkdir -p data/results/e2_smoke
pids=()
i=0
for tag in v3b v2d v3; do
  CUDA_VISIBLE_DEVICES=$i python 10_run_e2.py --mode smoke --tags "$tag" \
      --hmc-steps "$HMC_STEPS" --smoke-chains "$CH" \
      --out "data/results/e2_smoke/e2_smoke_${tag}.json" \
      > "data/results/e2_smoke/smoke_${tag}.log" 2>&1 &
  pids+=($!)
  i=$((i+1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== SMOKE LOGS ==="
for tag in v3b v2d v3; do
  echo "----- $tag -----"
  grep -E "SMOKE|steep\]|low  \]|HMC |all_starts_ok|built in" \
      "data/results/e2_smoke/smoke_${tag}.log" || true
done
exit $rc
