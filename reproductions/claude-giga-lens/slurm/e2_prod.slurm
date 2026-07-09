#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J cgl-e2-prod
#SBATCH -o slurm-%j.out
# E2 (P1c) production: two-stage re-preconditioned PHMC, both basins, per
# product. Runs the requested products IN PARALLEL, one per GPU (regular QOS
# bills the full 4-GPU node -> pack the node; the slowest product sets the
# wallclock + A100-h). ALL jobs charge deepsrch_g. Ledger the row (job id +
# est A100-h) in CAMPAIGN.md BEFORE reading results.
#
# PER-PRODUCT budgets (v3 fine is the expensive one): override KEEP/BURN/stages
# per tag so v3 doesn't force the whole node to run long. Uppercase suffix =
# tag (V3/V3B/V2D). Defaults apply where a per-tag var is unset.
#   sbatch -t HH:MM:SS \
#     --export=ALL,CHAINS=24,NLEAP=16,STEP=0.1,\
# KEEP_V3=600,STAGE1KEEP_V3=300,BURN_V3=300,STAGE1BURN_V3=300,\
# KEEP_V3B=2000,KEEP_V2D=2000  slurm/e2_prod.slurm
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"

E2_TAGS="${E2_TAGS:-v3 v3b v2d}"
CHAINS="${CHAINS:-24}"; NLEAP="${NLEAP:-16}"; STEP="${STEP:-0.1}"
mkdir -p data/results
pids=(); i=0
pertag() { # $1=varbase $2=tag $3=default -> value of ${VARBASE_TAGUPPER} or default
  local u; u=$(echo "$2" | tr '[:lower:]' '[:upper:]')
  local v="${1}_${u}"; echo "${!v:-$3}"
}
for tag in $E2_TAGS; do
  KEEP=$(pertag KEEP "$tag" "${KEEP:-2000}")
  BURN=$(pertag BURN "$tag" "${BURN:-500}")
  S1B=$(pertag STAGE1BURN "$tag" "${STAGE1BURN:-500}")
  S1K=$(pertag STAGE1KEEP "$tag" "${STAGE1KEEP:-500}")
  # per-tag basin control (e.g. BASINS_V3=steep to skip a blocked low basin);
  # default both basins. 10_run_e2.py --basins already supports single-basin runs.
  BASINS=$(pertag BASINS "$tag" "${BASINS:-low,steep}")
  echo "[e2_prod] $tag on GPU$i: chains=$CHAINS nleap=$NLEAP step=$STEP "\
"stage1=${S1B}+${S1K} stage2=${BURN}+${KEEP} basins=${BASINS}"
  CUDA_VISIBLE_DEVICES=$i python 10_run_e2.py --mode prod --tag "$tag" \
      --chains "$CHAINS" --num-leapfrog "$NLEAP" --step-size "$STEP" \
      --keep "$KEEP" --burn "$BURN" --stage1-burn "$S1B" --stage1-keep "$S1K" \
      --basins "$BASINS" \
      --out "data/results/e2_${tag}.json" \
      > "data/results/e2_${tag}_run.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
# GPU3 (otherwise idle, billed anyway): a SECOND INDEPENDENT SEED of the money
# product's low basin (v3 fine) for cross-seed robustness of gamma_fine(corr).
# Disable with GPU3_TASK=none; fallback GPU3_TASK=strict runs the strict-whitener
# native low basin at production budget. Uses the same per-tag v3 budget.
GPU3_TASK="${GPU3_TASK:-v3seed2}"
if [ "$i" -le 3 ] && [ "$GPU3_TASK" != "none" ]; then
  KEEP=$(pertag KEEP v3 "${KEEP:-2000}"); BURN=$(pertag BURN v3 "${BURN:-500}")
  S1B=$(pertag STAGE1BURN v3 "${STAGE1BURN:-500}"); S1K=$(pertag STAGE1KEEP v3 "${STAGE1KEEP:-500}")
  if [ "$GPU3_TASK" = "strict" ]; then
    echo "[e2_prod] GPU3: v2d native STRICT whitener low basin (converged info-cost check)"
    CUDA_VISIBLE_DEVICES=3 python 10_run_e2.py --mode prod --tag v2d \
        --whitener whitener_v2d.npz --basins low --seed 11 \
        --chains "$CHAINS" --num-leapfrog "$NLEAP" --step-size "$STEP" \
        --keep "$KEEP" --burn "$BURN" --stage1-burn "$S1B" --stage1-keep "$S1K" \
        --out data/results/e2_v2d_strict_low.json \
        > data/results/e2_v2d_strict_low_run.log 2>&1 &
    pids+=($!)
  else
    echo "[e2_prod] GPU3: v3 fine low basin SEED 2 (money-product cross-seed replicate)"
    CUDA_VISIBLE_DEVICES=3 python 10_run_e2.py --mode prod --tag v3 \
        --basins low --seed 7 \
        --chains "$CHAINS" --num-leapfrog "$NLEAP" --step-size "$STEP" \
        --keep "$KEEP" --burn "$BURN" --stage1-burn "$S1B" --stage1-keep "$S1K" \
        --out data/results/e2_v3_seed2_low.json \
        > data/results/e2_v3_seed2_low_run.log 2>&1 &
    pids+=($!)
  fi
fi
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== PROD SUMMARIES ==="
for tag in $E2_TAGS; do
  echo "----- $tag -----"
  grep -E "posterior gamma|MAP:|Laplace:|\[H1\]" "data/results/e2_${tag}_run.log" || true
done
exit $rc
