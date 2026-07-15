#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl-e2-prod
#SBATCH -o slurm-%j.out
# P1c METRIC-FIX PRODUCTION (2026-07-10). Per-product single-GPU SHARED job with
# the regularized-PD stage-1 metric (--metric, default diagraw). One product per
# submission (backfills fast, bills fractionally). ALL P1c jobs charge cosmo_g.
# Pass -t per product and override via --export. Example:
#   sbatch -t 04:30:00 --export=ALL,TAG=v3b,BASINS=low,steep,\
#     OUTNAME=e2_v3b_diagraw slurm/e2_prod_shared.slurm
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
mkdir -p data/results

TAG="${TAG:-v3b}"
BASINS="${BASINS:-low,steep}"
METRIC="${METRIC:-diagraw}"
CHAINS="${CHAINS:-24}"; NLEAP="${NLEAP:-16}"; STEP="${STEP:-0.1}"
S1B="${S1B:-500}"; S1K="${S1K:-500}"; BURN="${BURN:-500}"; KEEP="${KEEP:-2000}"
SEED="${SEED:-2}"
SVI_STEPS="${SVI_STEPS:-8000}"; SVI_PARTICLES="${SVI_PARTICLES:-32}"
WHITENER="${WHITENER:-}"
OUTNAME="${OUTNAME:-e2_${TAG}_${METRIC}}"

WFLAG=()
[ -n "$WHITENER" ] && WFLAG=(--whitener "$WHITENER")

echo "[e2_prod_shared] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1)"
echo "[e2_prod_shared] TAG=$TAG BASINS=$BASINS METRIC=$METRIC seed=$SEED "\
"stage1=${S1B}+${S1K} stage2=${BURN}+${KEEP} whitener='${WHITENER:-default}' out=$OUTNAME"

python 10_run_e2.py --mode prod --tag "$TAG" --basins "$BASINS" \
    --metric "$METRIC" --svi-steps "$SVI_STEPS" --svi-particles "$SVI_PARTICLES" \
    "${WFLAG[@]}" \
    --chains "$CHAINS" --num-leapfrog "$NLEAP" --step-size "$STEP" \
    --stage1-burn "$S1B" --stage1-keep "$S1K" --burn "$BURN" --keep "$KEEP" \
    --seed "$SEED" \
    --out "data/results/${OUTNAME}.json" \
    > "data/results/${OUTNAME}_run.log" 2>&1
echo "=== PROD SUMMARY ($OUTNAME) ==="
grep -E "MAP:|Laplace:|METRIC|SVI:|posterior gamma|\[H1\]" \
    "data/results/${OUTNAME}_run.log" || true
