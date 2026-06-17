#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -J cn-dr11score
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#
# DR11 stage-1 scoring (5 v2-lean members) — one array task per 160 part, scoring ALL shards.
# Pipeline with extraction via element-wise dependency:
#   sbatch --array=0-31 -t 02:00:00 --dependency=aftercorr:<EXTRACT_ARRAY_JID> \
#     --export=ALL,FOOT=south \
#     -o $SCRATCH/claudenet/logs/dr11score-south-%A_%a.out \
#     -e $SCRATCH/claudenet/logs/dr11score-south-%A_%a.err nersc/dr11_score.slurm
# Output: $SCRATCH/claudenet/scores/sweep_dr11_<FOOT>/stage1_part<NN>_lean_s*.parquet
# Resume-safe (161/112 skip a shard-range whose final parquet exists). HF offline (no download).
set -euo pipefail
cd "$HOME/claudenet"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"
export HF_HOME="${HF_HOME:-$SCRATCH/claudenet/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=$(printf %02d "$SLURM_ARRAY_TASK_ID")
FOOT="${FOOT:-south}"
echo "[dr11score] foot=$FOOT part=$P host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python 161_sweep_score.py \
    --cutout-root "$SCRATCH/claudenet/cutouts/sweep_dr11_${FOOT}/part${P}" \
    --stage1 lean --lean-ckpt-dir data/v2/ckpt_lean \
    --out-dir "$SCRATCH/claudenet/scores/sweep_dr11_${FOOT}" \
    --part-tag "part${P}"
