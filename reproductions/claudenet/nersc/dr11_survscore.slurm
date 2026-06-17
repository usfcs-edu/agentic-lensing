#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -J cn-dr11survscore
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Score the 8 v3blend8 members on the recalibrated survivors, reading the existing 160-part
# cutouts via --row-ids (no re-extraction). Paths expand here (no eval layer -> no $SCRATCH /
# glob quoting bugs). Submit:
#   sbatch --export=ALL,FOOT=south nersc/dr11_survscore.slurm
# Output: data/v3/scores_member_<art>_survdr11<f>.parquet (+ survdr11<f>_ids.parquet), f=FOOT[0].
set -euo pipefail
cd "$HOME/claudenet"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"
export HF_HOME="$SCRATCH/claudenet/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
FOOT="${FOOT:-south}"
TAG="survdr11${FOOT:0:1}"
echo "[dr11survscore] foot=$FOOT tag=$TAG host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python 315_score_mimic_eval.py \
    --scorelist data/v3/cv3_edf_8member_scorelist.json \
    --row-ids "$SCRATCH/claudenet/sweep_dr11/${FOOT}/survivors_dr10_recal.parquet" \
    --parts-glob "$SCRATCH/claudenet/cutouts/sweep_dr11_${FOOT}/part*" \
    --tag "$TAG" --out-dir data/v3
