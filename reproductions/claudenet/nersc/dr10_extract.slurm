#!/bin/bash
#SBATCH -A cosmo
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -J cn-dr10extract
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#
# DR10 full-sweep cutout extraction (101px grz) — one array task per 160 manifest part.
#   sbatch --array=0-15 -t 02:00:00 \
#     -o $SCRATCH/claudenet/logs/dr10extract-%A_%a.out \
#     -e $SCRATCH/claudenet/logs/dr10extract-%A_%a.err nersc/dr10_extract.slurm
# Each task -> $SCRATCH/claudenet/cutouts/dr10/partNN/ (shards + index.parquet). Resume-safe:
# 111 skips shards whose .npy already exists, so a re-run completes a partial part.
set -euo pipefail
cd "$HOME/claudenet"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"
P=$(printf %02d "$SLURM_ARRAY_TASK_ID")
echo "[dr10extract] part=$P host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
python 111_extract_cutouts_cfs.py \
    --manifest "$SCRATCH/claudenet/sweep_dr10/sweep_manifest_part${P}of16.parquet" \
    --out-root "$SCRATCH/claudenet/cutouts/dr10/part${P}" \
    --size 101 --bands grz --release dr10 --workers 64 --shard-size 50000
