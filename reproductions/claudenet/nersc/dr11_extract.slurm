#!/bin/bash
#SBATCH -A cosmo
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -J cn-dr11extract
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#
# DR11 full-sweep cutout extraction (101px grz) — one array task per 160 manifest part.
# Parameterized by FOOT (south|north) and NPARTS (must match 160 --chunk). Submit, e.g.:
#   sbatch --array=0-31 -t 04:00:00 --export=ALL,FOOT=south,NPARTS=32 \
#     -o $SCRATCH/claudenet/logs/dr11extract-south-%A_%a.out \
#     -e $SCRATCH/claudenet/logs/dr11extract-south-%A_%a.err nersc/dr11_extract.slurm
# Each task -> $SCRATCH/claudenet/cutouts/sweep_dr11_<FOOT>/partNN/ (shards + index.parquet).
# Resume-safe: 111 skips shards whose .npy already exists, so a re-run completes a partial part.
set -euo pipefail
cd "$HOME/claudenet"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"
P=$(printf %02d "$SLURM_ARRAY_TASK_ID")
FOOT="${FOOT:-south}"
NPARTS="${NPARTS:-32}"
NP=$(printf %02d "$NPARTS")
MANDIR="${MANDIR:-manifest}"
echo "[dr11extract] foot=$FOOT part=$P/$NP mandir=$MANDIR host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
python 111_extract_cutouts_cfs.py \
    --manifest "$SCRATCH/claudenet/sweep_dr11/${MANDIR}/sweep_manifest_part${P}of${NP}.parquet" \
    --out-root "$SCRATCH/claudenet/cutouts/sweep_dr11_${FOOT}/part${P}" \
    --size 101 --bands grz --release dr11 --workers 64 --shard-size 50000
