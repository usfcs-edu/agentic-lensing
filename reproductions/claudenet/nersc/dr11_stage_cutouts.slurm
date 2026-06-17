#!/bin/bash
#SBATCH -A cosmo
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -J cn-dr11stage
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Stage a small candidate set's grz cutouts: extract fresh from DR11 coadds (111) then pack to one
# npz (111b) for rsync back + local 120b -> per-object FITS (LensJudge CUTOUT_DIRS). Paths expand
# here (no eval layer). Submit:
#   sbatch --export=ALL,NAME=dr11s_cand500,FOOT=south nersc/dr11_stage_cutouts.slurm
# Reads  $SCRATCH/claudenet/sweep_dr11/<NAME>.parquet (row_id,RA,DEC,footprint,brick)
# Writes $SCRATCH/claudenet/<NAME>.npz (cutouts,row_ids,ok)
set -euo pipefail
cd "$HOME/claudenet"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"
NAME="${NAME:?set NAME}"
MAN="$SCRATCH/claudenet/sweep_dr11/${NAME}.parquet"
OUT="$SCRATCH/claudenet/cutouts/${NAME}"
echo "[dr11stage] NAME=$NAME manifest=$MAN host=$(hostname)"
python 111_extract_cutouts_cfs.py --manifest "$MAN" --out-root "$OUT" \
    --size 101 --bands grz --release dr11 --workers 32 --shard-size 2000
python 111b_dump_rows.py --out-root "$OUT" --row-ids "$MAN" --out "$SCRATCH/claudenet/${NAME}.npz"
echo "[dr11stage] done -> $SCRATCH/claudenet/${NAME}.npz"
