#!/bin/bash
#SBATCH -A cosmo
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH -J claudenet-cpu
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Generic shared-QOS CPU runner (extraction / IO-bound steps; charged per-core).
# Usage:
#   sbatch --export=ALL,CMD='python 111_extract_cutouts_cfs.py ...' nersc/shared_cpu.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"
echo "[slurm] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} cmd=$CMD"
eval "$CMD"
