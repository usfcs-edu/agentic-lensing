#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH -J claudenet
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Generic single-GPU ClaudeNet runner (shared QOS = charged per-GPU, 1/4 node).
# Usage:
#   sbatch --export=ALL,CMD='python 100_nersc_smoke.py --score ...' nersc/shared_gpu.slurm
# Debug-queue variant (30 min cap, fast scheduling):
#   sbatch -q debug -t 00:20:00 --gpus-per-node=1 ... (same CMD export)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"
echo "[slurm] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} cmd=$CMD"
eval "$CMD"
