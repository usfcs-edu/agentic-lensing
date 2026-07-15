#!/bin/bash -l
# Two-node sharding validation for gigalens ModellingSequence + MCLMC (NERSC Perlmutter).
# One process per GPU (8 procs over 2 nodes); jax.distributed auto-detects Slurm.
#
# Submit (from the repo, or pass GIGALENS_REPO):
#   sbatch --account=<acct> --export=ALL,GIGALENS_REPO=$PWD \
#          tests/multidevice/multinode_validate.slurm
# Override the interpreter with GIGALENS_PY (default: the gigalens_oldapi conda python,
# which has GPU JAX 0.9.1 + blackjax). This runs a SHARDING SMOKE, not real inference.
#
#SBATCH --job-name=gigalens_multinode_validate
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=multinode_validate_%j.out

set -eu

REPO="${GIGALENS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${GIGALENS_PY:-$HOME/.conda/envs/gigalens_oldapi/bin/python}"

# jax.distributed auto-detection uses these; hsn0 is the Perlmutter high-speed NIC.
export JAX_COORDINATOR_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export JAX_COORDINATOR_PORT=54321
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# src first so this checkout wins over the env's gigalens .pth.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

echo "REPO=$REPO"
echo "PY=$PY"
"$PY" -c "import jax; print('jax', jax.__version__)"

# gpu-bind=none: each task sees all local GPUs; jax pins one per rank via SLURM_LOCALID.
srun -n 8 -c 32 --cpu-bind=cores --gpu-bind=none --label \
  "$PY" -u "$REPO/tests/multidevice/multinode_validate.py"
