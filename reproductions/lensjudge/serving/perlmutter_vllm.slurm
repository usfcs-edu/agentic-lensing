#!/bin/bash
# Perlmutter A100 batch server: Qwen3-VL via vLLM in a podman-hpc container, OpenAI-compatible.
# Submit:  sbatch perlmutter_vllm.slurm     Account/QOS per your allocation (cosmo_g uncapped GPU).
#
# CRITICAL Perlmutter facts baked in:
#  - A100 is AMPERE => NO native FP8/W8A8. Use fp16 (small) or AWQ/GPTQ INT4 (big). Never MXFP4/FP8.
#  - Compute nodes have NO internet => pre-stage weights to $SCRATCH and HF_HUB_OFFLINE=1.
#  - 4 GPUs/node (40 GB each). 1 GPU: 8B/32B/30B-A3B INT4. TP=2: GLM-4.6V-106B-A12B INT4.
#    TP=4: Qwen3-VL-235B-A22B INT4 (referee tier).
#  - podman-hpc: build/pull the vLLM image on a login node, then `podman-hpc migrate` it so
#    compute nodes can run it.
#SBATCH -A m0000_g            # TODO: set your GPU allocation (e.g. cosmo_g)
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J ljv4-vllm
#SBATCH -o ljv4-vllm-%j.out

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$SCRATCH/ljv4/models/Qwen3-VL-235B-A22B-Instruct-AWQ}"  # pre-staged weights
PORT="${PORT:-8000}"
TP="${TP:-4}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"   # pre-pulled + `podman-hpc migrate`d on a login node

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# tiny-cutout resolution floor (see README.md "Why min_pixels matters")
MM_KW='{"min_pixels": 160000, "max_pixels": 1003520}'

srun podman-hpc run --rm --gpu \
  -v "${MODEL_DIR}:/model:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -p "${PORT}:${PORT}" \
  "${IMAGE}" \
  --model /model \
  --served-model-name "$(basename "${MODEL_DIR}")" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --quantization awq \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --limit-mm-per-prompt image=8 \
  --mm-processor-kwargs "${MM_KW}"

# LensJudge (same node or a login node that can reach the compute node) then uses:
#   export LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://<node>:${PORT}/v1
#   export LENSJUDGE_MODEL_GRADER="$(basename "${MODEL_DIR}")" LENSJUDGE_GUIDED_JSON=1
