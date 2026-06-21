#!/usr/bin/env bash
# Serve Qwen3-VL as an OpenAI-compatible endpoint (vision + tool calling) on a single GPU.
# Targets gpu3 (Titan RTX 24 GB, Turing) and phoenix (L4 24 GB, Ada). Run:  bash serve_vllm_qwen3vl.sh
#
# Then point LensJudge at it:
#   export LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:${PORT}/v1
#   export LENSJUDGE_MODEL_GRADER="${MODEL}"
#
# Requires vLLM >= 0.11 (Qwen3-VL support) in a torch venv with CUDA.
set -euo pipefail

# --- model: 8B fp16 fits 24 GB. For 30B-A3B on 24 GB use an INT4 (AWQ/GPTQ) repo. -----------
#   Titan RTX (Turing) and A100 (Ampere) have NO FP8 — use fp16 (8B) or AWQ/GPTQ INT4 (bigger).
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"        # e.g. Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ for a bump
PORT="${PORT:-8000}"
GPU="${GPU:-0}"                                      # CUDA_VISIBLE_DEVICES
MAXLEN="${MAXLEN:-32768}"
UTIL="${UTIL:-0.92}"
QUANT="${QUANT:-}"                                   # set to awq / gptq_marlin for INT4 repos

# Keep tiny 101->400px cutouts from being re-downsampled: floor the visual-token budget.
#   min_pixels ~= 400*400 = 160000 ; max_pixels caps cost. Tune + pin empirically.
MIN_PIXELS="${MIN_PIXELS:-160000}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"

export CUDA_VISIBLE_DEVICES="${GPU}"
extra=()
[[ -n "${QUANT}" ]] && extra+=(--quantization "${QUANT}")

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${UTIL}" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --limit-mm-per-prompt image=8 \
  --mm-processor-kwargs "{\"min_pixels\": ${MIN_PIXELS}, \"max_pixels\": ${MAX_PIXELS}}" \
  "${extra[@]}"
