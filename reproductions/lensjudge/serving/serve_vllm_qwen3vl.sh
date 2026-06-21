#!/usr/bin/env bash
# Serve Qwen3-VL as an OpenAI-compatible endpoint on a single GPU.
# Targets gpu3 (Titan RTX 24 GB, Turing) and phoenix (L4 24 GB, Ada). Run: bash serve_vllm_qwen3vl.sh
#
# Then point LensJudge at it:
#   export LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:${PORT}/v1
#   export LENSJUDGE_MODEL_GRADER="${MODEL}"
#
# Requires vLLM >= 0.11 (Qwen3-VL); validated on vLLM 0.23 + gpu3 (7x Titan RTX 24 GB).
#
# >>> TURING (gpu3) RECIPE <<< Titan RTX (sm_75) has NO bf16, and FlashInfer attention + the
# FlashInfer sampler are broken / JIT-fragile there. Use:
#     DTYPE=float16 ATTN_BACKEND=TRITON_ATTN ENFORCE_EAGER=1 bash serve_vllm_qwen3vl.sh
# Notes learned the hard way on vLLM 0.23:
#   - VLLM_ATTENTION_BACKEND env was REMOVED -> select via the --attention-backend flag (ATTN_BACKEND).
#   - FlashInfer attention -> CUDA "invalid argument" on Turing; TRITON_ATTN works.
#   - FlashInfer sampler -> needs `ninja` to JIT; disabled here via VLLM_USE_FLASHINFER_SAMPLER=0.
#   - --enforce-eager skips torch.compile + CUDA-graph capture (faster startup; avoids Turing hangs).
#   - --limit-mm-per-prompt takes JSON ('{"image": 8}'), not image=8.
# On Ada/Ampere you can leave ATTN_BACKEND/ENFORCE_EAGER unset and DTYPE=auto.
set -euo pipefail

# 8B fp16 fits 24 GB. For a quality bump on 24 GB use an INT4 (AWQ/GPTQ) repo (no FP8 on Turing/Ampere).
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"          # e.g. Qwen/Qwen3-VL-32B-Instruct-AWQ
PORT="${PORT:-8000}"
GPU="${GPU:-0}"                                       # CUDA_VISIBLE_DEVICES
MAXLEN="${MAXLEN:-16384}"
UTIL="${UTIL:-0.92}"
QUANT="${QUANT:-}"                                    # awq / gptq_marlin for INT4 repos
DTYPE="${DTYPE:-auto}"                                # Turing: float16
ATTN_BACKEND="${ATTN_BACKEND:-}"                      # Turing: TRITON_ATTN
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"                   # Turing: 1
ENABLE_TOOLS="${ENABLE_TOOLS:-0}"                     # 1 for the Phase-3 agentic tool loop
TOOL_PARSER="${TOOL_PARSER:-hermes}"
# Keep tiny 101->400px cutouts from being re-downsampled: floor the visual-token budget.
MIN_PIXELS="${MIN_PIXELS:-160000}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # native sampler, no ninja JIT

extra=()
[[ -n "${QUANT}" ]] && extra+=(--quantization "${QUANT}")
[[ -n "${ATTN_BACKEND}" ]] && extra+=(--attention-backend "${ATTN_BACKEND}")
[[ "${ENFORCE_EAGER}" == "1" ]] && extra+=(--enforce-eager)
[[ "${ENABLE_TOOLS}" == "1" ]] && extra+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER}")

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${UTIL}" \
  --limit-mm-per-prompt '{"image": 8}' \
  --mm-processor-kwargs "{\"min_pixels\": ${MIN_PIXELS}, \"max_pixels\": ${MAX_PIXELS}}" \
  "${extra[@]}"
