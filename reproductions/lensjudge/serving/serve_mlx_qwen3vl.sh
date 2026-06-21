#!/usr/bin/env bash
# Serve Qwen3-VL on the Mac Studio (M4 Max 128 GB) via MLX as an OpenAI-compatible endpoint.
# Run:  bash serve_mlx_qwen3vl.sh     Requires: pip install mlx-vlm  (Apple Silicon).
#
# Then point LensJudge at it:
#   export LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:${PORT}/v1
#   export LENSJUDGE_MODEL_GRADER="${MODEL}"
#
# Notes:
#  - 4-bit lets 8B (~6 GB) up to 32B / 30B-A3B (~20 GB) run comfortably in 128 GB.
#  - mlx-vlm's default max_pixels can shrink small images (mlx-vlm issue #1175): set the
#    resolution kwargs so the 400px cutout keeps detail. Verify the flag name for your mlx-vlm
#    version (some expose --resize / processor kwargs); pin it once confirmed.
#  - For a separate TEXT controller (Phase 3 / tool-delegated), serve gpt-oss-120b or
#    GLM-4.5-Air via `mlx_lm.server` on a second port.
set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen3-VL-8B-Instruct-4bit}"   # or ...-32B-... / ...-30B-A3B-...-4bit
PORT="${PORT:-8000}"

exec python -m mlx_vlm.server --model "${MODEL}" --port "${PORT}"
