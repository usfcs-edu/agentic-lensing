#!/bin/bash
# Perlmutter A100 serve + (optional) grade — VALIDATED 2026-06-30 (job 55339203):
# Qwen3-VL-8B bf16 on 1 A100 serves in ~90s and grades LensJudge cutouts SDK-free at ~5.8 s/cand,
# fully offline. Submit:  sbatch perlmutter_vllm.slurm
#
# >>> #1 GOTCHA (cost hours): route ALL caches/temp to $SCRATCH. Perlmutter $HOME has a tiny quota
#     (~40 GB, often full) and compute /tmp is small; vLLM's weight-loader + Triton JIT write there and
#     die with "OSError: [Errno 122] Disk quota exceeded". The exports below fix it.
# Other Perlmutter facts:
#  - A100 = AMPERE: bf16 is native (small models); for big models use AWQ/GPTQ INT4 (NO FP8/W8A8).
#  - Compute nodes have NO internet: pre-stage weights to $SCRATCH (hf download on a login node) +
#    HF_HUB_OFFLINE=1. 1 A100(40GB): 8B bf16 / 32B / 30B-A3B INT4. TP=4: 235B INT4 referee.
#  - The open path is SDK-free (claude_agent_sdk optional) — the Perlmutter lensjudge venv omits it.
#  - Prereqs (build once on a login node): $SCRATCH/venvs/vllm (uv venv + `uv pip install vllm ninja
#    hf_transfer`), $SCRATCH/venvs/lensjudge (pandas numpy astropy pillow pydantic scipy requests tqdm
#    pyarrow openai scikit-learn — anthropic NOT needed since Phase F), the model in $HF_HOME, code
#    rsync'd to $LJ_REPRO/lensjudge,
#    and cutouts staged as {name}.fits in $CUTOUT_DIR.
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -J ljv5
#SBATCH -o ljv5_%j.out
set -x

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"       # served id; big tier: a local INT4 model dir + --quantization awq --tensor-parallel-size 4
PORT="${PORT:-8000}"
VB="$SCRATCH/venvs/vllm/bin"
LJPY="$SCRATCH/venvs/lensjudge/bin/python"
LJ_REPRO="${LJ_REPRO:-$SCRATCH/ljv5/reproductions}"
CUTOUT_DIR="${CUTOUT_DIR:-$SCRATCH/ljv5/cutouts}"
MANIFEST="${MANIFEST:-$CUTOUT_DIR/lensbench_smoke.csv}"
OUT="${OUT:-$SCRATCH/ljv5/preds_perlmutter.parquet}"
DTYPE="${DTYPE:-bfloat16}"                          # A100 native; INT4 tier: leave model quantized + add --quantization awq

# --- route every cache/tmp to $SCRATCH (the #1 gotcha) ---
export HF_HOME="${HF_HOME:-$SCRATCH/claudenet/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TMPDIR="$SCRATCH/tmp" VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm" XDG_CACHE_HOME="$SCRATCH/.cache"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/inductor"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

"$VB/vllm" serve "$MODEL" --port "$PORT" --dtype "$DTYPE" --enforce-eager \
  --max-model-len 16384 --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":8}' --mm-processor-kwargs '{"min_pixels":160000,"max_pixels":1003520}' \
  --served-model-name "$MODEL" > "$SCRATCH/ljv5_vllm.log" 2>&1 &
SRV=$!
ready=0; for i in $(seq 1 90); do curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }; sleep 10; done
echo "SERVER_READY=$ready after ~$((i*10))s"

if [ "$ready" = "1" ]; then
  LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL="http://localhost:$PORT/v1" \
  LENSJUDGE_MODEL_GRADER="$MODEL" LENSJUDGE_CUTOUT_DIR="$CUTOUT_DIR" \
  HF_HOME="$HF_HOME" PYTHONPATH="$LJ_REPRO" \
  "$LJPY" -m lensjudge.imaging.run_batch --mode direct --manifest "$MANIFEST" --out "$OUT" --concurrency 8
  echo "GATE_RC=$?"
fi
kill $SRV 2>/dev/null; echo "JOB_DONE"
