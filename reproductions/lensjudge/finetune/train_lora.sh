#!/usr/bin/env bash
# Phase 4: QLoRA-distill Qwen3-VL-8B on the LensJudge SFT set (build_sft_data.py).
# Validated on gpu3 (Titan RTX 24 GB, sm_75) with ms-swift 4.3.1. Run: bash train_lora.sh
#
# TURING RECIPE (learned live): fp16 8B OOMs on 24 GB, so use QLoRA 4-bit (bitsandbytes nf4) ->
# ~14.5 GiB peak. fp16 compute (no bf16 on Turing), SDPA attention (no flash-attn), FROZEN ViT
# (LoRA only the LLM q/k/v/o/gate/up/down), gradient checkpointing. ms-swift uses --tuner_type
# (not --train_type) and defaults to ModelScope hub (set USE_HF=1 to reuse the HF cache).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-reproductions/lensjudge/outputs/sft}"
OUT="${OUT:-reproductions/lensjudge/outputs/sft/ckpt}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-1}"          # ~hours/epoch on one Titan RTX (~12 s/sample); 1 epoch is a sensible first run
LORA_RANK="${LORA_RANK:-16}"
SWIFT="${SWIFT:-/home2/benson/.venvs/ljtrain/bin/swift}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/home2/benson/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${SWIFT}" sft \
  --model "${MODEL}" \
  --dataset "${DATA_DIR}/sft_train.jsonl" \
  --val_dataset "${DATA_DIR}/sft_val.jsonl" \
  --tuner_type lora --quant_bits 4 --quant_method bnb \
  --torch_dtype float16 --attn_impl sdpa \
  --freeze_vit true --lora_rank "${LORA_RANK}" --lora_alpha $((LORA_RANK * 2)) --lora_dropout 0.05 \
  --max_length 6144 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --num_train_epochs "${EPOCHS}" --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --eval_strategy steps --eval_steps 100 --save_steps 100 --save_total_limit 2 \
  --logging_steps 5 --dataset_num_proc 2 \
  --output_dir "${OUT}"
