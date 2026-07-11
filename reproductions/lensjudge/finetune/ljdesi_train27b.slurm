#!/bin/bash
# LensJudge parity Phase C2: 27B bf16 LoRA + unfrozen ViT on the DESI-resolution corpus
# (12,250 rows: human-soft targets + dihedral copies + random_neg dilution; built by
# finetune/build_corpus_desi.py from parity_train_pool.csv), 2 FULL epochs, 4-GPU DDP
# (~766 steps). Identical recipe to ljv5_train27b_r3.slurm; only data/out/name changed.
#SBATCH -A deepsrch_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 480
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -J ljdesi
#SBATCH -o /pscratch/sd/g/gdbenson/ljdesi_train27b_%j.out
set -x
export HF_HOME="$SCRATCH/claudenet/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HF=1
export TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/.cache"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/inductor"
export MODELSCOPE_CACHE=/tmp/modelscope
export NPROC_PER_NODE=4
mkdir -p /tmp/modelscope "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SCRATCH/ljv5"
MODEL=Qwen/Qwen3.5-27B GPU=0,1,2,3 EPOCHS=2 SAVE_STEPS=75 SAVE_LIMIT=30 MAX_LENGTH=16384 \
  DTYPE=bfloat16 QUANT_BITS="" LR=5e-5 WARMUP=0.1 FREEZE_VIT=false VIT_LR=1e-5 \
  TRAIN_JSONL="$SCRATCH/ljdesi/corpus_desi/sft_desi_train_aug.jsonl" \
  VAL_JSONL="$SCRATCH/ljdesi/corpus_desi/sft_desi_val.jsonl" \
  OUT="$SCRATCH/ljdesi/ckpt_desi_27b" \
  SWIFT="$SCRATCH/venvs/ljtrain/bin/swift" \
  bash "$SCRATCH/ljv5/reproductions/lensjudge/finetune/train_lora.sh"
echo "TRAINDESI_RC=$?"
