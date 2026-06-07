#!/bin/bash
# WS3 local scaling ladder driver (model-size axis + seed replication).
# Runs each point sequentially on the 2xL4 (DDP eff-batch 64), then evals it
# leakage-free on its own seed-matched val split at n=4096. Robust: skips a
# point whose best.pt already exists, and a failed point does not abort the rest.
#
#   model axis (seed 42): d_model in {256,384,512,768}, head_dim fixed 64
#   seed replication (seed 123): d_model in {256,768} (endpoints) for noise bars
# All: 8k steps, eff-batch 64, lr 4e-4, V1 32x tokenizer, full mix, max_seq_len 512.
set -u
RS=/raid/benson/git/agentic-lensing/lensing-repos/redshifty
AL=/raid/benson/git/agentic-lensing
CKDIR=/raid/benson/data/desi_dr1_medium/checkpoints/checkpoints
RESDIR=$AL/experiments/runs/_ladder
TOK=/raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt
MANIFEST=/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl
PY=/home/benson/.venvs/redshifty/bin/python
mkdir -p "$RESDIR"
cd "$RS" || exit 1

# name d_model n_heads seed
configs=(
  "ladder_d256_s42 256 4 42"
  "ladder_d384_s42 384 6 42"
  "ladder_d512_s42 512 8 42"
  "ladder_d768_s42 768 12 42"
  "ladder_d256_s123 256 4 123"
  "ladder_d768_s123 768 12 123"
)

for cfg in "${configs[@]}"; do
  set -- $cfg
  NAME=$1; DM=$2; NH=$3; SEED=$4
  echo "================ LADDER TRAIN $NAME (d_model=$DM n_heads=$NH seed=$SEED) $(date +%H:%M) ================"
  if [ -f "$CKDIR/$NAME/best.pt" ]; then
    echo "[skip-train] $NAME best.pt already exists"
  else
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8,9 TORCHDYNAMO_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCRATCH=/raid/benson/data/desi_dr1_medium \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 nersc/train_transformer.py \
      --manifest "$MANIFEST" --tokenizer-ckpt "$TOK" --tokenizer-kind v1 \
      --approach a --steps 8000 --batch-size 32 --lr 4e-4 --warmup 500 \
      --d-model "$DM" --n-heads "$NH" --n-encoder-layers 6 --n-decoder-layers 6 --max-seq-len 512 \
      --healpix-holdout-frac 0.05 --seed "$SEED" \
      --val-every 1000 --save-every 4000 --log-every 200 --num-workers 4 --amp \
      --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 \
      --run-name "$NAME" --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints \
      --wandb-mode disabled --no-push-wandb-artifact || echo "[TRAIN-FAIL] $NAME"
  fi
  echo "================ LADDER EVAL $NAME (seed=$SEED, n=4096) ================"
  if [ -f "$CKDIR/$NAME/best.pt" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 TORCHDYNAMO_DISABLE=1 \
    "$PY" "$AL/tools/spectrumfm/eval_per_class.py" --n-spectra 4096 --seed "$SEED" \
      --checkpoints "$CKDIR/$NAME/best.pt" > "$RESDIR/$NAME.eval.txt" 2>&1 \
      && echo "[eval-ok] $NAME -> $RESDIR/$NAME.eval.txt" || echo "[EVAL-FAIL] $NAME"
  else
    echo "[EVAL-SKIP] $NAME has no best.pt"
  fi
  echo "[point-done] $NAME $(date +%H:%M)"
done
echo "================ LADDER COMPLETE $(date +%H:%M) ================"
