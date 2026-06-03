#!/usr/bin/env bash
# run_tier1.sh — Layer (b): bounded from-scratch training fidelity on MPS.
#
# Proves the TRAINING path (forward + backward + AdamW + bf16 AMP + DataLoader) is
# NaN-free and learning on MPS — the regime layer (a)'s inference check can't reach,
# and where the non_blocking-NaN bug manifests.
#   b1) V1 ConvNeXt+LFQ tokenizer from scratch        -> NaN-free + val_recon decreasing
#   b2) Approach-A transformer w/ the FROZEN phoenix tokenizer (the ignition trainer at
#       small scale)                                  -> NaN-free + val_loss decreasing
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
REPO=src-redshifty
MANIFEST=data/tier1_submanifest.jsonl
OUT=data/tier1
FROZEN_TOK=_raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt

TOK_STEPS="${TOK_STEPS:-400}"
A_STEPS="${A_STEPS:-300}"

mkdir -p "$OUT"

echo "===== b1: V1 tokenizer from scratch ($TOK_STEPS steps) ====="
[ -f "$OUT/checkpoints/tok_tier1/best.pt" ] || $PY $REPO/nersc/pretrain_tokenizer.py \
  --manifest "$MANIFEST" --steps "$TOK_STEPS" --batch-size 8 --lr 3e-4 --warmup 40 \
  --val-frac 0.1 --val-every 100 --save-every "$TOK_STEPS" --log-every 25 \
  --num-workers 0 --amp --run-name tok_tier1 --scratch-out "$OUT" \
  --wandb-mode disabled --no-push-wandb-artifact 2>&1 | tee data/tier1_tok.log

echo "===== b2: Approach-A w/ frozen phoenix tokenizer ($A_STEPS steps) ====="
[ -f "$OUT/checkpoints/approachA_tier1/final.pt" ] || $PY $REPO/nersc/train_transformer.py \
  --manifest "$MANIFEST" --tokenizer-ckpt "$FROZEN_TOK" --tokenizer-kind v1 --approach a \
  --steps "$A_STEPS" --batch-size 8 --lr 4e-4 --warmup 40 --healpix-holdout-frac 0.1 \
  --val-every 100 --save-every "$A_STEPS" --log-every 25 --num-workers 0 --amp \
  --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 2 --z-fit-files 16 \
  --run-name approachA_tier1 --scratch-out "$OUT" \
  --wandb-mode disabled --no-push-wandb-artifact 2>&1 | tee data/tier1_approachA.log

echo "===== tier-1 NaN scan (LOSS fields only) ====="
# NB: early Approach-A steps legitimately print mask_auc=nan / all_auc=nan (degenerate
# AUC before the model predicts >1 class) — that is NOT a training NaN. The non_blocking
# bug instead makes the LOSS itself nan from step 0, so scan only loss fields.
if grep -qiE '(^|[^a-z_])(loss|z_loss|spec_loss|recon|val_loss|val_recon|val_total)=[[:space:]]*-?(nan|inf)' data/tier1_run.log; then
  echo "TIER-1 FAIL: NaN/Inf in a LOSS field"; grep -niE 'loss[^,= ]*=[[:space:]]*-?(nan|inf)' data/tier1_run.log | head
else
  echo "TIER-1: no NaN/Inf in any loss field ✓"
fi
