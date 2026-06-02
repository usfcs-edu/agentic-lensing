#!/usr/bin/env bash
# run_tier1.sh — Huang-2021 Tier-1: from-scratch modeling + analysis on MPS.
#
# Trains all four models on MPS (shielded DR9/DR7 + L18/shielded north-aug),
# reproduces the north-aug false-positive collapse, cross-checks two-model MPS
# inference against phoenix to ~1e-4, reproduces the leak-aware recovery/catalog
# analysis from the phoenix full scores, and verifies. Resumable: each step skips
# if its output already exists. Run ./sync_from_phoenix.sh priority first.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1   # real-time logs (Python block-buffers stdout to a file otherwise)
PY=.venv/bin/python
# DataLoader workers for the training scripts. Default 0: on this Mac the
# num_workers>0 spawn DataLoader hangs at worker startup (macOS spawn + an
# MPS-initialized parent), same class as the 07 fix. The shielded/L18 nets are
# fast on MPS (~0.12 s/step), so synchronous FITS loading is the safe choice.
W="${TRAIN_WORKERS:-0}"

req() { [ -e "$1" ] || { echo "[error] missing required input: $1 (run ./sync_from_phoenix.sh priority)"; exit 1; }; }
req data/cutouts_fits_dr9
req data/cutouts_fits_dr7_train
req data/positives_huang2020.parquet
req data/negatives.parquet

echo "===== Phase 4a: shielded architecture (from scratch) $(date) ====="
[ -f data/checkpoint_best_shielded_dr9.pt ] || $PY 05_train_shielded.py --dr dr9 --workers "$W"
[ -f data/checkpoint_best_shielded_dr7.pt ] || $PY 05_train_shielded.py --dr dr7 --workers "$W"
$PY 06_compare_architectures.py || echo "[warn] 06 failed"
$PY 07_plot_training_curves.py  || echo "[warn] 07 failed"

echo "===== Phase 4b: north calibration (from scratch) $(date) ====="
req data/parent_dr8.parquet
if [ ! -d data/cutouts_fits_north ] || [ ! -f data/positives_north.parquet ]; then
  $PY 18_build_north_train_cutouts.py || echo "[warn] 18 failed"
fi
[ -f data/checkpoint_best_l18_northaug.pt ]      || $PY 05c_train_northaug.py --arch l18 --workers "$W"
[ -f data/checkpoint_best_shielded_northaug.pt ] || $PY 05c_train_northaug.py --arch shielded --workers "$W"
$PY northaug_fp_check.py || echo "[warn] northaug_fp_check failed"

echo "===== Bounded MPS-vs-phoenix two-model inference xcheck $(date) ====="
if [ ! -f data/mps_xcheck.json ]; then
  req data/ref/inference_scores_l18_dr8.parquet
  req data/ref/checkpoint_best_l18_northaug.pt
  # Use the EXACT phoenix deployment (northaug) checkpoints on ~300 bricks so any
  # score delta isolates MPS-vs-CUDA drift. keep-thresh high => few FITS written.
  $PY 11b_brick_inference_dr8.py --n-shards 1 --shard 0 --brick-workers 8 \
      --limit-bricks 300 --keep-thresh 0.95 \
      --ckpt-l18 data/ref/checkpoint_best_l18_northaug.pt \
      --ckpt-shielded data/ref/checkpoint_best_shielded_northaug.pt || echo "[warn] bounded 11b failed"
  $PY xcheck_mps_inference.py || echo "[warn] xcheck failed"
  # clear bounded-run scratch so it can't collide with a later Tier-2 sweep
  rm -f data/inference_scores_l18_shard0.parquet data/inference_scores_shielded_shard0.parquet \
        data/brick_manifest_shard0.csv
fi

echo "===== Phase 4c: analysis from phoenix full scores $(date) ====="
[ -f data/inference_scores_l18_dr8.parquet ]      || cp data/ref/inference_scores_l18_dr8.parquet      data/inference_scores_l18_dr8.parquet
[ -f data/inference_scores_shielded_dr8.parquet ] || cp data/ref/inference_scores_shielded_dr8.parquet data/inference_scores_shielded_dr8.parquet
$PY 13_extract_huang2021_catalog.py || echo "[warn] 13 failed"
$PY 14_crossmatch_recovery_dr8.py   || echo "[warn] 14 failed"
$PY 15_extended_crossmatch.py --model shielded --top-n 2000 || echo "[warn] 15 failed"

echo "===== verify $(date) ====="
$PY verify_against_reference.py || true   # writes data/REPRODUCTION_MPS_COMPARE.md
echo "[tier1-complete] $(date)"
