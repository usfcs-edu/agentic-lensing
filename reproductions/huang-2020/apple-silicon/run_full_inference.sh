#!/usr/bin/env bash
# run_full_inference.sh — full from-scratch DR7 deployment sweep on MPS for BOTH
# freshly-trained checkpoints, sequentially (they share the single-shard scratch
# files, so they cannot run concurrently). Resumable: 11b skips bricks already in
# its per-shard manifest; a checkpoint is skipped entirely once its final renamed
# parquet exists.
#
# Long pole: each checkpoint scores ~6.24M galaxies across ~113K bricks, streaming
# ~2.5 TB of transient brick coadds from portal.nersc.gov (deleted after scoring).
# Wall-clock is NERSC-download-bound (~20 h each on a single MPS device).
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
BW="${BRICK_WORKERS:-8}"

run_one() {
  local tag="$1" ckpt="$2"
  local final="data/inference_scores_${tag}.parquet"
  if [ -f "$final" ]; then
    echo "[skip] $tag already complete -> $final"; return 0
  fi
  if [ ! -f "data/$ckpt" ]; then
    echo "[error] missing checkpoint data/$ckpt"; return 1
  fi
  echo "[run] $tag  ckpt=data/$ckpt  brick-workers=$BW  $(date)"
  $PY 11b_brick_inference_dr7.py --ckpt "data/$ckpt" \
        --n-shards 1 --shard 0 --brick-workers "$BW" || return 1
  $PY 12_merge_shards.py || return 1
  mv -f data/inference_scores.parquet "$final"
  echo "[done] $tag -> $final"
  # clear single-shard scratch so the next checkpoint starts clean
  rm -f data/inference_scores_shard0.parquet data/brick_manifest_shard0.csv \
        data/inference_manifest_shard0.csv data/inference_manifest.csv
}

run_one dr9trained checkpoint_best.pt     || { echo "[abort] dr9 failed"; exit 1; }
run_one dr7trained checkpoint_best_dr7.pt || { echo "[abort] dr7 failed"; exit 1; }
echo "[all-done] both inference sweeps complete  $(date)"

# ---- downstream analysis + verification (quick) ----
echo "[analysis] recovery comparison, missing-7, viewer, extended crossmatch  $(date)"
$PY 14b_recovery_comparison.py      || echo "[warn] 14b failed"
$PY 15_diagnose_missing_seven.py    || echo "[warn] 15 failed"
$PY 16_build_inspection_viewer.py --top-n 2000 --per-page 50 || echo "[warn] 16 failed"
$PY 17_extended_crossmatch.py       || echo "[warn] 17 failed"
echo "[verify] comparing MPS results vs phoenix reference"
$PY verify_against_reference.py || true   # writes data/REPRODUCTION_MPS_COMPARE.md; nonzero = FAIL flagged
echo "[pipeline-complete] $(date)"
