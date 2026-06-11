#!/usr/bin/env bash
# run_dr7_sweep.sh — DR7-only from-scratch deployment sweep on MPS, then analysis.
#
# Per the chosen scope: run the full ~6.24M-galaxy inference for the paper-exact
# DR7-trained checkpoint on a single MPS device (resumable; ~20 h, NERSC-download
# bound), reproducing the DR7-trained recovery column from MPS-computed scores.
# The DR9-trained column is filled from the transferred phoenix scores (the gate
# already showed MPS inference matches phoenix to ~4e-4, so re-running DR9 on MPS
# would only re-derive identical numbers). Then recovery/diagnostics/viewer/
# crossmatch + verify.
#
# Resume after an interruption by simply re-running this script.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
BW="${BRICK_WORKERS:-8}"

[ -f data/parent_dr7.parquet ]      || { echo "[error] run 10 first (no parent_dr7.parquet)"; exit 1; }
[ -f data/checkpoint_best_dr7.pt ]  || { echo "[error] missing data/checkpoint_best_dr7.pt"; exit 1; }

FINAL=data/inference_scores_dr7trained.parquet
if [ ! -f "$FINAL" ]; then
  echo "[run] DR7-trained full MPS sweep  brick-workers=$BW  $(date)"
  $PY 11b_brick_inference_dr7.py --ckpt data/checkpoint_best_dr7.pt \
      --n-shards 1 --shard 0 --brick-workers "$BW" || { echo "[abort] 11b failed"; exit 1; }
  $PY 12_merge_shards.py || { echo "[abort] merge failed"; exit 1; }
  mv -f data/inference_scores.parquet "$FINAL"
  rm -f data/inference_scores_shard0.parquet data/brick_manifest_shard0.csv \
        data/inference_manifest_shard0.csv data/inference_manifest.csv
  echo "[done] $FINAL  $(date)"
else
  echo "[skip] $FINAL already exists"
fi

# DR9-trained column from the transferred phoenix scores (MPS-equivalent; see gate).
if [ ! -f data/inference_scores_dr9trained.parquet ]; then
  cp data/ref/inference_scores_dr9trained.parquet data/inference_scores_dr9trained.parquet
  echo "[fill] DR9-trained column from phoenix reference scores"
fi

echo "[analysis] recovery / missing-7 / viewer / crossmatch  $(date)"
$PY 14b_recovery_comparison.py   || echo "[warn] 14b failed"
$PY 15_diagnose_missing_seven.py || echo "[warn] 15 failed"
$PY 16_build_inspection_viewer.py --top-n 2000 --per-page 50 || echo "[warn] 16 failed"
$PY 17_extended_crossmatch.py    || echo "[warn] 17 failed"
echo "[verify] MPS results vs phoenix reference"
$PY verify_against_reference.py || true   # writes data/REPRODUCTION_MPS_COMPARE.md
echo "[pipeline-complete] $(date)"
