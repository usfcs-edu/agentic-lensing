#!/usr/bin/env bash
# run_dr8_sweep.sh — Huang-2021 Tier-2 (deferred): full from-scratch DR8 two-model
# deployment sweep on a single MPS device, then analysis. ~55-60 h, NERSC-download
# bound, resumable. This produces the deployment scores entirely on the Mac (vs
# Tier-1, which reuses the phoenix scores for the analysis). Mostly re-confirms the
# MPS==CUDA inference fidelity the bounded Tier-1 xcheck already shows.
#
# Prereqs: ./sync_from_phoenix.sh bulk  (724 GB dr8_sweep), and Tier-1 training done
# (checkpoint_best_{l18,shielded}_northaug.pt — the deployment weights).
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
BW="${BRICK_WORKERS:-8}"

[ -d data/dr8_sweep ]                              || { echo "[error] run ./sync_from_phoenix.sh bulk first"; exit 1; }
[ -f data/checkpoint_best_l18_northaug.pt ]        || { echo "[error] train north-aug L18 first (run_tier1.sh)"; exit 1; }
[ -f data/checkpoint_best_shielded_northaug.pt ]   || { echo "[error] train north-aug shielded first"; exit 1; }

[ -f data/parent_dr8.parquet ] || $PY 10_select_parent_sample_dr8.py
$PY 08_smoketest_dr8.py --mode brick || echo "[warn] brick-routing gate"

if [ ! -f data/inference_scores_shielded_dr8.parquet ]; then
  echo "[sweep] full DR8 two-model MPS sweep (~298,844 bricks, ~55-60 h)  $(date)"
  $PY 11b_brick_inference_dr8.py --n-shards 1 --shard 0 --brick-workers "$BW" \
      --ckpt-l18 data/checkpoint_best_l18_northaug.pt \
      --ckpt-shielded data/checkpoint_best_shielded_northaug.pt || { echo "[abort] 11b failed"; exit 1; }
  $PY 12_merge_shards.py || { echo "[abort] merge failed"; exit 1; }
fi

$PY 13_extract_huang2021_catalog.py || echo "[warn] 13"
$PY 14_crossmatch_recovery_dr8.py   || echo "[warn] 14"
$PY 15_extended_crossmatch.py --model shielded --top-n 2000 || echo "[warn] 15"
$PY 16_build_inspection_viewer.py --model shielded --top-n 2000 --per-page 50 || echo "[warn] 16"
$PY verify_against_reference.py || true
echo "[dr8-sweep-complete] $(date)"
