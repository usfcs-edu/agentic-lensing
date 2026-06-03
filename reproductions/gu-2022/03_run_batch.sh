#!/usr/bin/env bash
# 03_run_batch.sh -- fit a range of mock systems, cycling A16 GPUs 0-5.
#
# One process per system, pinned to ONE GPU (round-robin over 0-5), autotune off.
# Runs at most NPAR systems concurrently (one per GPU). Logs to data/logs/.
#
# Usage:
#   ./03_run_batch.sh 0 11      # validation batch: systems 0..11
#   ./03_run_batch.sh 0 99      # full 100-system benchmark
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/raid/benson/.venvs/gigalens/bin/python
GPUS=(0 1 2 3 4 5)
NPAR=${#GPUS[@]}
START=${1:-0}
END=${2:-11}
# Extra flags forwarded to 02_fit_system.py (e.g. "--gbtla" for the paper's
# adaptive-trajectory HMC, which mixes far better but is ~6x slower on the A16).
EXTRA="${3:-}"
mkdir -p "$HERE/data/logs"

idx=$START
while [ "$idx" -le "$END" ]; do
  # launch up to NPAR systems, one per GPU
  for g in "${GPUS[@]}"; do
    [ "$idx" -gt "$END" ] && break
    log="$HERE/data/logs/fit_${idx}.log"
    echo "launch system $idx on GPU $g -> $log"
    CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
      CUDA_VISIBLE_DEVICES=$g "$PY" "$HERE/02_fit_system.py" --idx "$idx" $EXTRA \
      > "$log" 2>&1 &
    idx=$((idx+1))
  done
  wait   # wait for this wave (one per GPU) before next wave
done
echo "batch $START..$END complete"
