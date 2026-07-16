#!/bin/bash
# B3 lane driver (Front C): one GPU, one system, ref arm THEN smc arm (same-device
# wall-clock pairing, research/checkpoints_b3.md). Usage: 22_run_b3_lane.sh <gpu> <sys> <devtag> <arm_timeout_s>
set -uo pipefail
GPU=$1; SYS=$2; TAG=$3; TMO=$4
ROOT=/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus
PY=/raid/benson/.venvs/cgl2/bin/python
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU GIGALENS_X64=1
for ARM in ref smc; do
  LOG="$ROOT/data/b3_lane_${ARM}_sys${SYS}_${TAG}.log"
  echo "[lane $TAG] $ARM sys$SYS start $(date)" | tee -a "$LOG"
  timeout "$TMO" "$PY" 22_run_b3.py --sys "$SYS" --arm "$ARM" --devtag "$TAG" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 124 ]; then
    echo "[lane $TAG] $ARM sys$SYS BLOCKED-BY-BUDGET (timeout ${TMO}s)" | tee -a "$LOG"
  elif [ $rc -ne 0 ]; then
    echo "[lane $TAG] $ARM sys$SYS EXIT rc=$rc" | tee -a "$LOG"
  else
    echo "[lane $TAG] $ARM sys$SYS done $(date)" | tee -a "$LOG"
  fi
done
echo "[lane $TAG] ALL DONE $(date)"
