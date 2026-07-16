#!/bin/bash
# 22_run_b3_lanes.sh -- B3 RESTART driver (Front C, 2026-07-16): all 4 cells run
# SEQUENTIALLY on the L4 (GPU 8). Why: the A16s (15.3 GB) OOM the reference arm at
# EVERY pre-registered fallback size (MAP 500/256/128 -> 26.9/13.9/12.6 GiB requests,
# see data/b3_lane_ref_sys*_a16-*.log and the restart entry in
# research/checkpoints_b3.md); the L4 (23 GB) fits from the 256 fallback on.
# ref THEN smc per system (same-device wall-clock pairing preserved, devtag l4-8).
# GPU 9 (L4) carries the L0 leg -- NEVER touched here (CUDA_VISIBLE_DEVICES=8).
#
# Launch (detached -- survives parent exit; lesson from the 2026-07-15 lanes):
#   nohup setsid bash 22_run_b3_lanes.sh >> data/b3_lane2_driver.log 2>&1 &
# Logs: data/b3_lane2_sys<i>_<arm>.log ; driver pid file: data/b3_lane2_driver.pid
# Idempotent: arms whose run-json already says status OK are skipped on re-launch.
# NO harvest here -- 22_harvest_b3.py runs after (orchestrator; figs before gates).
set -uo pipefail
ROOT=/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus
PY=/raid/benson/.venvs/cgl2/bin/python
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 GIGALENS_X64=1
TAG=l4-8
TMO=18000   # checkpoint budget fence: 5 h/arm on L4 -> BLOCKED-BY-BUDGET

echo $$ > "$ROOT/data/b3_lane2_driver.pid"
echo "[lane2] driver pid=$$ start $(date)"
for SYS in 0 1 2 3; do
  for ARM in ref smc; do
    LOG="$ROOT/data/b3_lane2_sys${SYS}_${ARM}.log"
    JSON="$ROOT/data/b3_run_${ARM}_sys${SYS}_${TAG}.json"
    if [ -f "$JSON" ] && grep -q '"status": "OK"' "$JSON"; then
      echo "[lane2] sys$SYS $ARM already OK ($JSON) -- skip"
      continue
    fi
    echo "[lane2] sys$SYS $ARM start $(date)" | tee -a "$LOG"
    timeout "$TMO" "$PY" 22_run_b3.py --sys "$SYS" --arm "$ARM" --devtag "$TAG" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 124 ]; then
      echo "[lane2] sys$SYS $ARM BLOCKED-BY-BUDGET (timeout ${TMO}s)" | tee -a "$LOG"
    elif [ $rc -ne 0 ]; then
      echo "[lane2] sys$SYS $ARM EXIT rc=$rc" | tee -a "$LOG"
    else
      echo "[lane2] sys$SYS $ARM done $(date)" | tee -a "$LOG"
    fi
  done
done
echo "[lane2] ALL RUNS DONE $(date) -- harvest: $PY 22_harvest_b3.py"
