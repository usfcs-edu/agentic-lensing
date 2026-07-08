#!/usr/bin/env bash
# 07_run_e1_batch.sh -- run an E1 experiment batch across GPUs {8,4,5,6,7}.
#
# One fit process per GPU (pinned via CUDA_VISIBLE_DEVICES, autotune off --
# the gu-2022 03_run_batch pattern), resumable (jobs whose output npz exists
# are skipped), one log per fit in data/logs/e1/.
#
# DOCUMENTED DEVIATION from the strict gu-2022 wave pattern: workers consume
# a flock-guarded FIFO queue (work stealing) instead of fixed waves, because
# the L4 is ~2x an A16 and corr/diag fit durations differ several-fold; it is
# still exactly one process per GPU at all times.
#
# Usage:
#   ./07_run_e1_batch.sh <pilot|e1a|e1b|e1c|e1d|all> [--dry-run] [EXTRA...]
#   CGL_E1_GPUS="8 4 5" ./07_run_e1_batch.sh e1c        # GPU subset override
# EXTRA args are forwarded verbatim to every 06_fit_mock.py call.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/raid/benson/.venvs/cgl/bin/python
EXP=${1:?usage: 07_run_e1_batch.sh <pilot|e1a|e1b|e1c|e1d|all> [--dry-run]}
shift
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
EXTRA=("$@")
read -r -a GPUS <<< "${CGL_E1_GPUS:-8 4 5 6 7}"

mkdir -p "$HERE/data/logs/e1" "$HERE/data/e1_fits" "$HERE/data/e1_kernels"

QUEUE=$(mktemp "${TMPDIR:-/tmp}/e1_queue_XXXXXX")
trap 'rm -f "$QUEUE" "$QUEUE.idx" "$QUEUE.lock"' EXIT

if [ -f "$EXP" ]; then
  # file-based queue: TSV of "name<TAB>args" (e.g. the diagnosis pass);
  # resumable -- lines whose --out target already exists are skipped
  while IFS=$'\t' read -r name args; do
    out=$(echo "$args" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    if [ -n "$out" ] && [ -e "$out" ]; then continue; fi
    printf '%s\t%s\n' "$name" "$args"
  done < "$EXP" > "$QUEUE"
else
  "$PY" - "$EXP" > "$QUEUE" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from cgl.e1 import build_job_manifest
for j in build_job_manifest(sys.argv[1]):
    print(j["name"] + "\t" + " ".join(j["args"]))
PYEOF
fi

N_JOBS=$(wc -l < "$QUEUE")
echo "experiment=$EXP jobs=$N_JOBS gpus=${GPUS[*]} (existing outputs skipped)"
if [ "$DRY" = "1" ]; then
  echo "DRY RUN -- no fits launched"
  while IFS=$'\t' read -r name args; do echo -e "JOB\t$name\t$args"; done < "$QUEUE"
  exit 0
fi
[ "$N_JOBS" -eq 0 ] && { echo "nothing to do"; exit 0; }

echo 0 > "$QUEUE.idx"
touch "$QUEUE.lock"

next_job() {  # atomically pop the next queue line (empty output = done)
  flock "$QUEUE.lock" bash -c '
    i=$(cat "'"$QUEUE"'.idx"); n=$(wc -l < "'"$QUEUE"'")
    if [ "$i" -ge "$n" ]; then exit 1; fi
    echo $((i+1)) > "'"$QUEUE"'.idx"
    sed -n "$((i+1))p" "'"$QUEUE"'"'
}

worker() {
  local gpu=$1 line name args rc
  while line=$(next_job); do
    name=${line%%$'\t'*}
    args=${line#*$'\t'}
    echo "[gpu $gpu] START $name $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID \
      XLA_FLAGS=--xla_gpu_autotune_level=0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$PY" "$HERE/06_fit_mock.py" $args "${EXTRA[@]}" \
      > "$HERE/data/logs/e1/${name}.log" 2>&1
    rc=$?
    echo "[gpu $gpu] DONE  $name rc=$rc $(date +%H:%M:%S)"
  done
}

cd "$HERE"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "batch $EXP complete $(date +%H:%M:%S)"
