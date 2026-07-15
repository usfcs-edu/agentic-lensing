#!/usr/bin/env bash
# 00_run_tests.sh — CPU unit toys, then (if a GPU is visible) the full
# validation battery: cross-stack parity (F1-F7) + whitener port + correlated
# term vs dense exact reference.
#
# Prereq for the parity stage: data/parity_refs.npz (generate once with the
# OLD venv:  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=9 \
#   /raid/benson/.venvs/cgl/bin/python 01a_gen_parity_refs.py )
set -uo pipefail
cd "$(dirname "$0")"
PY=/raid/benson/.venvs/cgl2/bin/python
FAIL=0

echo "=== [1/4] CPU unit tests (16x16 toys, CGL2_ALLOW_CPU=1) ==="
"$PY" -m pytest tests/ -q || FAIL=1

echo "=== [2/4] whitener port manifest (CPU) ==="
"$PY" 02_port_whiteners.py || FAIL=1

if nvidia-smi >/dev/null 2>&1; then
  export GIGALENS_X64=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CGL2_GPU:-9}"
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  if [ -f data/parity_refs.npz ]; then
    echo "=== [3/4] cross-stack parity F1-F7 (GPU ${CUDA_VISIBLE_DEVICES}) ==="
    "$PY" 01_parity_scene.py || FAIL=1
  else
    echo "=== [3/4] SKIPPED: data/parity_refs.npz missing (run 01a in cgl venv)"
    FAIL=1
  fi
  echo "=== [4/4] correlated term vs dense exact reference (GPU) ==="
  "$PY" 03_correlated_term_validation.py || FAIL=1
else
  echo "=== [3-4/4] SKIPPED: no GPU visible ==="
fi

echo
if [ "$FAIL" -eq 0 ]; then echo "00_run_tests: ALL GREEN"; else
  echo "00_run_tests: FAILURES (see above; F6 has a documented noise-floor"
  echo "  failure on v3b — see data/parity_report_scene.json + CAMPAIGN.md)"; fi
exit $FAIL
