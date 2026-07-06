#!/bin/bash
# Campaign test runner.
#   ./00_run_tests.sh          -> CPU unit tests (16x16 toys; sole sanctioned CPU exception)
#   ./00_run_tests.sh --gpu    -> also run GPU-marked tests (parity harness) on one pinned GPU
set -uo pipefail
cd "$(dirname "$0")"
PY=/raid/benson/.venvs/cgl/bin/python

echo "== CPU unit tests =="
"$PY" -m pytest tests -m "not gpu" -q
rc=$?

if [[ "${1:-}" == "--gpu" ]]; then
    echo "== GPU tests (pinned to CUDA_VISIBLE_DEVICES=${CGL_GPU:-8}, L4 default) =="
    # priority-fusion MUST be disabled in the process env (not per-test-file):
    # jaxlib 0.6.2's XLA priority-fusion pass livelocks on f64 random.normal
    # fused with a reduction (blackjax MCLMC momentum refresh), and XLA_FLAGS
    # is parsed once at backend init — which an earlier test module's imports
    # can trigger before a per-file append runs (CAMPAIGN.md, P0 stage log).
    CGL_TEST_GPU=1 GIGALENS_X64=1 \
    CUDA_VISIBLE_DEVICES="${CGL_GPU:-8}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion" \
    "$PY" -m pytest tests -m gpu -q
    rc=$((rc || $?))
fi

exit $rc
