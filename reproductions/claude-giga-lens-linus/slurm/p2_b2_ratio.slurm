#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC-wave production pin (carried flag)
#SBATCH -q shared
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b2-ratio
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B2 ratio CONTROL, STANDALONE (recovery lane, post-mortem R3): the
# dspl20_ratio exact-reparameterization control arm alone — job 55985447's
# 2:30 wall killed it at 79 min mid-SMC after the orig arm's 66.5 min (the
# "cheap, 1 h" estimate is REFUTED by that partial; honest wall 3:00, est
# 2.3). Same design as the lost arm: prior-seeded S1 N=512 seed 2 chunk 128.
# This control DECIDES the B2 falsifier: orig arm already evaluated m-hat =
# 0.000 (FAIL-as-written, interpretation PENDING-CONTROL).
# RC1 checkpoint retrofit enabled: --ckpt-dir + --wall-cap-h 2.70 (on cap:
# stage checkpoints + PARTIAL json + exit 3, resumable with --resume).
set -euo pipefail
WORK=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/campaign
VENV=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv
cd "$WORK"
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files)"
module load python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # slurm sets this on shared QOS; guard against the DEFAULT_GPU=9 phoenix fallback
source "$VENV/bin/activate"
export GIGALENS_X64=1 JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
mkdir -p "$OUTDIR" "$CFSDIR"
echo "[b2r] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1) ckpt md5=$(md5sum cgl2/samplers/ckpt.py | cut -d' ' -f1)"
TAG="b2_dspl20_ratio_s1_seed2"
RC=0
python 23_run_p2_scene.py --target dspl20_ratio --arm s1 --n 512 --seed 2 \
    --chunk 128 --ckpt-dir "$OUTDIR/${TAG}_ckpt" --wall-cap-h 2.70 \
    --out "$OUTDIR/$TAG" > "$OUTDIR/${TAG}_run.log" 2>&1 || RC=$?
# results back ALWAYS (PARTIAL json + checkpoints are results too)
cp "$OUTDIR/${TAG}_run.log" "$CFSDIR/" || true
cp "$OUTDIR/$TAG.npz" "$OUTDIR/$TAG.json" "$CFSDIR/" 2>/dev/null \
    || echo "[b2r] no npz/json (RC=$RC — PARTIAL under wall cap?)"
cp "$OUTDIR/$TAG.PARTIAL.json" "$CFSDIR/" 2>/dev/null || true
mkdir -p "$CFSDIR/${TAG}_ckpt"
cp "$OUTDIR/${TAG}_ckpt"/* "$CFSDIR/${TAG}_ckpt/" 2>/dev/null || true
echo "[b2r] exit=$RC (0=COMPLETE, 3=PARTIAL wall-cap protocol)"
echo "=== B2 RATIO SUMMARY ==="
grep -E "built|logZ|stages|PARTIAL|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR/${TAG}_run.log" | grep -vE "weakref" || true
exit $RC
