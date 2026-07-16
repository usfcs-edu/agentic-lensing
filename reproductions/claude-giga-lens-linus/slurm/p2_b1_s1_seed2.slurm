#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (55952482 OOM lesson; carried flag)
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b1-s1-s2
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B1 carousel33 S1 (prior-seeded MAMS-SMC, N=512, seed 2, mutation chunk 32, weight-step chunk 64 (sizing: 837 MB/particle grad, ~158 MB/particle fwd, login-A100 pre-flight)).
# DATA = the documented SEEDED MOCK stand-in (real MUSE cutouts permission-
# blocked; declared in the B1 design checkpoint, CAMPAIGN.md 2026-07-16).
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
echo "[b1-s1-s2] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1)"
python 23_run_p2_scene.py --target carousel33 --arm s1 --n 512 --seed 2 \
    --chunk 32 --forward-chunk 64 --out "$OUTDIR/b1_carousel33_s1_seed2" \
    > "$OUTDIR/b1_carousel33_s1_seed2_run.log" 2>&1
cp "$OUTDIR/b1_carousel33_s1_seed2.npz" "$OUTDIR/b1_carousel33_s1_seed2.json" \
   "$OUTDIR/b1_carousel33_s1_seed2_run.log" "$CFSDIR/"
echo "=== B1 S1 SEED2 SUMMARY ==="
grep -E "built|logZ|stages|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR/b1_carousel33_s1_seed2_run.log" | grep -vE "weakref" || true
