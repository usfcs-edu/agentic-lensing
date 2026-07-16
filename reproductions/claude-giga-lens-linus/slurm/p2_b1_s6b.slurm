#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC-wave production pin (carried flag)
#SBATCH -q shared
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b1-s6b
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B1 carousel33 S6b: MAMS-alone-warm baseline (MAP multistart billed in the
# grad ledger + ensemble-preconditioned MAMS at lambda=1). Mock stand-in data
# (declared, same as the S1 legs).
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
echo "[b1-s6b] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1)"
python 23_run_p2_scene.py --target carousel33 --arm s6b --seed 2 \
    --chunk 32 --out "$OUTDIR/b1_carousel33_s6b_seed2" \
    > "$OUTDIR/b1_carousel33_s6b_seed2_run.log" 2>&1
cp "$OUTDIR/b1_carousel33_s6b_seed2.npz" "$OUTDIR/b1_carousel33_s6b_seed2.json" \
   "$OUTDIR/b1_carousel33_s6b_seed2_run.log" "$CFSDIR/"
echo "=== B1 S6B SUMMARY ==="
grep -E "built|map|round|Error|Exhausted|Traceback" \
    "$OUTDIR/b1_carousel33_s6b_seed2_run.log" | tail -20 | grep -vE "weakref" || true
