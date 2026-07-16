#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC-wave production pin (carried flag)
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b2-dspl
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B2: dspl20_orig (headline arm, ORIGINAL pathological coords) +
# dspl20_ratio (control, exact ratio-coordinates reparameterization), both
# prior-seeded S1 N=512 seed 2, sequentially on the same GPU (same seeded data
# realization class per build; both builds are deterministic on this device).
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
echo "[b2] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1)"
for TGT in dspl20_orig dspl20_ratio; do
  echo "[b2] === $TGT ==="
  python 23_run_p2_scene.py --target "$TGT" --arm s1 --n 512 --seed 2 \
      --chunk 128 --out "$OUTDIR/b2_${TGT}_s1_seed2" \
      > "$OUTDIR/b2_${TGT}_s1_seed2_run.log" 2>&1
  cp "$OUTDIR/b2_${TGT}_s1_seed2.npz" "$OUTDIR/b2_${TGT}_s1_seed2.json" \
     "$OUTDIR/b2_${TGT}_s1_seed2_run.log" "$CFSDIR/"
done
echo "=== B2 SUMMARY ==="
grep -E "built|logZ|stages|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR"/b2_dspl20_*_run.log | grep -vE "weakref" || true
