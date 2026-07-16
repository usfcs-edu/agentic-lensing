#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # 128 x 367.6 MB/particle ~ 47 GB (l0_v3b_memory_smoke)
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-l0g2
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P3 L0-G2: v3b CORRELATED per-basin refit on the scene API (low @128 gamma
# gate leg + steep @96 logZ-sign comparator), production sampler recipe
# mirrored (cgl.e2.run_correlated_smc parameter-for-parameter), seed 2.
# Path A (cgl2-pm venv). Deferred from phoenix (L4 fits only ~56 particles).
set -euo pipefail
WORK=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/campaign
VENV=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv
X46=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/data/l0g2_basin_x46.npz
cd "$WORK"
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files)"
[ -f "$X46" ] || { echo "MISSING x46 export $X46"; exit 3; }
echo "[l0g2] x46 md5=$(md5sum $X46 | cut -d' ' -f1) whitener md5=$(md5sum ../claude-giga-lens/data/whitener_v3b.npz | cut -d' ' -f1)"
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
echo "[l0g2] host=$(hostname) runner md5=$(md5sum 25_run_l0g2_scene.py | cut -d' ' -f1)"
python 25_run_l0g2_scene.py --x46-npz "$X46" --seed 2 \
    --out "$OUTDIR/l0g2_v3b_scene_seed2" \
    > "$OUTDIR/l0g2_v3b_scene_seed2_run.log" 2>&1
cp "$OUTDIR/l0g2_v3b_scene_seed2.npz" "$OUTDIR/l0g2_v3b_scene_seed2.json" \
   "$OUTDIR/l0g2_v3b_scene_seed2_run.log" "$CFSDIR/"
echo "=== L0-G2 SUMMARY ==="
grep -E "devices|q\(|gamma=|dlogZ|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR/l0g2_v3b_scene_seed2_run.log" | grep -vE "weakref" || true
