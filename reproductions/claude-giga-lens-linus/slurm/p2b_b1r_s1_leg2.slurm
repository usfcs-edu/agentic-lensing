#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (standing SMC-wave rule)
#SBATCH -q shared
#SBATCH -t 03:55:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-p2b-b1r-s1-L2
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2b B1-REDUCED S1r leg 2/3 (research/checkpoint_b1_reduced.md). Submit with
#   --dependency=afterany:<leg1-jobid>
# Resumes from the newest stage checkpoint (BIT-IDENTICAL incl. full logZ,
# tests/test_ckpt_resume.py); NO-OP FAST-EXIT if the COMPLETE marker
# (b1r128_carousel33_s1_seed2.json) already exists. Same pins/fences as leg 1.
set -euo pipefail
WORK=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/campaign
VENV=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv
cd "$WORK"
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files)"
module load python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # guard vs DEFAULT_GPU=9 phoenix fallback
source "$VENV/bin/activate"
export GIGALENS_X64=1 JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
export CGL2_CAROUSEL_DATA="$WORK/newnewcutouts"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
NAME=b1r128_carousel33_s1_seed2
CKPT="$OUTDIR/${NAME}_ckpt"
mkdir -p "$OUTDIR" "$CFSDIR"
if [ -f "$OUTDIR/$NAME.json" ] || [ -f "$CFSDIR/$NAME.json" ]; then
  echo "[p2b-b1r-s1-L2] COMPLETE marker present ($NAME.json) — no-op fast-exit"
  exit 0
fi
finish() {
  cp -f "$OUTDIR/${NAME}_run.log" "$CFSDIR/" 2>/dev/null || true
  cp -f "$OUTDIR/${NAME}.PARTIAL.json" "$CFSDIR/" 2>/dev/null || true
  mkdir -p "$CFSDIR/${NAME}_ckpt" && cp -f "$CKPT"/* "$CFSDIR/${NAME}_ckpt/" 2>/dev/null || true
}
trap finish EXIT
echo "[p2b-b1r-s1-L2] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1) cutouts md5: $(cd newnewcutouts && md5sum source4-5.fits source9.fits | tr '\n' ' ')"
if [ ! -d "$CKPT" ]; then
  echo "[p2b-b1r-s1-L2] WARNING: no ckpt dir from leg 1 — --resume falls back to a FRESH run (first-leg semantics)"
fi
rc=0
python 23_run_p2_scene.py --target carousel33 --arm s1 --n 128 --seed 2 \
    --chunk 32 --forward-chunk 64 --wall-cap-h 3.55 --ckpt-dir "$CKPT" \
    --resume --out "$OUTDIR/$NAME" > "$OUTDIR/${NAME}_run.log" 2>&1 || rc=$?
if [ "$rc" -eq 0 ]; then
  cp "$OUTDIR/$NAME.npz" "$OUTDIR/$NAME.json" "$CFSDIR/"
  echo "[p2b-b1r-s1-L2] COMPLETE (lambda=1) — marker $NAME.json on CFS; leg 3 will fast-exit"
elif [ "$rc" -eq 3 ]; then
  echo "[p2b-b1r-s1-L2] PARTIAL_WALL_CAP (exit 3, l0 protocol — checkpoints kept); leg 3 resumes"
else
  echo "[p2b-b1r-s1-L2] FAILED rc=$rc"
  tail -30 "$OUTDIR/${NAME}_run.log" || true
  exit "$rc"
fi
echo "=== P2B B1R S1 LEG2 SUMMARY ==="
grep -E "built|resumed|logZ|stage|lam|Error|Exhausted|Traceback|PARTIAL" \
    "$OUTDIR/${NAME}_run.log" | tail -15 | grep -vE "weakref" || true
