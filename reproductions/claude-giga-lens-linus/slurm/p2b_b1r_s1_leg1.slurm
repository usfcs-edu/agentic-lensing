#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (standing SMC-wave rule)
#SBATCH -q shared
#SBATCH -t 03:55:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-p2b-b1r-s1-L1
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2b B1-REDUCED S1r leg 1/3 (DESCOPE amendment research/checkpoint_b1_reduced.md,
# pre-registered 2026-07-19; user decision 2026-07-18: P2b cap 18 OUTSIDE the
# exhausted P2 cap; S1r hard cap 12 = 3 x 3:55 walls, worst case 11.75).
# ARM: carousel33 REAL MUSE cutouts (UNPUBLISHED team data, D4/D6), prior-seeded
# MAMS-SMC N=128 seed 2 (descoped from N=512 x {2,3}; thresholds NOT moved),
# mutation chunk 32 / weight-step chunk 64, checkpoint retrofit ON: in-job wall
# cap 3.55 h -> exit-3 PARTIAL protocol, per-stage ckpts on $PSCRATCH.
# CHAIN SUBMISSION (orchestrator; watchdog-register each, on_stall=alert):
#   J1=$(sbatch --parsable p2b_b1r_s1_leg1.slurm)
#   J2=$(sbatch --parsable --dependency=afterany:$J1 p2b_b1r_s1_leg2.slurm)
#   J3=$(sbatch --parsable --dependency=afterany:$J2 p2b_b1r_s1_leg3.slurm)
# SUPERSEDES slurm/p2_b1r_s1_seed2.slurm + p2_b1r_s1_seed3.slurm (HELD
# full-scope designs — never submit those).
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
# audit trail even on a hard kill: log + PARTIAL json + (tiny) ckpt dir -> CFS
finish() {
  cp -f "$OUTDIR/${NAME}_run.log" "$CFSDIR/" 2>/dev/null || true
  cp -f "$OUTDIR/${NAME}.PARTIAL.json" "$CFSDIR/" 2>/dev/null || true
  mkdir -p "$CFSDIR/${NAME}_ckpt" && cp -f "$CKPT"/* "$CFSDIR/${NAME}_ckpt/" 2>/dev/null || true
}
trap finish EXIT
echo "[p2b-b1r-s1-L1] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1) cutouts md5: $(cd newnewcutouts && md5sum source4-5.fits source9.fits | tr '\n' ' ')"
if [ -e "$CKPT" ]; then
  echo "LEG1 REFUSES: $CKPT already exists (fresh-leg no-mixing fence — use legs 2/3 --resume or point at a fresh dir)"
  exit 2
fi
rc=0
python 23_run_p2_scene.py --target carousel33 --arm s1 --n 128 --seed 2 \
    --chunk 32 --forward-chunk 64 --wall-cap-h 3.55 --ckpt-dir "$CKPT" \
    --out "$OUTDIR/$NAME" > "$OUTDIR/${NAME}_run.log" 2>&1 || rc=$?
if [ "$rc" -eq 0 ]; then
  cp "$OUTDIR/$NAME.npz" "$OUTDIR/$NAME.json" "$CFSDIR/"
  echo "[p2b-b1r-s1-L1] COMPLETE (lambda=1) — marker $NAME.json on CFS; legs 2/3 will fast-exit"
elif [ "$rc" -eq 3 ]; then
  echo "[p2b-b1r-s1-L1] PARTIAL_WALL_CAP (exit 3, l0 protocol — checkpoints kept); leg 2 resumes"
else
  echo "[p2b-b1r-s1-L1] FAILED rc=$rc"
  tail -30 "$OUTDIR/${NAME}_run.log" || true
  exit "$rc"
fi
echo "=== P2B B1R S1 LEG1 SUMMARY ==="
grep -E "built|logZ|stage|lam|Error|Exhausted|Traceback|PARTIAL" \
    "$OUTDIR/${NAME}_run.log" | tail -15 | grep -vE "weakref" || true
