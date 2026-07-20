#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (standing SMC-wave rule)
#SBATCH -q shared
#SBATCH -t 04:55:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-p2b-b1r-s6b
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2b B1-REDUCED S6br — Track-A BUDGET-MATCHED MAMS-alone-warm baseline
# (DESCOPE amendment research/checkpoint_b1_reduced.md; replaces the
# until-converged 30+120 design whose measured true cost is 17-20 h).
# *** TEMPLATE — HELD until the S1r harvest fills GRAD_BUDGET below with
# *** B* = grad_evals.total from b1r128_carousel33_s1_seed2.json (the S1r
# *** COMPLETE result). Fill procedure + deploy.md5 update: slurm/README_p2b.md.
# *** The guards below refuse to run unfilled or without the S1r marker.
# Design: frozen MAP multistart 64x350 (billed) + 30 burn rounds (frozen) +
# sampling rounds until a fence stops the loop: grad fence at B* OR graceful
# in-job wall fence 4.5 h (cap 5 A100-h; wall 4:55 worst case 4.92). Env seams
# default OFF in the runner — only this job arms them.
# SUPERSEDES slurm/p2_b1r_s6b.slurm (HELD full-scope design — never submit).
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
S1R_JSON="$CFSDIR/b1r128_carousel33_s1_seed2.json"
NAME=b1r128_carousel33_s6b_seed2
mkdir -p "$OUTDIR" "$CFSDIR"

# ---- Track-A matched budget (TODO: harvest fills this) ----------------------
GRAD_BUDGET="TODO_FILL_FROM_S1R_HARVEST"  # <- B* = grad_evals.total from $S1R_JSON
# -----------------------------------------------------------------------------
if ! [[ "$GRAD_BUDGET" =~ ^[0-9]+$ ]]; then
  echo "S6BR TEMPLATE UNFILLED: GRAD_BUDGET='$GRAD_BUDGET' — fill B* from the"
  echo "S1r COMPLETE json (grad_evals.total) per slurm/README_p2b.md. Refusing."
  exit 2
fi
if [ ! -f "$S1R_JSON" ]; then
  echo "S1R COMPLETE MARKER MISSING ($S1R_JSON) — S6br is conditioned on S1r"
  echo "COMPLETE (amendment: no auto-burn against a failed primary arm). Refusing."
  exit 2
fi
export CGL2_S6B_GRAD_BUDGET="$GRAD_BUDGET"
export CGL2_S6B_WALL_CAP_H=4.5      # graceful round-boundary fence inside 4:55
export CGL2_S6B_SAMPLE_ROUNDS=100000  # ceiling raised so the FENCES govern the stop
finish() {
  cp -f "$OUTDIR/${NAME}_run.log" "$CFSDIR/" 2>/dev/null || true
}
trap finish EXIT
echo "[p2b-b1r-s6b] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1) cutouts md5: $(cd newnewcutouts && md5sum source4-5.fits source9.fits | tr '\n' ' ') B*=$GRAD_BUDGET"
python 23_run_p2_scene.py --target carousel33 --arm s6b --seed 2 \
    --chunk 32 --out "$OUTDIR/$NAME" > "$OUTDIR/${NAME}_run.log" 2>&1
cp "$OUTDIR/$NAME.npz" "$OUTDIR/$NAME.json" "$CFSDIR/"
echo "=== P2B B1R S6BR SUMMARY ==="
grep -E "built|map|round|fence STOP|Error|Exhausted|Traceback" \
    "$OUTDIR/${NAME}_run.log" | tail -20 | grep -vE "weakref" || true
