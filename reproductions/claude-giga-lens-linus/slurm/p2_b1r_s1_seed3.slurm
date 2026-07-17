#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (55952482 OOM lesson; carried flag)
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b1r-s1-s3
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B1 carousel33 S1 REAL DATA (prior-seeded MAMS-SMC, N=512, seed 3, mutation
# chunk 32, weight-step chunk 64 — same pins as the mock arm p2_b1_s1_seed3).
# DATA = the team's REAL MUSE cutouts (newnewcutouts/source4-5.fits + source9.fits,
# staged under $WORK/newnewcutouts, md5-audited via deploy.md5; UNPUBLISHED team
# data — D4/D6). Builder: cgl2.zoo.build_carousel33 real path via
# CGL2_CAROUSEL_DATA (raise-never-default).
# *** SUBMISSION HELD 2026-07-16: the mock-arm siblings 55985444/45 hit this same
# *** 04:00 walltime (TIMEOUT, no artifacts — runner writes npz only at the end).
# *** Walltime/budget redesign is an orchestrator decision (CAMPAIGN.md B1-real
# *** amendment 2026-07-16); do NOT sbatch as-is without it.
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
export CGL2_CAROUSEL_DATA="$WORK/newnewcutouts"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
mkdir -p "$OUTDIR" "$CFSDIR"
echo "[b1r-s1-s3] host=$(hostname) runner md5=$(md5sum 23_run_p2_scene.py | cut -d' ' -f1) cutouts md5: $(cd newnewcutouts && md5sum source4-5.fits source9.fits | tr '\n' ' ')"
python 23_run_p2_scene.py --target carousel33 --arm s1 --n 512 --seed 3 \
    --chunk 32 --forward-chunk 64 --out "$OUTDIR/b1r_carousel33_s1_seed3" \
    > "$OUTDIR/b1r_carousel33_s1_seed3_run.log" 2>&1
cp "$OUTDIR/b1r_carousel33_s1_seed3.npz" "$OUTDIR/b1r_carousel33_s1_seed3.json" \
   "$OUTDIR/b1r_carousel33_s1_seed3_run.log" "$CFSDIR/"
echo "=== B1R S1 SEED3 SUMMARY ==="
grep -E "built|logZ|stages|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR/b1r_carousel33_s1_seed3_run.log" | grep -vE "weakref" || true
