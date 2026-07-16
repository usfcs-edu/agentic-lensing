#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # pinned 2026-07-15: SMC p128/96 needs ~39GiB (job 55952482 OOM on hbm40g)
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-t02-steep-s3
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# claude-giga-lens-linus P1 T0.2 SEED-REPEAT (2026-07-15): EXACT clone of the
# P1c production e2_smc_canary_v3b_steep.slurm (frozen config: 96 particles,
# PREALLOCATE=false, correlated likelihood, q from the frozen production
# steep_draws in e2_v3b.npz) with ONLY (a) --seed 2 -> 3 and (b) output paths
# redirected to $PSCRATCH + the cgl2-linus CFS results dir (new-campaign
# storage rule). Purpose: sigma_seed(logZ_steep) for the T0.2 DlogZ gate
# (<5 nats keeps the 28.9-nat basin preference >5 sigma).
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
mkdir -p "$OUTDIR" "$CFSDIR"
SMC_PARTICLES="${SMC_PARTICLES:-96}"
echo "[t02-steep-s3] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1)"
echo "[t02-steep-s3] SMC_PARTICLES=$SMC_PARTICLES INFLATE=${INFLATE:-3.0} "\
"MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION PREALLOC=$XLA_PYTHON_CLIENT_PREALLOCATE"
python 10_run_e2.py --mode prod --tag v3b --basins steep --sampler smc \
    --smc-fit-npz data/results/e2_v3b.npz \
    --smc-particles "$SMC_PARTICLES" --smc-cov-inflate "${INFLATE:-3.0}" \
    --smc-mcmc-steps 4 --smc-integration-steps 8 --smc-step-size 0.1 \
    --smc-target-ess 0.7 --smc-max-lambda 400 --seed 3 \
    --out "$OUTDIR/e2_v3b_steep_smc_seed3.json" \
    > "$OUTDIR/e2_v3b_steep_smc_seed3_run.log" 2>&1
cp "$OUTDIR/e2_v3b_steep_smc_seed3.npz" "$OUTDIR/e2_v3b_steep_smc_seed3.json" \
   "$OUTDIR/e2_v3b_steep_smc_seed3_run.log" "$CFSDIR/"
echo "=== T0.2 SEED-3 STEEP SUMMARY ==="
grep -E "q from|SMC gamma|H1-SMC|Error|Exhausted|Out of memory|Traceback|did not reach" \
    "$OUTDIR/e2_v3b_steep_smc_seed3_run.log" | grep -vE "weakref|WeakStruct" || true
