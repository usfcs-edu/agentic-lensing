#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-t02-low-s3
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# claude-giga-lens-linus P1 T0.2 SEED-REPEAT (2026-07-15): EXACT clone of the
# P1c production e2_smc_canary_v3b_low.slurm (frozen config: 128 particles,
# correlated likelihood, q from svi_cov canary draws) with ONLY (a) --seed
# 2 -> 3 and (b) output paths redirected to $PSCRATCH + the cgl2-linus CFS
# results dir (new-campaign storage rule). Purpose: sigma_seed(gamma) /
# sigma_seed(logZ_low) for the T0.2 gate (<=0.008 / <5 nats).
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
mkdir -p "$OUTDIR" "$CFSDIR"
SMC_PARTICLES="${SMC_PARTICLES:-128}"
echo "[t02-low-s3] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1)"
echo "[t02-low-s3] SMC_PARTICLES=$SMC_PARTICLES INFLATE=${INFLATE:-3.0} "\
"MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION PREALLOC=${XLA_PYTHON_CLIENT_PREALLOCATE:-default}"
python 10_run_e2.py --mode prod --tag v3b --basins low --sampler smc \
    --smc-fit-npz data/results/e2_v3b_low_canary_svicov.npz \
    --smc-particles "$SMC_PARTICLES" --smc-cov-inflate "${INFLATE:-3.0}" \
    --smc-mcmc-steps 4 --smc-integration-steps 8 --smc-step-size 0.1 \
    --smc-target-ess 0.7 --smc-max-lambda 400 --seed 3 \
    --out "$OUTDIR/e2_v3b_low_smc_seed3.json" \
    > "$OUTDIR/e2_v3b_low_smc_seed3_run.log" 2>&1
cp "$OUTDIR/e2_v3b_low_smc_seed3.npz" "$OUTDIR/e2_v3b_low_smc_seed3.json" \
   "$OUTDIR/e2_v3b_low_smc_seed3_run.log" "$CFSDIR/"
echo "=== T0.2 SEED-3 LOW SUMMARY ==="
grep -E "q from|SMC gamma|H1-SMC|Error|Exhausted|Out of memory|Traceback|did not reach" \
    "$OUTDIR/e2_v3b_low_smc_seed3_run.log" | grep -vE "weakref|WeakStruct" || true
