#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl-e2-smc-v3blow
#SBATCH -o slurm-%j.out
# CORRELATED-SMC CANARY (2026-07-10; MEMORY-FIX re-run 2026-07-14): metric-free
# basin-local adaptive tempered SMC on v3b-LOW, reference q fit from the svi_cov
# canary draws (seeded at the gamma~1.10 DENSITY PEAK). Gate: converged gamma
# marginal (median + sigma + effective sample count) + logZ_low. Is
# gamma_binned(corr,low) cleanly ~1.10 with a REAL sigma now? cosmo_g / shared.
#
# MEMORY FIX (was OOM at 300 particles: 120.4 GB single op = ~400 MB/particle,
# A100=80GB). Peak scales linearly with n_particles (leapfrog steps run
# sequentially in the integrator scan; the tape is freed each step, so peak =
# n_particles x one grad-tape, NOT x integration_steps). So:
#   (1) SMC_PARTICLES 300 -> 128 (~51 GB single op; >=96 for a credible sigma);
#   (2) XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 (give JAX 76 GB of the 80 GB card).
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
SMC_PARTICLES="${SMC_PARTICLES:-128}"
echo "[smc-canary] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1)"
echo "[smc-canary] SMC_PARTICLES=$SMC_PARTICLES INFLATE=${INFLATE:-3.0} "\
"MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION PREALLOC=${XLA_PYTHON_CLIENT_PREALLOCATE:-default}"
python 10_run_e2.py --mode prod --tag v3b --basins low --sampler smc \
    --smc-fit-npz data/results/e2_v3b_low_canary_svicov.npz \
    --smc-particles "$SMC_PARTICLES" --smc-cov-inflate "${INFLATE:-3.0}" \
    --smc-mcmc-steps 4 --smc-integration-steps 8 --smc-step-size 0.1 \
    --smc-target-ess 0.7 --smc-max-lambda 400 --seed 2 \
    --out data/results/e2_v3b_low_smc_canary_fix.json \
    > data/results/e2_v3b_low_smc_canary_fix_run.log 2>&1
echo "=== SMC CANARY SUMMARY (v3b low, memfix) ==="
grep -E "q from|SMC gamma|H1-SMC|Error|Exhausted|Out of memory|Traceback|did not reach" \
    data/results/e2_v3b_low_smc_canary_fix_run.log | grep -vE "weakref|WeakStruct" || true
