#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl-e2-smc-v3bsteep
#SBATCH -o slurm-%j.out
# CORRELATED-SMC BASIN-EVIDENCE (2026-07-14): the STEEP-basin companion to the
# v3b-LOW SMC canary. Reference q fit from the frozen production steep_draws
# (e2_v3b.npz, steep MAP map_v11_v3b_warm gamma=2.672). Gives logZ_steep so the
# H1 linchpin is logZ_low vs logZ_steep = the correlated analog of the
# diagonal-SMC +162-nat steep preference: does correlated flip it to LOW?
# SMC log-evidence is q-robust (SMC-vs-SMC, the q log-det cancels in the ratio),
# so the imperfectly-mixed frozen steep_draws are an acceptable q.
#
# STEEP MEMORY RETRY (2026-07-14): the 128-particle steep run OOM'd — its 46.44
# GiB transient grad op (~363 MB/particle) on top of the ~39 GiB persistent set
# hit ~85 GiB > 80 GB (steep's wider q sits ~15 GiB past where LOW fit at 128).
# Drop to 96 particles (the credible-sigma floor): the transient scales to
# ~34.8 GiB, whole peak ~64 GiB, inside the 76 GiB MEM_FRACTION budget. Also set
# PREALLOCATE=false (on-demand growth) to ease large-transient fragmentation.
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
SMC_PARTICLES="${SMC_PARTICLES:-96}"
echo "[smc-steep] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1)"
echo "[smc-steep] SMC_PARTICLES=$SMC_PARTICLES INFLATE=${INFLATE:-3.0} "\
"MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION PREALLOC=$XLA_PYTHON_CLIENT_PREALLOCATE"
python 10_run_e2.py --mode prod --tag v3b --basins steep --sampler smc \
    --smc-fit-npz data/results/e2_v3b.npz \
    --smc-particles "$SMC_PARTICLES" --smc-cov-inflate "${INFLATE:-3.0}" \
    --smc-mcmc-steps 4 --smc-integration-steps 8 --smc-step-size 0.1 \
    --smc-target-ess 0.7 --smc-max-lambda 400 --seed 2 \
    --out data/results/e2_v3b_steep_smc_canary_p96.json \
    > data/results/e2_v3b_steep_smc_canary_p96_run.log 2>&1
echo "=== SMC STEEP SUMMARY (v3b steep, p96) ==="
grep -E "q from|SMC gamma|H1-SMC|Error|Exhausted|Out of memory|Traceback|did not reach" \
    data/results/e2_v3b_steep_smc_canary_p96_run.log | grep -vE "weakref|WeakStruct" || true
