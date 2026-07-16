#!/bin/bash
# deploy_f8_verify.slurm — P2 Path-A deployment verify (clone of
# parity_f8_nersc.slurm's shape, NATIVE-pin edition): 00_env_check + the
# 01_parity_scene gate battery under the NEW cgl2-pm venv (jax 0.6.2 cuda12,
# campaign pins) on one shared-QOS A100. Unlike the F8 template this does
# NOT set CGL2_F8 — jax==0.6.2 matches the pin, so require_jax_pin stays
# ACTIVE (strictly stronger than the report-only container cell). A PASS
# here is the "PM-native gate battery" row (plan research/p2_deployment_plan.md
# Path A step 3; F8 remains report-only / container-only).
# Budget: <= 0.2 A100-h est (30 min cap). Ledger row BEFORE results (house rule).
# Plain -C gpu on purpose: light job, do not waste the hbm80g pool.
#
#SBATCH -A cosmo_g
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -J cgl2-deploy-verify
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out

set -euo pipefail
WORK=${WORK:-/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/campaign}
VENV=${VENV:-/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv}
cd "$WORK"

# md5 deploy audit (canary before burning GPU time — the stale-remote lesson)
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files)"

module load python
source "$VENV/bin/activate"
export GIGALENS_X64=1 JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Same flag the phoenix L4 certification ran with (01_parity_scene docstring);
# the priority-fusion disable is the OLD-campaign A100 SMC flag, not needed here.
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export CGL2_GPU="${CUDA_VISIBLE_DEVICES:-0}"   # fallback only; shared QOS sets CVD

echo "=== 00_env_check ==="
python 00_env_check.py

echo "=== 01_parity_scene (native pins, jax-pin guard ACTIVE) ==="
rc=0
python 01_parity_scene.py || rc=$?   # keep going so the report is archived on FAIL too

cp data/parity_report_scene.json \
   "/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results/parity_report_scene_pmnative_${SLURM_JOB_ID}.json"
echo "PM-native gate report: results/parity_report_scene_pmnative_${SLURM_JOB_ID}.json (rc=$rc)"
exit $rc
