#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-t03-compmask
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# claude-giga-lens-linus P1 T0.3 COMPANION-MASK DISCRIMINATOR (2026-07-15):
# EXACT clone of the P1c production e2_smc_canary_v3b_low.slurm (frozen
# config: 128 particles, seed 2 = THE production seed, q from svi_cov canary
# draws) with ONLY (a) --whitener pointed at the companion-eroded variant
# bundle (whiten-then-drop: kernel h/e_op/M byte-identical to
# whitener_v3b.npz, keep_w additionally eroded by the LL2/LL3 companion
# region r<1.2" @ (-2.34,-2.86)" dilated by the (2M+1)^2 kernel support;
# 9273->8247 whitened dof; built+validated by
# claude-giga-lens-linus/05_build_companion_whitener.py, geometry cited from
# X1-G0 research/x1_g0_mechanism_check.md) and (b) output paths redirected to
# $PSCRATCH + the cgl2-linus CFS results dir. build_target loads the variant
# via np.load(DATA / wname) -- an absolute --whitener path overrides DATA
# (pathlib semantics), so the OLD campaign's staged tree is untouched.
# Discriminator: does gamma_binned(corr,low) move off 1.103 by >=3*sigma_stat
# (>=0.024) when the companion's localized chi2~9-15 misfit is excluded?
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
WHITENER="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/whitener_v3b_companion_eroded.npz"
mkdir -p "$OUTDIR" "$CFSDIR"
SMC_PARTICLES="${SMC_PARTICLES:-128}"
echo "[t03-compmask] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1)"
echo "[t03-compmask] whitener md5=$(md5sum "$WHITENER" | cut -d' ' -f1)"
echo "[t03-compmask] SMC_PARTICLES=$SMC_PARTICLES INFLATE=${INFLATE:-3.0} "\
"MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION PREALLOC=${XLA_PYTHON_CLIENT_PREALLOCATE:-default}"
python 10_run_e2.py --mode prod --tag v3b --basins low --sampler smc \
    --whitener "$WHITENER" \
    --smc-fit-npz data/results/e2_v3b_low_canary_svicov.npz \
    --smc-particles "$SMC_PARTICLES" --smc-cov-inflate "${INFLATE:-3.0}" \
    --smc-mcmc-steps 4 --smc-integration-steps 8 --smc-step-size 0.1 \
    --smc-target-ess 0.7 --smc-max-lambda 400 --seed 2 \
    --out "$OUTDIR/e2_v3b_low_smc_compmask.json" \
    > "$OUTDIR/e2_v3b_low_smc_compmask_run.log" 2>&1
cp "$OUTDIR/e2_v3b_low_smc_compmask.npz" "$OUTDIR/e2_v3b_low_smc_compmask.json" \
   "$OUTDIR/e2_v3b_low_smc_compmask_run.log" "$CFSDIR/"
echo "=== T0.3 COMPANION-MASK SUMMARY ==="
grep -E "q from|SMC gamma|H1-SMC|Error|Exhausted|Out of memory|Traceback|did not reach" \
    "$OUTDIR/e2_v3b_low_smc_compmask_run.log" | grep -vE "weakref|WeakStruct" || true
