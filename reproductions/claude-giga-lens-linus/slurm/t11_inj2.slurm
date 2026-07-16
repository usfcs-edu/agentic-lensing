#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # pinned 2026-07-15: SMC p128/96 needs ~39GiB (job 55952482 OOM on hbm40g)
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-t11-i2
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# claude-giga-lens-linus P1 T1.1 INJECTION-RECOVERY #2 (2026-07-15): the
# CONFIRMATORY experiment for the P1 mechanism synthesis. Data = t11_inj2_v3b
# (real v3b residual field [img - model_map_v3b_cold] + synthetic anchor-truth
# scene gamma_truth=1.43298, source shift (+0.030,-0.014)"; built+gated by
# claude-giga-lens-linus/06_build_injections.py, provenance
# t11_inj2_truth.json). Pipeline = UNMODIFIED production v3b-low correlated
# path with ONLY --data-file pointed at the injection (keyword-only clone of
# the --whitener pattern; default certified bit-for-bit): step 1 reproduces
# the e2_v3b_low_canary_svicov.npz generation (EXACT preserved config: 24ch,
# stage1 200+200, stage2 50+100, svi_cov 7000x16, seed 2) ON THE INJECTION;
# step 2 = the frozen production correlated SMC (p128, seed 2) warm-started
# from it. Whitener = PRODUCTION whitener_v3b (unchanged - testing it IS the
# experiment). Gate: median(gamma_rec - 1.433) < -0.072 confirms the
# stationary-whitener low bias; |bias| < 0.024 exonerates.
# Resilience: if walltime killed step 2, resubmit with SKIP_PREP=1 (canary
# npz persists on $PSCRATCH).
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
DATAFILE="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/t11_inj2_v3b.npz"
mkdir -p "$OUTDIR" "$CFSDIR"
SMC_PARTICLES="${SMC_PARTICLES:-128}"
echo "[t11-i2] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1) 10_run_e2.py md5=$(md5sum 10_run_e2.py | cut -d' ' -f1)"
echo "[t11-i2] datafile md5=$(md5sum "$DATAFILE" | cut -d' ' -f1)"
echo "[t11-i2] SMC_PARTICLES=$SMC_PARTICLES INFLATE=${INFLATE:-3.0} "\
"MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION SKIP_PREP=${SKIP_PREP:-0}"
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
python 10_run_e2.py --mode prod --tag v3b --basins low \
    --data-file "$DATAFILE" \
    --metric svi_cov --svi-steps 7000 --svi-particles 16 \
    --chains 24 --num-leapfrog 16 --step-size 0.1 \
    --stage1-burn 200 --stage1-keep 200 --burn 50 --keep 100 \
    --map-rounds 4 --map-iters 200 --seed 2 \
    --out "$OUTDIR/t11_inj2_canary_svicov.json" \
    > "$OUTDIR/t11_inj2_canary_svicov_run.log" 2>&1
cp "$OUTDIR/t11_inj2_canary_svicov.npz" "$OUTDIR/t11_inj2_canary_svicov.json" \
   "$OUTDIR/t11_inj2_canary_svicov_run.log" "$CFSDIR/"
fi
python 10_run_e2.py --mode prod --tag v3b --basins low --sampler smc \
    --data-file "$DATAFILE" \
    --smc-fit-npz "$OUTDIR/t11_inj2_canary_svicov.npz" \
    --smc-particles "$SMC_PARTICLES" --smc-cov-inflate "${INFLATE:-3.0}" \
    --smc-mcmc-steps 4 --smc-integration-steps 8 --smc-step-size 0.1 \
    --smc-target-ess 0.7 --smc-max-lambda 400 --seed 2 \
    --out "$OUTDIR/t11_inj2_smc.json" \
    > "$OUTDIR/t11_inj2_smc_run.log" 2>&1
cp "$OUTDIR/t11_inj2_smc.npz" "$OUTDIR/t11_inj2_smc.json" \
   "$OUTDIR/t11_inj2_smc_run.log" "$CFSDIR/"
echo "=== T1.1 INJ1 SUMMARY ==="
grep -E "MAP:|METRIC|SVI:|posterior gamma|q from|SMC gamma|Error|Exhausted|Out of memory|Traceback|did not reach" \
    "$OUTDIR/t11_inj2_canary_svicov_run.log" "$OUTDIR/t11_inj2_smc_run.log" \
    | grep -vE "weakref|WeakStruct" || true
