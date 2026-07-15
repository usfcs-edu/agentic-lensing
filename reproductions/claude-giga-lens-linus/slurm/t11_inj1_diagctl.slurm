#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-t11-i1d
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# claude-giga-lens-linus P1 T1.1 DIAGONAL CONTROL (2026-07-15): injection #1
# (t11_inj1_v3b, gamma_truth=1.43298) fit with the DIAGONAL masked marg
# likelihood via the delta whitener bundle (h=[[1]], M=0, keep_w=keep_mask;
# built by 06_build_injections.py; identity vs cgl.likelihood
# build_marg_model(whiten_fn=None) gated at 0.0 nats on 3 points). Same
# two-step prep+SMC as t11_inj1.slurm, ONLY --whitener added. Isolates the
# whitener: prediction gamma_rec ~ 1.29-1.43 (NOT dragged to ~1.1); also
# bounds any bias contributed by the residual field's bright-object
# scene-subtraction residue (same data, likelihood the only change).
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
DATAFILE="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/t11_inj1_v3b.npz"
WHITENER="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/whitener_v3b_delta_diag.npz"
mkdir -p "$OUTDIR" "$CFSDIR"
SMC_PARTICLES="${SMC_PARTICLES:-128}"
echo "[t11-i1d] host=$(hostname) e2.py md5=$(md5sum cgl/e2.py | cut -d' ' -f1) 10_run_e2.py md5=$(md5sum 10_run_e2.py | cut -d' ' -f1)"
echo "[t11-i1d] datafile md5=$(md5sum "$DATAFILE" | cut -d' ' -f1) whitener md5=$(md5sum "$WHITENER" | cut -d' ' -f1)"
echo "[t11-i1d] SMC_PARTICLES=$SMC_PARTICLES INFLATE=${INFLATE:-3.0} "\
"MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION SKIP_PREP=${SKIP_PREP:-0}"
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
python 10_run_e2.py --mode prod --tag v3b --basins low \
    --data-file "$DATAFILE" --whitener "$WHITENER" \
    --metric svi_cov --svi-steps 7000 --svi-particles 16 \
    --chains 24 --num-leapfrog 16 --step-size 0.1 \
    --stage1-burn 200 --stage1-keep 200 --burn 50 --keep 100 \
    --map-rounds 4 --map-iters 200 --seed 2 \
    --out "$OUTDIR/t11_inj1_diag_canary_svicov.json" \
    > "$OUTDIR/t11_inj1_diag_canary_svicov_run.log" 2>&1
cp "$OUTDIR/t11_inj1_diag_canary_svicov.npz" "$OUTDIR/t11_inj1_diag_canary_svicov.json" \
   "$OUTDIR/t11_inj1_diag_canary_svicov_run.log" "$CFSDIR/"
fi
python 10_run_e2.py --mode prod --tag v3b --basins low --sampler smc \
    --data-file "$DATAFILE" --whitener "$WHITENER" \
    --smc-fit-npz "$OUTDIR/t11_inj1_diag_canary_svicov.npz" \
    --smc-particles "$SMC_PARTICLES" --smc-cov-inflate "${INFLATE:-3.0}" \
    --smc-mcmc-steps 4 --smc-integration-steps 8 --smc-step-size 0.1 \
    --smc-target-ess 0.7 --smc-max-lambda 400 --seed 2 \
    --out "$OUTDIR/t11_inj1_diag_smc.json" \
    > "$OUTDIR/t11_inj1_diag_smc_run.log" 2>&1
cp "$OUTDIR/t11_inj1_diag_smc.npz" "$OUTDIR/t11_inj1_diag_smc.json" \
   "$OUTDIR/t11_inj1_diag_smc_run.log" "$CFSDIR/"
echo "=== T1.1 INJ1 DIAG-CONTROL SUMMARY ==="
grep -E "MAP:|METRIC|SVI:|posterior gamma|q from|SMC gamma|Error|Exhausted|Out of memory|Traceback|did not reach" \
    "$OUTDIR/t11_inj1_diag_canary_svicov_run.log" "$OUTDIR/t11_inj1_diag_smc_run.log" \
    | grep -vE "weakref|WeakStruct" || true
