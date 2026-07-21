#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu       # PLAIN gpu ON PURPOSE (amendment 2026-07-20): the 56006048
#SBATCH -q shared    # 21-GiB OOM was the N=128 MC-SMC MAMS mutate tape; this
# arm's tapes are MAP n=1 (trivial) + SVI <=250-draw ELBO grad with the B3
# OOM halving ladder 250->128->64 (in-script, recorded fallbacks) + HMC
# 50-chain batch (~50 x ~93 MB ~ 4.7 GB). No hbm80g burn needed.
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-l0arbc
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P3-L0 anchor arbitration — CLASSIC-RECIPE ARM (PRIMARY vehicle per the
# 2026-07-20 amendment in research/checkpoints_l0.md, written BEFORE this run):
# three MC-SMC attempts wall-capped below lambda=1 (L4 ~0.45 @ ~18 h; A100
# hbm80g 0.587 @ 3.5 h in 56168446 — healthy, budget-infeasible vehicle), so
# the pre-registered optional arm (b) — classic scene-API MAP -> SVI -> HMC,
# the SAME algorithm class that produced the 1.433 anchor on the old stack —
# is promoted to primary. Gates and bands UNCHANGED; the MC-SMC partials stay
# as secondary evidence (CFS results/l0arb/l0_smc_checkpoints + local L4
# ckpts), PARTIAL status recorded.
# Vehicle: 11_arbitration_classic.py — warm start from the mapped foundry-i
# MAP (param_map, S1-certified path), B3 reference-arm frozen settings
# (MAP 350 adabelief(1e-2); SVI 250x500; HMC 50 chains 250/750 + single
# 500/1500 escalation; Rhat<1.05, ESS>=200); stage-wise artifact writes so a
# walltime kill loses at most one stage. NO gate math in-job (plots-first).
# Execution pattern = p3_l0arb.slurm verbatim ($PSCRATCH audited-copy +
# PYTHONPATH vendor pin — the 56004205 path-identity lesson).
set -euo pipefail
CODE=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code
VENV=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv
CFSDIR=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results/l0arb
RUN="$PSCRATCH/cgl2-linus/l0arbc"
rm -rf "$RUN"; mkdir -p "$RUN" "$CFSDIR"
# audited-tree copy: campaign + the two sibling data trees paths.py resolves
cp -r "$CODE/campaign" "$CODE/foundry-i" "$CODE/claude-giga-lens" "$RUN/"
cd "$RUN/campaign"
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED (pscratch copy)"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files) on $RUN/campaign"
echo "[l0arbc] host=$(hostname) script md5=$(md5sum 11_arbitration_classic.py | cut -d' ' -f1) sanity md5=$(md5sum data/l0_sanity_report.json | cut -d' ' -f1)"
module load python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # slurm sets this on shared QOS; guard against the DEFAULT_GPU=9 phoenix fallback
source "$VENV/bin/activate"
# Vendor-guard path identity (the 56004205 lesson): PYTHONPATH precedes the
# venv's gigalens.pth (which targets the CFS tree), so the AUDITED COPY's
# vendor wins; the guard itself is UNCHANGED and still enforces ref + import
# identity against this copy's pinned vendor.
export PYTHONPATH="$RUN/campaign/vendor/gigalens-linus/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import gigalens; print('[l0arbc] gigalens ->', gigalens.__file__)"
export GIGALENS_X64=1 JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
RC=0
python 11_arbitration_classic.py > "$RUN/l0arbc_run.log" 2>&1 || RC=$?
echo "[l0arbc] python exit=$RC"
# results back to CFS ALWAYS (stage-wise partials are results too)
cp "$RUN/l0arbc_run.log" "$CFSDIR/" || true
cp data/l0_arbitration_classic.json "$CFSDIR/" 2>/dev/null \
    || echo "[l0arbc] no json (died before first stage write?)"
cp data/l0_arbitration_classic.npz "$CFSDIR/" 2>/dev/null \
    || echo "[l0arbc] no npz (died before first stage write?)"
echo "=== L0-ARB-CLASSIC SUMMARY ==="
grep -E "l0-classic|status|accepted|rhat|ess|Error|Exhausted|Traceback" \
    "$RUN/l0arbc_run.log" | tail -15 | grep -vE "weakref" || true
exit $RC
