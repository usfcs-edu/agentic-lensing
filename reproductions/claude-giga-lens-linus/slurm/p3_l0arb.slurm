#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu       # plain gpu ON PURPOSE: v2d diag MAMS grad tape ~93 MB/particle
#SBATCH -q shared    # (l0 L4 measurement) => N=128 ~ 12 GB — do not burn hbm80g
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-l0arb
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P3-L0 anchor arbitration, FRESH Perlmutter run per the pre-registered fallback
# (research/checkpoints_l0.md; the phoenix L4 leg hit its 12 h wall cap at
# lambda=0.418 per protocol — PARTIAL, checkpoints kept as cross-check).
# Protocol = 10_anchor_arbitration.py `run` stage UNCHANGED (frozen per-lambda
# tuning, MC-SMC MAMS prior-seeded, seed 2, v2d diagonal scene-API target);
# only N and device change via its documented fallback-ladder env knob:
# L0_N_OVERRIDE=128 (512->256->128 ladder; campaign-standard SMC count), A100.
# Wall-cap protocol: L0_WALL_CAP_H=3.5 inside -t 04:00 => on cap the script
# checkpoints, writes PARTIAL json, exits 3 (fail-loud; artifacts still copied).
# Hot I/O on $PSCRATCH via a full audited tree copy (script writes into its own
# data/ by design); results copied back to CFS results/l0arb/.
set -euo pipefail
CODE=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code
VENV=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv
CFSDIR=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results/l0arb
RUN="$PSCRATCH/cgl2-linus/l0arb"
rm -rf "$RUN"; mkdir -p "$RUN" "$CFSDIR"
# audited-tree copy: campaign + the two sibling data trees paths.py resolves
cp -r "$CODE/campaign" "$CODE/foundry-i" "$CODE/claude-giga-lens" "$RUN/"
cd "$RUN/campaign"
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED (pscratch copy)"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files) on $RUN/campaign"
echo "[l0arb] host=$(hostname) script md5=$(md5sum 10_anchor_arbitration.py | cut -d' ' -f1) sanity md5=$(md5sum data/l0_sanity_report.json | cut -d' ' -f1)"
module load python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # slurm sets this on shared QOS; guard against the DEFAULT_GPU=9 phoenix fallback
source "$VENV/bin/activate"
export GIGALENS_X64=1 JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
export L0_N_OVERRIDE=128
export L0_WALL_CAP_H=3.5
RC=0
python 10_anchor_arbitration.py run > "$RUN/l0arb_run.log" 2>&1 || RC=$?
echo "[l0arb] python exit=$RC (0=COMPLETE, 3=PARTIAL wall-cap protocol)"
# results back to CFS (PARTIAL json + checkpoints are results too — copy always)
cp "$RUN/l0arb_run.log" "$CFSDIR/" || true
cp data/l0_arbitration_smc.json "$CFSDIR/" || true
cp data/l0_arbitration_smc.npz "$CFSDIR/" 2>/dev/null || echo "[l0arb] no npz (PARTIAL run — expected under wall cap)"
mkdir -p "$CFSDIR/l0_smc_checkpoints"
cp data/l0_smc_checkpoints/stage_*.npz "$CFSDIR/l0_smc_checkpoints/" 2>/dev/null || true
echo "=== L0-ARB SUMMARY ==="
grep -E "l0-smc|status|lambda|stages|Error|Exhausted|Traceback" \
    "$RUN/l0arb_run.log" | tail -12 | grep -vE "weakref" || true
exit $RC
