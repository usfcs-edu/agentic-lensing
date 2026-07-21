#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (55952482 OOM lesson)
#SBATCH -q shared
#SBATCH -t 01:15:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b5-s1st
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B5 S1 STEEP LEG ONLY (recovery-of-the-recovery): job 56006049's low leg
# COMPLETED AND WROTE (b5_v3b74_low_mams_seed2.{npz,json} on CFS — low is NOT
# rerun here); the steep leg died at the FIRST mutate on
#   AssertionError: (96, 64) at 22_run_b3.py:135 (chunked-MAMS requires
#   n % chunk == 0; N=96 with --chunk 64 never divides).
# The bug was LATENT in the original 55985449 design too — that job's set -e
# killed it on the low leg's writer defect before steep ever executed, so this
# assert had never been reached.
# ATTEMPT HISTORY: #2 (56168443) died on the ORIGINAL (96, 64) signature —
# sbatch snapshots the script at submission and the chunk-48 fix had not
# landed in the file yet (ops lesson: verify the SUBMITTED SNAPSHOT via
# `scontrol show job <id> --Batch`, never later file state). #3 (56170614)
# died at 1:52 on AssertionError (64, 48): chunk 48 makes the FULL pass
# divisible (96 % 48 = 0) but breaks the PILOT subset — the dual-averaging
# step-size pilot (common.run_tempered_smc pilot_size=64, the runner's frozen
# 24_run_p2_oldstack.py:215 value) mutates a 64-particle subset through the
# SAME chunked kernel, and 64 % 48 != 0 trips the same 22_run_b3.py:135
# assert. CONSTRAINT: chunk must divide BOTH N (96, full pass) AND
# pilot_size (64, tuning pilot). FIX = --chunk 32 (divides both: three
# chunks of 32 at full-N, two at the pilot; 32 <= the low leg's 64-particle
# chunk memory — even safer). Chunking is EXECUTION ORDER ONLY by the
# runner's own declaration (24_run_p2_oldstack.py --chunk help; 22_run_b3.py
# splits per-particle keys BEFORE chunking — "== parent keys" — and
# bit-identity vs the stock kernel was verified in the deployment plan
# E-row), so the science design is UNCHANGED.
# Science otherwise UNCHANGED: T3 foundry_v3b74 per-basin S1 MAMS steep@96,
# seed 2, float32, Path B, checkpoint retrofit enabled (wall cap 0.95 h).
# The stale ckpt dir from the crashed run holds NO stage checkpoints (only
# run_meta + ll_000 written before the first mutate) — wiped for a clean
# fresh run (the no-mixing fence only inspects stage_*.npz; wiping removes
# any doubt).
set -euo pipefail
CAMPAIGN=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/campaign
cd "$CAMPAIGN"
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files)"
module load python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # slurm sets this on shared QOS; guard against the DEFAULT_GPU=9 phoenix fallback
source "$HOME/claude-giga-lens/venv/bin/activate"
export PYTHONPATH="$CAMPAIGN"
export GIGALENS_X64=0 JAX_ENABLE_X64=0 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
mkdir -p "$OUTDIR" "$CFSDIR"
echo "[b5-s1st] host=$(hostname) foundry_v3b md5=$(md5sum cgl/zoo/foundry_v3b.py | cut -d' ' -f1) _data_lib md5=$(md5sum ../foundry-i/_data_lib.py | cut -d' ' -f1) runner md5=$(md5sum $CAMPAIGN/24_run_p2_oldstack.py | cut -d' ' -f1) ckpt md5=$(md5sum $CAMPAIGN/cgl2/samplers/ckpt.py | cut -d' ' -f1)"
TAG="b5_v3b74_steep_mams_seed2"
rm -rf "$OUTDIR/${TAG}_ckpt"   # crashed-run residue: run_meta + ll_000 only, NO stage ckpts
RC=0
python "$CAMPAIGN/24_run_p2_oldstack.py" --cell b5 --basin steep \
    --n 96 --seed 2 --kernel mams --chunk 32 \
    --ckpt-dir "$OUTDIR/${TAG}_ckpt" --wall-cap-h 0.95 \
    --out "$OUTDIR/$TAG" > "$OUTDIR/${TAG}_run.log" 2>&1 || RC=$?
# results back ALWAYS (PARTIAL json + checkpoints are results too)
cp "$OUTDIR/${TAG}_run.log" "$CFSDIR/" || true
cp "$OUTDIR/$TAG.npz" "$OUTDIR/$TAG.json" "$CFSDIR/" 2>/dev/null \
    || echo "[b5-s1st] no npz/json (RC=$RC — PARTIAL under wall cap?)"
cp "$OUTDIR/$TAG.PARTIAL.json" "$CFSDIR/" 2>/dev/null || true
mkdir -p "$CFSDIR/${TAG}_ckpt"
cp "$OUTDIR/${TAG}_ckpt"/* "$CFSDIR/${TAG}_ckpt/" 2>/dev/null || true
echo "[b5-s1st] exit=$RC (0=COMPLETE, 3=PARTIAL wall-cap protocol)"
echo "=== B5 S1 STEEP SUMMARY ==="
grep -E "built|warmup|logZ|stages|PARTIAL|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR/${TAG}_run.log" | grep -vE "weakref" || true
exit $RC
