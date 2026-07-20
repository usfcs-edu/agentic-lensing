#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (55952482 OOM lesson)
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b5-s1
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B5: T3 foundry_v3b74 per-basin S1 (MAMS mutation), low @128 then steep
# @96, seed 2, float32 (the target is strict-f32; GIGALENS_X64=0). Basin q_k
# per the 24_basin_evidence_v3b (P2c) construction. Path B (OLD venv +
# PYTHONPATH=campaign). Legs run as SEPARATE processes (single-target-per-
# process dtype policy; also resets compile-cache memory pressure).
#
# RERUN of 55985449 (recovery lane, post-mortem R1): science UNCHANGED; the
# RC3 writer defect is fixed and the RC1 checkpoint retrofit is enabled
# (--ckpt-dir + --wall-cap-h; on cap: stage checkpoints kept + PARTIAL json +
# exit 3, resumable with --resume). Wall 2:30 vs measured legs ~1.0+0.75 h.
# Legs are independent: the steep leg runs even if the low leg wall-caps.
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
echo "[b5-s1] host=$(hostname) foundry_v3b md5=$(md5sum cgl/zoo/foundry_v3b.py | cut -d' ' -f1) _data_lib md5=$(md5sum ../foundry-i/_data_lib.py | cut -d' ' -f1) runner md5=$(md5sum $CAMPAIGN/24_run_p2_oldstack.py | cut -d' ' -f1) ckpt md5=$(md5sum $CAMPAIGN/cgl2/samplers/ckpt.py | cut -d' ' -f1)"

run_leg () {  # run_leg <basin> <n> <wall_cap_h>
  local BASIN=$1 NPART=$2 CAP=$3 RC=0
  local TAG="b5_v3b74_${BASIN}_mams_seed2"
  python "$CAMPAIGN/24_run_p2_oldstack.py" --cell b5 --basin "$BASIN" \
      --n "$NPART" --seed 2 --kernel mams --chunk 64 \
      --ckpt-dir "$OUTDIR/${TAG}_ckpt" --wall-cap-h "$CAP" \
      --out "$OUTDIR/$TAG" > "$OUTDIR/${TAG}_run.log" 2>&1 || RC=$?
  # results back ALWAYS (PARTIAL json + checkpoints are results too)
  cp "$OUTDIR/${TAG}_run.log" "$CFSDIR/" || true
  cp "$OUTDIR/$TAG.npz" "$OUTDIR/$TAG.json" "$CFSDIR/" 2>/dev/null \
      || echo "[b5-s1] $BASIN: no npz/json (RC=$RC — PARTIAL under wall cap?)"
  cp "$OUTDIR/$TAG.PARTIAL.json" "$CFSDIR/" 2>/dev/null || true
  mkdir -p "$CFSDIR/${TAG}_ckpt"
  cp "$OUTDIR/${TAG}_ckpt"/* "$CFSDIR/${TAG}_ckpt/" 2>/dev/null || true
  echo "[b5-s1] leg $BASIN exit=$RC (0=COMPLETE, 3=PARTIAL wall-cap protocol)"
  return $RC
}

RC_LOW=0; RC_STEEP=0
run_leg low 128 1.30 || RC_LOW=$?
run_leg steep 96 0.85 || RC_STEEP=$?
echo "=== B5 S1 SUMMARY ==="
grep -E "built|warmup|logZ|stages|PARTIAL|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR"/b5_v3b74_*_mams_seed2_run.log | grep -vE "weakref" || true
RC=$(( RC_LOW > RC_STEEP ? RC_LOW : RC_STEEP ))
exit $RC
