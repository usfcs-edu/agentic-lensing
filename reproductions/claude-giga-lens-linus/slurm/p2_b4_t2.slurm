#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (55952482 OOM lesson)
#SBATCH -q shared
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b4-t2
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B4: T2 foundry_marg46 prior-seeded S1 (MAMS-SMC) N=256 float64, seed 2.
# Path B: OLD venv + PYTHONPATH=campaign (vendor-free cgl2.samplers; E1/E2
# proven on Perlmutter). cwd = the OLD validated repo (md5-audited this wave).
set -euo pipefail
CAMPAIGN=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/campaign
cd "$CAMPAIGN"
md5sum -c --quiet deploy.md5 || { echo "DEPLOY AUDIT FAILED"; exit 2; }
echo "deploy.md5 audit: PASS ($(wc -l < deploy.md5) files)"
module load python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # slurm sets this on shared QOS; guard against the DEFAULT_GPU=9 phoenix fallback
source "$HOME/claude-giga-lens/venv/bin/activate"
export PYTHONPATH="$CAMPAIGN"
export GIGALENS_X64=1 JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${PREALLOC:-false}"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
OUTDIR="$PSCRATCH/cgl2-linus/results"
CFSDIR="/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/results"
mkdir -p "$OUTDIR" "$CFSDIR"
echo "[b4] host=$(hostname) foundry_marg md5=$(md5sum cgl/zoo/foundry_marg.py | cut -d' ' -f1) runner md5=$(md5sum $CAMPAIGN/24_run_p2_oldstack.py | cut -d' ' -f1)"
python "$CAMPAIGN/24_run_p2_oldstack.py" --cell b4 --n 256 --seed 2 \
    --kernel mams --chunk 64 --out "$OUTDIR/b4_marg46_s1_seed2" \
    > "$OUTDIR/b4_marg46_s1_seed2_run.log" 2>&1
cp "$OUTDIR/b4_marg46_s1_seed2.npz" "$OUTDIR/b4_marg46_s1_seed2.json" \
   "$OUTDIR/b4_marg46_s1_seed2_run.log" "$CFSDIR/"
echo "=== B4 SUMMARY ==="
grep -E "built|warmup|logZ|stages|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR/b4_marg46_s1_seed2_run.log" | grep -vE "weakref" || true
