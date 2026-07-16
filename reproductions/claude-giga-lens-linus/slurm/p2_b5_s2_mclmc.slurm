#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu&hbm80g  # SMC production pin (55952482 OOM lesson)
#SBATCH -q shared
#SBATCH -t 02:30:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl2-b5-s2
#SBATCH -o /global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/slurm-logs/slurm-%j.out
# P2 B5-G3: T3 foundry_v3b74 LOW basin, S2 MCLMC-mutation arm (the bias-
# laundering test: does SMC resampling launder unadjusted-MCLMC bias?
# pre-registered prediction 1.5-3x minor-mode distortion). N=128, seed 2,
# float32; MCLMC is DEMOTED to cost-frontier/diagnostic per the B0 bias screen
# (30 sigma on illcond46) -- this arm is diagnostic BY DESIGN, never evidence.
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
echo "[b5-s2] host=$(hostname) foundry_v3b md5=$(md5sum cgl/zoo/foundry_v3b.py | cut -d' ' -f1) runner md5=$(md5sum $CAMPAIGN/24_run_p2_oldstack.py | cut -d' ' -f1)"
python "$CAMPAIGN/24_run_p2_oldstack.py" --cell b5 --basin low --n 128 \
    --seed 2 --kernel mclmc --out "$OUTDIR/b5_v3b74_low_mclmc_seed2" \
    > "$OUTDIR/b5_v3b74_low_mclmc_seed2_run.log" 2>&1
cp "$OUTDIR/b5_v3b74_low_mclmc_seed2.npz" "$OUTDIR/b5_v3b74_low_mclmc_seed2.json" \
   "$OUTDIR/b5_v3b74_low_mclmc_seed2_run.log" "$CFSDIR/"
echo "=== B5 S2 MCLMC SUMMARY ==="
grep -E "built|warmup|logZ|stages|Error|Exhausted|Traceback|did not reach" \
    "$OUTDIR/b5_v3b74_low_mclmc_seed2_run.log" | grep -vE "weakref" || true
