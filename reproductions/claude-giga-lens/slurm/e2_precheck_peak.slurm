#!/bin/bash
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:25:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -J cgl-e2-precheck-peak
#SBATCH -o slurm-%j.out
# PRE-CHECK (2026-07-10): recompute the Laplace Hessian at the v3b-low DENSITY
# PEAK (gamma_best ~= 1.10, from the svi_cov canary draws), NOT the saddle MAP.
# If PD there -> the existing two-stage PHMC seeded at the peak converges and we
# skip SMC. If indefinite (expected: 5 neg eigs = source-center degeneracy) ->
# proceed to correlated-SMC. ~2-5 min A100. cosmo_g / shared.
set -euo pipefail
module load python
source "$HOME/claude-giga-lens/venv/bin/activate"
export GIGALENS_X64=1
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion"
cd "$HOME/claude-giga-lens/repo/reproductions/claude-giga-lens"
python - <<'PY'
import numpy as np, jax.numpy as jnp
from cgl import e2
z = np.load("data/results/e2_v3b_low_canary_svicov.npz", allow_pickle=True)
draws = np.asarray(z["low_draws"], dtype=np.float64)
T, C, ndim = draws.shape
flat = draws.reshape(-1, ndim)
tgt = e2.build_target("v3b")
n_probe = 96
step = max(1, flat.shape[0] // n_probe)
sub = flat[::step][:n_probe]
lp = np.asarray(tgt.batched_lp(jnp.asarray(sub)))
zb = sub[int(np.argmax(lp))]
g0 = float(np.asarray(tgt.model.to_physical_mass(zb[None, :])["gamma"])[0])
print(f"z_best (peak draw): gamma={g0:.4f} logp={float(lp.max()):.1f}", flush=True)
lapb = e2.laplace_evidence(tgt, zb)
print(f"HESSIAN @ z_best:       min_eig={lapb['min_eig']:.3e} n_neg={lapb['n_neg']} "
      f"is_pd={lapb['is_pd']}", flush=True)
mp = e2.map_polish(tgt, zb, rounds=6, iters=300)
zp = mp["z_map"]
gp = float(np.asarray(tgt.model.to_physical_mass(zp[None, :])["gamma"])[0])
print(f"polished peak: gamma={gp:.4f} logp {mp['logp0']:.1f}->{mp['logp_map']:.1f}", flush=True)
lapp = e2.laplace_evidence(tgt, zp)
print(f"HESSIAN @ polished peak: min_eig={lapp['min_eig']:.3e} n_neg={lapp['n_neg']} "
      f"is_pd={lapp['is_pd']}", flush=True)
print("PRECHECK VERDICT: " + ("PD -> two-stage PHMC@peak may work (skip SMC)"
      if lapp['is_pd'] else "INDEFINITE -> proceed to correlated-SMC"), flush=True)
PY
