"""Stage D: preconditioned HMC, the published GIGA-Lens recipe.

PHMC + ChEES trajectory adaptation + dual-averaging step size through the
library HMC (pmap across devices), momentum covariance = inv(SVI covariance).
The SVI covariance is eigenvalue-floored at float64 before inversion so the
momentum distribution is well defined (the upstream Bug-2 guard).

Chains start from SVI draws (the paper's protocol). Gate (via
44_diagnostics.py): split R-hat < 1.1 on all parameters, ESS >= 1e3.

Run:
  python 43_hmc_paper_scale.py --qz data/svi_v12_prod.npz --n-hmc 48 \
      --burn 250 --results 750 --tag prod
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import _data_lib as D

REPRO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qz", type=str, default="data/svi_v12_prod.npz")
    ap.add_argument("--n-hmc", type=int, default=48)
    ap.add_argument("--burn", type=int, default=250)
    ap.add_argument("--results", type=int, default=750)
    ap.add_argument("--init-eps", type=float, default=0.3)
    ap.add_argument("--init-l", type=int, default=3)
    ap.add_argument("--max-leapfrog", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=D.N_MAX)
    ap.add_argument("--data", type=str, default="cutout_v2.npz")
    ap.add_argument("--tag", type=str, default="prod")
    args = ap.parse_args()

    D.bootstrap_vendor()
    import jax
    import jax.experimental.shard_map  # noqa: F401
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.inference import ModellingSequence

    tfd = tfp.distributions
    print(f"devices: {jax.devices()}")

    d, prior, phys, prob, sim_config = D.build_all(n_max=args.n_max, data_file=args.data)
    seq = ModellingSequence(phys, prob, sim_config)

    z = np.load(REPRO / args.qz)
    loc = z["loc"].astype(np.float64)
    cov = z["cov"].astype(np.float64)

    # eigenvalue floor (float64) so inv(cov) in the momentum distribution is sane
    eigs, vecs = np.linalg.eigh(cov)
    floor = max(1e-10 * eigs[-1], 0.0)
    n_floored = int((eigs < floor).sum())
    eigs_reg = np.maximum(eigs, floor)
    cov_reg = (vecs * eigs_reg) @ vecs.T
    cov_reg = 0.5 * (cov_reg + cov_reg.T)
    scale_tril = np.linalg.cholesky(cov_reg)
    print(f"qz: dim {loc.size}, min eig {eigs[0]:.3e} -> floored {n_floored} "
          f"at {floor:.3e}, cond {eigs_reg[-1]/eigs_reg[0]:.2e}")

    qz = tfd.MultivariateNormalTriL(
        loc=jnp.asarray(loc, dtype=jnp.float32),
        scale_tril=jnp.asarray(scale_tril, dtype=jnp.float32))

    t0 = time.time()
    samples = seq.HMC(
        qz, init_eps=args.init_eps, init_l=args.init_l, n_hmc=args.n_hmc,
        num_burnin_steps=args.burn, num_results=args.results,
        max_leapfrog_steps=args.max_leapfrog, seed=args.seed)
    elapsed = time.time() - t0

    samples = np.asarray(samples)
    # library output: (num_steps, num_devices, n_hmc_per_device, dim)
    draws = samples.shape[0]
    samples = samples.reshape(draws, -1, samples.shape[-1])  # (draws, chains, dim)
    n_chains = samples.shape[1]
    print(f"HMC [{args.tag}]: {elapsed:.0f}s, samples {samples.shape} "
          f"(draws, chains, dim)")

    # physical mass parameters per draw for diagnostics + posterior table
    flat = samples.reshape(-1, samples.shape[-1])
    mp = D.mass_params_from_z(prob, flat)
    mass = {k: v.reshape(draws, n_chains) for k, v in mp.items()}

    out = dict(
        tag=args.tag, qz=args.qz, n_hmc=args.n_hmc, burn=args.burn,
        results=args.results, init_eps=args.init_eps, init_l=args.init_l,
        max_leapfrog=args.max_leapfrog, seed=args.seed,
        elapsed_s=elapsed, draws=draws, chains=n_chains,
        n_eig_floored=n_floored,
    )
    np.savez(REPRO / "data" / f"hmc_v13_{args.tag}.npz",
             samples=samples.astype(np.float32),
             **{f"mass_{k}": v.astype(np.float32) for k, v in mass.items()},
             meta=json.dumps(out))
    print(json.dumps(out, indent=2))
    print(f"wrote data/hmc_v13_{args.tag}.npz")


if __name__ == "__main__":
    main()
