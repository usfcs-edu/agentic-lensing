"""Stage D gate: convergence diagnostics for the paper-scale HMC run.

Split R-hat (TFP potential_scale_reduction, split_chains=True) and
cross-chain ESS (TFP effective_sample_size) over BOTH the 6 physical mass
parameters and all unconstrained dimensions -- the same machinery as
35_pool_chains.py, the same convergence definitions as the published paper
(R-hat < 1.10 lens/source, < 1.2 hard threshold; ESS O(1e3) "sufficiently
high", paper achieved 32,200-40,000).

Run (CPU is fine):
  python 44_diagnostics.py --run data/hmc_v13_prod.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _data_lib as D

REPRO = Path(__file__).resolve().parent
MASS_KEYS = ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="data/hmc_v13_prod.npz")
    ap.add_argument("--burn-extra", type=int, default=0,
                    help="extra draws to drop from the start")
    args = ap.parse_args()

    D.bootstrap_vendor()
    import tensorflow_probability.substrates.jax as tfp

    z = np.load(REPRO / args.run)
    meta = json.loads(str(z["meta"]))
    samples = z["samples"][args.burn_extra:]          # (draws, chains, dim)
    draws, chains, dim = samples.shape

    def diag(arr):  # arr: (draws, chains, k)
        rhat = np.asarray(tfp.mcmc.potential_scale_reduction(
            arr, independent_chain_ndims=1, split_chains=True))
        ess = np.asarray(tfp.mcmc.effective_sample_size(
            arr, cross_chain_dims=1))
        return rhat, ess

    mass = np.stack([z[f"mass_{k}"][args.burn_extra:] for k in MASS_KEYS], -1)
    rhat_m, ess_m = diag(mass)
    rhat_all, ess_all = diag(samples)

    table = {}
    for i, k in enumerate(MASS_KEYS):
        v = mass[..., i].ravel()
        lo, med, hi = np.percentile(v, [16, 50, 84])
        table[k] = dict(median=float(med), lo=float(lo), hi=float(hi),
                        rhat=float(rhat_m[i]), ess=float(ess_m[i]),
                        paper=D.PAPER[k])

    gate = dict(
        run=args.run, draws=int(draws), chains=int(chains), dim=int(dim),
        total_samples=int(draws * chains),
        rhat_max_mass=float(np.nanmax(rhat_m)),
        rhat_max_all=float(np.nanmax(rhat_all)),
        ess_min_mass=float(np.nanmin(ess_m)),
        ess_min_all=float(np.nanmin(ess_all)),
        gate_rhat_lt_1p1=bool(np.nanmax(rhat_all) < 1.1),
        gate_rhat_lt_1p2=bool(np.nanmax(rhat_all) < 1.2),
        gate_ess_ge_1e3=bool(np.nanmin(ess_m) >= 1000),
        elapsed_s=meta.get("elapsed_s"),
        mass_table=table,
    )
    out_path = Path(args.run).with_suffix("").name + "_diag.json"
    (REPRO / "data" / out_path).write_text(json.dumps(gate, indent=2))

    print(json.dumps(gate, indent=2))
    print(f"\n{'param':>10s} {'median':>9s} {'-1s':>8s} {'+1s':>8s} "
          f"{'paper':>9s} {'rhat':>7s} {'ess':>9s}")
    for k, r in table.items():
        print(f"{k:>10s} {r['median']:>+9.4f} {r['lo']:>+8.4f} {r['hi']:>+8.4f} "
              f"{r['paper']:>+9.4f} {r['rhat']:>7.3f} {r['ess']:>9.0f}")
    ok = gate["gate_rhat_lt_1p1"] and gate["gate_ess_ge_1e3"]
    print(f"\nGATE: R-hat_max(all)={gate['rhat_max_all']:.3f} (<1.1), "
          f"ESS_min(mass)={gate['ess_min_mass']:.0f} (>=1e3) -> "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
