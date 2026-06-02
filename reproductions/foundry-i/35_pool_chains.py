"""Pool & diagnose multiple HMC chains (CPU-only) for the 6 physical mass params.

Reads a glob of HMC npz files written by 34_fit_marg.py --mode hmc (one per chain,
varied --seed), pools the SIX physical mass params, and prints:

  * per-param SPLIT R-hat   (tfp.mcmc.potential_scale_reduction,
                             independent_chain_ndims=1, split_chains=True)
  * per-param POOLED ESS    (tfp.mcmc.effective_sample_size, cross_chain_dims=...)
  * pooled median + 16/84-percentile credible interval, side-by-side with the
    Huang 2025a (foundry-i / GIGA-Lens) published values.

WHY THIS EXISTS.  34_fit_marg.py runs ONE chain per process (one GPU / one --seed)
and only ever reports single-chain ESS.  Cross-chain convergence (split R-hat,
cross-chain ESS) needs ALL chains together.  This is the pooling/diagnostic step:
point it at the npz files (e.g. the parallel A16 production set
data/prod_diagraw_s*.npz, or the long run data/long_diagraw_s*.npz) and it gives
the honest multi-chain verdict.

The mass params are read from the per-chain ``mass_<name>`` arrays that
34_fit_marg.py already stores (the bijector-mapped physical mass params,
m.to_physical_mass(samples) at save time) -- no model rebuild, no GPU, fast.
Falls back to recomputing from raw ``samples`` only if mass_* keys are absent
(then it must rebuild the model on CPU, which is slow).

Chains of DIFFERING length are handled gracefully: every chain is TRUNCATED to the
minimum length across the matched files (TFP's R-hat / cross-chain ESS require a
rectangular [n_draws, n_chains] array).  A keep/drop note is printed.

CPU ONLY.  Run under JAX_PLATFORMS=cpu (and CUDA_VISIBLE_DEVICES="") -- the 8 A16s
are reserved for the production sampling chains; this is pure post-processing.

Examples
--------
Smoke-test on the existing 4x500 production set (reproduces gamma R-hat ~5):
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /raid/benson/.venvs/gigalens/bin/python 35_pool_chains.py \
      --glob 'data/prod_diagraw_s*.npz'

Pool the long 8-chain run once it lands:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    /raid/benson/.venvs/gigalens/bin/python 35_pool_chains.py \
      --glob 'data/long_diagraw_s*.npz' --burn 0
"""
import argparse
import glob
import os
from pathlib import Path

# CPU only.  Set BEFORE importing jax so it never touches a GPU.  (Respect an
# already-set value; default to cpu.)  JAX_PLATFORMS=cpu alone still triggers the
# CUDA *plugin discovery* (a harmless but noisy cuInit traceback on stderr when no
# GPU is visible); CUDA_VISIBLE_DEVICES="" + masking the cuda plugin keeps it quiet.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

REPRO = Path(__file__).parent

# 6 physical mass params, in report order.
MASS_PARAMS = ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2"]

# Huang 2025a (foundry-i / GIGA-Lens) published physical mass params (value,
# +/- 1-sigma if published else None).  Used only for side-by-side printing.
PAPER = {
    "theta_E": (2.6463, 0.0017),
    "gamma":   (1.372, 0.023),
    "e1":      (0.1091, None),
    "e2":      (-0.1320, None),
    "gamma1":  (0.0657, None),
    "gamma2":  (-0.0939, None),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True,
                    help="glob of HMC npz files, one per chain "
                         "(e.g. 'data/long_diagraw_s*.npz'). Relative globs are "
                         "resolved against this script's directory.")
    ap.add_argument("--burn", type=int, default=0,
                    help="drop this many leading samples from EACH chain before "
                         "pooling (the saved chains are already post-burn-in; "
                         "default 0). Applied before the truncate-to-min step.")
    return ap.parse_args()


def resolve_glob(pattern):
    """Resolve a glob both as-given and relative to the script dir; return sorted
    list of matched paths (sorted -> stable, seed-ordered chain order)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        paths = sorted(glob.glob(str(REPRO / pattern)))
    return paths


def load_chain_mass(path):
    """Return dict{param -> 1D array} for one chain.

    Prefer the pre-saved physical ``mass_<name>`` arrays (written by
    34_fit_marg.py at save time via m.to_physical_mass).  If those are absent,
    rebuild the marginal model on CPU and map raw ``samples`` -> physical (slow;
    only as a fallback)."""
    d = np.load(path, allow_pickle=True)
    have_mass = all(f"mass_{p}" in d.files for p in MASS_PARAMS)
    if have_mass:
        return {p: np.asarray(d[f"mass_{p}"], dtype=np.float64).ravel()
                for p in MASS_PARAMS}, "mass_* keys"

    if "samples" not in d.files:
        raise KeyError(
            f"{path}: no mass_* keys and no 'samples' array -- cannot extract "
            f"physical mass params. keys={d.files}")
    # Fallback: rebuild the model on CPU and map samples -> physical.
    print(f"  [{Path(path).name}] no mass_* keys; rebuilding marginal model on CPU "
          f"to map {d['samples'].shape} samples -> physical (slow) ...", flush=True)
    x64 = bool(d["x64"]) if "x64" in d.files else True
    if x64:
        os.environ["GIGALENS_X64"] = "1"
        import jax
        jax.config.update("jax_enable_x64", True)
    import _hmc_lib_marg  # noqa: E402  (deferred; only needed in fallback)
    m = _hmc_lib_marg.build_model_marg(x64=x64)
    combined = m.to_physical_mass(np.asarray(d["samples"]))
    return ({p: np.asarray(combined[p], dtype=np.float64).ravel()
             for p in MASS_PARAMS},
            "recomputed from raw samples via to_physical_mass")


def main():
    args = parse_args()
    paths = resolve_glob(args.glob)
    if not paths:
        raise SystemExit(f"no files matched glob: {args.glob!r} "
                         f"(also tried relative to {REPRO})")
    if len(paths) < 2:
        print(f"WARNING: only {len(paths)} chain matched; split R-hat is computed "
              f"per chain (split_chains=True splits each into halves) but "
              f"cross-chain diagnostics are weak with <2 independent chains.",
              flush=True)

    print(f"Pooling {len(paths)} chain(s) matched by {args.glob!r}:", flush=True)
    chains = []          # list of dict{param -> 1D array}
    lengths = []
    for p in paths:
        mass, src = load_chain_mass(p)
        n = len(mass[MASS_PARAMS[0]])
        lengths.append(n)
        chains.append(mass)
        print(f"  {Path(p).name:40s} n={n:6d}  ({src})", flush=True)

    burn = max(0, int(args.burn))
    if burn:
        chains = [{p: v[burn:] for p, v in c.items()} for c in chains]
        lengths = [n - burn for n in lengths]
        print(f"\nDropped {burn} leading samples/chain (burn).", flush=True)

    n_min = min(lengths)
    if len(set(lengths)) > 1:
        print(f"\nChain lengths differ {sorted(set(lengths))} -> truncating ALL to "
              f"min={n_min} (TFP needs a rectangular [draws, chains] array).",
              flush=True)
    n_chains = len(chains)
    print(f"\nUsing n_draws={n_min} x n_chains={n_chains} = {n_min*n_chains} pooled "
          f"draws per param.\n", flush=True)

    # Build [n_draws, n_chains] arrays per param (truncate each chain to n_min).
    stacked = {p: np.stack([c[p][:n_min] for c in chains], axis=1)  # (n_draws, n_chains)
               for p in MASS_PARAMS}

    # split R-hat: tfp.mcmc.potential_scale_reduction, chains along axis with
    # independent_chain_ndims=1 (last dim = chains), split_chains=True.
    # cross-chain ESS: tfp.mcmc.effective_sample_size, cross_chain_dims = chain axis.
    print(f"{'param':>9s} {'R-hat':>8s} {'ESS':>9s} | "
          f"{'pooled median':>14s} {'[16%':>10s} {'84%]':>10s} | "
          f"{'paper':>10s} {'(+/-)':>9s}", flush=True)
    print("-" * 96, flush=True)
    rhat_all, ess_all = {}, {}
    for p in MASS_PARAMS:
        arr = stacked[p]  # (n_draws, n_chains)
        rhat = float(np.asarray(tfp.mcmc.potential_scale_reduction(
            arr, independent_chain_ndims=1, split_chains=True)))
        ess = float(np.asarray(tfp.mcmc.effective_sample_size(
            arr, cross_chain_dims=1)))
        rhat_all[p], ess_all[p] = rhat, ess
        pooled = arr.ravel()
        med = float(np.median(pooled))
        lo, hi = (float(np.percentile(pooled, 16)),
                  float(np.percentile(pooled, 84)))
        pv, ps = PAPER[p]
        ps_str = f"{ps:.4f}" if ps is not None else "-"
        print(f"{p:>9s} {rhat:>8.3f} {ess:>9.1f} | "
              f"{med:>14.5f} {lo:>10.5f} {hi:>10.5f} | "
              f"{pv:>10.4f} {ps_str:>9s}", flush=True)

    rhat_max = max(rhat_all.values())
    ess_min = min(ess_all.values())
    print("-" * 96, flush=True)
    print(f"  max split R-hat = {rhat_max:.3f}   (want < 1.01 for convergence)",
          flush=True)
    print(f"  min pooled ESS  = {ess_min:.1f}", flush=True)
    converged = (rhat_max < 1.01)
    print(f"\n  CONVERGED (all R-hat < 1.01): {converged}", flush=True)
    if not converged:
        worst = max(rhat_all, key=rhat_all.get)
        print(f"  worst param: {worst} (R-hat={rhat_all[worst]:.3f}) -- chains "
              f"have NOT mixed; more samples / better preconditioning needed.",
              flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
