#!/usr/bin/env python
"""24_run_p2_oldstack.py -- P2 cells B4 (T2 foundry_marg46) and B5 (T3
foundry_v3b74): MC-SMC (cgl2.samplers, vendor-free -- deployment plan Path B,
PROVEN on Perlmutter 2026-07-16) against the OLD validated stack's zoo targets.

Runs in the OLD venv (~/claude-giga-lens/venv) with
PYTHONPATH=<cgl2-linus campaign root>, cwd = the OLD campaign repo
(~/claude-giga-lens/repo/reproductions/claude-giga-lens), so `cgl` resolves to
the validated 58ec9a7 stack and `cgl2.samplers` imports vendor-free
(gigalens-linus never enters the process; E1 record).

Cells / arms:
  b4       : prior-seeded S1 (MAMS-SMC) on foundry_marg46, N=256, float64.
             Gates (harvest): worst-param z<3 vs the P2c reference; ESS>=0.3N.
  b5       : per-basin S1 on foundry_v3b74 (float32 -- the target is strict-f32
             by construction, assert_dtype_env; GIGALENS_X64=0). Basin-local
             q_k -> posterior anneal, EXACTLY the 24_basin_evidence_v3b.py
             evidence construction (logprior := log q_k, loglike := log_prob -
             log q_k, q_k fit from the stored hmc_v13_v3b chains split at
             gamma=1.8 by chain mean, n_fit 8000, cov_inflate 2.0), with the
             mutation kernel swapped to OUR MAMS (S1) or MCLMC (S2, the
             B5-G3 bias-laundering arm) through the generalized
             common.run_tempered_smc driver (frozen P0 protocol).

No gate math here (harvest draws figs first). Artifacts: particles, gamma
(to_physical), logZ + boot sigma, traces, grad ledger, q_k provenance.

Run:
  b4:  GIGALENS_X64=1 python 24_run_p2_oldstack.py --cell b4 --n 256 --seed 2 \
           --kernel mams --chunk 64 --out $OUTDIR/b4_marg46_s1_seed2
  b5:  GIGALENS_X64=0 python 24_run_p2_oldstack.py --cell b5 --basin low \
           --n 128 --seed 2 --kernel mams --chunk 64 --out $OUTDIR/b5_low_mams
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

GAMMA_THRESHOLD = 1.8      # T3 basin split (foundry_v3b convention)
N_FIT = 8000               # 24_basin_evidence_v3b defaults (P2c convention)
COV_INFLATE = 2.0
RNG_OFFSET = 24_20260709   # 24_basin_evidence_v3b rng seed offset, mirrored


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=["b4", "b5"])
    ap.add_argument("--basin", choices=["low", "steep"], default="low")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--kernel", choices=["mams", "mclmc"], default="mams")
    ap.add_argument("--chunk", type=int, default=64,
                    help="MAMS mutation chunk (execution order only); "
                         "mclmc runs the stock kernel")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def main():
    args = parse_args()
    target_name = "foundry_marg46" if args.cell == "b4" else "foundry_v3b74"

    # dtype env BEFORE any jax import (single-target-per-process policy)
    from cgl.zoo import get_target, get_target_info
    from cgl.zoo.runtime import setup_jax_cache, setup_process_env

    info = get_target_info(target_name)
    _flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_disable_hlo_passes" not in _flags:   # A100 priority-fusion policy
        os.environ["XLA_FLAGS"] = (
            _flags + " --xla_disable_hlo_passes=priority-fusion").strip()
    setup_process_env(info.dtype, args.gpu)

    import jax
    import jax.numpy as jnp

    from cgl2.samplers import common, smc_micro   # vendor-free (Path B, E1)

    # chunked MAMS subclass (identical math/keys; 22_run_b3.py, imported by path)
    import importlib.util
    campaign_root = Path(os.environ["CGL2_CAMPAIGN_ROOT"]) \
        if "CGL2_CAMPAIGN_ROOT" in os.environ else \
        Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "run_b3", campaign_root / "22_run_b3.py")
    _run_b3 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_run_b3)

    setup_jax_cache(Path.cwd())
    print(f"[24] devices={jax.devices()} x64={jax.config.jax_enable_x64} "
          f"dtype={info.dtype}", flush=True)

    t0 = time.time()
    target = get_target(target_name)
    dim = int(target.dim)
    print(f"[24] built {target.name} dim={dim} in {time.time()-t0:.0f}s",
          flush=True)
    fdt = jnp.float64 if info.dtype == "float64" else jnp.float32

    qmeta = {}
    if args.cell == "b4":
        # prior-seeded S1 (the lambda=0 law IS the z-space prior)
        z0 = np.asarray(target.init.prior_sample_fn(
            jax.random.PRNGKey(int(args.seed)), int(args.n)), dtype=np.float64)

        def logprior_fn(z):
            return target.log_prior_batch(jnp.reshape(
                jnp.asarray(z, dtype=fdt), (1, -1)))[0]

        def loglik_fn(z):
            zz = jnp.reshape(jnp.asarray(z, dtype=fdt), (1, -1))
            return (target.log_prob_batch(zz)[0]
                    - target.log_prior_batch(zz)[0])
    else:
        # b5: basin-local q_k -> posterior anneal (24_basin construction,
        # mirrored parameter-for-parameter; kernel is the only change)
        import tensorflow_probability.substrates.jax as tfp

        from cgl import guards as old_guards
        from cgl.paths import HMC_V13_V3B

        tfd = tfp.distributions
        g = np.load(HMC_V13_V3B, allow_pickle=True)
        samples_ref = np.asarray(g["samples"], dtype=np.float64)
        chain_gamma = np.asarray(g["mass_gamma"], dtype=np.float64)
        chain_mean_g = chain_gamma.mean(axis=0)
        chains = (np.where(chain_mean_g < GAMMA_THRESHOLD)[0]
                  if args.basin == "low"
                  else np.where(chain_mean_g >= GAMMA_THRESHOLD)[0])
        assert len(chains), f"no {args.basin} chains in the stored reference"
        rng = np.random.default_rng(int(args.seed) + RNG_OFFSET)
        pool = samples_ref[:, chains, :].reshape(-1, dim)
        if pool.shape[0] > N_FIT:
            pool = pool[rng.choice(pool.shape[0], N_FIT, replace=False)]
        loc = pool.mean(axis=0)
        cov = np.cov(pool, rowvar=False) * COV_INFLATE
        cov_reg, chol, n_floored = old_guards.floor_svi_covariance(cov)
        qmeta = dict(basin=args.basin, n_chains=int(len(chains)),
                     n_fit=int(pool.shape[0]), cov_inflate=COV_INFLATE,
                     n_floored=int(n_floored),
                     q_source="hmc_v13_v3b.npz (24_basin_evidence_v3b fit "
                              "convention, mirrored)")
        q = tfd.MultivariateNormalTriL(
            loc=jnp.asarray(loc, dtype=fdt),
            scale_tril=jnp.asarray(chol, dtype=fdt))
        k_seed = jax.random.PRNGKey(int(args.seed))
        k_part, _ = jax.random.split(k_seed)
        eps = jax.random.normal(k_part, (int(args.n), dim), dtype=fdt)
        z0 = np.asarray(
            jnp.asarray(loc, dtype=fdt)[None, :]
            + eps @ jnp.asarray(chol, dtype=fdt).T, dtype=np.float64)

        def logprior_fn(z):
            return q.log_prob(jnp.asarray(z, dtype=fdt))

        def loglik_fn(z):
            zz = jnp.asarray(z, dtype=fdt)
            return (target.log_prob_batch(jnp.reshape(zz, (1, -1)))[0]
                    - q.log_prob(zz))

    # warmup: single-row batch compile probe (fail-loud on wiring errors)
    lp0 = float(np.asarray(target.log_prob_batch(
        jnp.asarray(z0[:1], dtype=fdt))).reshape(-1)[0])
    assert np.isfinite(lp0), f"non-finite log_prob at first particle: {lp0}"
    print(f"[24] warmup log_prob(z0[0]) = {lp0:.3f}", flush=True)

    if args.kernel == "mams":
        kern = _run_b3.make_chunked_mams(int(args.chunk))  # num_mcmc_steps=4 frozen
    else:
        kern = smc_micro.make_kernel(
            "mclmc", num_mcmc_steps=smc_micro.NUM_MCMC_STEPS)

    t0 = time.time()
    res = common.run_tempered_smc(
        logprior_fn, loglik_fn, z0, jax.random.PRNGKey(int(args.seed)),
        kernel=kern, target_ess=smc_micro.TARGET_ESS, max_stages=400,
        n_boot=200, pilot_size=64, eps0=0.5, precondition=True,
        boot_seed=20260715 + int(args.seed))
    wall = time.time() - t0

    parts = np.asarray(res.pop("particles"), dtype=np.float64)
    phys = target.to_physical(parts.astype(np.float64))
    gam = np.asarray(phys["gamma"], dtype=np.float64).reshape(-1)
    frac_low = float((gam < GAMMA_THRESHOLD).mean())
    st = jax.local_devices()[0].memory_stats() or {}
    out = dict(
        script="24_run_p2_oldstack.py",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cell=args.cell, target=target.name, dim=dim, dtype=info.dtype,
        kernel=kern.name, config=vars(args), q=qmeta,
        wall_s=round(wall, 1),
        peak_mb=round(float(st.get("peak_bytes_in_use", 0)) / 2**20, 1),
        gamma_eqw_median=float(np.median(gam)),
        frac_gamma_lt_split=frac_low,
        jax=jax.__version__, device=str(jax.local_devices()[0]),
        **_jsonable(res))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp.with_suffix(".npz"), particles=parts,
             gamma=gam, labels=np.array(list(target.labels), dtype=object))
    json.dump(out, open(outp.with_suffix(".json"), "w"), indent=1, default=float)
    print(f"[24] wrote {outp}.npz + .json (logZ={out['logZ']:.3f} "
          f"stages={out['n_stages']} frac_low={frac_low:.3f})", flush=True)


if __name__ == "__main__":
    main()
