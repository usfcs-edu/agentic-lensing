#!/usr/bin/env python
"""24_basin_evidence_v3b.py -- per-basin evidence for the T3 (foundry_v3b74)
bimodal posterior via basin-local adaptive tempered SMC.

Second, INDEPENDENT estimator of the v3b low/steep mode weights (the first is
23_pt_reference_v3b's parallel-tempering cold occupancy). For each basin k we
build a basin-local reference Gaussian q_k (mean + inflated, guard-floored
covariance fit to that basin's stored chains), seed N particles inside it, and
run blackjax adaptive tempered SMC from q_k to the (unnormalized) posterior:

    logprior := log q_k,   loglike := log_prob - log q_k

so the lambda=0 law is q_k and the lambda=1 law is exp(log_prob) restricted to
q_k's basin. The accumulated log-evidence is therefore

    logZ_k = log integral_{basin k} exp(log_prior + log_like) dz

and the posterior mode weight is  w_low = Z_low / (Z_low + Z_steep). Because the
basins are well separated in gamma (ZERO migrations even under HMC), particles
seeded in basin k stay there, so logZ_k is the basin-restricted evidence. This
is the same valid-logZ construction as glnt's anneal_init="svi" path, localized
per basin.

Uncertainty: --n-repeats independent SMC runs per basin give mean/std of logZ_k;
the weight sigma is propagated by sampling logZ_low, logZ_steep from their
Gaussian summaries. If a 23_pt_reference output is present (auto-found or
--pt-json), the script reports w_low(SMC) vs w_low(PT) and the n_sigma
agreement -- the P2c acceptance gate (accept the reference iff <= 2 sigma).

NOTE (deviation): thermodynamic integration along the PT ladder is NOT done
here -- it needs the full per-beta replica trace (not exported by 23, memory-
heavy), so it is not "cheap". The cross-script PT-vs-SMC weight agreement IS the
two-estimator cross-check the brief requires.

Run (phoenix smoke, GPU 5):
  /raid/benson/.venvs/cgl/bin/python 24_basin_evidence_v3b.py --gpu 5 \
      --n-particles 200 --n-repeats 1 --num-mcmc-steps 2 --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parent
TARGET = "foundry_v3b74"
GAMMA_THRESHOLD = 1.8


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--n-particles", type=int, default=1000)
    ap.add_argument("--n-repeats", type=int, default=3,
                    help="independent SMC runs per basin (logZ_k mean/std)")
    ap.add_argument("--n-fit", type=int, default=8000,
                    help="stored draws subsampled per basin for the q_k fit")
    ap.add_argument("--cov-inflate", type=float, default=2.0,
                    help="multiply the basin covariance (fatter q_k tails)")
    ap.add_argument("--target-ess", type=float, default=0.7)
    ap.add_argument("--num-mcmc-steps", type=int, default=4)
    ap.add_argument("--hmc-integration-steps", type=int, default=8)
    ap.add_argument("--hmc-step-size", type=float, default=0.1)
    ap.add_argument("--max-lambda-steps", type=int, default=400)
    ap.add_argument("--pt-json", default=None,
                    help="23_pt_reference output for the 2-sigma cross-check "
                         "(default: auto-find data/results/pt_reference/"
                         "v3b_s<seed>.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    from cgl.zoo import get_target_info
    from cgl.zoo.runtime import setup_process_env

    info = get_target_info(TARGET)
    assert info.dtype == "float32", info.dtype
    import os as _os
    _flags = _os.environ.get("XLA_FLAGS", "")
    if "--xla_disable_hlo_passes" not in _flags:   # A100 priority-fusion policy
        _os.environ["XLA_FLAGS"] = (
            _flags + " --xla_disable_hlo_passes=priority-fusion").strip()
    setup_process_env(info.dtype, args.gpu)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorflow_probability.substrates.jax as tfp

    from cgl import guards, metrics
    from cgl.paths import HMC_V13_V3B
    from cgl.samplers import common
    from cgl.zoo import get_target
    from cgl.zoo.api import np_dtype
    from cgl.zoo.runtime import setup_jax_cache

    tfd = tfp.distributions
    setup_jax_cache(REPRO)
    guards.require_single_device()
    print(f"devices={jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)
    fdtype = np_dtype(info.dtype)

    t0 = time.time()
    target = get_target(TARGET)
    dim = target.dim
    print(f"built {target.name} dim={dim} in {time.time()-t0:.0f}s", flush=True)
    common.warmup_batch(target, 1)

    # ---- basin membership from the stored chains (mass_gamma; no recompute) --
    assert HMC_V13_V3B.exists(), f"missing reference chains {HMC_V13_V3B}"
    g = np.load(HMC_V13_V3B, allow_pickle=True)
    samples_ref = np.asarray(g["samples"], dtype=np.float64)     # (T, C, dim)
    chain_gamma = np.asarray(g["mass_gamma"], dtype=np.float64)  # (T, C)
    Tr, Cr = chain_gamma.shape
    chain_mean_g = chain_gamma.mean(axis=0)
    low_chains = np.where(chain_mean_g < GAMMA_THRESHOLD)[0]
    steep_chains = np.where(chain_mean_g >= GAMMA_THRESHOLD)[0]
    print(f"stored chains {samples_ref.shape}: {len(low_chains)} low / "
          f"{len(steep_chains)} steep (by chain-mean gamma)", flush=True)
    assert len(low_chains) and len(steep_chains), "need both basins populated"

    rng = np.random.default_rng(int(args.seed) + 24_20260709)

    def fit_qk(chains):
        pool = samples_ref[:, chains, :].reshape(-1, dim)
        if pool.shape[0] > int(args.n_fit):
            pool = pool[rng.choice(pool.shape[0], int(args.n_fit), replace=False)]
        loc = pool.mean(axis=0)
        cov = np.cov(pool, rowvar=False) * float(args.cov_inflate)
        cov_reg, chol, n_floored = guards.floor_svi_covariance(cov)
        return loc, cov_reg, chol, int(n_floored), int(pool.shape[0])

    # ---- per-basin SMC evidence ---------------------------------------------
    def basin_logZ(loc, cov_reg, chol, k_seed):
        q = tfd.MultivariateNormalTriL(
            loc=jnp.asarray(loc, dtype=fdtype),
            scale_tril=jnp.asarray(chol, dtype=fdtype))
        eps = jax.random.normal(k_seed, (int(args.n_particles), dim),
                                dtype=fdtype)
        particles = (jnp.asarray(loc, dtype=fdtype)[None, :]
                     + eps @ jnp.asarray(chol, dtype=fdtype).T)

        def logprior_pt(z):
            return q.log_prob(jnp.asarray(z, dtype=fdtype))

        def loglike_pt(z):
            return (target.log_prob_batch(jnp.reshape(z, (1, -1)))[0]
                    - q.log_prob(jnp.asarray(z, dtype=fdtype)))

        _, k_smc = jax.random.split(k_seed)
        out = common.run_adaptive_tempered_smc(
            logprior_pt, loglike_pt, particles, k_smc,
            target_ess=float(args.target_ess),
            num_mcmc_steps=int(args.num_mcmc_steps),
            hmc_step_size=float(args.hmc_step_size),
            inverse_mass_matrix=jnp.asarray(cov_reg, dtype=fdtype),
            hmc_num_integration_steps=int(args.hmc_integration_steps),
            max_lambda_steps=int(args.max_lambda_steps))
        # basin containment check: fraction of final particles still low-gamma
        fin = np.asarray(out["particles"], dtype=np.float64)
        fin_g = np.asarray(target.to_physical(fin)["gamma"], dtype=np.float64)
        frac_low = float((fin_g < GAMMA_THRESHOLD).mean())
        return (float(out["log_evidence"]), int(out["n_steps"]),
                float(np.mean(fin_g)), frac_low)

    results = {}
    key = jax.random.PRNGKey(int(args.seed))
    for name, chains in (("low", low_chains), ("steep", steep_chains)):
        loc, cov_reg, chol, n_floored, n_pool = fit_qk(chains)
        logZs, nsteps, gmeans, fraclows = [], [], [], []
        for r in range(int(args.n_repeats)):
            key, ksub = jax.random.split(key)
            t0 = time.time()
            lz, ns, gm, fl = basin_logZ(loc, cov_reg, chol, ksub)
            logZs.append(lz); nsteps.append(ns); gmeans.append(gm)
            fraclows.append(fl)
            print(f"  basin={name} rep={r}: logZ={lz:.3f} lambda_steps={ns} "
                  f"final_gamma_mean={gm:.3f} frac_low={fl:.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        results[name] = dict(
            logZ_mean=float(np.mean(logZs)),
            logZ_std=float(np.std(logZs)) if len(logZs) > 1 else 0.0,
            logZ_reps=[float(x) for x in logZs],
            lambda_steps=nsteps, n_fit_pool=n_pool, n_floored=n_floored,
            final_gamma_mean=[float(x) for x in gmeans],
            frac_low_final=[float(x) for x in fraclows])
        print(f"basin {name}: logZ = {np.mean(logZs):.3f} "
              f"+/- {results[name]['logZ_std']:.3f}", flush=True)

    # ---- mode weights + propagated sigma ------------------------------------
    lz_low_m, lz_low_s = results["low"]["logZ_mean"], results["low"]["logZ_std"]
    lz_st_m, lz_st_s = results["steep"]["logZ_mean"], results["steep"]["logZ_std"]

    def _wlow(d):
        return 1.0 / (1.0 + np.exp(np.clip(-d, -700, 700)))   # sigmoid(logZlow-logZsteep)
    w_low = float(_wlow(lz_low_m - lz_st_m))
    bs = rng.normal(lz_low_m, max(lz_low_s, 1e-9), 20000) \
        - rng.normal(lz_st_m, max(lz_st_s, 1e-9), 20000)
    w_low_sigma = float(np.std(_wlow(bs)))
    print(f"\nSMC mode weights: w_low={w_low:.4f} +/- {w_low_sigma:.4f} "
          f"(w_steep={1-w_low:.4f})", flush=True)

    # ---- optional cross-check vs the PT reference (the acceptance gate) ------
    cross = None
    pt_path = Path(args.pt_json) if args.pt_json else \
        REPRO / "data" / "results" / "pt_reference" / f"v3b_s{args.seed}.json"
    if pt_path.exists():
        pt = json.loads(pt_path.read_text())
        w_pt = float(pt["mode_weights"]["w_low"])
        s_pt = float(pt["mode_weights"]["sigma_w_low"])
        denom = float(np.sqrt(w_low_sigma ** 2 + s_pt ** 2)) or 1e-9
        n_sigma = abs(w_low - w_pt) / denom
        cross = dict(pt_json=str(pt_path), w_low_pt=w_pt, sigma_pt=s_pt,
                     w_low_smc=w_low, sigma_smc=w_low_sigma,
                     n_sigma=float(n_sigma), agree_2sigma=bool(n_sigma <= 2.0))
        print(f"CROSS-CHECK vs PT reference: w_low(PT)={w_pt:.4f}+/-{s_pt:.4f} "
              f"vs w_low(SMC)={w_low:.4f}+/-{w_low_sigma:.4f}  "
              f"n_sigma={n_sigma:.2f}  agree_2sigma={cross['agree_2sigma']}",
              flush=True)
    else:
        print(f"(no PT reference at {pt_path}; run 23_pt_reference_v3b.py for "
              "the 2-sigma cross-check)", flush=True)

    out_json = Path(args.out) if args.out else \
        REPRO / "data" / "results" / "basin_evidence" / f"v3b_s{args.seed}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        script="24_basin_evidence_v3b.py", target=TARGET, seed=int(args.seed),
        smoke=bool(args.smoke),
        config=dict(n_particles=int(args.n_particles),
                    n_repeats=int(args.n_repeats), n_fit=int(args.n_fit),
                    cov_inflate=float(args.cov_inflate),
                    target_ess=float(args.target_ess),
                    num_mcmc_steps=int(args.num_mcmc_steps),
                    hmc_integration_steps=int(args.hmc_integration_steps),
                    hmc_step_size=float(args.hmc_step_size)),
        basins=results,
        mode_weights=dict(w_low=w_low, w_steep=float(1.0 - w_low),
                          sigma_w_low=w_low_sigma, estimator="basin_smc_logZ",
                          gamma_threshold=GAMMA_THRESHOLD),
        n_low_chains=int(len(low_chains)), n_steep_chains=int(len(steep_chains)),
        cross_check=cross,
        provenance="basin-local adaptive tempered SMC (q_k -> posterior) per "
                   "basin; w_low = Z_low/(Z_low+Z_steep). Accept the T3 mode-"
                   "weight reference iff |w_low(SMC)-w_low(PT)| <= 2 sigma.")
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
