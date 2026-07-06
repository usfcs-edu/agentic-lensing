"""S9 adapter: GLNT ("GIGA-Lens Neural Transport") -- the CGL recipe
candidate. Pipeline, every stage cost-ledgered separately:

  1. MAP -> SVI: the shared init-cache contract (billed like every consumer;
     analytic bundle on T0).
  2. SHORT adaptive tempered-SMC anneal (bj_smc machinery, ~1-2k particles),
     HMC inner kernel preconditioned with the SVI covariance. Two anneal
     paths behind a config switch (frozen on the dev split):
       anneal_init="prior": particles ~ prior, path prior -> posterior
         (real mode discovery; valid logZ vs Reference.logZ);
       anneal_init="svi":   particles ~ floored SVI q, path q -> posterior
         (logprior_fn := log q, loglike_fn := log_prob - log q; ALSO a valid
         logZ estimator for the same integral, but mode discovery is limited
         to q's basin -- the honest tradeoff the dev split arbitrates).
  3. NSF fit on the (equal-weight resampled) anneal output -- zero target
     evals, wallclock timed.
  4. S0-style ChEES-HMC in the flow-pullback space; draws pushed to z.

The headline bet, run under the SAME frozen-policy discipline as everyone
else (budgets, freeze assertion, ledger conventions).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cgl import flows, guards, metrics
from cgl.io import CellResult
from cgl.samplers import common
from cgl.zoo.api import LensPosterior, np_dtype

SCALE_KEYS = ("n_keep",)

DEFAULT_CONFIG = dict(
    # stage 2: anneal
    anneal_init="prior",            # "prior" | "svi"
    n_particles=1000,
    target_ess=0.6,
    num_mcmc_steps=3,
    hmc_num_integration_steps=5,
    hmc_step_size=0.2,
    mass="svi",                     # inner-kernel inverse mass: "svi"|"identity"
    max_lambda_steps=100,
    # stage 3: flow
    flow_layers=6,
    knots=8,
    interval=5.0,
    nn_width=64,
    nn_depth=1,
    max_epochs=100,
    max_patience=8,
    flow_batch=256,
    flow_lr=3e-4,
    # stage 4: ChEES-HMC (S0 conventions)
    eps0=0.3,
    init_l=3,
    max_leapfrog=30,
    use_gbtla=True,
    target_accept=None,
    adapt_frac=0.8,
    start_scale=1.0,
)

DEFAULT_BUDGET = dict(n_chains=32, n_burn=200, n_keep=700)


def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    tfd = tfp.distributions
    guards.require_single_device()
    budget = {**DEFAULT_BUDGET, **(budget or {})}
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    fdtype = np_dtype(target.dtype)
    if target.dtype == "float64":
        guards.require_x64()

    freeze_check = common.assert_freeze_fidelity(target, freeze_points, fdtype)
    ledger = metrics.BudgetLedger()
    timing = {}
    extras = {}
    key = jax.random.PRNGKey(int(seed))
    k_part, k_smc, k_start, k_hmc = jax.random.split(key, 4)
    n_chains = int(budget["n_chains"])
    n_burn, n_keep = int(budget["n_burn"]), int(budget["n_keep"])
    n_particles = int(budget.get("n_particles") or cfg["n_particles"])
    dim = target.dim

    # ---- stage 1: MAP+SVI (shared init contract) ----------------------------------
    ginit = common.gaussian_init(target, seed, ledger, timing)
    extras["init_source"] = ginit.source
    extras["svi_cov_n_floored"] = ginit.n_floored

    # ---- stage 2: short tempered-SMC anneal ------------------------------------------
    imm = ginit.cov_reg if cfg["mass"] == "svi" else np.eye(dim)
    common.warmup_batch(target, 1)

    if cfg["anneal_init"] == "prior":
        particles = target.init.prior_sample_fn(k_part, n_particles)

        def logprior_pt(z):
            return target.log_prior_batch(jnp.reshape(z, (1, -1)))[0]

        def loglike_pt(z):
            return target.log_like_batch(jnp.reshape(z, (1, -1)))[0]
    elif cfg["anneal_init"] == "svi":
        q = tfd.MultivariateNormalTriL(
            loc=jnp.asarray(ginit.loc, dtype=fdtype),
            scale_tril=jnp.asarray(ginit.chol, dtype=fdtype))
        particles = common.draw_gaussian_starts(ginit, k_part, n_particles,
                                                fdtype)

        def logprior_pt(z):
            return q.log_prob(jnp.asarray(z, dtype=fdtype))

        def loglike_pt(z):
            return (target.log_prob_batch(jnp.reshape(z, (1, -1)))[0]
                    - q.log_prob(jnp.asarray(z, dtype=fdtype)))
    else:
        raise ValueError(f"unknown anneal_init {cfg['anneal_init']!r}")

    t0 = time.time()
    out = common.run_adaptive_tempered_smc(
        logprior_pt, loglike_pt, particles, k_smc,
        target_ess=cfg["target_ess"], num_mcmc_steps=cfg["num_mcmc_steps"],
        hmc_step_size=cfg["hmc_step_size"],
        inverse_mass_matrix=jnp.asarray(imm, dtype=fdtype),
        hmc_num_integration_steps=cfg["hmc_num_integration_steps"],
        max_lambda_steps=cfg["max_lambda_steps"])
    timing["smc_s"] = time.time() - t0
    ledger.add("smc_anneal", n_grad=out["n_grad"], n_logp=out["n_logp"],
               note=f"{out['n_steps']} lambda-steps x {n_particles} "
                    f"particles ({cfg['anneal_init']} path)")
    extras.update(logZ_smc=out["log_evidence"],
                  n_lambda_steps=out["n_steps"],
                  lambda_schedule=out["lambdas"],
                  anneal_weight_ess=common.weight_ess(out["weights"]))
    if target.reference is not None and target.reference.logZ is not None:
        extras["logz_compare"] = metrics.compare_logZ(
            out["log_evidence"], target.reference.logZ)

    # ---- stage 3: NSF on weighted anneal output ------------------------------------------
    rng = np.random.default_rng(int(seed) + 9_20260706)
    w = out["weights"] / out["weights"].sum()
    idx = rng.choice(n_particles, size=n_particles, p=w)
    anneal_eq = np.asarray(out["particles"], dtype=np.float64)[idx]

    t0 = time.time()
    bundle = flows.fit_nsf(
        int(seed) + 99, anneal_eq,
        flow_layers=int(cfg["flow_layers"]), knots=int(cfg["knots"]),
        interval=float(cfg["interval"]), nn_width=int(cfg["nn_width"]),
        nn_depth=int(cfg["nn_depth"]), max_epochs=int(cfg["max_epochs"]),
        max_patience=int(cfg["max_patience"]),
        batch_size=int(cfg["flow_batch"]), learning_rate=float(cfg["flow_lr"]))
    timing["flow_s"] = time.time() - t0
    ledger.add("flow_fit",
               note=f"NSF on {n_particles} anneal particles, "
                    f"{len(bundle.losses['train'])} epochs, 0 target evals")
    extras["flow_final_val_loss"] = float(bundle.losses["val"][-1]) \
        if bundle.losses.get("val") else None

    # ---- stage 4: ChEES-HMC in flow space ---------------------------------------------
    pullback = bundle.make_pullback(target.log_prob_batch)
    momentum = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(dim, dtype=fdtype),
        scale_diag=jnp.ones(dim, dtype=fdtype))
    start_u = float(cfg["start_scale"]) * jax.random.normal(
        k_start, (n_chains, dim), dtype=fdtype)

    t0 = time.time()
    samples_u_all, diagnostics, g_hmc, exact = common.run_chees_hmc(
        pullback, start_u, momentum, k_hmc, fdtype, n_burn, n_keep,
        eps0=cfg["eps0"], init_l=cfg["init_l"],
        max_leapfrog=cfg["max_leapfrog"], use_gbtla=cfg["use_gbtla"],
        target_accept=cfg["target_accept"], adapt_frac=cfg["adapt_frac"])
    timing["hmc_s"] = time.time() - t0
    ledger.add("hmc_pullback", n_grad=g_hmc,
               n_logp=(n_burn + n_keep) * n_chains,
               note="ChEES-HMC in flow space"
                    + (", leapfrog traced" if exact else ", ANALYTIC"))

    t0 = time.time()
    samples = bundle.push_samples(samples_u_all[n_burn:])
    timing["push_s"] = time.time() - t0

    diagnostics["lambda_schedule"] = np.asarray(out["lambdas"])
    accepted = diagnostics["is_accepted"][n_burn:]
    return common.finalize_cell(
        sampler="glnt", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        acceptance_rate=float(np.mean(accepted)),
        final_step_size=float(np.ravel(diagnostics["step_size"])[-1]))
