"""S4 adapter: blackjax adaptive tempered SMC (prior -> posterior anneal).

Path: pi_lambda proportional to prior * like^lambda with the adaptive lambda
schedule (next lambda chosen so the incremental-weight ESS hits target_ess),
HMC inner mutation kernel (shared parameters via extend_params), SYSTEMATIC
resampling. Uses the zoo's exact log_prior/log_like split (log_prior includes
the bijector fwd log-det-J, so the lambda=0 distribution is exactly the
z-space prior that prior_sample_fn draws from, and the accumulated
log-likelihood increments estimate Reference.logZ's integral exactly).

Outputs: final particles (harvested at lambda=1) + logZ estimate.

ESS semantics (documented, applies to nautilus too): the final particles are
EXCHANGEABLE, so autocorrelation ESS on pseudo-chains is not meaningful. The
efficiency numbers use ESS_est = min(importance-weight ESS of the final
state, number of UNIQUE particles surviving the final equal-weight
resample); rank diagnostics on shuffled pseudo-chains are stored for
reference only (ess_source records the override).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cgl import guards, metrics
from cgl.io import CellResult
from cgl.samplers import common
from cgl.zoo.api import LensPosterior, np_dtype

SCALE_KEYS = ("n_particles",)

DEFAULT_CONFIG = dict(
    target_ess=0.7,
    num_mcmc_steps=4,
    hmc_num_integration_steps=8,
    hmc_step_size=0.15,
    mass="identity",              # "identity" | "svi" (billed init cache)
    max_lambda_steps=200,
    n_pseudo_chains=8,
)

DEFAULT_BUDGET = dict(n_particles=1000)


def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    import jax
    import jax.numpy as jnp

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
    k_part, k_smc = jax.random.split(key)
    n_particles = int(budget["n_particles"])
    dim = target.dim

    # ---- mass matrix ---------------------------------------------------------------
    if cfg["mass"] == "svi":
        ginit = common.gaussian_init(target, seed, ledger, timing)
        imm = ginit.cov_reg            # inverse mass = covariance
        extras["mass_source"] = ginit.source
    else:
        imm = np.eye(dim)
        extras["mass_source"] = "identity"

    # ---- prior particles + single-point closures -------------------------------------
    particles = target.init.prior_sample_fn(k_part, n_particles)
    common.warmup_batch(target, 1)

    def logprior_pt(z):
        return target.log_prior_batch(jnp.reshape(z, (1, -1)))[0]

    def loglike_pt(z):
        return target.log_like_batch(jnp.reshape(z, (1, -1)))[0]

    # ---- anneal --------------------------------------------------------------------
    t0 = time.time()
    out = common.run_adaptive_tempered_smc(
        logprior_pt, loglike_pt, particles, k_smc,
        target_ess=cfg["target_ess"],
        num_mcmc_steps=cfg["num_mcmc_steps"],
        hmc_step_size=cfg["hmc_step_size"],
        inverse_mass_matrix=jnp.asarray(imm, dtype=fdtype),
        hmc_num_integration_steps=cfg["hmc_num_integration_steps"],
        max_lambda_steps=cfg["max_lambda_steps"])
    timing["smc_s"] = time.time() - t0
    ledger.add("smc", n_grad=out["n_grad"], n_logp=out["n_logp"],
               note=f"{out['n_steps']} lambda-steps x {n_particles} particles"
                    f" x ({cfg['num_mcmc_steps']}x"
                    f"{cfg['hmc_num_integration_steps']} leapfrog + 1 init); "
                    "logp = weight evals")

    # ---- harvest ---------------------------------------------------------------------
    rng = np.random.default_rng(int(seed) + 20260706)
    w = out["weights"] / out["weights"].sum()
    w_ess = common.weight_ess(w)
    idx = rng.choice(n_particles, size=n_particles, p=w)   # equal-weight draw
    eq = out["particles"][idx]
    n_unique = int(len(np.unique(idx)))
    ess_est = float(min(w_ess, n_unique))

    samples = common.particles_to_chains(eq, int(cfg["n_pseudo_chains"]), rng)
    extras.update(
        logZ=out["log_evidence"], n_lambda_steps=out["n_steps"],
        lambda_schedule=out["lambdas"], weight_ess=w_ess,
        n_unique_after_resample=n_unique,
        pseudo_chains=True,
    )
    if target.reference is not None and target.reference.logZ is not None:
        extras["logz_compare"] = metrics.compare_logZ(
            out["log_evidence"], target.reference.logZ)

    diagnostics = dict(
        lambda_schedule=np.asarray(out["lambdas"]),
        final_weights=out["weights"],
    )
    return common.finalize_cell(
        sampler="bj_smc", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        ess_override=dict(source="smc_weight_ess_min_unique", ess=ess_est),
        final_step_size=float(cfg["hmc_step_size"]))
