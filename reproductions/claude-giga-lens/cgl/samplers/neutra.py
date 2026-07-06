"""S8 adapter: neural-transport (NeuTra-style) ChEES-HMC.

Recipe: fit a flowjax coupling-NSF (typed keys) to samples drawn from the
FLOORED SVI Gaussian of the init bundle (init cache for T1, analytic bundle
for T0), then run the S0-style batched ChEES-HMC in the flow-PULLBACK space:

    logp_u(u) = logp(T(u)) + log|det J_T(u)|,   T = destandardize o flow

with identity momentum in u-space and N(0, I) starts (the flow base). Draws
are pushed forward to z-space for the uniform metrics.

Cost accounting: the flow fit performs ZERO target evaluations (it fits to
Gaussian samples); its wallclock is timed (flow_s) and counts in ESS/s per
the budget convention. The SVI init is billed via the shared init-cache
contract. Every leapfrog gradient of the pullback density contains one
target gradient (the flow-Jacobian part is sampler overhead, not a target
eval) -- counted exactly via the traced leapfrog counts.
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
    n_flow_train=4096,
    flow_layers=6,
    knots=8,
    interval=5.0,
    nn_width=64,
    nn_depth=1,
    max_epochs=100,
    max_patience=8,
    flow_batch=256,
    flow_lr=3e-4,
    # ChEES-HMC phase (S0 conventions)
    eps0=0.3,
    init_l=3,
    max_leapfrog=30,
    use_gbtla=True,
    target_accept=None,
    adapt_frac=0.8,
    start_scale=1.0,
)

DEFAULT_BUDGET = dict(n_chains=32, n_burn=250, n_keep=750)


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
    k_start, k_hmc = jax.random.split(key)
    n_chains = int(budget["n_chains"])
    n_burn, n_keep = int(budget["n_burn"]), int(budget["n_keep"])
    dim = target.dim

    # ---- SVI Gaussian -> flow training samples ---------------------------------------
    ginit = common.gaussian_init(target, seed, ledger, timing)
    extras["init_source"] = ginit.source
    extras["svi_cov_n_floored"] = ginit.n_floored
    rng = np.random.default_rng(int(seed) + 8_20260706)
    eps = rng.standard_normal((int(cfg["n_flow_train"]), dim))
    train = ginit.loc[None, :] + eps @ ginit.chol.T

    # ---- NSF fit (typed keys inside flows.fit_nsf) -------------------------------------
    t0 = time.time()
    bundle = flows.fit_nsf(
        int(seed) + 88, train,
        flow_layers=int(cfg["flow_layers"]), knots=int(cfg["knots"]),
        interval=float(cfg["interval"]), nn_width=int(cfg["nn_width"]),
        nn_depth=int(cfg["nn_depth"]), max_epochs=int(cfg["max_epochs"]),
        max_patience=int(cfg["max_patience"]),
        batch_size=int(cfg["flow_batch"]), learning_rate=float(cfg["flow_lr"]))
    timing["flow_s"] = time.time() - t0
    ledger.add("flow_fit",
               note=f"NSF on {cfg['n_flow_train']} floored-SVI samples, "
                    f"{len(bundle.losses['train'])} epochs, 0 target evals")
    extras["flow_final_val_loss"] = float(bundle.losses["val"][-1]) \
        if bundle.losses.get("val") else None

    # ---- ChEES-HMC in pullback space ---------------------------------------------------
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

    # ---- push forward to z-space ---------------------------------------------------------
    t0 = time.time()
    samples = bundle.push_samples(samples_u_all[n_burn:])
    timing["push_s"] = time.time() - t0

    accepted = diagnostics["is_accepted"][n_burn:]
    return common.finalize_cell(
        sampler="neutra", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        acceptance_rate=float(np.mean(accepted)),
        final_step_size=float(np.ravel(diagnostics["step_size"])[-1]))
