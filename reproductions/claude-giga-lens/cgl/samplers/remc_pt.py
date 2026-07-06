"""S7 adapter: TFP ReplicaExchangeMC (parallel tempering) over the S0
batched PHMC kernel, with LIKELIHOOD-ONLY tempering.

Tempering uses the zoo's exact prior/likelihood split via TFP's native
untempered/tempered decomposition: per-replica target = log_prior +
beta_r * log_like (the prior is NEVER tempered, so the hottest replica is
the z-space prior and the beta ladder interpolates data strength -- the
correct construction for evidence-bearing lens posteriors). Both closures
are wrapped by the reshape-to-2D contract (common.flatten_leading): the
inner kernel evaluates replica-stacked states (R, C, dim) and each chain of
the C-batch carries its own independent R-replica system.

`make_replica_logp` exposes the composed per-replica density for the
hand-built-tempering unit test (tests/test_samplers.py).

Ladder: geometric, beta_r = beta_min^(r/(R-1)), r=0..R-1 (beta_0=1).
Per-replica step sizes eps_r = eps0 * beta_r^eps_scale_power (hotter ->
flatter -> larger steps), broadcast (R, 1, 1) against the stacked state.

Metrics: adjacent-swap acceptance traced per step/pair/chain; mode round
trips on the retained beta=1 chain come from the uniform 22_run_cell mode
metrics (cgl.metrics.count_mode_round_trips).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cgl import guards, metrics
from cgl.io import CellResult
from cgl.samplers import common
from cgl.zoo.api import LensPosterior, np_dtype

SCALE_KEYS = ("n_keep",)

DEFAULT_CONFIG = dict(
    n_replicas=6,
    beta_min=0.01,
    num_leapfrog=8,
    eps0=0.2,
    eps_scale_power=-0.5,
    eps_max=2.0,
    start="svi",                  # "svi" | "prior"
    mass="svi",                   # "svi" | "identity"
)

DEFAULT_BUDGET = dict(n_chains=8, n_burn=300, n_keep=900)


def geometric_ladder(n_replicas: int, beta_min: float) -> np.ndarray:
    """Geometric inverse-temperature ladder: beta_0=1 ... beta_{R-1}=beta_min."""
    assert n_replicas >= 2 and 0.0 < beta_min < 1.0
    r = np.arange(n_replicas, dtype=np.float64)
    return beta_min ** (r / (n_replicas - 1))


def make_replica_logp(target: LensPosterior, betas: np.ndarray):
    """(tempered_fn, untempered_fn, composed per-replica logp) closures.

    composed(Z) for Z of shape (R, ..., dim) returns
    log_prior(Z) + betas[:, None...] * log_like(Z) -- the hand-built
    beta-tempered density the unit test checks against.
    """
    import jax.numpy as jnp

    tempered = common.flatten_leading(target.log_like_batch)
    untempered = common.flatten_leading(target.log_prior_batch)

    def composed(Z):
        Z = jnp.asarray(Z)
        b = jnp.asarray(betas, dtype=Z.dtype)
        b = b.reshape((-1,) + (1,) * (Z.ndim - 2))
        return untempered(Z) + b * tempered(Z)

    return tempered, untempered, composed


def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    tfd = tfp.distributions
    tfe = tfp.experimental
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
    k_start, k_chain = jax.random.split(key)
    n_chains = int(budget["n_chains"])
    n_burn, n_keep = int(budget["n_burn"]), int(budget["n_keep"])
    R = int(cfg["n_replicas"])
    L = int(cfg["num_leapfrog"])
    dim = target.dim

    betas = geometric_ladder(R, float(cfg["beta_min"]))
    extras["betas"] = betas.tolist()

    # ---- init: starts + momentum preconditioner --------------------------------------
    if cfg["start"] == "svi" or cfg["mass"] == "svi":
        ginit = common.gaussian_init(target, seed, ledger, timing)
        extras["init_source"] = ginit.source
    if cfg["start"] == "svi":
        start = common.draw_gaussian_starts(ginit, k_start, n_chains, fdtype)
    else:
        start = target.init.prior_sample_fn(k_start, n_chains)
        extras["init_source"] = "prior"
    if cfg["mass"] == "svi":
        prec = np.linalg.solve(ginit.cov_reg, np.eye(dim))
        prec = 0.5 * (prec + prec.T)
        momentum = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros(dim, dtype=fdtype),
            covariance_matrix=jnp.asarray(prec, dtype=fdtype))
    else:
        momentum = tfd.MultivariateNormalDiag(
            loc=jnp.zeros(dim, dtype=fdtype),
            scale_diag=jnp.ones(dim, dtype=fdtype))

    # ---- tempered/untempered closures + kernel ---------------------------------------
    tempered, untempered, _ = make_replica_logp(target, betas)
    # zoo batch-size warmup contract: the inner kernel sees (R, C, dim) which
    # the reshape-to-2D wrapper flattens to batch R*C; TFP also evaluates the
    # unstacked (C, dim) state at bootstrap.
    common.warmup_batch(target, n_chains, R * n_chains)

    eps = np.minimum(float(cfg["eps0"])
                     * betas ** float(cfg["eps_scale_power"]),
                     float(cfg["eps_max"]))
    extras["step_size_ladder"] = eps.tolist()
    eps_arr = jnp.asarray(eps, dtype=fdtype).reshape(R, 1, 1)

    def make_kernel_fn(target_log_prob_fn):
        return tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            momentum_distribution=momentum,
            step_size=eps_arr,
            num_leapfrog_steps=L)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=None,
        inverse_temperatures=jnp.asarray(betas, dtype=fdtype),
        make_kernel_fn=make_kernel_fn,
        tempered_log_prob_fn=tempered,
        untempered_log_prob_fn=untempered,
    )

    def trace_fn(_, pkr):
        return {
            "is_swap_accepted_adjacent": pkr.is_swap_accepted_adjacent,
            "is_swap_proposed_adjacent": pkr.is_swap_proposed_adjacent,
        }

    @jax.jit
    def run_chain(seed_key, start):
        return tfp.mcmc.sample_chain(
            num_results=n_burn + n_keep, num_burnin_steps=0,
            current_state=start, kernel=remc, trace_fn=trace_fn,
            seed=seed_key)

    t0 = time.time()
    samples_all, trace = run_chain(k_chain, start)
    samples_all = jax.block_until_ready(samples_all)
    timing["remc_s"] = time.time() - t0

    n_steps = n_burn + n_keep
    g = n_steps * R * n_chains * L
    ledger.add("remc", n_grad=g, n_logp=n_steps * R * n_chains,
               note=f"{n_steps} steps x {R} replicas x {n_chains} chains x "
                    f"L={L}; logp = per-step tempered/untempered bookkeeping")

    samples = np.asarray(samples_all)[n_burn:]            # beta=1 chain only
    acc = np.asarray(trace["is_swap_accepted_adjacent"])   # (T, R-1, C)
    prop = np.asarray(trace["is_swap_proposed_adjacent"])
    with np.errstate(invalid="ignore"):
        pair_rates = (acc.astype(np.float64).sum(axis=(0,) + tuple(
            range(2, acc.ndim)))
            / np.maximum(prop.astype(np.float64).sum(axis=(0,) + tuple(
                range(2, prop.ndim))), 1.0))
    extras["swap_acceptance_per_pair"] = pair_rates.tolist()
    extras["swap_acceptance_mean"] = float(np.mean(pair_rates))

    diagnostics = dict(
        is_swap_accepted_adjacent=acc[n_burn:],
        is_swap_proposed_adjacent=prop[n_burn:],
    )
    return common.finalize_cell(
        sampler="remc_pt", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        acceptance_rate=float(np.mean(pair_rates)),
        final_step_size=float(eps[0]))
