"""S2 adapter: blackjax MCLMC (microcanonical Langevin Monte Carlo).

Two variants behind one config switch (frozen per tier on the dev split):
  * adjusted=False: UNADJUSTED MCLMC -- mclmc_find_L_and_step_size tuner
    (single tuning chain, diagonal preconditioning) then a vmapped
    multi-chain scan of the unadjusted kernel. Small step-size-controlled
    bias (desired_energy_var) -- recorded in extras, standard for the method.
  * adjusted=True: adjusted (Metropolis-corrected) STATIC MCLMC
    (blackjax.adjusted_mclmc) with step_size/num_integration_steps derived
    from the same tuner output (adj_step_scale * eps, L/eps steps).

x64 targets inherit --xla_disable_hlo_passes=priority-fusion from
cgl.zoo.runtime.setup_process_env (jaxlib 0.6.2 livelock: f64 random.normal
fused with a reduction = exactly MCLMC's partially_refresh_momentum;
CAMPAIGN.md P0).

Gradient ledger: isokinetic mclachlan = 2 NEW gradients per integrator step
(the state carries the current gradient); tuner steps are kernel steps (2
grads each) + 1 init grad; chain init = 1 grad per chain.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cgl import guards, metrics
from cgl.io import CellResult
from cgl.samplers import common
from cgl.zoo.api import LensPosterior, np_dtype

SCALE_KEYS = ("n_steps",)

GRADS_PER_STEP = 2      # isokinetic mclachlan (see common.GRADS_PER_STEP)

DEFAULT_CONFIG = dict(
    adjusted=False,
    start="svi",                  # "svi" | "prior"
    desired_energy_var=5e-4,
    diagonal_preconditioning=True,
    # mass="tuner": the tuner's own diagonal preconditioning.
    # mass="svi_diag": SEED the tuner with inverse_mass_matrix =
    #   diag(floored SVI covariance) and diagonal_preconditioning=False.
    #   Needed on extreme-conditioning targets: on t0_illcond46 (cond 1e14)
    #   the tuner's Welford variance estimate produced a NEGATIVE entry
    #   (-2e-16) -> L=9e-11, step=5e-10, all-NaN chains (dev-split finding,
    #   2026-07-06).
    # mass="auto": tuner first; on collapse (non-positive/non-finite mass
    #   entries or degenerate L/step) DETERMINISTICALLY re-tune seeded with
    #   the SVI diagonal. Part of the frozen algorithm, not a hand-tune.
    mass="auto",
    adj_step_scale=1.0,           # adjusted: eps = adj_step_scale * tuned eps
    adj_max_steps=64,             # adjusted: cap on num_integration_steps
    adj_target_min_steps=2,
)

DEFAULT_BUDGET = dict(n_chains=8, n_tune=2000, n_steps=10000)


def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    import blackjax
    import jax
    import jax.numpy as jnp
    from blackjax.mcmc.integrators import isokinetic_mclachlan

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
    k_start, k_init, k_tune, k_run = jax.random.split(key, 4)
    n_chains = int(budget["n_chains"])
    n_tune = int(budget["n_tune"])
    n_steps = int(budget["n_steps"])
    dim = target.dim

    # ---- starts -----------------------------------------------------------------
    ginit = None
    if cfg["start"] == "svi" or cfg["mass"] in ("svi_diag", "auto"):
        ginit = common.gaussian_init(target, seed, ledger, timing)
        extras["init_source"] = ginit.source
    if cfg["start"] == "svi":
        start = common.draw_gaussian_starts(ginit, k_start, n_chains, fdtype)
    else:
        start = target.init.prior_sample_fn(k_start, n_chains)
        extras["init_source"] = "prior"

    common.warmup_batch(target, 1)
    logdensity = common.batched_logdensity(target)

    # ---- tuner (single chain; params shared across chains) ------------------------
    t0 = time.time()
    state0 = blackjax.mcmc.mclmc.init(start[0], logdensity, k_init)

    def kernel_builder(inverse_mass_matrix):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=logdensity,
            integrator=isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix)

    def _svidiag_params():
        from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

        return MCLMCAdaptationState(
            L=jnp.asarray(np.sqrt(dim), dtype=fdtype),
            step_size=jnp.asarray(0.25 * np.sqrt(dim), dtype=fdtype),
            inverse_mass_matrix=jnp.asarray(np.diag(ginit.cov_reg),
                                            dtype=fdtype))

    def _run_tuner(params0, diag_precond):
        return blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel_builder, num_steps=n_tune, state=state0,
            rng_key=k_tune,
            desired_energy_var=float(cfg["desired_energy_var"]),
            diagonal_preconditioning=diag_precond, params=params0)

    def _collapsed(params):
        L_, eps_ = float(params.L), float(params.step_size)
        imm_ = np.asarray(params.inverse_mass_matrix, dtype=np.float64)
        return (not np.isfinite(L_) or not np.isfinite(eps_)
                or eps_ <= 1e-10 or L_ <= 1e-10
                or not np.all(np.isfinite(imm_)) or np.min(imm_) <= 0.0)

    n_tunes_run = 1
    if cfg["mass"] == "svi_diag":
        _, params, _ = _run_tuner(_svidiag_params(), False)
    else:
        _, params, _ = _run_tuner(
            None, bool(cfg["diagonal_preconditioning"]))
        if cfg["mass"] == "auto" and _collapsed(params):
            # deterministic fallback (frozen-algorithm robustness guard):
            # re-tune seeded with the SVI diagonal, no re-preconditioning
            extras["tuner_fallback_svi_diag"] = True
            _, params, _ = _run_tuner(_svidiag_params(), False)
            n_tunes_run = 2
    jax.block_until_ready(params.L)
    timing["tune_s"] = time.time() - t0
    L, eps = float(params.L), float(params.step_size)
    if _collapsed(params):
        raise RuntimeError(
            f"MCLMC tuner failed (post-fallback): L={L}, step_size={eps}, "
            f"imm_min={float(np.min(np.asarray(params.inverse_mass_matrix)))}")
    ledger.add("tune", n_grad=(GRADS_PER_STEP * n_tune + 1) * n_tunes_run,
               note=f"mclmc_find_L_and_step_size {n_tune} steps x 2 grads "
                    f"+ init (x{n_tunes_run} tuner passes)")
    extras.update(tuned_L=L, tuned_step_size=eps,
                  imm_minmax=[float(np.min(np.asarray(
                      params.inverse_mass_matrix))),
                      float(np.max(np.asarray(params.inverse_mass_matrix)))])

    # ---- chains -------------------------------------------------------------------
    t0 = time.time()
    ledger.add("chain_init", n_grad=n_chains, note="1 grad per chain init")

    if not cfg["adjusted"]:
        init_keys = jax.random.split(k_init, n_chains)
        states = jax.vmap(lambda p, k: blackjax.mcmc.mclmc.init(
            p, logdensity, k))(start, init_keys)
        algo = blackjax.mclmc(logdensity, L=params.L,
                              step_size=params.step_size,
                              inverse_mass_matrix=params.inverse_mass_matrix)

        def one_step(states, k):
            ks = jax.random.split(k, n_chains)
            states, infos = jax.vmap(algo.step)(ks, states)
            return states, (states.position, infos.energy_change)

        keys = jax.random.split(k_run, n_steps)
        _, (pos, dE) = jax.jit(
            lambda st, ks: jax.lax.scan(one_step, st, ks))(states, keys)
        pos = jax.block_until_ready(pos)
        timing["mclmc_s"] = time.time() - t0
        g_run = GRADS_PER_STEP * n_steps * n_chains
        ledger.add("mclmc", n_grad=g_run,
                   note=f"unadjusted, {n_steps} steps x {n_chains} chains "
                        "x 2 grads (mclachlan)")
        diagnostics = dict(energy_change=np.asarray(dE))
        acc_rate = float("nan")
        variant = "unadjusted"
    else:
        nis = int(np.clip(round(L / eps), cfg["adj_target_min_steps"],
                          cfg["adj_max_steps"]))
        eps_adj = float(cfg["adj_step_scale"]) * eps
        algo = blackjax.adjusted_mclmc(
            logdensity, step_size=eps_adj, num_integration_steps=nis,
            inverse_mass_matrix=params.inverse_mass_matrix)
        states = jax.vmap(algo.init)(start)

        def one_step(states, k):
            ks = jax.random.split(k, n_chains)
            states, infos = jax.vmap(algo.step)(ks, states)
            return states, (states.position, infos.acceptance_rate)

        keys = jax.random.split(k_run, n_steps)
        _, (pos, acc) = jax.jit(
            lambda st, ks: jax.lax.scan(one_step, st, ks))(states, keys)
        pos = jax.block_until_ready(pos)
        timing["mclmc_s"] = time.time() - t0
        g_run = GRADS_PER_STEP * nis * n_steps * n_chains
        ledger.add("mclmc", n_grad=g_run,
                   note=f"adjusted, {n_steps} steps x {nis} int-steps x "
                        f"{n_chains} chains x 2 grads")
        diagnostics = dict(acceptance_rate=np.asarray(acc))
        acc_rate = float(np.asarray(acc).mean())
        extras["adj_num_integration_steps"] = nis
        extras["adj_step_size"] = eps_adj
        variant = "adjusted_static"

    extras["variant"] = variant
    samples = np.asarray(pos)                             # (T, C, dim)
    return common.finalize_cell(
        sampler="bj_mclmc", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        acceptance_rate=acc_rate, final_step_size=eps)
