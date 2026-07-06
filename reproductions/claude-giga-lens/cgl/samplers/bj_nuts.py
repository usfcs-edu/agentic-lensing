"""S1 adapter: blackjax window-adapted NUTS (Stan-style warmup), vmapped
over chains against the zoo single-point logdensity view.

Recipe: window_adaptation(blackjax.nuts) warmup per chain (step size via dual
averaging + diagonal mass matrix via the 3-window schedule), then a NUTS scan.
Chain parallelism is jax.vmap over (key, state); the underlying zoo target is
evaluated through the batch-1 view (eagerly warmed per the zoo batch-size
warmup contract -- under vmap the batch-1 trace vectorizes over chains).

Policy knobs (frozen on the dev split):
  * start: "svi" (init-cache / analytic-bundle Gaussian; billed) | "prior"
  * pool_params: "median" (share median step size + mass matrix across
    chains -- robust to the odd badly-adapted chain) | "per_chain"
  * target_accept, max_num_doublings

Gradient ledger: warmup = sum of traced per-step num_integration_steps + 1
init gradient per chain; sampling = sum of traced num_integration_steps
(velocity-verlet: 1 NEW gradient per leapfrog step, integrator state carries
the current gradient).
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
    start="svi",                 # "svi" | "prior"
    pool_params="median",        # "median" | "per_chain"
    target_accept=0.8,
    max_num_doublings=10,
    is_mass_matrix_diagonal=True,
    initial_step_size=0.1,
)

DEFAULT_BUDGET = dict(n_chains=16, n_warmup=500, n_keep=1000)


def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    import blackjax
    import jax
    import jax.numpy as jnp
    from blackjax.adaptation.base import get_filter_adapt_info_fn

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
    k_start, k_warm, k_sample = jax.random.split(key, 3)
    n_chains = int(budget["n_chains"])
    n_warmup = int(budget.get("n_warmup") or budget.get("n_burn") or 500)
    n_keep = int(budget["n_keep"])
    dim = target.dim

    # ---- starts -----------------------------------------------------------------
    if cfg["start"] == "svi":
        ginit = common.gaussian_init(target, seed, ledger, timing)
        start = common.draw_gaussian_starts(ginit, k_start, n_chains, fdtype)
        extras["init_source"] = ginit.source
    else:
        start = target.init.prior_sample_fn(k_start, n_chains)
        extras["init_source"] = "prior"

    common.warmup_batch(target, 1)          # batch-1 view under vmap
    logdensity = common.batched_logdensity(target)

    # ---- warmup (window adaptation, vmapped over chains) --------------------------
    t0 = time.time()
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity,
        is_mass_matrix_diagonal=bool(cfg["is_mass_matrix_diagonal"]),
        initial_step_size=float(cfg["initial_step_size"]),
        target_acceptance_rate=float(cfg["target_accept"]),
        progress_bar=False,
        adaptation_info_fn=get_filter_adapt_info_fn(
            info_keys={"num_integration_steps", "acceptance_rate",
                       "is_divergent"}),
        max_num_doublings=int(cfg["max_num_doublings"]),
    )

    def run_warmup(k, pos):
        (state, params), info = warmup.run(k, pos, num_steps=n_warmup)
        return state, params, info

    warm_keys = jax.random.split(k_warm, n_chains)
    states, params, winfo = jax.jit(jax.vmap(run_warmup))(warm_keys, start)
    jax.block_until_ready(states.position)
    timing["warmup_s"] = time.time() - t0
    w_nis = np.asarray(winfo.info.num_integration_steps)     # (C, n_warmup)
    g_warm = int(w_nis.sum()) + n_chains                     # + init grads
    ledger.add("warmup", n_grad=g_warm,
               note=f"window_adaptation({n_warmup} steps x {n_chains} chains,"
                    " leapfrogs traced + 1 init grad/chain)")
    extras["warmup_divergences"] = int(
        np.asarray(winfo.info.is_divergent).sum())

    step_sizes = np.asarray(params["step_size"], dtype=np.float64)
    imm = np.asarray(params["inverse_mass_matrix"], dtype=np.float64)
    extras["step_size_per_chain"] = step_sizes.tolist()
    if cfg["pool_params"] == "median":
        ss = jnp.asarray(np.median(step_sizes), dtype=fdtype)
        mm = jnp.asarray(np.median(imm, axis=0), dtype=fdtype)
    else:
        ss = jnp.asarray(step_sizes, dtype=fdtype)
        mm = jnp.asarray(imm, dtype=fdtype)

    # ---- sampling ------------------------------------------------------------------
    t0 = time.time()
    kernel = blackjax.nuts.build_kernel()

    def one_step(states, k):
        ks = jax.random.split(k, n_chains)
        if cfg["pool_params"] == "median":
            states, infos = jax.vmap(
                lambda kk, st: kernel(kk, st, logdensity, ss, mm,
                                      max_num_doublings=int(
                                          cfg["max_num_doublings"])))(
                ks, states)
        else:
            states, infos = jax.vmap(
                lambda kk, st, s1, m1: kernel(
                    kk, st, logdensity, s1, m1,
                    max_num_doublings=int(cfg["max_num_doublings"])))(
                ks, states, ss, mm)
        return states, (states.position, infos.num_integration_steps,
                        infos.acceptance_rate, infos.is_divergent)

    keys = jax.random.split(k_sample, n_keep)
    _, (pos, nis, acc, div) = jax.jit(
        lambda st, ks: jax.lax.scan(one_step, st, ks))(states, keys)
    pos = jax.block_until_ready(pos)
    timing["nuts_s"] = time.time() - t0

    samples = np.asarray(pos)                                # (T, C, dim)
    nis = np.asarray(nis)                                    # (T, C)
    g_sample = int(nis.sum())
    ledger.add("nuts", n_grad=g_sample,
               note=f"{n_keep} draws x {n_chains} chains, leapfrogs traced")
    extras["n_divergent"] = int(np.asarray(div).sum())
    extras["mean_tree_leapfrogs"] = float(nis.mean())

    diagnostics = dict(
        num_integration_steps=nis,
        acceptance_rate=np.asarray(acc),
        is_divergent=np.asarray(div),
        step_size_per_chain=step_sizes,
        inverse_mass_matrix=imm,
    )
    return common.finalize_cell(
        sampler="bj_nuts", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        acceptance_rate=float(np.asarray(acc).mean()),
        final_step_size=float(np.median(step_sizes)))
