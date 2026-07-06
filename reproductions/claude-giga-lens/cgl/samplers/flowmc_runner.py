"""S3 adapter: flowMC 0.4.5 (RQSpline_MALA resource-strategy bundle).

Local MALA moves + global normalizing-flow (rational-quadratic spline)
Metropolis-Hastings moves, with the flow trained on the accumulated chains
during the training loops. Production positions are harvested from
sampler.resources per the P0 smoke test.

MANDATORY STACK WORKAROUND (CAMPAIGN.md P0, do not remove): flowMC 0.4.5
initializes its optax adamw state over eqx.filter(model, eqx.is_array),
which keeps the RQSpline's non-trainable bool mask arrays, while
train_step's filtered grads have None at those leaves -- a tree mismatch on
jax 0.6.2. The bundle's optimizer is re-initialized as
optax.chain(clip_by_global_norm(1.0), adam(lr)) over
eqx.filter(model, eqx.is_inexact_array) BEFORE Sampler construction.

Ledger convention: local MALA = 1 gradient per step per chain (proposal
gradient; matches the 1-new-grad-per-leapfrog HMC convention) with the
accept-side density folded into n_logp; global flow moves = 1 target logp
per step per chain (no gradients). Both training and production loops are
counted (training chains hit the target too). Flow training itself performs
zero target evaluations (wallclock only).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cgl import guards, metrics
from cgl.io import CellResult
from cgl.samplers import common
from cgl.zoo.api import LensPosterior, np_dtype

SCALE_KEYS = ("n_production_loops",)

DEFAULT_CONFIG = dict(
    start="svi",                    # "svi" | "prior"
    mala_step_size=0.15,
    rq_spline_hidden_units=[64, 64],
    rq_spline_n_bins=8,
    rq_spline_n_layers=4,
    n_epochs=8,
    batch_size=1024,
    n_max_examples=30000,
    learning_rate=1e-3,
    verbose=False,
)

DEFAULT_BUDGET = dict(n_chains=32, n_local_steps=50, n_global_steps=50,
                      n_training_loops=8, n_production_loops=8)


def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    import equinox as eqx
    import jax
    import optax
    from flowMC.resource_strategy_bundle.RQSpline_MALA import \
        RQSpline_MALA_Bundle
    from flowMC.Sampler import Sampler

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
    k_start, k_bundle, k_sampler = jax.random.split(key, 3)
    n_chains = int(budget["n_chains"])
    n_local = int(budget["n_local_steps"])
    n_global = int(budget["n_global_steps"])
    n_train_loops = int(budget["n_training_loops"])
    n_prod_loops = int(budget["n_production_loops"])
    dim = target.dim

    # ---- starts -----------------------------------------------------------------
    if cfg["start"] == "svi":
        ginit = common.gaussian_init(target, seed, ledger, timing)
        start = common.draw_gaussian_starts(ginit, k_start, n_chains, fdtype)
        extras["init_source"] = ginit.source
    else:
        start = target.init.prior_sample_fn(k_start, n_chains)
        extras["init_source"] = "prior"

    common.warmup_batch(target, 1)
    single = common.batched_logdensity(target)

    def logpdf(x, data):
        return single(x)

    # ---- bundle + the mandatory optimizer re-init workaround -----------------------
    t0 = time.time()
    bundle = RQSpline_MALA_Bundle(
        k_bundle, n_chains, dim, logpdf,
        n_local_steps=n_local, n_global_steps=n_global,
        n_training_loops=n_train_loops, n_production_loops=n_prod_loops,
        n_epochs=int(cfg["n_epochs"]),
        mala_step_size=float(cfg["mala_step_size"]),
        rq_spline_hidden_units=list(cfg["rq_spline_hidden_units"]),
        rq_spline_n_bins=int(cfg["rq_spline_n_bins"]),
        rq_spline_n_layers=int(cfg["rq_spline_n_layers"]),
        batch_size=int(cfg["batch_size"]),
        n_max_examples=int(cfg["n_max_examples"]),
        verbose=bool(cfg["verbose"]),
    )
    opt = bundle.resources["optimizer"]
    model = bundle.resources["model"]
    opt.optim = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adam(float(cfg["learning_rate"])))
    opt.optim_state = opt.optim.init(eqx.filter(model, eqx.is_inexact_array))

    sampler = Sampler(dim, n_chains, k_sampler,
                      resource_strategy_bundles=bundle)
    timing["setup_s"] = time.time() - t0

    # ---- run ------------------------------------------------------------------------
    t0 = time.time()
    sampler.sample(np.asarray(start, dtype=fdtype), {})
    timing["flowmc_s"] = time.time() - t0

    n_loops = n_train_loops + n_prod_loops
    g_local = n_loops * n_local * n_chains
    l_global = n_loops * n_global * n_chains
    ledger.add("local_mala", n_grad=g_local, n_logp=g_local,
               note=f"{n_loops} loops x {n_local} steps x {n_chains} chains "
                    "(1 grad + 1 accept logp per proposal)")
    ledger.add("global_flow", n_logp=l_global,
               note=f"{n_loops} loops x {n_global} steps x {n_chains} chains"
                    " (flow MH: target logp only)")
    ledger.add("flow_training", note="no target evals (wallclock only)")

    positions = np.asarray(
        sampler.resources["positions_production"].data)   # (C, T, dim)
    log_prob = np.asarray(sampler.resources["log_prob_production"].data)
    if not np.all(np.isfinite(positions)):
        extras["nonfinite_positions"] = int(
            (~np.isfinite(positions)).sum())
    samples = np.transpose(positions, (1, 0, 2))          # (T, C, dim)

    diagnostics = dict(log_prob_production=np.transpose(log_prob, (1, 0)))

    def _finite_mean(a):
        # flowMC pads the acceptance buffers with -inf on the steps where
        # the OTHER move type ran; average the finite entries only.
        a = np.asarray(a)
        m = np.isfinite(a)
        return float(a[m].mean()) if m.any() else float("nan")

    acc = float("nan")
    for key_name in ("local_accs_production", "global_accs_production",
                     "local_accs_training", "global_accs_training"):
        if key_name in sampler.resources:
            diagnostics[key_name] = np.asarray(
                sampler.resources[key_name].data)
            extras[key_name + "_mean"] = _finite_mean(
                diagnostics[key_name])
    acc = extras.get("local_accs_production_mean", float("nan"))

    return common.finalize_cell(
        sampler="flowmc_runner", target=target, seed=seed, budget=budget,
        config=cfg, samples=samples, diagnostics=diagnostics, ledger=ledger,
        timing=timing, freeze_check=freeze_check, extras=extras,
        acceptance_rate=acc,
        final_step_size=float(cfg["mala_step_size"]))
