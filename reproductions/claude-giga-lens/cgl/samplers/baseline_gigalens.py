"""S0 baseline adapter: the published GIGA-Lens recipe (MAP -> SVI -> PHMC),
reimplemented against the zoo `LensPosterior.log_prob_batch` surface.

Kernel stack fidelity:
  * "chees" mode (default; T0/T1-class f32 targets) follows gu-2022
    02_fit_system.py / the canonical gigalens demo: K-multistart adabelief
    (1e-2, b1=.95, b2=.99) MAP from prior draws -> full-rank MVN SVI
    (adabelief 1e-3, FillScaleTriL(diag=Exp, shift=1e-6) parameterization,
    init_scales=1e-3, best-loss iterate kept) -> batched single-device
    PreconditionedHMC (momentum covariance = inv(SVI cov)) +
    GradientBasedTrajectoryLengthAdaptation (ChEES; max_leapfrog_steps=30) +
    DualAveragingStepSizeAdaptation over the first 80% of burn-in.
  * "precond_fixedL" mode (default for float64 targets with Hessian init,
    i.e. T2) follows foundry-i 34_fit_marg.py --mode hmc --massmatrix
    diagraw: start at InitBundle.map_z, momentum = MVNDiag(scale_diag =
    sqrt(max(|diag H_raw|, 1))) (the float64-safe momentum precision-factor
    guard: per-param scalars only, no decomposition of the cond~1e14
    matrix), fixed leapfrog L (default 16) + DualAveraging. MAP/SVI are
    SKIPPED (the artifacts already encode them).

Deviations from the copied recipes (complete list, pre-registered):
  1. MAP loss is -mean(logp) WITHOUT gigalens's 1/n_pixels normalization
     (the zoo surface has no image; adabelief's per-parameter adaptive
     scaling absorbs the constant factor).
  2. The SVI covariance is ALWAYS passed through guards.floor_svi_covariance
     before inversion (the Bug-2 guard; gu-2022 inverted the raw covariance).
  3. guards.check_svi_schedule enforces >=150 SVI steps/param, so the default
     schedule is max(500, 150*dim) steps (gu-2022 ran 500 for 22 params --
     below the guard floor that the foundry-i incident established).
  4. sample_chain runs with num_results = burn+keep and num_burnin_steps=0,
     discarding the first `burn` draws afterwards -- identical chain, but the
     trace (leapfrog counts for the gradient ledger) covers burn-in too.
  5. Chain starts are drawn from the FLOORED SVI q (loc + chol_floored @ eps),
     consistent with deviation 2 (gu-2022 sampled the raw qz).

Gradient accounting: MAP steps*K; SVI steps*n_vi; HMC sum_t(L_t)*chains with
L_t traced from the kernel results (exact even under ChEES jittering).
ChEES criterion gradients (w.r.t. trajectory length, computed from already-
evaluated states) are not extra logp gradients and are not counted.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cgl import fitting, guards, metrics
from cgl.io import CellResult, env_info
from cgl.zoo.api import LensPosterior, np_dtype

DEFAULT_CONFIG = dict(
    mode="auto",              # "chees" | "precond_fixedL" | "auto"
    n_map=128,                # K_MAP multistart (gu-2022 hardware-proxy value)
    map_steps=250,
    map_lr=1e-2,
    n_vi=200,                 # ELBO MC batch (gu-2022 hardware-proxy value)
    svi_steps=None,           # None -> max(500, 150*dim) (guard floor)
    svi_lr=1e-3,
    init_scales=1e-3,
    eps0=0.3,                 # init step size (paper Table 1)
    init_l=3,                 # init leapfrog steps under ChEES (demo default)
    max_leapfrog=30,          # ChEES cap (paper/demo default)
    use_gbtla=True,           # ChEES trajectory adaptation (paper stack)
    num_leapfrog=16,          # fixed L for precond_fixedL (34_fit_marg)
    target_accept=None,       # None -> TFP DualAveraging default (chees mode)
                              #         0.8 in precond_fixedL (34_fit_marg)
    adapt_frac=0.8,           # DualAveraging/ChEES over first 80% of burn
    start_jitter=0.0,         # precond_fixedL: N(0, jitter^2) start scatter
)

DEFAULT_BUDGET = dict(n_chains=50, n_burn=250, n_keep=750)


# --------------------------------------------------------------------------- #
# HMC phase (MAP/SVI live in cgl.fitting, shared with future adapters)
# --------------------------------------------------------------------------- #
def _trace_fn_factory(use_gbtla):
    from tensorflow_probability.substrates.jax.internal import unnest

    def trace_fn(_, pkr):
        out = {
            "step_size": unnest.get_innermost(pkr, "step_size"),
            "is_accepted": unnest.get_innermost(pkr, "is_accepted"),
            "target_log_prob": unnest.get_innermost(pkr, "target_log_prob"),
        }
        # exact leapfrog count for the gradient ledger (GBTLA re-sets it per
        # step; fixed-L PHMC stores the constant). Absent on exotic kernels ->
        # the caller falls back to the analytic count.
        try:
            out["num_leapfrog_steps"] = unnest.get_innermost(
                pkr, "num_leapfrog_steps")
        except (AttributeError, ValueError):
            pass
        if use_gbtla:
            try:
                out["max_trajectory_length"] = unnest.get_innermost(
                    pkr, "max_trajectory_length")
            except (AttributeError, ValueError):
                pass
        return out

    return trace_fn


def _run_hmc(target: LensPosterior, start, momentum_distribution, key, cfg,
             budget, fdtype, use_gbtla: bool, num_leapfrog: int):
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    tfe = tfp.experimental
    n_burn, n_keep = int(budget["n_burn"]), int(budget["n_keep"])
    num_adapt = int(cfg["adapt_frac"] * n_burn)
    n_chains = int(start.shape[0])
    # eager warmup of the (n_chains, dim) batch path: gigalens targets build a
    # LensSimulator per batch size, which cannot happen inside the TFP trace.
    target.log_prob_batch(start)

    kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob_batch,
        momentum_distribution=momentum_distribution,
        step_size=jnp.asarray(cfg["eps0"], dtype=fdtype),
        num_leapfrog_steps=int(cfg["init_l"]) if use_gbtla else num_leapfrog,
    )
    if use_gbtla:
        kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
            kernel, num_adaptation_steps=num_adapt,
            max_leapfrog_steps=int(cfg["max_leapfrog"]),
        )
    da_kwargs = dict(num_adaptation_steps=num_adapt)
    if cfg["target_accept"] is not None:
        da_kwargs["target_accept_prob"] = jnp.asarray(cfg["target_accept"],
                                                      dtype=fdtype)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, **da_kwargs)

    trace_fn = _trace_fn_factory(use_gbtla)

    @jax.jit
    def run_chain(seed_key, start):
        return tfp.mcmc.sample_chain(
            num_results=n_burn + n_keep,
            num_burnin_steps=0,          # deviation 4: trace burn-in too
            current_state=start,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=seed_key,
        )

    samples, trace = run_chain(key, start)
    samples = jax.block_until_ready(samples)
    samples = np.asarray(samples)                    # (burn+keep, C, dim)
    if "num_leapfrog_steps" in trace:
        leapfrog = np.asarray(trace["num_leapfrog_steps"]).astype(np.int64)
        leapfrog_exact = True
    else:                                            # analytic fallback
        leapfrog = np.full(n_burn + n_keep, num_leapfrog, dtype=np.int64)
        leapfrog_exact = False
    n_grad = int(leapfrog.sum() * n_chains)
    diagnostics = {
        "step_size": np.asarray(trace["step_size"]),
        "is_accepted": np.asarray(trace["is_accepted"]),
        "target_log_prob": np.asarray(trace["target_log_prob"]),
        "num_leapfrog_steps": leapfrog,
    }
    if "max_trajectory_length" in trace:
        diagnostics["max_trajectory_length"] = np.asarray(
            trace["max_trajectory_length"])
    return samples, diagnostics, n_grad, leapfrog_exact


# --------------------------------------------------------------------------- #
# the uniform adapter entry point
# --------------------------------------------------------------------------- #
def run_cell(target: LensPosterior, seed: int, budget: Optional[dict] = None,
             config: Optional[dict] = None,
             freeze_points: Optional[list] = None) -> CellResult:
    """Run one S0 cell. See module docstring for the recipe + deviations.

    budget: dict(n_chains, n_burn, n_keep)  (Track-A default 50/250/750)
    freeze_points: list of dicts with keys "z" (list) and "logp" (float) from
        data/zoo_freeze.json; the ADAPTER-FIDELITY pre-run assertion checks
        the adapter's own logp view against them (required before benchmarks;
        pass None only for unit-test smokes).
    """
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

    mode = cfg["mode"]
    if mode == "auto":
        mode = ("precond_fixedL"
                if (target.dtype == "float64"
                    and target.init.hess_diag_raw is not None)
                else "chees")
    cfg["mode"] = mode

    # ---- adapter-fidelity assertion (pre-registered) --------------------------
    freeze_check = {"checked": False}
    if freeze_points:
        zs = np.asarray([p["z"] for p in freeze_points], dtype=np.float64)
        want = np.asarray([p["logp"] for p in freeze_points], dtype=np.float64)
        got = np.asarray(target.log_prob_batch(
            jnp.asarray(zs, dtype=fdtype)), dtype=np.float64)
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
        tol = 1e-10 if target.dtype == "float64" else 1e-6
        freeze_check = dict(checked=True, n_points=len(freeze_points),
                            max_rel_err=float(rel.max()), tol=tol,
                            logp_recomputed=got.tolist(),
                            logp_frozen=want.tolist())
        if rel.max() > tol:
            raise RuntimeError(
                f"ADAPTER-FIDELITY FAILURE on {target.name}: adapter logp "
                f"disagrees with the zoo freeze (max rel {rel.max():.3e} > "
                f"{tol:g}). Refusing to run the cell.")

    ledger = metrics.BudgetLedger()
    timing = {}
    key = jax.random.PRNGKey(seed)
    k_map, k_svi, k_start, k_hmc = jax.random.split(key, 4)
    n_chains = int(budget["n_chains"])
    dim = target.dim
    extras = {}

    if mode == "chees":
        # ---- MAP ---------------------------------------------------------------
        t0 = time.time()
        z0 = target.init.prior_sample_fn(k_map, int(cfg["n_map"]))
        z_map, lp_map, map_hist, g_map = fitting.run_map(
            target.log_prob_batch, z0, int(cfg["map_steps"]), cfg["map_lr"],
            fdtype)
        timing["map_s"] = time.time() - t0
        ledger.add("map", n_grad=g_map,
                   note=f"{cfg['map_steps']} steps x {cfg['n_map']} starts")
        extras["map_best_logp"] = lp_map

        # ---- SVI ---------------------------------------------------------------
        t0 = time.time()
        svi_steps = int(cfg["svi_steps"] or max(500, 150 * dim))
        guards.check_svi_schedule(svi_steps, dim)
        svi_loc, svi_cov, best_neg_elbo, svi_hist, g_svi = fitting.run_svi(
            target.log_prob_batch, z_map, k_svi, dim, svi_steps,
            int(cfg["n_vi"]), cfg["svi_lr"], cfg["init_scales"], fdtype)
        timing["svi_s"] = time.time() - t0
        ledger.add("svi", n_grad=g_svi,
                   note=f"{svi_steps} steps x {cfg['n_vi']} MC samples")
        extras["svi_best_neg_elbo"] = best_neg_elbo

        # ---- floored covariance -> momentum + starts (Bug-2 guard) --------------
        cov_reg, chol, n_floored = guards.floor_svi_covariance(svi_cov)
        extras["svi_cov_n_floored"] = n_floored
        prec = np.linalg.solve(cov_reg, np.eye(dim))
        prec = 0.5 * (prec + prec.T)
        momentum = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros(dim, dtype=fdtype),
            covariance_matrix=jnp.asarray(prec, dtype=fdtype))
        eps = jax.random.normal(k_start, (n_chains, dim), dtype=fdtype)
        start = (jnp.asarray(svi_loc, dtype=fdtype)[None, :]
                 + eps @ jnp.asarray(chol, dtype=fdtype).T)
        use_gbtla, num_leapfrog = bool(cfg["use_gbtla"]), int(cfg["init_l"])

    elif mode == "precond_fixedL":
        # ---- T2-style: artifact init + diagraw diagonal mass ---------------------
        init = target.init
        assert init.map_z is not None and init.hess_diag_raw is not None, \
            "precond_fixedL needs InitBundle.map_z + hess_diag_raw"
        diag = np.maximum(np.abs(np.asarray(init.hess_diag_raw,
                                            dtype=np.float64)), 1.0)
        momentum = tfd.MultivariateNormalDiag(
            loc=jnp.zeros(dim, dtype=fdtype),
            scale_diag=jnp.asarray(np.sqrt(diag), dtype=fdtype))
        base = jnp.asarray(init.map_z, dtype=fdtype)
        jit_amp = float(cfg["start_jitter"])
        eps = jax.random.normal(k_start, (n_chains, dim), dtype=fdtype)
        start = base[None, :] + jit_amp * eps / jnp.asarray(
            np.sqrt(diag), dtype=fdtype)[None, :]
        use_gbtla, num_leapfrog = False, int(cfg["num_leapfrog"])
        if cfg["target_accept"] is None:
            cfg["target_accept"] = 0.8              # 34_fit_marg default
        extras["massmatrix"] = "diagraw"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # ---- HMC ------------------------------------------------------------------
    t0 = time.time()
    samples_all, diagnostics, g_hmc, leapfrog_exact = _run_hmc(
        target, start, momentum, k_hmc, cfg, budget, fdtype,
        use_gbtla=use_gbtla, num_leapfrog=num_leapfrog)
    timing["hmc_s"] = time.time() - t0
    n_burn = int(budget["n_burn"])
    ledger.add("hmc", n_grad=g_hmc,
               n_logp=(n_burn + int(budget["n_keep"])) * n_chains,
               note=("PHMC+ChEES+DualAveraging" if use_gbtla else
                     f"PHMC fixed L={num_leapfrog}+DualAveraging")
                    + (", leapfrog traced" if leapfrog_exact
                       else ", leapfrog ANALYTIC fallback")
                    + ", burn-in included")
    samples = samples_all[n_burn:]

    timing["total_s"] = sum(v for k, v in timing.items() if k.endswith("_s"))

    hardware = env_info()
    diag_summary = metrics.rank_diagnostics(samples, target.labels,
                                            target.mass_labels)
    eff = metrics.efficiency(diag_summary, ledger.n_grad, ledger.n_logp,
                             timing["total_s"],
                             hardware=hardware["device_kind"])
    accepted = diagnostics["is_accepted"][n_burn:]
    result_metrics = dict(
        diagnostics=diag_summary,
        efficiency=eff,
        acceptance_rate=float(np.mean(accepted)),
        final_step_size=float(np.ravel(diagnostics["step_size"])[-1]),
        extras=extras,
    )

    return CellResult(
        sampler="s0_baseline", target=target.name, seed=int(seed),
        track=str(budget.get("track", "A")),
        samples=samples.astype(np_dtype(target.dtype)),
        labels=list(target.labels), mass_labels=list(target.mass_labels),
        diagnostics=diagnostics,
        budget={**ledger.as_dict(), "requested": {k: v for k, v in
                                                  budget.items()}},
        timing=timing,
        config=cfg,
        env=hardware,
        freeze_check=freeze_check,
        metrics=result_metrics,
    )
