"""Shared machinery for the P2b contender adapters (S1-S9).

Everything here is factored out of (or adapted from) the VALIDATED S0 adapter
cgl/samplers/baseline_gigalens.py so all contenders share one implementation
of the pre-registered protocol pieces:

  * freeze-point fidelity assertion (identical logic + tolerances to S0);
  * the init-cache contract: MAP+SVI computed ONCE per (T1 target, seed) by
    cgl.samplers.init_cache, and BILLED IDENTICALLY to every consumer
    (ledger phases "init_map"/"init_svi" with the cached gradient counts;
    cached wallclock added to the cell's timing so ESS/s is honest too).
    T0 targets use the analytic InitBundle (free by protocol -- noted in the
    ledger so the report can flag the T0 init-cost asymmetry vs S0);
  * ChEES-HMC runner (adapted copy of baseline_gigalens._run_hmc that takes a
    plain batched log-prob callable, so neutra/glnt can run it in flow space);
  * an adaptive tempered-SMC driver (blackjax) shared by bj_smc and glnt;
  * budget-ledger / CellResult assembly helpers, including the weight-based
    ESS override used by the particle methods (SMC, nautilus) where
    autocorrelation ESS on exchangeable particles would be inflated;
  * the frozen-policy hash helpers used by 22_run_cell's eval-split assertion
    (kept here so tests/test_samplers.py can unit-test the logic on CPU).

Gradient-counting conventions (extends the S0 ledger convention):
  * velocity-verlet leapfrog: 1 NEW gradient per step (both TFP and blackjax
    carry the current-position gradient in the integrator state);
  * isokinetic mclachlan (MCLMC family): 2 NEW gradients per step;
  * blackjax hmc/nuts/mclmc init: 1 gradient per (re)init;
  * flow training/pushforward: 0 target evals (wallclock only, timed).
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Callable, Optional

import numpy as np

from cgl import metrics
from cgl.io import CellResult, config_hash, env_info
from cgl.paths import DATA
from cgl.zoo.api import LensPosterior, np_dtype

INIT_CACHE_DIR = DATA / "init_cache"
POLICIES_FROZEN = DATA / "policies_frozen.json"

# gradient cost of one integrator step (NEW gradient evaluations; the
# integrator state carries the current-position gradient in blackjax/TFP)
GRADS_PER_STEP = {
    "velocity_verlet": 1,
    "mclachlan": 2,
    "isokinetic_mclachlan": 2,
    "yoshida": 3,
    "omelyan": 4,
}


# --------------------------------------------------------------------------- #
# freeze-point fidelity (identical to the validated S0 logic)
# --------------------------------------------------------------------------- #
def assert_freeze_fidelity(target: LensPosterior, freeze_points, fdtype) -> dict:
    """ADAPTER-FIDELITY pre-run assertion (pre-registered; see S0)."""
    import jax.numpy as jnp

    if not freeze_points:
        return {"checked": False}
    zs = np.asarray([p["z"] for p in freeze_points], dtype=np.float64)
    want = np.asarray([p["logp"] for p in freeze_points], dtype=np.float64)
    got = np.asarray(target.log_prob_batch(
        jnp.asarray(zs, dtype=fdtype)), dtype=np.float64)
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
    tol = 1e-10 if target.dtype == "float64" else 1e-6
    check = dict(checked=True, n_points=len(freeze_points),
                 max_rel_err=float(rel.max()), tol=tol,
                 logp_recomputed=got.tolist(), logp_frozen=want.tolist())
    if rel.max() > tol:
        raise RuntimeError(
            f"ADAPTER-FIDELITY FAILURE on {target.name}: adapter logp "
            f"disagrees with the zoo freeze (max rel {rel.max():.3e} > "
            f"{tol:g}). Refusing to run the cell.")
    return check


# --------------------------------------------------------------------------- #
# target-surface helpers
# --------------------------------------------------------------------------- #
def batched_logdensity(target: LensPosterior) -> Callable:
    """Single-point (dim,) -> scalar logdensity for blackjax/flowMC kernels.

    Under vmap-over-chains the inner call sees batch size 1, so callers MUST
    eagerly warm the batch-1 path first (warmup_batch(target, 1)) per the
    zoo batch-size warmup contract.
    """
    def logdensity(z):
        import jax.numpy as jnp
        return target.log_prob_batch(jnp.reshape(z, (1, -1)))[0]

    return logdensity


def warmup_batch(target: LensPosterior, *sizes) -> None:
    """Eagerly build the target's per-batch-size machinery (zoo contract)."""
    import jax.numpy as jnp

    fdtype = np_dtype(target.dtype)
    for n in sorted({int(s) for s in sizes}):
        target.log_prob_batch(jnp.zeros((n, target.dim), dtype=fdtype))


def flatten_leading(fn_batch: Callable) -> Callable:
    """Wrap a (N, dim)->(N,) batch fn to accept any (..., dim) input.

    This is the reshape-to-2D contract for TFP ReplicaExchangeMC, whose inner
    kernel evaluates the target on replica-stacked states (R, C, dim).
    """
    def fn(Z):
        import jax.numpy as jnp
        Z = jnp.asarray(Z)
        lead = Z.shape[:-1]
        out = fn_batch(Z.reshape((-1, Z.shape[-1])))
        return out.reshape(lead)

    return fn


# --------------------------------------------------------------------------- #
# init cache (MAP+SVI once per (T1 target, seed); billed to every consumer)
# --------------------------------------------------------------------------- #
def init_cache_path(target_name: str, seed: int):
    return INIT_CACHE_DIR / f"{target_name}_s{int(seed)}.npz"


def load_init_cache(target_name: str, seed: int) -> dict:
    p = init_cache_path(target_name, seed)
    if not p.exists():
        raise FileNotFoundError(
            f"init cache missing: {p}. Build it first: "
            f"python -m cgl.samplers.init_cache --target {target_name} "
            f"--seed {seed}")
    z = np.load(p, allow_pickle=True)
    return dict(
        z_map=np.asarray(z["z_map"], dtype=np.float64),
        lp_map=float(z["lp_map"]),
        svi_loc=np.asarray(z["svi_loc"], dtype=np.float64),
        svi_cov=np.asarray(z["svi_cov"], dtype=np.float64),
        n_grad_map=int(z["n_grad_map"]),
        n_grad_svi=int(z["n_grad_svi"]),
        map_s=float(z["map_s"]),
        svi_s=float(z["svi_s"]),
        config=json.loads(str(z["config_json"])),
        provenance=str(p),
    )


@dataclasses.dataclass
class GaussianInit:
    loc: np.ndarray                # (dim,) float64
    cov_reg: np.ndarray            # (dim, dim) floored covariance
    chol: np.ndarray               # (dim, dim) floored Cholesky
    n_floored: int
    z_map: Optional[np.ndarray]
    source: str                    # "init_cache" | "analytic_bundle"
    billed_grads: int


def gaussian_init(target: LensPosterior, seed: int, ledger, timing: dict,
                  bill: bool = True) -> GaussianInit:
    """Uniform SVI-Gaussian init for init-consuming adapters.

    T1 -> data/init_cache (billed: ledger phases init_map/init_svi with the
    cached gradient counts; cached wallclock added to timing["init_cache_s"]).
    T0/analytic -> InitBundle svi_loc/svi_scale_tril (free per protocol).
    """
    from cgl import guards

    if target.tier == "T1":
        c = load_init_cache(target.name, seed)
        cov_reg, chol, n_floored = guards.floor_svi_covariance(c["svi_cov"])
        if bill:
            ledger.add("init_map", n_grad=c["n_grad_map"],
                       note=f"init_cache MAP ({c['config']['n_map']}x"
                            f"{c['config']['map_steps']}), billed per protocol")
            ledger.add("init_svi", n_grad=c["n_grad_svi"],
                       note=f"init_cache SVI ({c['config']['svi_steps']}x"
                            f"{c['config']['n_vi']}), billed per protocol")
            timing["init_cache_s"] = timing.get("init_cache_s", 0.0) \
                + c["map_s"] + c["svi_s"]
        return GaussianInit(loc=c["svi_loc"], cov_reg=cov_reg, chol=chol,
                            n_floored=n_floored, z_map=c["z_map"],
                            source="init_cache",
                            billed_grads=(c["n_grad_map"] + c["n_grad_svi"])
                            if bill else 0)

    init = target.init
    if init.svi_loc is None or init.svi_scale_tril is None:
        raise RuntimeError(
            f"target {target.name} (tier {target.tier}) has no analytic SVI "
            "bundle and no init cache path is defined for this tier")
    chol = np.asarray(init.svi_scale_tril, dtype=np.float64)
    cov = chol @ chol.T
    return GaussianInit(loc=np.asarray(init.svi_loc, dtype=np.float64),
                        cov_reg=cov, chol=chol,
                        n_floored=int(init.svi_n_floored),
                        z_map=None if init.map_z is None
                        else np.asarray(init.map_z, dtype=np.float64),
                        source="analytic_bundle", billed_grads=0)


def draw_gaussian_starts(ginit: GaussianInit, key, n: int, fdtype):
    """(n, dim) starts from the floored SVI q (S0 deviation-5 convention)."""
    import jax
    import jax.numpy as jnp

    dim = ginit.loc.shape[0]
    eps = jax.random.normal(key, (n, dim), dtype=fdtype)
    return (jnp.asarray(ginit.loc, dtype=fdtype)[None, :]
            + eps @ jnp.asarray(ginit.chol, dtype=fdtype).T)


# --------------------------------------------------------------------------- #
# ChEES-HMC runner (adapted WITH ATTRIBUTION from baseline_gigalens._run_hmc;
# takes a plain batched log-prob callable so it can run in flow space)
# --------------------------------------------------------------------------- #
def run_chees_hmc(log_prob_batch: Callable, start, momentum_distribution,
                  key, fdtype, n_burn: int, n_keep: int, *, eps0=0.3,
                  init_l=3, max_leapfrog=30, use_gbtla=True,
                  target_accept=None, adapt_frac=0.8, num_leapfrog=16):
    """Batched single-device PHMC (+ChEES+DualAveraging). Returns
    (samples_all (burn+keep, C, dim) np, diagnostics dict, n_grad, exact)."""
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    from cgl.samplers.baseline_gigalens import _trace_fn_factory

    tfe = tfp.experimental
    n_burn, n_keep = int(n_burn), int(n_keep)
    num_adapt = int(adapt_frac * n_burn)
    n_chains = int(start.shape[0])
    log_prob_batch(start)          # eager batch-shape warmup (zoo contract)

    kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_batch,
        momentum_distribution=momentum_distribution,
        step_size=jnp.asarray(eps0, dtype=fdtype),
        num_leapfrog_steps=int(init_l) if use_gbtla else int(num_leapfrog),
    )
    if use_gbtla:
        kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
            kernel, num_adaptation_steps=num_adapt,
            max_leapfrog_steps=int(max_leapfrog),
        )
    da_kwargs = dict(num_adaptation_steps=num_adapt)
    if target_accept is not None:
        da_kwargs["target_accept_prob"] = jnp.asarray(target_accept,
                                                      dtype=fdtype)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, **da_kwargs)

    trace_fn = _trace_fn_factory(use_gbtla)

    @jax.jit
    def run_chain(seed_key, start):
        return tfp.mcmc.sample_chain(
            num_results=n_burn + n_keep, num_burnin_steps=0,
            current_state=start, kernel=kernel, trace_fn=trace_fn,
            seed=seed_key)

    samples, trace = run_chain(key, start)
    samples = jax.block_until_ready(samples)
    samples = np.asarray(samples)
    if "num_leapfrog_steps" in trace:
        leapfrog = np.asarray(trace["num_leapfrog_steps"]).astype(np.int64)
        exact = True
    else:
        leapfrog = np.full(n_burn + n_keep,
                           int(init_l) if use_gbtla else int(num_leapfrog),
                           dtype=np.int64)
        exact = False
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
    return samples, diagnostics, n_grad, exact


# --------------------------------------------------------------------------- #
# adaptive tempered SMC driver (blackjax; shared by bj_smc and glnt)
# --------------------------------------------------------------------------- #
def run_adaptive_tempered_smc(logprior_pt: Callable, loglike_pt: Callable,
                              particles, key, *, target_ess: float,
                              num_mcmc_steps: int, hmc_step_size: float,
                              inverse_mass_matrix, hmc_num_integration_steps: int,
                              max_lambda_steps: int = 200):
    """Run blackjax adaptive_tempered_smc (HMC inner kernel, systematic
    resampling) until the tempering parameter reaches 1.

    logprior_pt/loglike_pt: SINGLE-POINT (dim,) -> scalar closures (blackjax
    vmaps them over particles internally).

    Returns dict(particles, weights, log_evidence, lambdas, n_steps, n_grad,
    n_logp). Gradient count: per lambda-step, every particle pays
    num_mcmc_steps * hmc_num_integration_steps leapfrog gradients + 1 hmc
    (re)init gradient; weights cost 1 loglike eval per particle per step.
    """
    import blackjax
    import blackjax.smc.resampling as resampling
    import jax
    import jax.numpy as jnp
    from blackjax.smc import extend_params

    n_particles = int(particles.shape[0])
    hmc_parameters = extend_params(dict(
        step_size=hmc_step_size,
        inverse_mass_matrix=jnp.asarray(inverse_mass_matrix,
                                        dtype=particles.dtype),
        num_integration_steps=int(hmc_num_integration_steps),
    ))
    alg = blackjax.adaptive_tempered_smc(
        logprior_pt, loglike_pt,
        blackjax.hmc.build_kernel(), blackjax.hmc.init, hmc_parameters,
        resampling.systematic, target_ess=float(target_ess),
        num_mcmc_steps=int(num_mcmc_steps),
    )
    state = alg.init(particles)
    step = jax.jit(alg.step)
    log_evidence = 0.0
    lambdas = []
    n_steps = 0
    while float(state.tempering_param) < 1.0 and n_steps < max_lambda_steps:
        key, sub = jax.random.split(key)
        state, info = step(sub, state)
        log_evidence += float(info.log_likelihood_increment)
        lambdas.append(float(state.tempering_param))
        n_steps += 1
    if float(state.tempering_param) < 1.0:
        raise RuntimeError(
            f"tempered SMC did not reach lambda=1 in {max_lambda_steps} steps "
            f"(lambda={float(state.tempering_param):.4f}); raise target_ess "
            "or max_lambda_steps")
    per_step_grad = n_particles * (num_mcmc_steps * hmc_num_integration_steps
                                   + 1)
    return dict(
        particles=np.asarray(state.particles),
        weights=np.asarray(state.weights, dtype=np.float64),
        log_evidence=float(log_evidence),
        lambdas=lambdas, n_steps=n_steps,
        n_grad=per_step_grad * n_steps,
        n_logp=n_particles * n_steps,
    )


def weight_ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def particles_to_chains(particles: np.ndarray, n_pseudo_chains: int,
                        rng: np.random.Generator) -> np.ndarray:
    """(N, dim) exchangeable particles -> (T, C, dim) SHUFFLED pseudo-chains.

    The shuffle prevents systematic-resampling duplicate runs from faking
    between-chain structure. R-hat/autocorr-ESS on the result is a WEAK
    diagnostic (exchangeable draws); the particle methods override the
    efficiency ESS with the importance-weight ESS.
    """
    p = np.asarray(particles)
    if p.shape[0] < 1:
        raise RuntimeError("particles_to_chains: empty particle set")
    # degenerate posteriors (e.g. nautilus on cond-1e14 returning < C
    # equal-weight points) fall back to fewer pseudo-chains rather than
    # producing an empty (T=0) sample array (which crashes the uniform
    # mode metrics downstream)
    n_pseudo_chains = max(1, min(int(n_pseudo_chains), p.shape[0]))
    n = (p.shape[0] // n_pseudo_chains) * n_pseudo_chains
    idx = rng.permutation(p.shape[0])[:n]
    return p[idx].reshape(n // n_pseudo_chains, n_pseudo_chains, p.shape[1])


# --------------------------------------------------------------------------- #
# result assembly
# --------------------------------------------------------------------------- #
def device_tag() -> str:
    """Honest device name. jax device_kind mislabels every phoenix GPU as
    'NVIDIA L4' (CAMPAIGN.md P0); with CUDA_DEVICE_ORDER=PCI_BUS_ID indices
    0-7 are A16s and 8-9 are L4s."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip()
    if first.isdigit():
        return "NVIDIA A16" if int(first) <= 7 else "NVIDIA L4"
    return "unknown"


def finalize_cell(*, sampler: str, target: LensPosterior, seed: int,
                  budget: dict, config: dict, samples: np.ndarray,
                  diagnostics: dict, ledger, timing: dict, freeze_check: dict,
                  extras: dict, acceptance_rate: float = float("nan"),
                  final_step_size: float = float("nan"),
                  ess_override: Optional[dict] = None,
                  notes: str = "") -> CellResult:
    """Uniform CellResult assembly (mirrors the validated S0 tail).

    ess_override: dict(source=str, ess=float) replaces the autocorrelation
    ESS in the EFFICIENCY numbers (particle methods); the rank diagnostics
    are still computed and stored for reference.
    """
    timing["total_s"] = sum(v for k, v in timing.items() if k.endswith("_s")
                            and k != "total_s")
    hardware = env_info()
    hardware["device_tag"] = device_tag()
    diag_summary = metrics.rank_diagnostics(samples, target.labels,
                                            target.mass_labels)
    eff = metrics.efficiency(diag_summary, ledger.n_grad, ledger.n_logp,
                             timing["total_s"], hardware=hardware["device_tag"])
    if ess_override is not None:
        ess = float(ess_override["ess"])
        eff["ess_source"] = ess_override["source"]
        eff["ess_autocorr_mass_min"] = eff["ess_mass_min"]
        eff["ess_mass_min"] = ess
        eff["ess_all_min"] = ess
        eff["ess_per_grad_mass"] = ess / max(ledger.n_grad, 1)
        eff["ess_per_grad_all"] = ess / max(ledger.n_grad, 1)
        eff["ess_per_sec_mass"] = ess / max(timing["total_s"], 1e-9)
        eff["ess_per_sec_all"] = ess / max(timing["total_s"], 1e-9)
    else:
        eff["ess_source"] = "rank_normalized_bulk"

    result_metrics = dict(
        diagnostics=diag_summary,
        efficiency=eff,
        acceptance_rate=float(acceptance_rate),
        final_step_size=float(final_step_size),
        extras=extras,
    )
    return CellResult(
        sampler=sampler, target=target.name, seed=int(seed),
        track=str(budget.get("track", "A")),
        samples=np.asarray(samples).astype(np_dtype(target.dtype)),
        labels=list(target.labels), mass_labels=list(target.mass_labels),
        diagnostics=diagnostics,
        budget={**ledger.as_dict(),
                "requested": {k: v for k, v in budget.items()}},
        timing=timing, config=config, env=hardware,
        freeze_check=freeze_check, metrics=result_metrics, notes=notes,
    )


# --------------------------------------------------------------------------- #
# frozen-policy hashing (22_run_cell eval-split assertion; unit-tested on CPU)
# --------------------------------------------------------------------------- #
def policy_sha(config: dict, budget: Optional[dict] = None) -> str:
    """sha256 of the canonical policy blob. With budget=None hashes the
    config alone (Track-B assertion: config frozen, budget scaled 4x)."""
    if budget is None:
        return config_hash({"config": config})
    return config_hash({"config": config, "budget": budget})


def load_frozen_policies(path=POLICIES_FROZEN) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; the policy freeze has not been written yet")
    return json.loads(path.read_text())


def frozen_policy_for(policies: dict, sampler: str, tier: str) -> dict:
    try:
        return policies["methods"][sampler][tier]
    except KeyError:
        raise KeyError(
            f"no frozen policy for sampler={sampler!r} tier={tier!r} in "
            f"{sorted(policies.get('methods', {}))}")


def assert_frozen_policy(policies: dict, sampler: str, tier: str,
                         config: dict, budget: dict, track: str) -> dict:
    """Raise unless (config, budget) matches the frozen policy.

    Track A: full (config+budget) hash must match sha_policy.
    Track B: config hash must match sha_config (budget is the frozen one
    scaled 4x on the adapter's SCALE_KEYS, per the pre-registered protocol).
    """
    entry = frozen_policy_for(policies, sampler, tier)
    if track == "B":
        got = policy_sha(config, None)
        want = entry["sha_config"]
        kind = "config (Track B)"
    else:
        clean_budget = {k: v for k, v in budget.items()
                        if k not in ("track", "n_grad_budget")}
        got = policy_sha(config, clean_budget)
        want = entry["sha_policy"]
        kind = "config+budget (Track A)"
    if got != want:
        raise RuntimeError(
            f"FROZEN-POLICY VIOLATION for {sampler}/{tier}: {kind} hash "
            f"{got[:16]} != frozen {want[:16]}. Eval cells must run the "
            f"frozen policy exactly (data/policies_frozen.json).")
    return entry
