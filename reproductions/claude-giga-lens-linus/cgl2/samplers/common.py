"""Shared SMC machinery: the adaptive tempered-SMC driver with a GENERALIZED
mutation-kernel interface (P0 task #13).

Copied with attribution from
/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens/cgl/samplers/common.py
(the validated claude-giga-lens P2b machinery) and from
/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens/cgl/samplers/bj_smc.py
(the proven S4 adaptive tempered-SMC config: adaptive lambda via incremental-weight
ESS target 0.7, SYSTEMATIC resampling, logZ increments, gradient accounting).
The Stan-style covariance regularizer is copied with attribution from the vendored
scene API's kernel reference implementation (see `regularize_cov`).

What changed vs the proven copy (frozen at P0, plans/PLAN.md §4 "MC-SMC design"):

  * The mutation kernel is GENERALIZED: `run_tempered_smc` drives any object with
    the `MutationKernel` interface (cgl2.samplers.smc_micro provides hmc / mams /
    mclmc). The proven blackjax `adaptive_tempered_smc` top-level driver is kept
    verbatim below (`run_adaptive_tempered_smc`) as the reference implementation,
    but the generalized driver re-implements the loop at python level because the
    per-lambda tuning (ensemble mass matrix + step-size pilot) cannot be expressed
    through blackjax's fixed `mcmc_parameters`.
  * Per-lambda ensemble preconditioning: the mass matrix is the REGULARIZED
    WEIGHTED PARTICLE COVARIANCE, implemented as an exact affine whitening
    u = chol^{-1}(z - mu) (mathematically identical to a dense mass matrix; this
    is the "affine per-lambda preconditioning" named in plans/PLAN.md §6 B1).
  * logZ bootstrap: per-stage particle bootstrap over the stored incremental
    log-weights (B reps). LIMITATION (documented, frozen): the bootstrap treats
    stages as independent and ignores the resampling genealogy, so it tends to
    UNDERESTIMATE sigma; gates that compare |logZ - ref| <= 3 sigma_boot are
    therefore conservative. Multi-seed spread is the stronger check (P1 sigma_seed).

Gradient-counting conventions (carried verbatim from the proven copy):
  * velocity-verlet leapfrog: 1 NEW gradient per step;
  * isokinetic mclachlan (MCLMC/MAMS family): 2 NEW gradients per step;
  * blackjax hmc/mclmc/adjusted-mclmc init: 1 gradient per (re)init;
  * weight evaluations: 1 log-likelihood eval per particle per lambda stage
    (counted as n_logp, not n_grad).
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

# gradient cost of one integrator step (NEW gradient evaluations; the
# integrator state carries the current-position gradient in blackjax/TFP)
# Copied with attribution from ../claude-giga-lens/cgl/samplers/common.py
GRADS_PER_STEP = {
    "velocity_verlet": 1,
    "mclachlan": 2,
    "isokinetic_mclachlan": 2,
    "yoshida": 3,
    "omelyan": 4,
}


# --------------------------------------------------------------------------- #
# small numerics
# --------------------------------------------------------------------------- #
def weight_ess(weights: np.ndarray) -> float:
    """Importance-weight effective sample size 1 / sum(w_norm^2).

    Copied with attribution from ../claude-giga-lens/cgl/samplers/common.py.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return float(m + np.log(np.mean(np.exp(x - m))))


def systematic_resample(weights: np.ndarray, u: float) -> np.ndarray:
    """Systematic resampling indices for normalized `weights` given ONE uniform
    draw u in [0,1). Deterministic given (weights, u) -- unit-testable invariant:
    each index i is drawn n_i in {floor(N w_i), ceil(N w_i)} times.

    Same scheme as blackjax.smc.resampling.systematic (the proven S4 config);
    reimplemented in numpy so the python-level driver does not round-trip
    device arrays for the resampling step.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = w.shape[0]
    w = w / w.sum()
    positions = (float(u) + np.arange(n, dtype=np.float64)) / n
    cum = np.cumsum(w)
    cum[-1] = 1.0  # guard rounding
    return np.searchsorted(cum, positions, side="right").astype(np.int64)


def weighted_mean_cov(particles: np.ndarray, weights: np.ndarray):
    """Weighted ensemble mean/covariance (normalized weights)."""
    z = np.asarray(particles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mu = w @ z
    d = z - mu[None, :]
    cov = (d * w[:, None]).T @ d
    return mu, cov


def regularize_cov(cov: np.ndarray, n: float,
                   rel_floor: float = 1e-12) -> np.ndarray:
    """Shrinkage + PSD floor of an ensemble covariance, SCALE-AWARE variant.

    Adapted with attribution from the vendored scene API's kernel reference
    implementation `_regularize_cov` in
    vendor/gigalens-linus/src/gigalens/jax/experimental/mams.py (identical copy in
    mclmc.py) -- copied, NOT imported: their experimental modules import private
    blackjax internals and the old simulator (plans/PLAN.md §4). Unlike their
    opt-in flag, we apply it ALWAYS (frozen P0 choice): the weighted particle
    covariance at small ESS is exactly the ill-estimated case the shrinkage exists
    for. `n` is the effective sample count (we pass the weight ESS).

    DOCUMENTED DEVIATION from their formula (P0 finding, 2026-07-15): their Stan
    window shrinkage adds an ABSOLUTE identity floor `1e-3 * 5/(n+5) * I` and clips
    eigenvalues at that absolute value -- correct at unit scales, but it DESTROYS
    ensemble preconditioning on extreme-anisotropy targets: on t0_illcond46
    (posterior cond 1e14) the floored directions are ~1e9 x too wide, mutation
    stalls, and the adaptive lambda schedule never reaches 1 (measured: stall at
    lambda=0.22 after 200 stages). This variant keeps their n/(n+5) shrinkage
    weight but shrinks toward the DIAGONAL of the ensemble covariance (scale-
    preserving) and floors eigenvalues RELATIVE to the largest one (`rel_floor`,
    frozen 1e-12 -- comfortably above f64 eigh noise, ~100x whitened-scale spread
    left for genuinely floored directions). NOTE for the handoff: their windowed
    MAMS/MCLMC `regularize_mass_matrix=True` path has the same absolute floor and
    would hit the same wall on cond>~1e6 posteriors.
    """
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)                              # symmetrize
    n = float(max(n, 1.0))
    lam = n / (n + 5.0)                                    # their shrink weight
    reg = lam * cov + (1.0 - lam) * np.diag(np.diag(cov))  # scale-preserving
    w, v = np.linalg.eigh(reg)                             # PSD + relative floor
    w = np.clip(w, rel_floor * max(w.max(), 1e-300), None)
    return (v * w[None, :]) @ v.T


def solve_delta_lambda(loglik: np.ndarray, lam: float, target_ess: float,
                       rel_tol: float = 1e-3, max_bisect: int = 2000) -> float:
    """Adaptive tempering step: largest delta <= (1 - lam) whose incremental
    weights w_i = exp(delta * loglik_i) have ESS >= target_ess * N (bisection;
    ESS is monotone decreasing in delta). This is the same adaptive-lambda rule
    as blackjax adaptive_tempered_smc (the proven S4 config), reimplemented at
    python level so the mutation kernel can be re-tuned per stage.

    The bisection runs to RELATIVE convergence (rel_tol on the bracket), not a
    fixed iteration count: an extreme-anisotropy target's first delta can be
    ~1e-16 (loglik spread ~1e16 across prior draws -- t0_illcond46), and any
    absolute floor larger than that silently degenerates the weights to a
    single particle (P0 incident, 2026-07-15: an 1e-10 floor collapsed stage 0
    to unique=1 and produced logZ ~ -8e14).
    """
    ll = np.asarray(loglik, dtype=np.float64)
    n = ll.shape[0]
    target = float(target_ess) * n

    def ess_at(delta):
        x = delta * ll
        x = x - x.max()
        w = np.exp(x)
        return (w.sum() ** 2) / np.sum(w ** 2)

    hi = 1.0 - lam
    if ess_at(hi) >= target:
        return hi
    lo = 0.0
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) >= target:
            lo = mid
        else:
            hi = mid
        if lo > 0 and (hi - lo) <= rel_tol * lo:
            break
    if lo <= 0.0:
        raise RuntimeError(
            "solve_delta_lambda: no delta > 0 reaches the ESS target -- "
            "the incremental log-likelihoods are degenerate (inf spread?)")
    return lo


# --------------------------------------------------------------------------- #
# dual averaging (step-size pilot)
# --------------------------------------------------------------------------- #
class DualAveraging:
    """Nesterov dual averaging to a target acceptance rate (Stan defaults).

    Standard scheme (Hoffman & Gelman 2014 §3.2), reimplemented rather than
    importing blackjax.adaptation internals (the vendored reference implementations
    import private blackjax modules -- copy-logic-never-import rule).
    """

    def __init__(self, eps0: float, target: float, gamma: float = 0.05,
                 t0: float = 10.0, kappa: float = 0.75):
        self.mu = float(np.log(10.0 * eps0))
        self.target = float(target)
        self.gamma, self.t0, self.kappa = gamma, t0, kappa
        self.t = 0
        self.h_bar = 0.0
        self.log_eps = float(np.log(eps0))
        self.log_eps_bar = float(np.log(eps0))

    @property
    def eps(self) -> float:
        return float(np.exp(self.log_eps))

    @property
    def eps_final(self) -> float:
        return float(np.exp(self.log_eps_bar))

    def update(self, accept: float) -> None:
        self.t += 1
        frac = 1.0 / (self.t + self.t0)
        self.h_bar = (1.0 - frac) * self.h_bar + frac * (self.target - accept)
        self.log_eps = self.mu - np.sqrt(self.t) / self.gamma * self.h_bar
        eta = self.t ** (-self.kappa)
        self.log_eps_bar = eta * self.log_eps + (1.0 - eta) * self.log_eps_bar


# --------------------------------------------------------------------------- #
# the generalized mutation-kernel interface
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class StageTuning:
    """Per-lambda tuning result handed back by MutationKernel.tune."""
    step_size: float
    n_int: int                      # integration steps per kernel draw
    n_grad: int = 0                 # gradients spent tuning (billed)
    n_logp: int = 0
    info: dict = dataclasses.field(default_factory=dict)


class MutationKernel:
    """Interface the tempered-SMC driver mutates through (GENERALIZED seam).

    A kernel is built ONCE per run against single-point closures
    logprior_fn / loglik_fn ((dim,) -> scalar; the exact blackjax split: the
    lambda=0 distribution is the z-space prior). The driver whitens per stage
    (u = chol^{-1}(z - mu)) and calls:

      tune(key, lam, mu, chol, U_pilot, eps0) -> StageTuning
      mutate(key, lam, mu, chol, U, tuning)   -> (U_new, stats dict)

    with lam/mu/chol traced arguments of the kernel's jitted internals, so a
    stage never triggers a recompile unless n_int changes (bounded by the
    kernel's discrete n_int ladder). stats must contain 'accept_mean',
    'n_grad', 'n_logp'.
    """

    name: str = "abstract"

    def build(self, logprior_fn: Callable, loglik_fn: Callable, dim: int) -> None:
        raise NotImplementedError

    def tune(self, key, lam, mu, chol, U_pilot, eps0) -> StageTuning:
        raise NotImplementedError

    def mutate(self, key, lam, mu, chol, U, tuning: StageTuning):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# the proven blackjax driver (verbatim reference copy)
# --------------------------------------------------------------------------- #
def run_adaptive_tempered_smc(logprior_pt: Callable, loglike_pt: Callable,
                              particles, key, *, target_ess: float,
                              num_mcmc_steps: int, hmc_step_size: float,
                              inverse_mass_matrix, hmc_num_integration_steps: int,
                              max_lambda_steps: int = 200):
    """Run blackjax adaptive_tempered_smc (HMC inner kernel, systematic
    resampling) until the tempering parameter reaches 1.

    Copied VERBATIM with attribution from
    ../claude-giga-lens/cgl/samplers/common.py::run_adaptive_tempered_smc
    (the validated S4 path). Kept as the reference implementation for the
    generalized driver's bookkeeping unit tests.
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


# --------------------------------------------------------------------------- #
# the generalized adaptive tempered-SMC driver
# --------------------------------------------------------------------------- #
def run_tempered_smc(logprior_fn: Callable, loglik_fn: Callable,
                     z0_particles, key, *, kernel: MutationKernel,
                     target_ess: float = 0.7, max_stages: int = 400,
                     n_boot: int = 200, pilot_size: int = 64,
                     eps0: float = 0.5, precondition: bool = True,
                     boot_seed: int = 20260715) -> dict:
    """Adaptive tempered SMC (prior -> posterior anneal) with a generalized,
    per-lambda-tuned mutation kernel.

    Frozen P0 protocol (plans/PLAN.md §4 "MC-SMC design"; README §verification):
      * adaptive lambda: incremental-weight ESS target `target_ess` (0.7), bisection;
      * SYSTEMATIC resampling every stage;
      * logZ increment per stage m: logmeanexp(delta_m * loglik_i) over the
        pre-resample (equal-weight) particle set;
      * per-lambda tuning: mass matrix = regularized weighted particle covariance
        (affine whitening), step size via short dual-averaging pilot on a particle
        subset, trajectory length proportional to sqrt(tr(Sigma_ensemble))
        (= sqrt(dim) in whitened coordinates) -- all owned by the kernel adapter;
      * mutation: kernel.mutate (4-6 kernel draws per stage, kernel-frozen).

    Args:
      logprior_fn / loglik_fn: SINGLE-POINT (dim,) -> scalar jax closures.
        log_prior must include the bijector forward log-det-Jacobian so the
        lambda=0 distribution is exactly the z-space prior of z0_particles.
      z0_particles: (N, dim) prior draws.
      kernel: a MutationKernel (smc_micro.make_kernel('hmc'|'mams'|'mclmc')).

    Returns dict(particles, logZ, logZ_boot_sigma, lambda_schedule, ess_trace,
      unique_particle_trace, accept_trace, step_size_trace, n_int_trace,
      grad_evals dict, n_logp, n_stages, logZ_increments).
    """
    import jax
    import jax.numpy as jnp

    z = np.asarray(z0_particles, dtype=np.float64)
    n, dim = z.shape
    kernel.build(logprior_fn, loglik_fn, dim)

    loglik_batch = jax.jit(jax.vmap(loglik_fn))

    rng = np.random.default_rng(boot_seed)
    lam = 0.0
    logz = 0.0
    stage_deltas, stage_logliks, increments = [], [], []
    lambdas, ess_trace, uniq_trace = [], [], []
    acc_trace, eps_trace, nint_trace = [], [], []
    grad_evals = dict(tune=0, mutate=0)
    n_logp = 0
    eps_warm = float(eps0)
    n_stages = 0

    while lam < 1.0 and n_stages < max_stages:
        # ---- weight step: batched loglik at current particles -------------------
        ll = np.asarray(loglik_batch(jnp.asarray(z)), dtype=np.float64)
        if not np.all(np.isfinite(ll)):
            bad = int(np.sum(~np.isfinite(ll)))
            raise RuntimeError(
                f"stage {n_stages}: {bad}/{n} non-finite log-likelihoods at the "
                "current particle set -- prior draws outside the likelihood domain?")
        n_logp += n

        delta = solve_delta_lambda(ll, lam, target_ess)
        lam_new = min(1.0, lam + delta)
        inc = logmeanexp(delta * ll)
        logz += inc
        increments.append(inc)
        stage_deltas.append(delta)
        stage_logliks.append(ll.copy())

        logw = delta * ll
        w = np.exp(logw - logw.max())
        w = w / w.sum()
        ess_trace.append(weight_ess(w))

        # ---- per-lambda ensemble preconditioner (weighted, regularized) ---------
        if precondition:
            mu, cov = weighted_mean_cov(z, w)
            cov_reg = regularize_cov(cov, weight_ess(w))
            chol = np.linalg.cholesky(cov_reg)
        else:
            mu = np.zeros(dim)
            chol = np.eye(dim)

        # ---- systematic resample -------------------------------------------------
        key, k_u = jax.random.split(key)
        u_draw = float(jax.random.uniform(k_u))
        idx = systematic_resample(w, u_draw)
        uniq_trace.append(int(len(np.unique(idx))))
        z = z[idx]

        # ---- whiten, tune, mutate, unwhiten ---------------------------------------
        mu_j = jnp.asarray(mu)
        chol_j = jnp.asarray(chol)
        from scipy.linalg import solve_triangular
        U = jnp.asarray(solve_triangular(chol, (z - mu[None, :]).T,
                                         lower=True).T)

        key, k_pilot, k_mut = jax.random.split(key, 3)
        pilot_idx = rng.choice(n, size=min(pilot_size, n), replace=False)
        tuning = kernel.tune(k_pilot, lam_new, mu_j, chol_j, U[pilot_idx],
                             eps_warm)
        grad_evals["tune"] += tuning.n_grad
        n_logp += tuning.n_logp
        eps_warm = tuning.step_size

        U_new, stats = kernel.mutate(k_mut, lam_new, mu_j, chol_j, U, tuning)
        grad_evals["mutate"] += int(stats["n_grad"])
        n_logp += int(stats.get("n_logp", 0))
        z = np.asarray(mu[None, :]
                       + np.asarray(U_new, dtype=np.float64) @ chol.T)

        acc_trace.append(float(stats["accept_mean"]))
        eps_trace.append(float(tuning.step_size))
        nint_trace.append(int(tuning.n_int))
        lam = lam_new
        lambdas.append(lam)
        n_stages += 1

    if lam < 1.0:
        raise RuntimeError(
            f"tempered SMC did not reach lambda=1 in {max_stages} stages "
            f"(lambda={lam:.6f}); raise max_stages or target_ess")

    # ---- particle bootstrap of logZ (documented limitation: genealogy ignored) --
    boots = np.empty(n_boot)
    for b in range(n_boot):
        tot = 0.0
        for delta, ll in zip(stage_deltas, stage_logliks):
            bidx = rng.integers(0, n, n)
            tot += logmeanexp(delta * ll[bidx])
        boots[b] = tot
    logz_boot_sigma = float(np.std(boots))

    grad_evals["total"] = grad_evals["tune"] + grad_evals["mutate"]
    return dict(
        particles=z,
        logZ=float(logz),
        logZ_boot_sigma=logz_boot_sigma,
        logZ_increments=increments,
        lambda_schedule=lambdas,
        ess_trace=ess_trace,
        unique_particle_trace=uniq_trace,
        accept_trace=acc_trace,
        step_size_trace=eps_trace,
        n_int_trace=nint_trace,
        grad_evals=grad_evals,
        n_logp=n_logp,
        n_stages=n_stages,
        kernel=kernel.name,
    )
