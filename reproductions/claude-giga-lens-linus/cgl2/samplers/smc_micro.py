"""MC-SMC v0: tempered SMC with microcanonical mutation kernels (P0 task #13).

Mutation kernels (kernel= argument of run_smc):

  'hmc'   : baseline -- blackjax HMC, velocity-verlet, the PROVEN claude-giga-lens
            S4 mutation constants (8 leapfrog steps per draw, 4 draws per lambda
            stage, incremental-ESS target 0.7, systematic resampling; see
            ../claude-giga-lens/cgl/samplers/bj_smc.py DEFAULT_CONFIG). One
            DOCUMENTED deviation from the proven config: the fixed step size 0.15 /
            identity mass in raw z-space is replaced by the shared per-lambda
            tuning below, because all kernels run in per-lambda WHITENED
            coordinates here (that is precisely what P2's S3 arm needs: the same
            tuning frame across kernels isolates the kernel question).
  'mams'  : PRIMARY -- blackjax.adjusted_mclmc (Metropolis-Adjusted Microcanonical
            Sampler; Robnik, Cohn-Gordon & Seljak). Metropolis-adjusted, so every
            tempered target pi_lambda is EXACTLY invariant => unbiased evidence
            increments.
  'mclmc' : unadjusted MCLMC, DIAGNOSTIC-ONLY behind the pre-registered B0 bias
            screen (README §verification: MCLMC-vs-MAMS delta-logZ > 3 sigma =>
            "MCLMC demoted" to cost-frontier-only).

Per-lambda tuning -- FROZEN at P0 (plans/PLAN.md §4 "MC-SMC design"):
  * mass matrix = regularized weighted particle covariance
    (cgl2.samplers.common.regularize_cov -- Stan-window shrinkage COPIED with
    attribution from vendor/gigalens-linus/src/gigalens/jax/experimental/mams.py's
    `_regularize_cov`; their experimental modules are never imported), applied as
    exact affine whitening by the driver;
  * step size: short dual-averaging pilot on a particle subset --
    target acceptance 0.90 for MAMS (the MAMS paper's value for the 2nd-order
    McLachlan integrator, same as their reference implementation), 0.80 for HMC
    (Stan default); for MCLMC an EEVPD pilot (energy-variance-per-dimension
    target 5e-4, blackjax's default setpoint) with multiplicative ^(1/4) updates;
  * trajectory length L proportional to sqrt(tr(Sigma_ensemble)): in whitened
    coordinates tr(Sigma) ~= dim, so L = l_factor * sqrt(dim) with l_factor=1.0
    (matches the vendored MAMS reference init_L = sqrt(dim));
    n_int = round(L / eps) snapped to a discrete ladder (bounds jit recompiles);
  * mutation steps per lambda stage: 4 kernel draws (frozen inside the 4-6 band;
    = the proven S4 num_mcmc_steps).

Gradient accounting (common.GRADS_PER_STEP conventions): velocity-verlet 1/step,
isokinetic mclachlan 2/step, +1 per (re)init; pilot draws billed identically.

API (task #13 deliverable):
  run_smc(logprior_fn, loglik_fn, z0_particles, kernel='mams', n_particles=None,
          seed=0, ...) -> dict(particles, logZ, logZ_boot_sigma, lambda_schedule,
                               ess_trace, unique_particle_trace, grad_evals, ...)
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from cgl2.samplers import common
from cgl2.samplers.common import MutationKernel, StageTuning

# FROZEN P0 constants (see module docstring)
NUM_MCMC_STEPS = 4          # kernel draws per lambda stage (4-6 band; proven S4 value)
TARGET_ESS = 0.7            # incremental-weight ESS target (proven S4 value)
L_FACTOR = 1.0              # trajectory length = L_FACTOR * sqrt(dim) (whitened)
HMC_N_INT = 8               # proven S4 leapfrog steps per draw
HMC_DA_TARGET = 0.80        # Stan default
MAMS_DA_TARGET = 0.90       # MAMS paper value for the McLachlan integrator
MCLMC_EEVPD = 5e-4          # blackjax desired_energy_var default
PILOT_ITERS = 10            # dual-averaging pilot iterations (per lambda stage)
MCLMC_PILOT_ITERS = 6
MAX_N_INT = 64
# discrete n_int ladder: bounds the number of distinct jit compilations per run
_N_INT_LADDER = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _snap_ladder(x: float) -> int:
    x = float(np.clip(x, 1, MAX_N_INT))
    return min(_N_INT_LADDER, key=lambda v: abs(v - x))


# --------------------------------------------------------------------------- #
# kernel adapters
# --------------------------------------------------------------------------- #
class _HMCKernel(MutationKernel):
    """Baseline HMC mutation (velocity-verlet, identity mass in whitened space)."""

    name = "hmc"
    grads_per_step = common.GRADS_PER_STEP["velocity_verlet"]

    def __init__(self, num_mcmc_steps: int = NUM_MCMC_STEPS,
                 da_target: float = HMC_DA_TARGET,
                 pilot_iters: int = PILOT_ITERS):
        self.num_mcmc_steps = int(num_mcmc_steps)
        self.da_target = float(da_target)
        self.pilot_iters = int(pilot_iters)
        self._cache = {}

    # ---- jitted mutate factory ---------------------------------------------------
    def build(self, logprior_fn, loglik_fn, dim):
        self.logprior_fn, self.loglik_fn, self.dim = logprior_fn, loglik_fn, dim
        self._cache.clear()

    def _get_mutate(self, n_int: int, n_mcmc: int):
        key = (n_int, n_mcmc)
        if key in self._cache:
            return self._cache[key]
        import blackjax
        import jax
        import jax.numpy as jnp

        logprior_fn, loglik_fn, dim = self.logprior_fn, self.loglik_fn, self.dim
        kern = blackjax.hmc.build_kernel()
        imm = jnp.ones(dim)

        def mutate(rng, U, lam, mu, chol, eps):
            def logdens(u):
                z = mu + chol @ u
                return logprior_fn(z) + lam * loglik_fn(z)

            def one(u, k):
                st = blackjax.hmc.init(u, logdens)

                def body(st, kk):
                    st2, info = kern(kk, st, logdens, eps, imm, n_int)
                    return st2, info.acceptance_rate

                st, accs = jax.lax.scan(body, st, jax.random.split(k, n_mcmc))
                return st.position, jnp.mean(accs)

            keys = jax.random.split(rng, U.shape[0])
            U_new, accs = jax.vmap(one)(U, keys)
            return U_new, jnp.mean(accs)

        fn = jax.jit(mutate)
        self._cache[key] = fn
        return fn

    def _n_int(self, eps: float) -> int:
        # proven S4 constant; NOT scaled with eps (frozen baseline behavior)
        return HMC_N_INT

    def tune(self, key, lam, mu, chol, U_pilot, eps0) -> StageTuning:
        import jax
        da = common.DualAveraging(max(eps0, 1e-6), self.da_target)
        n_sub = int(U_pilot.shape[0])
        n_grad = 0
        for _ in range(self.pilot_iters):
            n_int = self._n_int(da.eps)
            step1 = self._get_mutate(n_int, 1)
            key, k = jax.random.split(key)
            _, acc = step1(k, U_pilot, lam, mu, chol, da.eps)
            da.update(float(acc))
            n_grad += n_sub * (1 + n_int * self.grads_per_step)
        eps = da.eps_final
        return StageTuning(step_size=eps, n_int=self._n_int(eps),
                           n_grad=n_grad,
                           info=dict(da_target=self.da_target))

    def mutate(self, key, lam, mu, chol, U, tuning: StageTuning):
        fn = self._get_mutate(tuning.n_int, self.num_mcmc_steps)
        U_new, acc = fn(key, U, lam, mu, chol, tuning.step_size)
        n = int(U.shape[0])
        n_grad = n * (1 + self.num_mcmc_steps * tuning.n_int
                      * self.grads_per_step)
        return U_new, dict(accept_mean=float(acc), n_grad=n_grad, n_logp=0)


class _MAMSKernel(_HMCKernel):
    """PRIMARY: Metropolis-adjusted microcanonical mutation
    (blackjax.adjusted_mclmc, isokinetic McLachlan integrator).

    Metropolis adjustment makes each tempered target exactly invariant, so the
    SMC evidence increments are unbiased -- the reason this kernel (not raw
    MCLMC) is the primary (plans/PLAN.md §4).
    """

    name = "mams"
    grads_per_step = common.GRADS_PER_STEP["isokinetic_mclachlan"]

    def __init__(self, num_mcmc_steps: int = NUM_MCMC_STEPS,
                 da_target: float = MAMS_DA_TARGET,
                 pilot_iters: int = PILOT_ITERS,
                 l_factor: float = L_FACTOR):
        super().__init__(num_mcmc_steps, da_target, pilot_iters)
        self.l_factor = float(l_factor)

    def _n_int(self, eps: float) -> int:
        L_traj = self.l_factor * float(np.sqrt(self.dim))
        return _snap_ladder(L_traj / max(eps, 1e-6))

    def _get_mutate(self, n_int: int, n_mcmc: int):
        key = (n_int, n_mcmc)
        if key in self._cache:
            return self._cache[key]
        import blackjax
        import jax
        import jax.numpy as jnp
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        logprior_fn, loglik_fn = self.logprior_fn, self.loglik_fn

        def mutate(rng, U, lam, mu, chol, eps):
            def logdens(u):
                z = mu + chol @ u
                return logprior_fn(z) + lam * loglik_fn(z)

            kern = blackjax.adjusted_mclmc.build_kernel(
                logdens, integrator=isokinetic_mclachlan,
                inverse_mass_matrix=1.0)

            def one(u, k):
                st = blackjax.adjusted_mclmc.init(u, logdens)

                def body(st, kk):
                    st2, info = kern(kk, st, step_size=eps,
                                     num_integration_steps=n_int)
                    return st2, info.acceptance_rate

                st, accs = jax.lax.scan(body, st, jax.random.split(k, n_mcmc))
                return st.position, jnp.mean(accs)

            keys = jax.random.split(rng, U.shape[0])
            U_new, accs = jax.vmap(one)(U, keys)
            return U_new, jnp.mean(accs)

        fn = jax.jit(mutate)
        self._cache[key] = fn
        return fn


class _MCLMCKernel(MutationKernel):
    """DIAGNOSTIC-ONLY: unadjusted MCLMC mutation (blackjax.mclmc).

    Unadjusted dynamics leave pi_lambda only approximately invariant (O(eps^2)
    bias) => the SMC evidence is NOT guaranteed unbiased. Admitted solely under
    the pre-registered B0 bias screen; step size tuned to the EEVPD setpoint
    (blackjax desired_energy_var default), momentum-decoherence length
    L = l_factor * sqrt(dim) in whitened coordinates. Integration cost per
    stage is matched to MAMS (num_mcmc_steps * n_int single steps).
    Non-finite trajectories are reverted to their start point (no MH layer to
    reject them) -- reverts are counted in stats['n_reverted'].
    """

    name = "mclmc"
    grads_per_step = common.GRADS_PER_STEP["isokinetic_mclachlan"]

    def __init__(self, num_mcmc_steps: int = NUM_MCMC_STEPS,
                 eevpd: float = MCLMC_EEVPD,
                 pilot_iters: int = MCLMC_PILOT_ITERS,
                 l_factor: float = L_FACTOR):
        self.num_mcmc_steps = int(num_mcmc_steps)
        self.eevpd = float(eevpd)
        self.pilot_iters = int(pilot_iters)
        self.l_factor = float(l_factor)
        self._cache = {}

    def build(self, logprior_fn, loglik_fn, dim):
        self.logprior_fn, self.loglik_fn, self.dim = logprior_fn, loglik_fn, dim
        self._cache.clear()

    def _n_int(self, eps: float) -> int:
        L_traj = self.l_factor * float(np.sqrt(self.dim))
        return _snap_ladder(L_traj / max(eps, 1e-6))

    def _get_run(self, n_steps: int):
        if n_steps in self._cache:
            return self._cache[n_steps]
        import blackjax
        import jax
        import jax.numpy as jnp
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        logprior_fn, loglik_fn, dim = self.logprior_fn, self.loglik_fn, self.dim
        imm = jnp.ones(dim)

        def run(rng, U, lam, mu, chol, eps, L):
            def logdens(u):
                z = mu + chol @ u
                return logprior_fn(z) + lam * loglik_fn(z)

            kern = blackjax.mclmc.build_kernel(
                logdens, inverse_mass_matrix=imm,
                integrator=isokinetic_mclachlan)

            def one(u, keys):
                st0 = blackjax.mclmc.init(u, logdens, keys[0])

                def body(st, kk):
                    st2, info = kern(kk, st, L=L, step_size=eps)
                    return st2, info.energy_change

                st, dE = jax.lax.scan(body, st0, keys[1:])
                ok = jnp.all(jnp.isfinite(st.position))
                pos = jnp.where(ok, st.position, u)
                return pos, dE, ~ok

            keys = jax.vmap(lambda k: jax.random.split(k, n_steps + 1))(
                jax.random.split(rng, U.shape[0]))
            U_new, dE, reverted = jax.vmap(one)(U, keys)
            return U_new, dE, jnp.sum(reverted)

        fn = jax.jit(run)
        self._cache[n_steps] = fn
        return fn

    def tune(self, key, lam, mu, chol, U_pilot, eps0) -> StageTuning:
        import jax
        eps = max(float(eps0), 1e-6)
        L_traj = self.l_factor * float(np.sqrt(self.dim))
        n_sub = int(U_pilot.shape[0])
        n_pilot_steps = 8
        run = self._get_run(n_pilot_steps)
        n_grad = 0
        for _ in range(self.pilot_iters):
            key, k = jax.random.split(key)
            _, dE, _ = run(k, U_pilot, lam, mu, chol, eps, L_traj)
            evpd = float(np.mean(np.asarray(dE) ** 2)) / self.dim
            # blackjax mclmc-adaptation style multiplicative update, clipped
            ratio = (self.eevpd / max(evpd, 1e-300)) ** 0.25
            eps *= float(np.clip(ratio, 1.0 / 3.0, 3.0))
            n_grad += n_sub * (1 + n_pilot_steps * self.grads_per_step)
        return StageTuning(step_size=eps, n_int=self._n_int(eps),
                           n_grad=n_grad,
                           info=dict(eevpd_target=self.eevpd, L=L_traj))

    def mutate(self, key, lam, mu, chol, U, tuning: StageTuning):
        n_steps = self.num_mcmc_steps * tuning.n_int
        run = self._get_run(n_steps)
        L_traj = float(tuning.info.get("L",
                                       self.l_factor * np.sqrt(self.dim)))
        U_new, dE, n_rev = run(key, U, lam, mu, chol, tuning.step_size, L_traj)
        n = int(U.shape[0])
        n_grad = n * (1 + n_steps * self.grads_per_step)
        evpd = float(np.mean(np.asarray(dE) ** 2)) / self.dim
        # 'accept_mean' has no MH meaning here; report the fraction of finite
        # trajectories so the trace stays comparable in shape.
        return U_new, dict(accept_mean=1.0 - float(n_rev) / n, n_grad=n_grad,
                           n_logp=0, n_reverted=int(n_rev), evpd=evpd)


_KERNELS = {"hmc": _HMCKernel, "mams": _MAMSKernel, "mclmc": _MCLMCKernel}


def make_kernel(name: str, **opts) -> MutationKernel:
    try:
        cls = _KERNELS[name]
    except KeyError:
        raise ValueError(f"unknown mutation kernel {name!r}; "
                         f"choose from {sorted(_KERNELS)}")
    return cls(**opts)


# --------------------------------------------------------------------------- #
# public API (task #13 deliverable)
# --------------------------------------------------------------------------- #
def run_smc(logprior_fn: Callable, loglik_fn: Callable, z0_particles,
            kernel: str = "mams", n_particles: Optional[int] = None,
            seed: int = 0, *, target_ess: float = TARGET_ESS,
            num_mcmc_steps: int = NUM_MCMC_STEPS, max_stages: int = 400,
            n_boot: int = 200, pilot_size: int = 64, eps0: float = 0.5,
            precondition: bool = True, kernel_opts: Optional[dict] = None
            ) -> dict:
    """Tempered SMC from prior particles to the posterior with a microcanonical
    (or baseline HMC) mutation kernel. See module docstring for the frozen
    protocol; see common.run_tempered_smc for the driver contract.

    Args:
      logprior_fn / loglik_fn: single-point (dim,) -> scalar jax closures
        (log_prior INCLUDES the bijector forward log-det-Jacobian).
      z0_particles: (N, dim) prior draws (the lambda=0 ensemble).
      kernel: 'hmc' | 'mams' | 'mclmc'.
      n_particles: optional check/truncation; None uses all rows of z0_particles.
      seed: python int; seeds the jax PRNG chain of the run.

    Returns: dict with keys particles (N, dim), logZ, logZ_boot_sigma,
      lambda_schedule, ess_trace, unique_particle_trace, accept_trace,
      step_size_trace, n_int_trace, grad_evals {tune, mutate, total}, n_logp,
      n_stages, kernel.
    """
    import jax

    z0 = np.asarray(z0_particles, dtype=np.float64)
    if n_particles is not None:
        if z0.shape[0] < int(n_particles):
            raise ValueError(
                f"z0_particles has {z0.shape[0]} rows < n_particles={n_particles}")
        z0 = z0[: int(n_particles)]
    opts = dict(kernel_opts or {})
    opts.setdefault("num_mcmc_steps", num_mcmc_steps)
    kern = make_kernel(kernel, **opts)
    key = jax.random.PRNGKey(int(seed))
    return common.run_tempered_smc(
        logprior_fn, loglik_fn, z0, key, kernel=kern,
        target_ess=target_ess, max_stages=max_stages, n_boot=n_boot,
        pilot_size=pilot_size, eps0=eps0, precondition=precondition,
        boot_seed=20260715 + int(seed))
