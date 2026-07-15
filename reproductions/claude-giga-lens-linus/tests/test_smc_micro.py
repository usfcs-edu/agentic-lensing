"""CPU unit toys for the MC-SMC v0 machinery (P0 task #13).

Pre-registered coverage (task spec): (1) logZ on a 2-dim analytic Gaussian
within 3 sigma_boot for ALL kernels; (2) systematic-resampling invariants;
(3) evidence-increment bookkeeping (+ gradient-ledger consistency).
All CPU (conftest pins JAX_PLATFORMS=cpu, x64 ON) with small particle counts.
"""
import numpy as np
import pytest

from cgl2.samplers import common, smc_micro
from cgl2 import zoo

KERNELS = ("hmc", "mams", "mclmc")


# --------------------------------------------------------------------------- #
# (1) 2-dim analytic Gaussian: logZ within 3 sigma_boot for all kernels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", KERNELS)
def test_gauss2_logz(kernel):
    import jax

    t = zoo.build_gauss2()
    z0 = np.asarray(t.prior_sample_fn(jax.random.PRNGKey(7), 256))
    out = smc_micro.run_smc(t.logprior_fn, t.loglik_fn, z0, kernel=kernel,
                            seed=3, n_boot=200, pilot_size=32)
    err = abs(out["logZ"] - t.reference["logZ"])
    sig = out["logZ_boot_sigma"]
    assert sig > 0
    assert err <= 3 * sig, (
        f"{kernel}: |logZ-analytic|={err:.4f} > 3*sigma_boot={3 * sig:.4f}")
    # posterior mean sanity (loose, 6 se)
    mean = out["particles"].mean(0)
    se = np.sqrt(np.asarray(t.reference["var"])) / np.sqrt(64.0)
    assert np.all(np.abs(mean - np.asarray(t.reference["mean"])) < 6 * se + 0.1)
    # lambda schedule reaches exactly 1
    assert out["lambda_schedule"][-1] == 1.0
    # traces all have one entry per stage
    n = out["n_stages"]
    for k in ("lambda_schedule", "ess_trace", "unique_particle_trace",
              "accept_trace", "step_size_trace", "n_int_trace",
              "logZ_increments"):
        assert len(out[k]) == n, k


# --------------------------------------------------------------------------- #
# (2) systematic resampling invariants
# --------------------------------------------------------------------------- #
def test_systematic_resample_invariants():
    rng = np.random.default_rng(0)
    n = 400
    w = rng.exponential(size=n)
    w /= w.sum()
    for u in (0.0, 0.25, 0.999):
        idx = common.systematic_resample(w, u)
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n
        counts = np.bincount(idx, minlength=n)
        # systematic property: every index drawn floor(Nw) or ceil(Nw) times
        # (1e-9 slop: cumsum rounding at exact stratum boundaries)
        lo = np.floor(n * w - 1e-9).astype(int)
        hi = np.ceil(n * w + 1e-9).astype(int)
        assert np.all(counts >= lo) and np.all(counts <= hi)
    # uniform weights + interior u -> exactly one draw per index (identity
    # counts; u=0.5 keeps strata strictly away from rounding boundaries)
    idx = common.systematic_resample(np.full(n, 1.0 / n), 0.5)
    assert np.array_equal(np.bincount(idx, minlength=n), np.ones(n, dtype=int))


def test_weight_ess():
    assert common.weight_ess(np.ones(50)) == pytest.approx(50.0)
    w = np.zeros(50)
    w[0] = 1.0
    assert common.weight_ess(w) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# (3) evidence-increment bookkeeping
# --------------------------------------------------------------------------- #
def test_constant_likelihood_single_stage_exact_logz():
    """lik == const c: ESS never drops => ONE stage, logZ == c exactly, and the
    increments sum to logZ (bookkeeping identity)."""
    import jax
    import jax.numpy as jnp

    c = -3.7
    t = zoo.build_gauss2()

    def loglik_const(z):
        return jnp.asarray(c, dtype=jnp.float64) + 0.0 * jnp.sum(z)

    z0 = np.asarray(t.prior_sample_fn(jax.random.PRNGKey(1), 128))
    out = smc_micro.run_smc(t.logprior_fn, loglik_const, z0, kernel="hmc",
                            seed=0, n_boot=50, pilot_size=16)
    assert out["n_stages"] == 1
    assert out["logZ"] == pytest.approx(c, abs=1e-12)
    assert sum(out["logZ_increments"]) == pytest.approx(out["logZ"], abs=1e-12)
    assert out["logZ_boot_sigma"] == pytest.approx(0.0, abs=1e-12)


def test_increments_sum_and_grad_ledger():
    """Increments sum to logZ; the gradient ledger matches the closed-form count
    for the hmc kernel (per stage: pilot n_sub*(1+8) + mutate N*(1+4*8))."""
    import jax

    t = zoo.build_gauss2()
    n, n_sub = 128, 16
    z0 = np.asarray(t.prior_sample_fn(jax.random.PRNGKey(2), n))
    out = smc_micro.run_smc(t.logprior_fn, t.loglik_fn, z0, kernel="hmc",
                            seed=1, n_boot=50, pilot_size=n_sub)
    assert sum(out["logZ_increments"]) == pytest.approx(out["logZ"], rel=1e-12)
    st = out["n_stages"]
    pilot = st * smc_micro.PILOT_ITERS * n_sub * (1 + smc_micro.HMC_N_INT)
    mutate = st * n * (1 + smc_micro.NUM_MCMC_STEPS * smc_micro.HMC_N_INT)
    assert out["grad_evals"]["tune"] == pilot
    assert out["grad_evals"]["mutate"] == mutate
    assert out["grad_evals"]["total"] == pilot + mutate
    # weight evals: one batched loglik per stage
    assert out["n_logp"] == st * n


def test_solve_delta_lambda_monotone_and_capped():
    ll = np.linspace(-50, 0, 200)
    d1 = common.solve_delta_lambda(ll, 0.0, 0.7)
    assert 0 < d1 <= 1.0
    # near lambda=1 the step is capped at the remainder
    d2 = common.solve_delta_lambda(0.001 * ll, 0.9, 0.7)
    assert d2 == pytest.approx(0.1)
    # flat likelihood: full jump
    assert common.solve_delta_lambda(np.zeros(64), 0.0, 0.7) == 1.0


def test_regularize_cov_psd_and_symmetric():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T + np.diag([1e-18, 1, 2, 3, 4])  # near-singular direction
    reg = common.regularize_cov(cov, n=10.0)
    assert np.allclose(reg, reg.T)
    w = np.linalg.eigvalsh(reg)
    assert w.min() > 0  # PSD floor engaged


def test_dual_averaging_converges():
    """DA on a synthetic acceptance curve a(eps) = exp(-eps) converges near the
    eps* with a(eps*) = target."""
    da = common.DualAveraging(1.0, target=0.8)
    for _ in range(60):
        da.update(float(np.exp(-da.eps)))
    assert abs(np.exp(-da.eps_final) - 0.8) < 0.05


# --------------------------------------------------------------------------- #
# T0 target self-consistency (analytic references)
# --------------------------------------------------------------------------- #
def test_t0_targets_logprob_split_finite():
    import jax

    for name in zoo.T0_NAMES:
        t = zoo.build(name)
        z = np.asarray(t.prior_sample_fn(jax.random.PRNGKey(0), 8))
        lp = np.asarray(t.logprior_batch(z))
        ll = np.asarray(t.loglik_batch(z))
        assert np.all(np.isfinite(lp)) and np.all(np.isfinite(ll)), name
        # single-point closures agree with the batched surface
        lp0 = float(t.logprior_fn(z[0]))
        ll0 = float(t.loglik_fn(z[0]))
        assert lp0 == pytest.approx(float(lp[0]), rel=1e-12)
        assert ll0 == pytest.approx(float(ll[0]), rel=1e-12)


def test_mix2_analytics_match_quadrature():
    """Independent check of the mix2 analytic logZ by 2-D grid quadrature."""
    t = zoo.build_mix2()
    g = np.linspace(-8, 8, 401)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    Z = np.stack([xx.ravel(), yy.ravel()], axis=1)
    lp = np.asarray(t.logprior_batch(Z)) + np.asarray(t.loglik_batch(Z))
    dz = (g[1] - g[0]) ** 2
    logz_quad = float(np.log(np.exp(lp - lp.max()).sum() * dz) + lp.max())
    assert logz_quad == pytest.approx(t.reference["logZ"], abs=1e-6)
