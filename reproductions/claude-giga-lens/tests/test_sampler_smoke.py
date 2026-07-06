"""GPU smoke tests: prove each P2 sampler library RUNS on the pinned stack.

Runtime-compat gate, NOT a statistics test. Each test uses a tiny 2-D target
(Neal funnel: theta ~ N(0, 3), x ~ N(0, exp(theta/2)); or a Gaussian where
simpler) with tiny sample counts, and asserts only basic sanity: finite
samples, expected shapes, and the odd very loose moment check.

Pinned stack under test: jax 0.6.2 (cuda12), tfp 0.25.0 (jax substrate),
blackjax 1.3, flowMC 0.4.5, flowjax 19.0.0, nautilus-sampler 1.0.6.

Run with:
    CGL_TEST_GPU=1 GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=9 \
        python -m pytest tests/test_sampler_smoke.py -m gpu -q

KNOWN STACK DEFECT (worked around below, do not remove silently):
jaxlib 0.6.2's XLA `priority-fusion` pass livelocks (infinite compile,
~600 spinning threads) when fusing float64 `jax.random.normal` (erf_inv
expansion) with a reduction on this platform (NVIDIA L4, linux-aarch64).
Minimal repro: jit(lambda k, m: (m + normal(k, m.shape, f64)) / norm(m + z)).
This is exactly the `partially_refresh_momentum` pattern in blackjax MCLMC,
and potentially any f64 sampler kernel. Workaround: disable the pass via
XLA_FLAGS. Must be set before the first JAX computation in the process,
which is why it is done at module import here.
"""

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("CGL_TEST_GPU") != "1",
        reason="GPU smoke tests require CGL_TEST_GPU=1 (conftest forces CPU otherwise)",
    ),
]

# --- XLA priority-fusion livelock workaround (see module docstring) ---------
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_disable_hlo_passes" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_disable_hlo_passes=priority-fusion"
    ).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def funnel_logdensity(v):
    """Neal funnel, 2-D: theta ~ N(0, 3); x ~ N(0, exp(theta/2)).

    Indexes the last axis so it broadcasts over leading batch dimensions --
    TFP's ReplicaExchangeMC evaluates the target on a stacked (n_replica, 2)
    state, while the blackjax/flowMC callers pass an unbatched (2,) vector.
    """
    theta, x = v[..., 0], v[..., 1]
    logp_theta = jax.scipy.stats.norm.logpdf(theta, loc=0.0, scale=3.0)
    logp_x = jax.scipy.stats.norm.logpdf(x, loc=0.0, scale=jnp.exp(theta / 2.0))
    return logp_theta + logp_x


# ---------------------------------------------------------------------------
# 1. blackjax NUTS + window_adaptation
# ---------------------------------------------------------------------------


def test_blackjax_nuts_window_adaptation():
    import blackjax

    warm_key, sample_key = jax.random.split(jax.random.PRNGKey(0))

    warmup = blackjax.window_adaptation(
        blackjax.nuts, funnel_logdensity, progress_bar=False
    )
    (last_state, parameters), _ = warmup.run(
        warm_key, jnp.array([0.5, 0.1]), num_steps=200
    )
    assert np.isfinite(float(parameters["step_size"]))

    algo = blackjax.nuts(funnel_logdensity, **parameters)

    def one_step(state, key):
        state, info = algo.step(key, state)
        return state, (state.position, info.acceptance_rate)

    keys = jax.random.split(sample_key, 300)
    _, (positions, accept) = jax.lax.scan(one_step, last_state, keys)

    positions = np.asarray(positions)
    assert positions.shape == (300, 2)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(np.asarray(accept)))
    # theta ~ N(0, 3): very loose location check.
    assert abs(positions[:, 0].mean()) < 3.0


# ---------------------------------------------------------------------------
# 2. blackjax unadjusted MCLMC (+ mclmc_find_L_and_step_size) and adjusted MCLMC
# ---------------------------------------------------------------------------


def test_blackjax_mclmc_unadjusted_with_tuning():
    import blackjax
    from blackjax.mcmc.integrators import isokinetic_mclachlan

    init_key, tune_key, run_key = jax.random.split(jax.random.PRNGKey(1), 3)

    initial_state = blackjax.mcmc.mclmc.init(
        jnp.array([0.5, 0.1]), funnel_logdensity, init_key
    )

    def kernel(inverse_mass_matrix):
        return blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=funnel_logdensity,
            integrator=isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix,
        )

    state, params, _n_tune = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=400,
        state=initial_state,
        rng_key=tune_key,
    )
    assert np.isfinite(float(params.L))
    assert np.isfinite(float(params.step_size))

    algo = blackjax.mclmc(
        funnel_logdensity,
        L=params.L,
        step_size=params.step_size,
        inverse_mass_matrix=params.inverse_mass_matrix,
    )

    def one_step(state, key):
        state, info = algo.step(key, state)
        return state, state.position

    keys = jax.random.split(run_key, 300)
    _, positions = jax.lax.scan(one_step, state, keys)

    positions = np.asarray(positions)
    assert positions.shape == (300, 2)
    assert np.all(np.isfinite(positions))


def test_blackjax_mclmc_adjusted():
    """Adjusted (Metropolis-corrected) MCLMC, fixed hyperparameters.

    blackjax 1.3 ships blackjax.adjusted_mclmc (static #steps) and
    blackjax.adjusted_mclmc_dynamic; the smoke test runs the static variant
    with fixed L/step_size -- adaptation is exercised in the unadjusted test.
    """
    import blackjax

    run_key = jax.random.PRNGKey(2)

    algo = blackjax.adjusted_mclmc(
        funnel_logdensity,
        step_size=0.3,
        num_integration_steps=8,
    )
    state = algo.init(jnp.array([0.5, 0.1]))

    def one_step(state, key):
        state, info = algo.step(key, state)
        return state, state.position

    keys = jax.random.split(run_key, 300)
    _, positions = jax.lax.scan(one_step, state, keys)

    positions = np.asarray(positions)
    assert positions.shape == (300, 2)
    assert np.all(np.isfinite(positions))


# ---------------------------------------------------------------------------
# 3. blackjax adaptive tempered SMC
# ---------------------------------------------------------------------------


def test_blackjax_adaptive_tempered_smc():
    import blackjax
    import blackjax.smc.resampling as resampling
    from blackjax.smc import extend_params

    dim = 2
    prior_cov = 25.0 * jnp.eye(dim)  # N(0, 5^2 I) prior
    lik_cov = 0.5 * jnp.eye(dim)  # N(1, 0.5 I) likelihood

    def logprior_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, jnp.zeros(dim), prior_cov
        )

    def loglikelihood_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, jnp.ones(dim), lik_cov
        )

    hmc_parameters = extend_params(
        dict(
            step_size=0.2,
            inverse_mass_matrix=jnp.eye(dim),
            num_integration_steps=10,
        )
    )

    smc_alg = blackjax.adaptive_tempered_smc(
        logprior_fn,
        loglikelihood_fn,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        hmc_parameters,
        resampling.systematic,
        target_ess=0.75,
        num_mcmc_steps=5,
    )

    key = jax.random.PRNGKey(3)
    key, init_key = jax.random.split(key)
    n_particles = 300
    particles = jax.random.multivariate_normal(
        init_key, jnp.zeros(dim), prior_cov, (n_particles,)
    )
    state = smc_alg.init(particles)

    step = jax.jit(smc_alg.step)
    log_evidence = 0.0
    n_steps = 0
    while float(state.tempering_param) < 1.0 and n_steps < 100:
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        log_evidence += float(info.log_likelihood_increment)
        n_steps += 1

    assert float(state.tempering_param) >= 1.0, "tempering never reached lmbda=1"
    assert n_steps < 100
    particles_out = np.asarray(state.particles)
    assert particles_out.shape == (n_particles, dim)
    assert np.all(np.isfinite(particles_out))
    assert np.all(np.isfinite(np.asarray(state.weights)))
    assert np.isfinite(log_evidence)


# ---------------------------------------------------------------------------
# 4. TFP-on-JAX ReplicaExchangeMC (HMC inner kernel, 4 replicas)
# ---------------------------------------------------------------------------


def test_tfp_replica_exchange_mc():
    import tensorflow_probability.substrates.jax as tfp

    tfm = tfp.mcmc

    inverse_temperatures = 0.5 ** jnp.arange(4.0)  # geometric ladder

    def make_kernel_fn(target_log_prob_fn):
        return tfm.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=0.3,
            num_leapfrog_steps=3,
        )

    remc = tfm.ReplicaExchangeMC(
        target_log_prob_fn=funnel_logdensity,
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn,
    )

    samples = tfm.sample_chain(
        num_results=300,
        num_burnin_steps=100,
        current_state=jnp.array([0.5, 0.1]),
        kernel=remc,
        trace_fn=None,
        seed=jax.random.PRNGKey(4),
    )

    samples = np.asarray(samples)
    assert samples.shape == (300, 2)
    assert np.all(np.isfinite(samples))
    assert abs(samples[:, 0].mean()) < 3.0


# ---------------------------------------------------------------------------
# 5. flowMC 0.4.5 (RQSpline_MALA resource-strategy bundle)
# ---------------------------------------------------------------------------


def test_flowmc_rqspline_mala_bundle():
    from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
    from flowMC.Sampler import Sampler

    def logpdf(x, data):
        return funnel_logdensity(x)

    n_chains, n_dims = 4, 2
    n_local_steps, n_global_steps = 10, 10
    n_production_loops = 2

    rng_key, bundle_key, init_key = jax.random.split(jax.random.PRNGKey(5), 3)

    bundle = RQSpline_MALA_Bundle(
        bundle_key,
        n_chains,
        n_dims,
        logpdf,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_training_loops=2,
        n_production_loops=n_production_loops,
        n_epochs=2,
        mala_step_size=0.2,
        rq_spline_hidden_units=[16, 16],
        rq_spline_n_bins=4,
        rq_spline_n_layers=2,
        batch_size=64,
        n_max_examples=1000,
        verbose=False,
    )

    # STACK DEFECT WORKAROUND: flowMC 0.4.5's Optimizer resource initializes
    # its optax adamw state on eqx.filter(model, eqx.is_array), which keeps the
    # RQSpline's non-trainable bool mask arrays, while train_step's
    # eqx.filter_value_and_grad produces grads with None at those leaves.
    # jax 0.6.2 no longer accepts None as a tree-prefix of a non-None leaf, so
    # optim.update() raises ValueError inside optax (and adamw's weight decay
    # would hit the same mismatch against full params). Re-initialize the
    # bundle's optimizer with inexact-filtered params and decay-free adam so
    # the update trees are structurally consistent.
    import equinox as eqx
    import optax

    opt = bundle.resources["optimizer"]
    model = bundle.resources["model"]
    opt.optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    opt.optim_state = opt.optim.init(eqx.filter(model, eqx.is_inexact_array))

    sampler = Sampler(
        n_dims,
        n_chains,
        rng_key,
        resource_strategy_bundles=bundle,
    )

    initial_position = jax.random.normal(init_key, (n_chains, n_dims)) * 0.5
    sampler.sample(initial_position, {})

    n_production_steps = (n_local_steps + n_global_steps) * n_production_loops
    positions = np.asarray(sampler.resources["positions_production"].data)
    log_prob = np.asarray(sampler.resources["log_prob_production"].data)
    assert positions.shape == (n_chains, n_production_steps, n_dims)
    assert np.all(np.isfinite(positions))
    assert log_prob.shape == (n_chains, n_production_steps)
    assert np.all(np.isfinite(log_prob))


# ---------------------------------------------------------------------------
# 6. flowjax: fit a small masked autoregressive flow to 2-D banana data
# ---------------------------------------------------------------------------


def test_flowjax_fit_to_data():
    from flowjax.distributions import Normal
    from flowjax.flows import masked_autoregressive_flow
    from flowjax.train import fit_to_data

    # flowjax 19 requires new-style typed PRNG keys (jax.random.key); the
    # legacy uint32 jax.random.PRNGKey raises TypeError in Distribution.sample.
    data_key, flow_key, train_key, sample_key = jax.random.split(
        jax.random.key(6), 4
    )

    # 2-D banana: x2 curved along x1.
    z = jax.random.normal(data_key, (500, 2))
    x = jnp.stack([z[:, 0], z[:, 1] + 0.5 * z[:, 0] ** 2], axis=1)

    flow = masked_autoregressive_flow(
        flow_key,
        base_dist=Normal(jnp.zeros(2)),
        flow_layers=2,
        nn_width=16,
    )

    flow, losses = fit_to_data(
        train_key,
        flow,
        x,
        max_epochs=5,
        batch_size=64,
        show_progress=False,
    )

    assert np.all(np.isfinite(np.asarray(losses["train"])))
    log_probs = np.asarray(flow.log_prob(x))
    assert log_probs.shape == (500,)
    assert np.all(np.isfinite(log_probs))

    samples = np.asarray(flow.sample(sample_key, (200,)))
    assert samples.shape == (200, 2)
    assert np.all(np.isfinite(samples))


# ---------------------------------------------------------------------------
# 7. nautilus: tiny 2-D Gaussian, vectorized numpy likelihood
# ---------------------------------------------------------------------------


def test_nautilus_2d_gaussian_evidence():
    from nautilus import Sampler

    # Unit cube -> uniform prior over [-5, 5]^2 (prior volume 100).
    def prior(u):
        return 10.0 * u - 5.0

    # Normalized N(0, I) log-likelihood, vectorized over (n, 2) batches.
    def likelihood(x):
        return -0.5 * np.sum(x**2, axis=-1) - np.log(2.0 * np.pi)

    # Analytic evidence: Z = (1/100) * integral N(0,I) ~= 1/100.
    log_z_true = -np.log(100.0)

    sampler = Sampler(
        prior,
        likelihood,
        n_dim=2,
        n_live=200,
        vectorized=True,
        seed=42,
    )
    success = sampler.run(verbose=False, n_eff=500)

    assert success
    assert sampler.log_z is not None
    assert np.isfinite(sampler.log_z)
    assert abs(sampler.log_z - log_z_true) < 1.0

    points, log_w, log_l = sampler.posterior()
    assert points.shape[1] == 2
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(log_w))
