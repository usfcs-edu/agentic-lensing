"""ProbModel validation (Phase 4): the single-dataset Gaussian pixel likelihood.

Three distinct claims, kept separate (method-discipline §2 — "name the link"):

  1. **Reconstruction (noiseless):** at the true shapes the lstsq model reproduces the
     exact forward-rendered data. This is a residual-magnitude check, NOT a χ²: with no
     noise a good fit drives Σ(r/σ)² to ~0, not 1, so it is not a calibrated GoF.
  2. **Likelihood code correctness:** ProbModel's internal Σ(r/σ)² equals a numpy
     recomputation (valid regardless of noise).
  3. **Calibrated χ² (with noise):** on a genuine noise realization, reduced χ² ≈ 1 at
     truth — the proper goodness-of-fit / likelihood-normalization diagnostic
     (inference-diagnostics.md: reduced χ² near 1; residuals ~ N(0,1)).

Residual plots are the standard model−data diagnostic; structure ⇒ mismodeling.
"""
import numpy as np
import pytest
from jax import numpy as jnp

pytestmark = pytest.mark.physics

EPL0 = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
LL = dict(R_sersic=1.2, n_sersic=3.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
SRC = dict(R_sersic=0.3, n_sersic=2.0, e1=0.05, e2=-0.02, center_x=0.05, center_y=-0.04)
_TRUTH = {(0, "mass", 0): EPL0, (0, "light", 0): LL, (1, "light", 0): SRC}
_BG, _EXP = 0.01, 1000.0


def _gauss_psf(sigma=2.0, half=10):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def _cfg(num_pix=60, supersample=2):
    from gigalens.simulator import SimulatorConfig
    return SimulatorConfig(delta_pix=0.05, num_pix=num_pix, supersample=supersample,
                           kernel=_gauss_psf(), likelihood_precision="float64")


def _free(d):
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    return {k: tfd.Normal(float(v), 0.3) for k, v in d.items()}


def _scene():
    """Returns (clean_observed, err_true, model, sim, z_true, true_unique)."""
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    cfg = _cfg()
    fwd = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)],
              light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=500.0))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=False), dict(SRC, Ie=200.0))]),
    ])
    clean = np.asarray(SceneSimulator(fwd, cfg).simulate(fwd.to_params({})))
    err_true = np.sqrt(_BG**2 + np.clip(clean, 0, np.inf) / _EXP)  # true per-pixel sigma

    model = LensModel([
        Plane(mass=[Component(EPL(50), _free(EPL0))],
              light=[Component(SersicEllipse(use_lstsq=True), _free(LL))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=True), _free(SRC))]),
    ])
    sim = SceneSimulator(model, cfg)

    def truth_val(key):
        p = key.split("/")
        return float(_TRUTH[(int(p[1]), p[2], int(p[3]))][p[4]])

    true_unique = {k: jnp.asarray(truth_val(k)) for k in model.prior.model.keys()}
    z_true = model.bijector.inverse(true_unique)
    return clean, err_true, model, sim, z_true, true_unique


def test_reconstruction_at_truth_and_likelihood_code(plotter):
    """Noiseless: model reproduces the exact data (residual magnitude), and ProbModel's
    Σ(r/σ)² matches a numpy recomputation. NOT a χ² goodness-of-fit (no noise)."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    clean, err_true, model, sim, z_true, _ = _scene()
    ds = ImageData(clean, _cfg(), error_map=err_true, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")

    log_like, resid_norm = prob.log_like(z_true, simulator=sim)
    params = model.to_params(model.bijector.forward(z_true))
    im = np.asarray(sim.lstsq_simulate(params, jnp.asarray(clean),
                                       jnp.asarray(err_true), ds.mask))

    # (1) reconstruction accuracy — residual magnitude, not a calibrated statistic.
    plotter.image("reconstruction_residual_noiseless", clean, im, rtol=1e-6, atol=1e-9,
                  err=err_true, labels=("data (noiseless)", "model (truth)"))
    assert np.max(np.abs(im - clean)) / np.max(np.abs(clean)) < 1e-4, \
        f"reconstruction max|Δ|/peak = {np.max(np.abs(im-clean))/np.max(np.abs(clean)):.2e}"

    # (2) likelihood-code correctness: the internal Σ(r/σ)² matches numpy.
    ssq_np = float(np.sum(((im - clean) / err_true) ** 2))
    ssq_pm = float(resid_norm) * prob.event_size
    assert np.isclose(ssq_pm, ssq_np, rtol=1e-6), f"pm {ssq_pm} vs numpy {ssq_np}"
    lp, _ = prob.log_prob(z_true, simulator=sim)
    assert np.isfinite(float(lp))
    assert np.isclose(float(lp), float(log_like) + float(prob.log_prior(z_true)), rtol=1e-6)


def test_reduced_chi2_calibrated_with_noise(plotter):
    """Proper χ²: with a real noise realization, reduced χ² ≈ 1 at truth.

    This is the calibrated goodness-of-fit / likelihood-normalization check. The
    observed image is clean + N(0, err_true), and err_true is the *true* per-pixel
    sigma. At truth, E[reduced χ²] = (N − n_amp)/N ≈ 1, with spread √(2/N). Falsifier:
    reduced χ² far from 1 ⇒ the likelihood mis-weights the noise (a normalization bug).
    """
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    clean, err_true, model, sim, z_true, _ = _scene()
    rng = np.random.default_rng(20260623)
    noisy = clean + rng.standard_normal(clean.shape) * err_true
    ds = ImageData(noisy, _cfg(), error_map=err_true, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")

    _, red_chi2 = prob.log_like(z_true, simulator=sim)
    red = float(red_chi2)
    n = ds.event_size
    spread = np.sqrt(2.0 / n)  # std of reduced chi2

    params = model.to_params(model.bijector.forward(z_true))
    im = np.asarray(sim.lstsq_simulate(params, jnp.asarray(noisy), jnp.asarray(err_true), ds.mask))
    plotter.image("calibrated_chi2_residual_noisy", noisy, im, rtol=1e-6, atol=1e-9,
                  err=err_true, labels=("data (noisy)", "model (truth)"))
    # within ~6 sigma of 1 (robust to the RNG draw); a normalization bug would be O(1) off.
    assert abs(red - 1.0) < 6 * spread, \
        f"reduced chi2 at truth = {red:.4f}, expected 1 ± {spread:.4f}"


def test_residual_grows_when_source_displaced(plotter):
    """Displacing the source sharply increases the residual norm — the likelihood is
    informative (a relative check, valid with or without noise)."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    clean, err_true, model, sim, z_true, true_unique = _scene()
    ds = ImageData(clean, _cfg(), error_map=err_true, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    _, r0 = prob.log_like(z_true, simulator=sim)

    pert = dict(true_unique)
    skey = next(k for k in pert if k.startswith("planes/1/light/0/center_x"))
    pert[skey] = jnp.asarray(float(true_unique[skey]) + 0.3)
    z_pert = model.bijector.inverse(pert)
    _, rp = prob.log_like(z_pert, simulator=sim)

    params = model.to_params(model.bijector.forward(z_pert))
    im = np.asarray(sim.lstsq_simulate(params, jnp.asarray(clean), jnp.asarray(err_true), ds.mask))
    plotter.image("residual_source_displaced", clean, im, rtol=1e-6, atol=1e-9,
                  err=err_true, labels=("data", "model (source +0.3\")"))
    assert float(rp) > 100 * float(r0) + 1.0, \
        f"perturbed residual norm {float(rp):.3e} not >> truth {float(r0):.3e}"
