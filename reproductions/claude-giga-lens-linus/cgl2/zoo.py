"""Zoo target adapters for the MC-SMC benchmark (P0 task #13).

Every target exposes the same surface (class `Target`): single-point and batched
(logprior, loglik) closures over a flat unconstrained z, a z-space prior sampler,
z_dim, name, provenance. `log_prior` ALWAYS includes the bijector forward
log-det-Jacobian, so the lambda=0 tempering distribution is exactly the z-space
prior that `prior_sample_fn` draws from and the SMC evidence is the z-space
log integral exp(loglik) d prior -- the claude-giga-lens zoo convention
(../claude-giga-lens/cgl/zoo/api.py), which is also exactly the vendored scene
API's ProbModel.log_prior/log_like split.

Tiers here:
  (a) T0 analytic targets t0_mix2 / t0_funnel10 / t0_illcond46 (+ t0_gauss2 for
      CPU unit tests) -- ported with attribution from
      ../claude-giga-lens/cgl/zoo/synthetic.py (conjugate constructions with
      analytic logZ / moments / mode weights), all float64 here (the cgl2
      process is x64-always; the old f32 variants existed for tfp-f32 parity
      with stored chains, which does not apply to these fresh targets).
  (b) Scene-API targets carousel33 / dspl20_orig / dspl20_ratio / hs2_sys{0..7}
      -- model construction REPLICATED from the team's research repo
      /raid/benson/lensing-repos/GIGALens-Code @ eb2a09b6b2ff20ed868ffe92b5b5b68b615f4dbe
      (read-only; builder logic copied with attribution, gigalens_research never
      installed), wrapping the VENDORED scene API's ProbModel.log_prior(z) /
      log_like(z). Where their on-disk data is absent from the local mirror the
      builder renders a SEEDED MOCK stand-in and says so in `provenance` +
      `meta['data']` (honesty rule; the B0 gate that consumes these targets is
      the adapter-vs-native logp parity, which is data-agnostic).
  (c) T2/T3 old-stack targets are NOT built here (they run in the old cgl venv).

Scene-target configs are OUTSIDE the parity-certified class (multi-plane, NFW,
Shapelets, forward-mode Sersic): they are sampler-benchmark targets, marked
UNCERTIFIED (external) per the campaign vocabulary -- never science payloads.

Module is import-light: jax / gigalens are imported inside builders only.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Callable, Optional

import numpy as np

GIGALENS_CODE_COMMIT = "eb2a09b6b2ff20ed868ffe92b5b5b68b615f4dbe"
_LOG2PI = float(np.log(2.0 * np.pi))


@dataclasses.dataclass
class Target:
    """One zoo target (see module docstring for the contract)."""
    name: str
    z_dim: int
    logprior_fn: Callable            # (dim,) -> scalar (jax)
    loglik_fn: Callable              # (dim,) -> scalar (jax)
    logprior_batch: Callable         # (N, dim) -> (N,)
    loglik_batch: Callable           # (N, dim) -> (N,)
    prior_sample_fn: Callable        # (key, n) -> (n, dim) z draws
    provenance: str
    reference: dict = dataclasses.field(default_factory=dict)
    # native combined logp for the B0 parity gate (scene targets: ProbModel.log_prob)
    native_logprob_batch: Optional[Callable] = None
    prob_model: Optional[object] = None      # scene targets: the vendored ProbModel
    meta: dict = dataclasses.field(default_factory=dict)


# =========================================================================== #
# (a) T0 analytic targets
#     Ported with attribution from ../claude-giga-lens/cgl/zoo/synthetic.py
#     (same conjugate constructions, same constants; f64; trimmed to the
#     Target surface -- InitBundle/CellResult machinery not carried).
# =========================================================================== #
def _gauss_prior(dim: int, sigma_p: float):
    import jax
    import jax.numpy as jnp

    sp2 = sigma_p ** 2
    c_prior = -0.5 * dim * (_LOG2PI + np.log(sp2))

    def logprior_batch(Z):
        Z = jnp.asarray(Z, dtype=jnp.float64)
        return c_prior - 0.5 / sp2 * jnp.sum(Z ** 2, axis=-1)

    def logprior_fn(z):
        z = jnp.asarray(z, dtype=jnp.float64)
        return c_prior - 0.5 / sp2 * jnp.sum(z ** 2)

    def prior_sample_fn(key, n):
        return sigma_p * jax.random.normal(key, (n, dim), dtype=jnp.float64)

    return logprior_fn, logprior_batch, prior_sample_fn


def build_mix2(dim: int = 2, weights=(0.8, 0.2), mode_x0: float = 2.0,
               comp_sigma: float = 0.5, prior_sigma: float = 5.0) -> Target:
    """Two-mode Gaussian mixture, conjugate: logZ / moments / posterior mode
    weights all analytic. Copied construction (constants included) from
    cgl/zoo/synthetic.py::build_mixture."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    w = np.asarray(weights, dtype=np.float64)
    assert abs(w.sum() - 1.0) < 1e-12
    mus = np.zeros((2, dim))
    mus[0, 0], mus[1, 0] = mode_x0, -mode_x0
    s2, sp2 = comp_sigma ** 2, prior_sigma ** 2

    post_var = 1.0 / (1.0 / s2 + 1.0 / sp2)
    post_means = mus * (post_var / s2)
    ev2 = s2 + sp2
    log_ev = -0.5 * dim * (_LOG2PI + np.log(ev2)) - 0.5 * (mus ** 2).sum(1) / ev2
    logZ = float(np.logaddexp(np.log(w[0]) + log_ev[0],
                              np.log(w[1]) + log_ev[1]))
    post_w = np.exp(np.log(w) + log_ev - logZ)   # == weights (symmetric modes)
    mean_vec = (post_w[:, None] * post_means).sum(0)
    second = (post_w[:, None] * (post_means ** 2 + post_var)).sum(0)
    var_vec = second - mean_vec ** 2

    mus_j = jnp.asarray(mus)
    logw_j = jnp.asarray(np.log(w))
    c_like = -0.5 * dim * (_LOG2PI + np.log(s2))

    def loglik_batch(Z):
        Z = jnp.asarray(Z, dtype=jnp.float64)
        d2 = jnp.sum((Z[:, None, :] - mus_j[None, :, :]) ** 2, axis=-1)
        return logsumexp(logw_j[None, :] - 0.5 / s2 * d2, axis=-1) + c_like

    def loglik_fn(z):
        z = jnp.asarray(z, dtype=jnp.float64)
        d2 = jnp.sum((z[None, :] - mus_j) ** 2, axis=-1)
        return logsumexp(logw_j - 0.5 / s2 * d2) + c_like

    logprior_fn, logprior_batch, prior_sample_fn = _gauss_prior(dim, prior_sigma)

    def exact_sample_fn(seed, n):
        rng = np.random.default_rng(seed)
        comp = rng.choice(2, size=n, p=post_w)
        return post_means[comp] + np.sqrt(post_var) * rng.standard_normal((n, dim))

    return Target(
        name="t0_mix2", z_dim=dim,
        logprior_fn=logprior_fn, loglik_fn=loglik_fn,
        logprior_batch=logprior_batch, loglik_batch=loglik_batch,
        prior_sample_fn=prior_sample_fn,
        provenance=("analytic: conjugate Gaussian mixture, ported with "
                    "attribution from ../claude-giga-lens/cgl/zoo/synthetic.py::"
                    f"build_mixture (w={tuple(w)}, mu0=+-{mode_x0} e0, "
                    f"comp_sigma={comp_sigma}, prior_sigma={prior_sigma}); f64"),
        reference=dict(logZ=logZ, minor_mode_weight=float(post_w[1]),
                       mode_axis=0, mode_boundary=0.0,
                       mean=mean_vec.tolist(), var=var_vec.tolist(),
                       exact_sample_fn=exact_sample_fn),
        meta=dict(family="gaussian_mixture", tier="T0"),
    )


def build_funnel10(dim: int = 10, prior_sigma: float = 3.0) -> Target:
    """Neal funnel: theta=z0 ~ N(0,3), z_i ~ N(0, e^{theta/2}); prior
    N(0, prior_sigma^2 I); like = funnel - prior so logZ = 0 EXACTLY.
    Copied construction from cgl/zoo/synthetic.py::build_funnel."""
    import jax.numpy as jnp

    k = dim - 1
    sp2 = prior_sigma ** 2
    c_prior = -0.5 * dim * (_LOG2PI + np.log(sp2))

    def _funnel_logpdf(Z):
        th, x = Z[..., 0], Z[..., 1:]
        lp_th = -0.5 * _LOG2PI - jnp.log(3.0) - 0.5 * (th / 3.0) ** 2
        scale = jnp.exp(0.5 * th)[..., None]
        lp_x = jnp.sum(-0.5 * _LOG2PI - 0.5 * th[..., None]
                       - 0.5 * (x / scale) ** 2, axis=-1)
        return lp_th + lp_x

    def loglik_batch(Z):
        Z = jnp.asarray(Z, dtype=jnp.float64)
        return _funnel_logpdf(Z) - (c_prior
                                    - 0.5 / sp2 * jnp.sum(Z ** 2, axis=-1))

    def loglik_fn(z):
        z = jnp.asarray(z, dtype=jnp.float64)
        return _funnel_logpdf(z) - (c_prior - 0.5 / sp2 * jnp.sum(z ** 2))

    logprior_fn, logprior_batch, prior_sample_fn = _gauss_prior(dim, prior_sigma)

    def exact_sample_fn(seed, n):
        rng = np.random.default_rng(seed)
        th = 3.0 * rng.standard_normal(n)
        x = np.exp(th / 2.0)[:, None] * rng.standard_normal((n, k))
        return np.concatenate([th[:, None], x], axis=1)

    return Target(
        name="t0_funnel10", z_dim=dim,
        logprior_fn=logprior_fn, loglik_fn=loglik_fn,
        logprior_batch=logprior_batch, loglik_batch=loglik_batch,
        prior_sample_fn=prior_sample_fn,
        provenance=(f"analytic: {dim}-D Neal funnel, ported with attribution "
                    "from ../claude-giga-lens/cgl/zoo/synthetic.py::build_funnel "
                    f"(prior N(0,{prior_sigma}^2 I); logZ=0 exact); f64"),
        reference=dict(logZ=0.0, var_theta=9.0, mean_theta=0.0,
                       exact_sample_fn=exact_sample_fn),
        meta=dict(family="neal_funnel", tier="T0"),
    )


def build_illcond46(dim: int = 46, cond: float = 1e14,
                    prior_sigma: float = 10.0) -> Target:
    """Ill-conditioned Gaussian: likelihood N(0, diag(lam)) with lam log-spaced
    1..1/cond (the T2 structure); prior N(0, prior_sigma^2 I); logZ analytic.
    Copied construction from cgl/zoo/synthetic.py::build_illcond."""
    import jax.numpy as jnp

    lam = np.logspace(0.0, -np.log10(cond), dim)
    sp2 = prior_sigma ** 2
    post_var = 1.0 / (1.0 / lam + 1.0 / sp2)
    logZ = float(np.sum(-0.5 * (_LOG2PI + np.log(lam + sp2))))
    c_like = float(-0.5 * dim * _LOG2PI - 0.5 * np.log(lam).sum())
    lam_j = jnp.asarray(lam)

    def loglik_batch(Z):
        Z = jnp.asarray(Z, dtype=jnp.float64)
        return c_like - 0.5 * jnp.sum(Z ** 2 / lam_j[None, :], axis=-1)

    def loglik_fn(z):
        z = jnp.asarray(z, dtype=jnp.float64)
        return c_like - 0.5 * jnp.sum(z ** 2 / lam_j)

    logprior_fn, logprior_batch, prior_sample_fn = _gauss_prior(dim, prior_sigma)

    def exact_sample_fn(seed, n):
        rng = np.random.default_rng(seed)
        return np.sqrt(post_var)[None, :] * rng.standard_normal((n, dim))

    return Target(
        name="t0_illcond46", z_dim=dim,
        logprior_fn=logprior_fn, loglik_fn=loglik_fn,
        logprior_batch=logprior_batch, loglik_batch=loglik_batch,
        prior_sample_fn=prior_sample_fn,
        provenance=(f"analytic: {dim}-D ill-conditioned Gaussian (cond {cond:g},"
                    " ported with attribution from ../claude-giga-lens/cgl/zoo/"
                    f"synthetic.py::build_illcond; prior N(0,{prior_sigma}^2 I))"),
        reference=dict(logZ=logZ, mean=[0.0] * dim, var=post_var.tolist(),
                       exact_sample_fn=exact_sample_fn,
                       cond_posterior=float(post_var.max() / post_var.min())),
        meta=dict(family="illconditioned_gaussian", tier="T0", cond=cond),
    )


def build_gauss2(dim: int = 2, mu0=(0.5, -0.3), comp_sigma: float = 1.0,
                 prior_sigma: float = 2.0) -> Target:
    """2-D conjugate Gaussian for the CPU unit toys (tests/test_smc_micro.py):
    likelihood N(mu0, comp_sigma^2 I), prior N(0, prior_sigma^2 I); logZ and
    posterior moments in closed form."""
    import jax.numpy as jnp

    mu0 = np.asarray(mu0, dtype=np.float64)[:dim]
    s2, sp2 = comp_sigma ** 2, prior_sigma ** 2
    ev2 = s2 + sp2
    logZ = float(np.sum(-0.5 * (_LOG2PI + np.log(ev2)) - 0.5 * mu0 ** 2 / ev2))
    post_var = 1.0 / (1.0 / s2 + 1.0 / sp2)
    post_mean = mu0 * (post_var / s2)
    c_like = -0.5 * dim * (_LOG2PI + np.log(s2))
    mu_j = jnp.asarray(mu0)

    def loglik_batch(Z):
        Z = jnp.asarray(Z, dtype=jnp.float64)
        return c_like - 0.5 / s2 * jnp.sum((Z - mu_j[None, :]) ** 2, axis=-1)

    def loglik_fn(z):
        z = jnp.asarray(z, dtype=jnp.float64)
        return c_like - 0.5 / s2 * jnp.sum((z - mu_j) ** 2)

    logprior_fn, logprior_batch, prior_sample_fn = _gauss_prior(dim, prior_sigma)
    return Target(
        name="t0_gauss2", z_dim=dim,
        logprior_fn=logprior_fn, loglik_fn=loglik_fn,
        logprior_batch=logprior_batch, loglik_batch=loglik_batch,
        prior_sample_fn=prior_sample_fn,
        provenance="analytic: 2-D conjugate Gaussian (unit-test toy)",
        reference=dict(logZ=logZ, mean=post_mean.tolist(),
                       var=[post_var] * dim),
        meta=dict(family="gaussian", tier="T0"),
    )


# =========================================================================== #
# (b) scene-API targets (vendored gigalens-linus @ 80916d2)
# =========================================================================== #
def _wrap_prob_model(name: str, prob_model, provenance: str, meta: dict,
                     reference: Optional[dict] = None) -> Target:
    """Adapter around a vendored scene ProbModel.

    logprior = ProbModel.log_prior(z)  (prior log-prob + fwd log-det-Jacobian)
    loglik   = ProbModel.log_like(z)[0]
    native   = ProbModel.log_prob(z)[0]  (parity surface for gate B0-parity)
    prior_sample_fn: model.prior.sample -> ZBijector.inverse (flat z), the
    scene-native analogue of the old MAP multistart convention.
    """
    import jax.numpy as jnp

    pm = prob_model
    dim = int(pm.model.num_free_params)

    def logprior_batch(Z):
        return pm.log_prior(jnp.asarray(Z, dtype=jnp.float64))

    def loglik_batch(Z):
        return pm.log_like(jnp.asarray(Z, dtype=jnp.float64))[0]

    def native_logprob_batch(Z):
        return pm.log_prob(jnp.asarray(Z, dtype=jnp.float64))[0]

    def logprior_fn(z):
        return pm.log_prior(jnp.asarray(z, dtype=jnp.float64))

    def loglik_fn(z):
        return pm.log_like(jnp.asarray(z, dtype=jnp.float64))[0]

    def prior_sample_fn(key, n):
        x = pm.model.prior.sample(int(n), seed=key)
        z = pm.model.bijector.inverse(x)
        return jnp.asarray(z, dtype=jnp.float64)

    return Target(
        name=name, z_dim=dim,
        logprior_fn=logprior_fn, loglik_fn=loglik_fn,
        logprior_batch=logprior_batch, loglik_batch=loglik_batch,
        prior_sample_fn=prior_sample_fn,
        provenance=provenance, reference=reference or {},
        native_logprob_batch=native_logprob_batch, prob_model=pm,
        meta={**meta, "z_param_names": list(pm.z_param_names or []),
              "certification": "UNCERTIFIED (external) scene config"},
    )


def _gaussian_psf_kernel(fwhm_pix: float = 2.0, half: int = 7) -> np.ndarray:
    """Replicates photutils.psf.GaussianPSF(x_fwhm=f, y_fwhm=f) evaluated on
    np.mgrid[-half:half+1, -half:half+1] (the dspl_arm_init recipe): a unit-flux
    2-D Gaussian f(x,y) = exp(-(x^2+y^2)/(2 sigma^2)) / (2 pi sigma^2) with
    sigma = fwhm / (2 sqrt(2 ln 2)). photutils is NOT installed in the cgl2
    venv (deliberate: no new deps); this closed form is the whole of what the
    recipe uses. Any normalization difference is inert if the simulator
    renormalizes the kernel; the FUNCTIONAL shape is exact."""
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    return np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)) / (
        2.0 * np.pi * sigma ** 2)


# --------------------------------------------------------------------------- #
# DSPL cosmology system (sample_cosmology)
# --------------------------------------------------------------------------- #
# Constants verbatim from GIGALens-Code@eb2a09b6
# experiments/sample_cosmology/dspl_arm_init.py (themselves verbatim from
# dspl_cosmology_newapi.ipynb):
DSPL_Z_LENS = 0.5
DSPL_Z_SOURCE1 = 1.0
DSPL_Z_SOURCE2 = 1.5
DSPL_NUM_PIX, DSPL_DELTA_PIX = 100, 0.065
DSPL_EXP_TIME, DSPL_BACKGROUND_RMS = 1000, 0.1
DSPL_TRUE_TRUTH_SCENE = {
    "planes": {
        0: {"geometry": {"redshift": DSPL_Z_LENS},
            "mass": {0: {"theta_E": 1.1, "gamma": 2.0, "e1": 0.05, "e2": 0.02,
                         "center_x": 0.0, "center_y": 0.0}}},
        1: {"geometry": {"redshift": DSPL_Z_SOURCE1},
            "light": {0: {"R_sersic": 0.25, "n_sersic": 2., "e1": 0.05,
                          "e2": 0., "center_x": 0.05, "center_y": 0.,
                          "Ie": 50.}}},
        2: {"geometry": {"redshift": DSPL_Z_SOURCE2},
            "light": {0: {"R_sersic": 1., "n_sersic": 6., "e1": 0.0,
                          "e2": 0.05, "center_x": 0., "center_y": 0.05,
                          "Ie": 15.}}},
    },
    "cosmo": dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0, wa=0.0),
}
DSPL_OM0_BOUNDS = (0.0, 1.0)
DSPL_W0_BOUNDS = (-2.0, -1.0 / 3.0)
# Run C validator tolerances, verbatim from dspl_ratio_coords.py (provenance in
# that module's docstring: measured sliver bounds on their 201x201 grid).
DSPL_DU_DW_ATOL = 1.0e-4
DSPL_EXCURSION_ATOL = 3.0e-6


def _dspl_components(coords: str):
    """Copied with attribution from GIGALens-Code@eb2a09b6
    experiments/sample_cosmology/dspl_arm_init.py::build_components (orig
    coords) + dspl_ratio_coords.py::build_grouped_model / build_u_fn (ratio
    coords). Verbatim priors; the ratio variant swaps ONLY the cosmology
    Component's (Om0, w0) scalar priors for the grouped ratio-coordinates
    prior (identical uniform box density, different sampling coordinates)."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    from gigalens.jax.cosmo import w0waCDM_Cosmo
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.scene import Component

    tfd, tfb = tfp.distributions, tfp.bijectors

    lens = Component(
        EPL(),
        dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
            gamma=2.0,
            e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            center_x=tfd.Normal(0, 0.05),
            center_y=tfd.Normal(0, 0.05),
        ),
    )

    def _make_source():
        return Component(
            SersicEllipse(use_lstsq=False),
            dict(
                center_x=tfd.Normal(0, 2),
                center_y=tfd.Normal(0, 2),
                e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                n_sersic=tfd.Uniform(1, 10),
                R_sersic=tfd.LogNormal(jnp.log(1.), 0.15),
                Ie=tfd.LogNormal(jnp.log(150), 1),
            ),
        )

    source1, source2 = _make_source(), _make_source()

    if coords == "orig":
        def tNCDF_bij(low, high):
            return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low),
                              tfb.NormalCDF()])

        class UniformBij(tfd.Uniform):
            def __init__(self, *args, event_space_bijector_class=tNCDF_bij,
                         **kwargs):
                self._esb = event_space_bijector_class(*args)
                super().__init__(*args, **kwargs)

            def _default_event_space_bijector(self):
                return self._esb

        cosmo = Component(
            w0waCDM_Cosmo(z_lens=DSPL_Z_LENS, z_source_ref=DSPL_Z_SOURCE1),
            dict(
                H0=70.,
                Om0=UniformBij(jnp.float64(DSPL_OM0_BOUNDS[0]),
                               jnp.float64(DSPL_OM0_BOUNDS[1])),
                w0=UniformBij(jnp.float64(DSPL_W0_BOUNDS[0]),
                              jnp.float64(DSPL_W0_BOUNDS[1])),
                wa=0.0,
                k=0.0,
            ),
        )
    elif coords == "ratio":
        from cgl2._ratio_coords_copy import (RatioCoordsUniform,
                                             deflection_ratio_u_fn)

        cosmo_profile = w0waCDM_Cosmo(z_lens=DSPL_Z_LENS,
                                      z_source_ref=DSPL_Z_SOURCE1)
        u_fn = deflection_ratio_u_fn(cosmo_profile, (DSPL_Z_SOURCE2,), (1.0,),
                                     fixed=dict(H0=70.0, k=0.0, wa=0.0))
        grouped = RatioCoordsUniform(
            u_fn, DSPL_OM0_BOUNDS, DSPL_W0_BOUNDS,
            du_dw_atol=DSPL_DU_DW_ATOL, excursion_atol=DSPL_EXCURSION_ATOL,
        )
        cosmo = Component(
            w0waCDM_Cosmo(z_lens=DSPL_Z_LENS, z_source_ref=DSPL_Z_SOURCE1),
            {
                "H0": 70.0,
                "k": 0.0,
                "wa": 0.0,
                ("Om0", "w0"): grouped,
            },
        )
    else:
        raise ValueError(f"coords must be 'orig' or 'ratio'; got {coords!r}")
    return lens, source1, source2, cosmo


def _dspl_observed_images(model, source1, source2, data_seed: int = 0):
    """Replicates GIGALens-Code@eb2a09b6 dspl_arm_init.py::make_observed_images
    (fresh-realization branch): render TRUE_TRUTH_SCENE, then
    np.random.seed(data_seed) + lenstronomy add_poisson/add_background --
    the same seeded (seed 0) realization their Runs A/B/C used (their baseline
    notebook realization is unreproducible by their own docstring). Deviation:
    the Gaussian PSF kernel is computed in closed form (photutils not
    installed) -- see _gaussian_psf_kernel."""
    from lenstronomy.Util import image_util

    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.simulator import SimulatorConfig

    kernel = _gaussian_psf_kernel(2.0, 7)
    sim_config = SimulatorConfig(
        delta_pix=DSPL_DELTA_PIX, num_pix=DSPL_NUM_PIX, kernel=kernel,
        supersample=1, likelihood_precision="float64")

    sim1 = SceneSimulator(model, sim_config, sees=[source1])
    sim2 = SceneSimulator(model, sim_config, sees=[source2])

    np.random.seed(data_seed)

    def add_noise(img, exp_time, sigma_bkd):
        poisson = image_util.add_poisson(img, exp_time=exp_time)
        bkg = image_util.add_background(img, sigma_bkd=sigma_bkd)
        return img + poisson + bkg

    img1 = np.asarray(sim1.simulate(DSPL_TRUE_TRUTH_SCENE))
    img2 = np.asarray(sim2.simulate(DSPL_TRUE_TRUTH_SCENE))
    img1 = add_noise(img1, exp_time=DSPL_EXP_TIME, sigma_bkd=DSPL_BACKGROUND_RMS)
    img2 = add_noise(img2, exp_time=DSPL_EXP_TIME, sigma_bkd=DSPL_BACKGROUND_RMS)
    return img1, img2, sim_config


def build_dspl20(coords: str = "orig", data_seed: int = 0) -> Target:
    """DSPL two-source-plane cosmology system (their LAPS/MCLMC pathology bed).

    coords='orig'  : the ORIGINAL pathological (Om0, w0)+NormalCDF coordinates
                     (PLAN §6 B2 headline arm).
    coords='ratio' : the Run C exact ratio-coordinates reparameterization
                     (control arm; grouped tuple-key prior, copied module).
    """
    from gigalens.jax.scene import LensModel, Plane
    from gigalens.jax.scene_prob_model import ImageData, ProbModel

    lens, source1, source2, cosmo = _dspl_components(coords)
    model = LensModel(
        [
            Plane(redshift=DSPL_Z_LENS, mass=[lens]),
            Plane(redshift=DSPL_Z_SOURCE1, light=[source1]),
            Plane(redshift=DSPL_Z_SOURCE2, light=[source2]),
        ],
        cosmo=cosmo,
    )
    img1, img2, sim_config = _dspl_observed_images(model, source1, source2,
                                                   data_seed)
    d1 = ImageData(img1, sim_config, background_rms=DSPL_BACKGROUND_RMS,
                   exp_time=DSPL_EXP_TIME, sees=[source1])
    d2 = ImageData(img2, sim_config, background_rms=DSPL_BACKGROUND_RMS,
                   exp_time=DSPL_EXP_TIME, sees=[source2])
    pm = ProbModel(model, [d1, d2], mode="forward")

    name = "dspl20_orig" if coords == "orig" else "dspl20_ratio"
    return _wrap_prob_model(
        name, pm,
        provenance=(
            f"scene-API replication of GIGALens-Code@{GIGALENS_CODE_COMMIT} "
            "experiments/sample_cosmology/dspl_arm_init.py (build_components, "
            "TRUE_TRUTH_SCENE, make_observed_images data_seed="
            f"{data_seed}) " + (
                "+ dspl_ratio_coords.py grouped ratio-coordinates prior "
                "(cgl2/_ratio_coords_copy.py)" if coords == "ratio" else
                "in the ORIGINAL (Om0,w0)+NormalCDF coordinates") +
            "; PSF kernel: closed-form Gaussian FWHM=2pix replicating "
            "photutils.psf.GaussianPSF (photutils not installed); noise: "
            "np.random.seed + lenstronomy add_poisson/add_background "
            "(their fresh-realization branch; bit-level match depends on "
            "the lenstronomy version). Substrate: vendored gigalens-linus "
            "@80916d2, forward mode. UNCERTIFIED (external)."),
        meta=dict(tier="scene", system="dspl", coords=coords,
                  data="seeded mock (their own generation recipe; their "
                       "baseline realization unreproducible per their note)",
                  truth=DSPL_TRUE_TRUTH_SCENE),
        reference=dict(truth_cosmo=dict(Om0=0.3, w0=-1.0),
                       arm_mass_below_om0_0146=0.103),
    )


# --------------------------------------------------------------------------- #
# carousel (sim_carousel): multi-plane MUSE-like system, their hardest
# ESS-limited target
# --------------------------------------------------------------------------- #
def _carousel_dataset_from_fits(path):
    """Load one real carousel MUSE cutout, conventions VERBATIM from
    GIGALens-Code@eb2a09b6 experiments/sim_carousel/_h1h2_diag/
    build_model.py::dataset_from_dir: observed = DATA (f64), error_map =
    sqrt(STAT) (f64), psf = PSF (f64), mask = MASK as bool. RAISES on any
    missing file/extension or non-finite/non-positive noise — never defaults.
    Returns (img, err, psf, mask, md5)."""
    import hashlib

    from astropy.io import fits  # in both pinned venvs (cgl2 + cgl2-pm)

    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"carousel real cutout missing: {path}")
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    with fits.open(path) as hdul:
        names = [h.name for h in hdul]
        for ext in ("DATA", "STAT", "PSF", "MASK"):
            if ext not in names:
                raise KeyError(f"{path}: FITS extension {ext!r} missing "
                               f"(have {names})")
        img = np.asarray(hdul["DATA"].data, dtype=np.float64)
        stat = np.asarray(hdul["STAT"].data, dtype=np.float64)
        psf = np.asarray(hdul["PSF"].data, dtype=np.float64)
        mask = np.asarray(hdul["MASK"].data).astype(bool)
    if img.shape != (300, 300) or stat.shape != img.shape \
            or mask.shape != img.shape:
        raise ValueError(f"{path}: unexpected shapes img={img.shape} "
                         f"stat={stat.shape} mask={mask.shape} "
                         "(carousel convention is 300x300)")
    if not np.all(np.isfinite(img)):
        raise ValueError(f"{path}: non-finite DATA pixels")
    if not (np.all(np.isfinite(stat)) and np.all(stat > 0)):
        raise ValueError(f"{path}: STAT must be finite and positive "
                         "(err = sqrt(STAT))")
    if not (np.all(np.isfinite(psf)) and psf.ndim == 2):
        raise ValueError(f"{path}: bad PSF extension")
    if not mask.any():
        raise ValueError(f"{path}: MASK keeps zero pixels")
    return img, np.sqrt(stat), psf, mask, md5


def build_carousel33(mock_data_seed: int = 33,
                     data_dir: Optional[str] = None) -> Target:
    """Carousel scene model, replicated verbatim from GIGALens-Code@eb2a09b6
    experiments/sim_carousel/_h1h2_diag/build_model.py (itself 'EXACTLY as
    prelim_sim_carousel.ipynb cells 0-8'): 3 planes (z=0.49 mass: NFW_ELLIPSE +
    2x EPL + shear; z=1.432 light: Shapelets n_max=8 + n_max=6; z=1.506 light:
    SersicEllipse), wCDM cosmology fixed, lstsq amplitudes.

    DATA — two modes, selected by `data_dir` (argument, else env
    CGL2_CAROUSEL_DATA; B1-real extension 2026-07-16):

    REAL (data_dir set): loads the team's MUSE cutouts
    ``<data_dir>/source4-5.fits`` + ``<data_dir>/source9.fits``
    (DATA/STAT/PSF/MASK extensions, conventions verbatim from their
    build_model.py::dataset_from_dir: err = sqrt(STAT), per-dataset PSF
    kernel, bool mask). RAISE-NEVER-DEFAULT: any missing/ill-formed input
    raises — this path NEVER falls back to the mock. FITS md5s recorded in
    meta. The data are the team's UNPUBLISHED cutouts: results to the team
    first, publication sign-off-gated (D4/D6).

    MOCK (default, unchanged code path — the declared stand-in of the
    RUNNING B1 mock arms): unit-amplitude render of the model at a prior
    draw (PRNGKey(mock_data_seed)), Gaussian PSF stand-in (FWHM 3 pix),
    constant error map sigma = 0.05*max(img), all-true mask. The POSTERIOR
    GEOMETRY IS THEREFORE NOT THEIRS -- valid for the B0 adapter-parity gate
    and machinery smoke only."""
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.light.shapelets import Shapelets
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.scene import Component, LensModel, Plane
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.simulator import SimulatorConfig

    tfd = tfp.distributions

    # -- components: verbatim from build_model.py ----------------------------------
    NFW0 = Component(NFW_ELLIPSE(), dict(
        Rs=tfd.Uniform(20, 100), alpha_Rs=tfd.Uniform(10, 40),
        e1=tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
        e2=tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
        center_x=tfd.Normal(5.344, 0.05), center_y=tfd.Normal(3.805, 0.05)))
    EPL_Le = Component(EPL(50), dict(
        theta_E=tfd.TruncatedNormal(2.4, 0.1, 1, 3),
        gamma=tfd.TruncatedNormal(2.2, 0.5, 1, 3),
        e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        center_x=tfd.Normal(-22.1, 0.1), center_y=tfd.Normal(-24.7, 0.1)))
    EPL_Lf = Component(EPL(50), dict(
        center_x=tfd.Normal(-15.10088063, 0.1),
        center_y=tfd.Normal(-4.66657821, 0.1),
        e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        theta_E=tfd.TruncatedNormal(0.8151327, 0.05, 0.2, 1.5),
        gamma=tfd.TruncatedNormal(2.2266, 0.5, 1, 3)))
    shear = Component(Shear(), dict(
        gamma1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        gamma2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3)))
    src4 = Component(Shapelets(n_max=8, use_lstsq=True), dict(
        center_x=tfd.Normal(3.7, 1), center_y=tfd.Normal(3.2, 1),
        beta=tfd.LogNormal(jnp.log(0.4), 0.15)))
    src5 = Component(Shapelets(n_max=6, use_lstsq=True), dict(
        center_x=tfd.Normal(3.0, 1), center_y=tfd.Normal(0., 1),
        beta=tfd.LogNormal(jnp.log(0.1), 0.15)))
    src9 = Component(SersicEllipse(use_lstsq=True), dict(
        center_x=tfd.Normal(-10, 1), center_y=tfd.Normal(-16, 1),
        n_sersic=tfd.Uniform(0.1, 10),
        R_sersic=tfd.LogNormal(jnp.log(0.4), 0.15),
        e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3)))

    z4_5, z9, z_lens = 1.432, 1.506, 0.49
    cosmo = Component(wCDM_Cosmo(z_lens=z_lens, z_source_ref=z4_5),
                      dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel([
        Plane(redshift=z_lens, mass=[NFW0, EPL_Le, EPL_Lf, shear]),
        Plane(redshift=z4_5, light=[src4, src5]),
        Plane(redshift=z9, light=[src9]),
    ], cosmo=cosmo)

    # -- data: REAL cutouts if requested (raise-never-default), else MOCK -----------
    if data_dir is None:
        data_dir = os.environ.get("CGL2_CAROUSEL_DATA") or None

    if data_dir is not None:
        # REAL DATA path (B1-real, 2026-07-16). Conventions verbatim from their
        # build_model.py::ds(): per-dataset SimulatorConfig with the FITS PSF,
        # delta_pix=0.2, num_pix=300, ss=1, likelihood f64 / conv f32.
        img45, err45, psf45, mask45, md5_45 = _carousel_dataset_from_fits(
            os.path.join(str(data_dir), "source4-5.fits"))
        img9, err9, psf9, mask9, md5_9 = _carousel_dataset_from_fits(
            os.path.join(str(data_dir), "source9.fits"))

        def _cfg(psf):
            return SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1,
                                   kernel=psf, likelihood_precision="float64",
                                   conv_precision="float32")

        d4_5 = ImageData(img45, _cfg(psf45), error_map=err45, mask=mask45,
                         sees=[src4, src5])
        d9 = ImageData(img9, _cfg(psf9), error_map=err9, mask=mask9,
                       sees=[src9])
        pm = ProbModel(model, [d4_5, d9], mode="lstsq")
        return _wrap_prob_model(
            "carousel33", pm,
            provenance=(
                f"scene-API replication of GIGALens-Code@{GIGALENS_CODE_COMMIT} "
                "experiments/sim_carousel/_h1h2_diag/build_model.py (model "
                "components/planes/cosmo VERBATIM; delta_pix=0.2 num_pix=300 "
                "ss=1 lstsq, likelihood f64 / conv f32). DATA = REAL team MUSE "
                f"cutouts from {data_dir}: source4-5.fits (md5 {md5_45}) + "
                f"source9.fits (md5 {md5_9}), DATA/STAT/PSF/MASK conventions "
                "verbatim (err=sqrt(STAT), FITS PSF per dataset, bool mask). "
                "UNPUBLISHED team data (D4/D6: results to the team first, "
                "sign-off-gated). Substrate: vendored gigalens-linus "
                "@80916d2. UNCERTIFIED (external)."),
            meta=dict(tier="scene", system="carousel",
                      data="REAL team MUSE cutouts (newnewcutouts)",
                      data_dir=str(data_dir),
                      fits_md5=dict({"source4-5.fits": md5_45,
                                     "source9.fits": md5_9})),
        )

    # -- MOCK data stand-in (unchanged; the declared B1 mock-arm record) ------------
    psf = _gaussian_psf_kernel(3.0, 7)   # stand-in for the MUSE PSF extension
    cfg = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=psf,
                          likelihood_precision="float64",
                          conv_precision="float32")
    key = jax.random.PRNGKey(int(mock_data_seed))
    z_true = model.bijector.inverse(model.prior.sample(seed=key))
    params_true = model.constrained(z_true)
    rng = np.random.default_rng(int(mock_data_seed) + 1)

    def _mock(sees):
        # The carousel sources are all lstsq components, so `simulate()` does
        # not apply (it is documented non-lstsq; measured: basis-stack
        # broadcast TypeError). The mock is generated with the PUBLIC
        # `lstsq_simulate` seam instead: solve the basis amplitudes against a
        # smooth broad seed image (a wide Gaussian blob) at params_true -- the
        # returned best-fit model image is a valid in-model render (a linear
        # combination of the deflected basis at solved amplitudes). Arbitrary
        # but documented; the mock is a machinery stand-in, not their
        # posterior.
        import jax.numpy as jnp

        npx = int(cfg.num_pix)
        yy, xx = np.mgrid[:npx, :npx]
        r2 = (xx - npx / 2) ** 2 + (yy - npx / 2) ** 2
        blob = np.exp(-r2 / (2.0 * (npx / 6.0) ** 2))
        ones = np.ones((npx, npx))
        sim = SceneSimulator(model, cfg, sees=sees)
        img = np.asarray(
            sim.lstsq_simulate(params_true, jnp.asarray(blob),
                               jnp.asarray(ones)),
            dtype=np.float64)
        img = np.squeeze(img)
        sigma = 0.05 * float(np.abs(img).max() or 1.0)
        err = np.full_like(img, sigma)
        return img + rng.normal(size=img.shape) * sigma, err

    img45, err45 = _mock([src4, src5])
    img9, err9 = _mock([src9])
    d4_5 = ImageData(img45, cfg, error_map=err45, sees=[src4, src5])
    d9 = ImageData(img9, cfg, error_map=err9, sees=[src9])
    pm = ProbModel(model, [d4_5, d9], mode="lstsq")

    return _wrap_prob_model(
        "carousel33", pm,
        provenance=(
            f"scene-API replication of GIGALens-Code@{GIGALENS_CODE_COMMIT} "
            "experiments/sim_carousel/_h1h2_diag/build_model.py (model "
            "components/planes/cosmo VERBATIM; delta_pix=0.2 num_pix=300 "
            "ss=1 lstsq, likelihood f64 / conv f32). DATA = SEEDED MOCK "
            f"stand-in (seed {mock_data_seed}): real MUSE cutouts "
            "newnewcutouts/*.fits are absent from the local mirror (BLOCKED; "
            "on Perlmutter under linusu's GIGALens-Code). Posterior geometry "
            "is NOT theirs. Substrate: vendored gigalens-linus @80916d2. "
            "UNCERTIFIED (external)."),
        meta=dict(tier="scene", system="carousel",
                  data="MOCK STAND-IN (real cutouts BLOCKED)",
                  blocked="newnewcutouts/source4-5.fits + source9.fits "
                          "(DATA/STAT/PSF/MASK) not on local mirror",
                  z_true=np.asarray(z_true).tolist()),
    )


# --------------------------------------------------------------------------- #
# hundred-systems (hundred_systems_GL2), systems 0..7, scene-API re-expression
# --------------------------------------------------------------------------- #
def _hs2_prior_components():
    """Priors copied verbatim (int literals kept) from GIGALens-Code@eb2a09b6
    experiments/hundred_systems_GL2/setup.py::make_default_prior ('the default
    prior from the original GIGALens paper'), re-expressed as scene Components
    for the old-API PhysicalModel([EPL(50), Shear()], [SersicEllipse],
    [SersicEllipse]) of hundredsystems.py."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.scene import Component

    tfd = tfp.distributions
    epl = Component(EPL(50), dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
        gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
        e1=tfd.Normal(0, 0.2), e2=tfd.Normal(0, 0.2),
        center_x=tfd.Normal(0, 0.06), center_y=tfd.Normal(0, 0.06)))
    shear = Component(Shear(), dict(gamma1=tfd.Normal(0, 0.1),
                                    gamma2=tfd.Normal(0, 0.1)))
    lens_light = Component(SersicEllipse(use_lstsq=False), dict(
        R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
        n_sersic=tfd.Uniform(0.5, 8),
        e1=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
        e2=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
        center_x=tfd.Normal(0, 0.02), center_y=tfd.Normal(0, 0.02),
        Ie=tfd.LogNormal(jnp.log(300.0), 0.5)))
    source = Component(SersicEllipse(use_lstsq=False), dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
        n_sersic=tfd.Uniform(0.5, 8),
        e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
        center_x=tfd.Normal(0, 0.5), center_y=tfd.Normal(0, 0.5),
        Ie=tfd.LogNormal(jnp.log(150.0), 0.9)))
    return epl, shear, lens_light, source


HS2_BACKGROUND_RMS, HS2_EXP_TIME = 0.2, 100
HS2_DELTA_PIX, HS2_NUM_PIX, HS2_SUPERSAMPLE = 0.065, 80, 2


def build_hs2(sys_idx: int) -> Target:
    """hundred_systems_GL2 system re-expressed on the scene API (single mass
    plane EPL(50)+Shear with Sersic lens light; Sersic source on the default
    deflection_ratio=1.0 plane; forward mode, background_rms=0.2 exp_time=100,
    delta_pix=0.065 num_pix=80 ss=2, PSF = vendored gigalens/assets/psf.npy --
    the same asset their hundredsystems.py loads).

    DATA partially BLOCKED: their observed images
    (SystemSaves/100SystemsStandard80px.npz) and truth params yaml are ABSENT
    from the local mirror, so system `sys_idx` here is OUR seeded mock: truth
    drawn from the same prior (PRNGKey(1000+sys_idx)), rendered with the scene
    simulator, noise added with np.random.seed(3000+sys_idx) via the same
    lenstronomy add_poisson/add_background recipe as their simulated datasets.
    Model construction + priors are theirs; the specific realizations are not."""
    from lenstronomy.Util import image_util

    import jax

    from gigalens.jax.scene import LensModel, Plane
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.simulator import SimulatorConfig

    from cgl2 import paths

    sys_idx = int(sys_idx)
    if not (0 <= sys_idx <= 7):
        raise ValueError("hs2 systems built here are sys 0..7")

    epl, shear, lens_light, source = _hs2_prior_components()
    model = LensModel([
        Plane(mass=[epl, shear], light=[lens_light]),
        Plane(light=[source]),          # single lensed source: ratio 1.0 default
    ])

    kernel = np.load(paths.VENDOR_DIR / "src" / "gigalens" / "assets"
                     / "psf.npy").astype(np.float64)
    cfg = SimulatorConfig(delta_pix=HS2_DELTA_PIX, num_pix=HS2_NUM_PIX,
                          supersample=HS2_SUPERSAMPLE, kernel=kernel)

    # seeded mock observation (their npz BLOCKED -- see docstring)
    key = jax.random.PRNGKey(1000 + sys_idx)
    z_true = model.bijector.inverse(model.prior.sample(seed=key))
    params_true = model.constrained(z_true)
    sim = SceneSimulator(model, cfg, sees=[lens_light, source])
    img = np.asarray(sim.simulate(params_true), dtype=np.float64)
    np.random.seed(3000 + sys_idx)
    img = (img + image_util.add_poisson(img, exp_time=HS2_EXP_TIME)
           + image_util.add_background(img, sigma_bkd=HS2_BACKGROUND_RMS))

    ds = ImageData(img, cfg, background_rms=HS2_BACKGROUND_RMS,
                   exp_time=HS2_EXP_TIME, sees="all")
    pm = ProbModel(model, ds, mode="forward")

    return _wrap_prob_model(
        f"hs2_sys{sys_idx}", pm,
        provenance=(
            f"scene-API re-expression of GIGALens-Code@{GIGALENS_CODE_COMMIT} "
            "experiments/hundred_systems_GL2/hundredsystems.py (old-API "
            "PhysicalModel EPL(50)+Shear / Sersic / Sersic, ForwardProbModel "
            "bg_rms=0.2 exp_time=100, SimulatorConfig 0.065\"/80px/ss2, "
            "assets/psf.npy) with the make_default_prior priors from their "
            "setup.py, VERBATIM. DATA = OUR seeded mock (their "
            "SystemSaves/100SystemsStandard80px.npz + params yaml absent from "
            "the local mirror -- BLOCKED); truth from the same prior, "
            f"PRNGKey({1000 + sys_idx}), noise np.random.seed({3000 + sys_idx})"
            " + lenstronomy add_poisson/add_background. Substrate: vendored "
            "gigalens-linus @80916d2, forward mode. UNCERTIFIED (external)."),
        meta=dict(tier="scene", system="hundred_systems", sys_idx=sys_idx,
                  data="OUR seeded mock (their npz BLOCKED)",
                  blocked="SystemSaves/100SystemsStandard80px.npz + "
                          "100SystemsStandardParams.yaml not on local mirror",
                  z_true=np.asarray(z_true).tolist()),
    )


# =========================================================================== #
# registry
# =========================================================================== #
_BUILDERS = {
    "t0_gauss2": build_gauss2,
    "t0_mix2": build_mix2,
    "t0_funnel10": build_funnel10,
    "t0_illcond46": build_illcond46,
    "dspl20_orig": lambda: build_dspl20("orig"),
    "dspl20_ratio": lambda: build_dspl20("ratio"),
    "carousel33": build_carousel33,
    **{f"hs2_sys{i}": (lambda i=i: build_hs2(i)) for i in range(8)},
}

T0_NAMES = ("t0_mix2", "t0_funnel10", "t0_illcond46")
SCENE_NAMES = ("dspl20_orig", "dspl20_ratio", "carousel33",
               *[f"hs2_sys{i}" for i in range(8)])


def build(name: str) -> Target:
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(f"unknown zoo target {name!r}; "
                       f"known: {sorted(_BUILDERS)}")
    return builder()
