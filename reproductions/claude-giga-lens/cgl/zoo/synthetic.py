r"""T0 synthetic zoo targets: fully-analytic posteriors that validate the
metrics + evidence machinery before any GPU burn.

All targets use an explicit conjugate construction so that log_prior,
log_like, log_prob, logZ, posterior moments and (for mixtures) posterior
mode weights are ALL analytic:

    prior     p(z)   = N(0, sigma_p^2 I)
    likelihood l(z)  = declared density (mixture / funnel-ratio / Gaussian)
    log_prob         = log p + log l        (unnormalized posterior)
    logZ             = log \int l(z) p(z) dz   (analytic / exact by construction)

Targets (registry names):
    t0_mix2       2-D  f32  two-mode Gaussian mixture, weights 0.8/0.2
    t0_mix2_f64   2-D  f64  same (f64 kernel-path testing, MCLMC family)
    t0_mix22      22-D f32  same construction at gu-2022 dimensionality
    t0_funnel10   10-D f32  Neal funnel (theta ~ N(0,3), x_i ~ N(0, e^{theta/2}))
    t0_illcond46  46-D f64  ill-conditioned Gaussian, cond 1e14 (mimics T2)

Every closure uses EXPLICIT dtypes so f32 targets build correctly even in an
x64 process (the CPU unit tests run under conftest's global x64).
"""
from __future__ import annotations

import numpy as np

from cgl import guards
from cgl.zoo.api import InitBundle, LensPosterior, Reference, assert_dtype_env, np_dtype

_LOG2PI = float(np.log(2.0 * np.pi))


def _labels(dim: int) -> list:
    return [f"z{i:02d}" for i in range(dim)]


def _to_physical_factory(labels, mass_labels):
    idx = [labels.index(m) for m in mass_labels]

    def to_physical(Z):
        Z = np.asarray(Z)
        flat = Z.reshape(-1, Z.shape[-1])
        return {labels[i]: flat[:, i] for i in idx}

    return to_physical


def _floored_svi(cov: np.ndarray):
    cov_reg, chol, n_floored = guards.floor_svi_covariance(cov)
    return chol, n_floored


def _gauss_prior_bundle(dim, dtype, sigma_p, map_z, svi_loc, svi_cov,
                        multi_starts=None, notes=""):
    fdtype = np_dtype(dtype)

    def prior_sample_fn(key, n: int):
        import jax
        import jax.numpy as jnp
        return jnp.asarray(sigma_p, dtype=fdtype) * jax.random.normal(
            key, (n, dim), dtype=fdtype)

    chol, n_floored = _floored_svi(svi_cov)
    return InitBundle(
        prior_sample_fn=prior_sample_fn,
        map_z=np.asarray(map_z, dtype=np.float64),
        svi_loc=np.asarray(svi_loc, dtype=np.float64),
        svi_scale_tril=chol,
        svi_n_floored=n_floored,
        multi_starts=(None if multi_starts is None
                      else np.asarray(multi_starts, dtype=np.float64)),
        notes=notes,
    )


def _gauss_prior_transform(dim, sigma_p):
    """Unit cube -> z (identity bijector: physical x == z) for nautilus."""
    from scipy.special import ndtri

    def prior_transform(u):
        return sigma_p * ndtri(np.clip(np.asarray(u, dtype=np.float64),
                                       1e-14, 1.0 - 1e-14))

    return prior_transform


# --------------------------------------------------------------------------- #
# (a) two-mode Gaussian mixture (conjugate; weights/logZ/moments analytic)
# --------------------------------------------------------------------------- #
def build_mixture(name: str, dim: int, dtype: str, weights=(0.8, 0.2),
                  mode_x0: float = 2.0, comp_sigma: float = 0.5,
                  prior_sigma: float = 5.0) -> LensPosterior:
    """likelihood = w1 N(mu1, s^2 I) + w2 N(mu2, s^2 I), mu_k = ±mode_x0 e0;
    prior = N(0, sigma_p^2 I). Symmetric |mu_k| keeps the POSTERIOR mode
    weights exactly = `weights`."""
    assert_dtype_env(dtype)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    fdtype = np_dtype(dtype)
    w = np.asarray(weights, dtype=np.float64)
    assert abs(w.sum() - 1.0) < 1e-12
    mus = np.zeros((2, dim))
    mus[0, 0], mus[1, 0] = mode_x0, -mode_x0
    s2, sp2 = comp_sigma ** 2, prior_sigma ** 2

    # conjugate posterior per component
    post_var = 1.0 / (1.0 / s2 + 1.0 / sp2)
    post_means = mus * (post_var / s2)
    # component evidences  \int N(z; mu_k, s^2 I) N(z; 0, sp^2 I) dz
    ev2 = s2 + sp2
    log_ev = -0.5 * dim * (_LOG2PI + np.log(ev2)) - 0.5 * (mus ** 2).sum(1) / ev2
    logZ = float(np.logaddexp(np.log(w[0]) + log_ev[0], np.log(w[1]) + log_ev[1]))
    post_w = np.exp(np.log(w) + log_ev - logZ)          # == weights (symmetric)

    # analytic posterior moments
    mean_vec = (post_w[:, None] * post_means).sum(0)
    second = (post_w[:, None] * (post_means ** 2 + post_var)).sum(0)
    var_vec = second - mean_vec ** 2

    mus_j = jnp.asarray(mus, dtype=fdtype)
    logw_j = jnp.asarray(np.log(w), dtype=fdtype)
    c_like = jnp.asarray(-0.5 * dim * (_LOG2PI + np.log(s2)), dtype=fdtype)
    c_prior = jnp.asarray(-0.5 * dim * (_LOG2PI + np.log(sp2)), dtype=fdtype)
    inv2s2 = jnp.asarray(0.5 / s2, dtype=fdtype)
    inv2sp2 = jnp.asarray(0.5 / sp2, dtype=fdtype)

    def log_like_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        d2 = jnp.sum((Z[:, None, :] - mus_j[None, :, :]) ** 2, axis=-1)  # (N,2)
        return logsumexp(logw_j[None, :] - inv2s2 * d2, axis=-1) + c_like

    def log_prior_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        return c_prior - inv2sp2 * jnp.sum(Z ** 2, axis=-1)

    def log_prob_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        d2 = jnp.sum((Z[:, None, :] - mus_j[None, :, :]) ** 2, axis=-1)
        lik = logsumexp(logw_j[None, :] - inv2s2 * d2, axis=-1) + c_like
        pri = c_prior - inv2sp2 * jnp.sum(Z ** 2, axis=-1)
        return lik + pri

    def exact_sample_fn(seed: int, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        comp = rng.choice(2, size=n, p=post_w)
        return (post_means[comp]
                + np.sqrt(post_var) * rng.standard_normal((n, dim)))

    def log_like_x(x):
        return np.asarray(log_like_batch(np.asarray(x, dtype=fdtype)))

    labels = _labels(dim)
    mass_labels = ["z00"]
    reference = Reference(
        provenance=("analytic: conjugate Gaussian mixture, "
                    f"w={tuple(w)}, mu0=±{mode_x0} e0, comp_sigma={comp_sigma}, "
                    f"prior_sigma={prior_sigma}; logZ/moments/mode weights in "
                    "closed form (cgl.zoo.synthetic.build_mixture)"),
        mode_labels=("mode_pos", "mode_neg"),
        mode_weights=post_w,
        mode_weights_trusted=True,
        mode_centers=post_means,
        mode_covs=np.stack([np.eye(dim) * post_var] * 2),
        mode_assigner={"method": "mahalanobis"},
        logZ=logZ,
        truth={"mean": mean_vec.tolist(), "var": var_vec.tolist(),
               "post_var_component": post_var},
        exact_sample_fn=exact_sample_fn,
    )
    init = _gauss_prior_bundle(
        dim, dtype, prior_sigma,
        map_z=post_means[0], svi_loc=post_means[0],
        svi_cov=np.eye(dim) * post_var,
        multi_starts=post_means,
        notes="map_z/svi = DOMINANT posterior mode (what a mode-seeking "
              "MAP->SVI finds); multi_starts = both mode means.")
    return LensPosterior(
        name=name, tier="T0", dim=dim, dtype=dtype, noise_model="diag",
        log_prob_batch=log_prob_batch, log_prior_batch=log_prior_batch,
        log_like_batch_fn=log_like_batch,
        labels=labels, mass_labels=mass_labels,
        to_physical=_to_physical_factory(labels, mass_labels),
        init=init, bijector=None,
        prior_transform=_gauss_prior_transform(dim, prior_sigma),
        log_like_x=log_like_x,
        reference=reference,
        meta={"family": "gaussian_mixture", "prior_sigma": prior_sigma,
              "comp_sigma": comp_sigma, "mode_x0": mode_x0},
    )


# --------------------------------------------------------------------------- #
# (b) Neal funnel 10-D
# --------------------------------------------------------------------------- #
def build_funnel(name: str = "t0_funnel10", dim: int = 10,
                 dtype: str = "float32", prior_sigma: float = 3.0
                 ) -> LensPosterior:
    """posterior = Neal funnel: theta=z0 ~ N(0,3), z_i ~ N(0, e^{theta/2}).
    prior = N(0, prior_sigma^2 I); like = funnel - prior, so logZ = 0 EXACTLY
    (funnel is normalized)."""
    assert_dtype_env(dtype)
    import jax.numpy as jnp

    fdtype = np_dtype(dtype)
    k = dim - 1
    c_half = jnp.asarray(0.5, dtype=fdtype)
    log3 = jnp.asarray(np.log(3.0), dtype=fdtype)
    c2pi = jnp.asarray(0.5 * _LOG2PI, dtype=fdtype)
    sp2 = prior_sigma ** 2
    c_prior = jnp.asarray(-0.5 * dim * (_LOG2PI + np.log(sp2)), dtype=fdtype)
    inv2sp2 = jnp.asarray(0.5 / sp2, dtype=fdtype)

    def _funnel_logpdf(Z):
        th, x = Z[:, 0], Z[:, 1:]
        lp_th = -c2pi - log3 - c_half * (th / 3.0) ** 2
        scale = jnp.exp(c_half * th)[:, None]                # e^{theta/2}
        lp_x = jnp.sum(-c2pi - c_half * th[:, None] - c_half * (x / scale) ** 2,
                       axis=-1)
        return lp_th + lp_x

    def log_prob_batch(Z):
        return _funnel_logpdf(jnp.asarray(Z, dtype=fdtype))

    def log_prior_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        return c_prior - inv2sp2 * jnp.sum(Z ** 2, axis=-1)

    def log_like_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        return _funnel_logpdf(Z) - (c_prior - inv2sp2 * jnp.sum(Z ** 2, axis=-1))

    def exact_sample_fn(seed: int, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        th = 3.0 * rng.standard_normal(n)
        x = np.exp(th / 2.0)[:, None] * rng.standard_normal((n, k))
        return np.concatenate([th[:, None], x], axis=1)

    def log_like_x(x):
        return np.asarray(log_like_batch(np.asarray(x, dtype=fdtype)))

    labels = _labels(dim)
    mass_labels = ["z00", "z01"]
    reference = Reference(
        provenance=(f"analytic: {dim}-D Neal funnel (theta~N(0,3), "
                    "x_i~N(0, e^{theta/2})); prior N(0, %g^2 I); logZ=0 exact "
                    "by construction" % prior_sigma),
        logZ=0.0,
        truth={"var_theta": 9.0, "mean_theta": 0.0,
               "var_x": float(np.exp(4.5)),
               # E[log x_i^2] = E[theta] + E[log chi2_1] = psi(1/2) + log 2
               "mean_log_x2": float(-1.2703628454614782)},
        exact_sample_fn=exact_sample_fn,
    )
    init = _gauss_prior_bundle(
        dim, dtype, prior_sigma,
        map_z=np.zeros(dim), svi_loc=np.zeros(dim),
        svi_cov=np.diag([9.0] + [np.exp(4.5)] * k),
        notes="svi cov = marginal posterior variances (a funnel has no useful "
              "global Gaussian approx; this is the honest moment-matched one).")
    return LensPosterior(
        name=name, tier="T0", dim=dim, dtype=dtype, noise_model="diag",
        log_prob_batch=log_prob_batch, log_prior_batch=log_prior_batch,
        log_like_batch_fn=log_like_batch,
        labels=labels, mass_labels=mass_labels,
        to_physical=_to_physical_factory(labels, mass_labels),
        init=init, bijector=None,
        prior_transform=_gauss_prior_transform(dim, prior_sigma),
        log_like_x=log_like_x,
        reference=reference,
        meta={"family": "neal_funnel", "prior_sigma": prior_sigma},
    )


# --------------------------------------------------------------------------- #
# (c) ill-conditioned Gaussian, dim 46, cond 1e14 (mimics T2)
# --------------------------------------------------------------------------- #
def build_illcond(name: str = "t0_illcond46", dim: int = 46,
                  dtype: str = "float64", cond: float = 1e14,
                  prior_sigma: float = 10.0) -> LensPosterior:
    """likelihood = N(0, diag(lam)), lam_i log-spaced 1..1/cond (DIAGONAL
    conditioning, exactly the T2 structure); prior = N(0, prior_sigma^2 I).
    Posterior var_i = (1/lam_i + 1/sp^2)^-1; logZ analytic."""
    assert_dtype_env(dtype)
    import jax.numpy as jnp

    fdtype = np_dtype(dtype)
    lam = np.logspace(0.0, -np.log10(cond), dim)      # variances, cond = 1e14
    sp2 = prior_sigma ** 2
    post_var = 1.0 / (1.0 / lam + 1.0 / sp2)
    logZ = float(np.sum(-0.5 * (_LOG2PI + np.log(lam + sp2))))
    c_like = float(-0.5 * dim * _LOG2PI - 0.5 * np.log(lam).sum())
    c_prior = float(-0.5 * dim * (_LOG2PI + np.log(sp2)))

    lam_j = jnp.asarray(lam, dtype=fdtype)
    c_like_j = jnp.asarray(c_like, dtype=fdtype)
    c_prior_j = jnp.asarray(c_prior, dtype=fdtype)
    inv2sp2 = jnp.asarray(0.5 / sp2, dtype=fdtype)

    def log_like_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        return c_like_j - 0.5 * jnp.sum(Z ** 2 / lam_j[None, :], axis=-1)

    def log_prior_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        return c_prior_j - inv2sp2 * jnp.sum(Z ** 2, axis=-1)

    def log_prob_batch(Z):
        Z = jnp.asarray(Z, dtype=fdtype)
        return (c_like_j - 0.5 * jnp.sum(Z ** 2 / lam_j[None, :], axis=-1)
                + c_prior_j - inv2sp2 * jnp.sum(Z ** 2, axis=-1))

    def exact_sample_fn(seed: int, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.sqrt(post_var)[None, :] * rng.standard_normal((n, dim))

    def log_like_x(x):
        return np.asarray(log_like_batch(np.asarray(x, dtype=fdtype)))

    labels = _labels(dim)
    mass_labels = ["z00", "z22", "z45"]
    reference = Reference(
        provenance=(f"analytic: {dim}-D Gaussian, likelihood cov diag "
                    f"logspace(1..1e-{int(np.log10(cond))}), prior N(0, "
                    f"{prior_sigma}^2 I); posterior cond ~ "
                    f"{post_var.max() / post_var.min():.2e}"),
        logZ=logZ,
        truth={"mean": [0.0] * dim, "var": post_var.tolist(),
               "cond_posterior": float(post_var.max() / post_var.min())},
        exact_sample_fn=exact_sample_fn,
        mode_centers=np.zeros((1, dim)),
        mode_covs=np.diag(post_var)[None, :, :],
        mode_labels=("mode0",),
        mode_weights=np.array([1.0]),
        mode_weights_trusted=True,
        mode_assigner={"method": "mahalanobis"},
    )
    init = _gauss_prior_bundle(
        dim, dtype, prior_sigma,
        map_z=np.zeros(dim), svi_loc=np.zeros(dim),
        svi_cov=np.diag(post_var),
        notes="svi cov = exact posterior cov BEFORE the guard floor; "
              "floor_svi_covariance raises eigenvalues < 1e-10*max "
              "(svi_n_floored records how many) -- the documented Bug-2 "
              "guard behavior, deliberately kept.")
    # diagraw-analog for the T2-style preconditioned path
    init.hess_diag_raw = 1.0 / post_var
    return LensPosterior(
        name=name, tier="T0", dim=dim, dtype=dtype, noise_model="diag",
        log_prob_batch=log_prob_batch, log_prior_batch=log_prior_batch,
        log_like_batch_fn=log_like_batch,
        labels=labels, mass_labels=mass_labels,
        to_physical=_to_physical_factory(labels, mass_labels),
        init=init, bijector=None,
        prior_transform=_gauss_prior_transform(dim, prior_sigma),
        log_like_x=log_like_x,
        reference=reference,
        meta={"family": "illconditioned_gaussian", "cond": cond,
              "prior_sigma": prior_sigma},
    )


# --------------------------------------------------------------------------- #
# registry factories
# --------------------------------------------------------------------------- #
def build_mix2() -> LensPosterior:
    return build_mixture("t0_mix2", dim=2, dtype="float32")


def build_mix2_f64() -> LensPosterior:
    return build_mixture("t0_mix2_f64", dim=2, dtype="float64")


def build_mix22() -> LensPosterior:
    return build_mixture("t0_mix22", dim=22, dtype="float32")


def build_funnel10() -> LensPosterior:
    return build_funnel("t0_funnel10", dim=10, dtype="float32")


def build_illcond46() -> LensPosterior:
    return build_illcond("t0_illcond46", dim=46, dtype="float64")
