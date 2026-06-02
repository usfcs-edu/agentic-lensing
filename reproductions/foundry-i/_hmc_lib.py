"""Shared HMC/diagnostic library for the foundry-i gigalens reproduction.

Public API (used by 26_compile_diagnostic.py and later scan scripts):
    build_model() -> SimpleNamespace
    sigma_hat(jitter=1e-9) -> jnp (74,74) float32 jittered empirical cov
    momentum_distribution(mode='inv') -> tfd.Distribution over R^74
    to_physical_mass(samples_np, prob_model=None) -> dict of 6 mass params
    print_comparison(combined, extra_cols=None) -> None
    PAPER, V10  (reference dicts, verbatim from 25_fit_nuts_v11f.py)

The model build (EPL(50)+Shear, 4x SersicEllipse lens light, SersicEllipse +
Shapelets(n_max=6) source) and the masked single-device log-prob are an exact
copy of 25_fit_nuts_v11f.py so target_log_prob_fn is identical to v11f.
"""
import functools as _ft
import types as _types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.model import ForwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions
_REPRO = Path(__file__).parent
_DATA = _REPRO / "data"
N_MAX = 6
EXP_TIME = 1197.7

# Reference dicts (verbatim from 25_fit_nuts_v11f.py lines 251-260)
PAPER = dict(
    theta_E=(2.6463, 0.0017), gamma=(1.372, 0.023),
    e1=(0.1091, 0.0020), e2=(-0.1320, 0.0020),
    gamma1=(0.0657, 0.0024), gamma2=(-0.0939, 0.0022),
)
V10 = dict(
    theta_E=(+2.5659, 0.0003), gamma=(+2.1507, 0.0004),
    e1=(+0.1064, 0.0004), e2=(-0.0533, 0.0002),
    gamma1=(+0.0399, 0.0003), gamma2=(-0.0676, 0.0002),
)


def _sersic_prior(R_med, R_sig, Ie_med, Ie_sig, n_lo=0.5, n_hi=8.0,
                  cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
        Ie=tfd.LogNormal(jnp.log(Ie_med), Ie_sig),
    ))


def build_model():
    """Return a SimpleNamespace with the v11f-identical forward model + log-prob."""
    with fits.open(_DATA / "cutout_F140W.fits") as h:
        sci = h["SCI"].data.astype(np.float32)
        wht = h["WHT"].data.astype(np.float32)
    sky = float(np.median(sci))
    data_arr = sci - sky
    background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
    kernel = np.load(_DATA / "empirical_psf.npy").astype(np.float32)
    nb = np.load(_DATA / "nearby_galaxy_loc.npz")
    NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])
    NUM_PIX = data_arr.shape[0]
    sim_config = SimulatorConfig(delta_pix=0.13, num_pix=NUM_PIX, supersample=2,
                                 kernel=kernel)

    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)] * 4,
        [sersic.SersicEllipse(use_lstsq=False),
         shapelets.Shapelets(n_max=N_MAX, use_lstsq=False, interpolate=False)],
    )

    lens_mass_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
            gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
            e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
            center_x=tfd.Normal(0.0, 0.02), center_y=tfd.Normal(0.0, 0.02),
        )),
        tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05),
                                         gamma2=tfd.Normal(0.0, 0.05))),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        _sersic_prior(0.4, 0.3, 5.0, 0.5, c_sig=0.02),
        _sersic_prior(2.0, 0.3, 2.0, 0.5, c_sig=0.02),
        _sersic_prior(0.3, 0.3, 1.0, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
        _sersic_prior(0.6, 0.3, 0.5, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
    ])
    src_sersic_prior = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
        Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
    ))
    shp_amp_names = shapelets.Shapelets(n_max=N_MAX)._amp_names
    amp_priors = {name: tfd.Normal(0.0, 5.0 / float(jnp.sqrt(i + 1)))
                  for i, name in enumerate(shp_amp_names)}
    src_shp_prior = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
        **amp_priors,
    ))
    source_light_prior = tfd.JointDistributionSequential(
        [src_sersic_prior, src_shp_prior])
    prior = tfd.JointDistributionSequential(
        [lens_mass_prior, lens_light_prior, source_light_prior])

    yy, xx = np.indices(data_arr.shape)
    r_center = np.sqrt((xx - NUM_PIX // 2) ** 2 + (yy - NUM_PIX // 2) ** 2)
    keep_mask = r_center > 1.5
    prob_model = ForwardProbModel(prior, data_arr, background_rms=background_rms,
                                  exp_time=EXP_TIME)

    INF_ERR = jnp.float32(1e10)
    mask_jax = jnp.asarray(keep_mask)

    @_ft.partial(jax.jit, static_argnums=(0, 1))
    def masked_log_prob(self, simulator, z):
        z = list(z.T)
        x = self.bij.forward(z)
        im_sim = simulator.simulate(x)
        err_map = jnp.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
        err_map = jnp.where(mask_jax, err_map, INF_ERR)
        log_like = tfd.Independent(
            tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
        ).log_prob(self.observed_image)
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
        chisq = jnp.mean(
            jnp.where(mask_jax, ((im_sim - self.observed_image) / err_map) ** 2, 0.0),
            axis=(-2, -1))
        return log_like + log_prior, chisq

    prob_model.log_prob = _types.MethodType(masked_log_prob, prob_model)
    lens_sim_bs1 = LensSimulator(phys_model, sim_config, bs=1)

    @jax.jit
    def target_log_prob_fn(z):
        z_batched = z[None, :]
        lp, _ = prob_model.log_prob(lens_sim_bs1, z_batched)
        return jnp.squeeze(lp)

    v7pm = np.load(_DATA / "map_v7_paper_mode.npz")
    qz_start = jnp.asarray(v7pm["best_params"].astype(np.float32))
    ndim = int(qz_start.shape[0])

    return _types.SimpleNamespace(
        target_log_prob_fn=target_log_prob_fn,
        prob_model=prob_model,
        phys_model=phys_model,
        sim_config=sim_config,
        ndim=ndim,
        data_arr=data_arr,
        qz_start=qz_start,
    )


def sigma_hat(jitter=1e-9):
    """Jittered empirical covariance of v11f NUTS samples (74x74 float32)."""
    s = np.load(_DATA / "nuts_v11f_samples.npz")["samples"]
    cov = np.cov(s.T).astype(np.float32)
    cov = 0.5 * (cov + cov.T)
    raw_min = float(np.linalg.eigvalsh(cov).min())
    cov_j = cov + jitter * np.eye(cov.shape[0], dtype=cov.dtype)
    cov_j = 0.5 * (cov_j + cov_j.T)
    eig = np.linalg.eigvalsh(cov_j)
    min_eig = float(eig.min())
    assert min_eig > 0, f"jittered min eig not positive: {min_eig}"
    cond = float(eig.max() / eig.min())
    print(f"[sigma_hat] raw min eig={raw_min:.3e}, jittered min eig={min_eig:.3e}, "
          f"cond={cond:.3e}", flush=True)
    return jnp.asarray(cov_j, dtype=jnp.float32)


_HESS_CACHE = _DATA / "hess_massmatrix.npz"


def laplace_hessian(force=False, eig_floor_frac=1e-6):
    """Compute (or load) the Laplace mass matrix H_reg at the v7 MAP.

    The textbook Laplace approximation uses the curvature of the negative
    log-posterior at the mode as the inverse posterior covariance:

        H = d^2/dz^2 [ -target_log_prob_fn(z) ]  at z = qz_start

    By the TFP preconditioned-HMC convention the optimal mass matrix is
    M = Sigma_post^-1, which under Laplace is M ~= H.  Unlike Sigma_hat
    (the empirical cov of the v11f NUTS chain), H needs no mixed chain to
    estimate the posterior scales -- it is the exact LOCAL precision.

    Returns a dict with H_reg (74,74 float32, PD) plus the raw spectrum
    diagnostics.  Result is cached to data/hess_massmatrix.npz; pass
    force=True to recompute.
    """
    if _HESS_CACHE.exists() and not force:
        d = np.load(_HESS_CACHE)
        return {k: d[k] for k in d.files}

    m = build_model()
    # Hessian of the NEGATIVE log-posterior at the MAP start (full 74x74).
    neg_lp = lambda z: -m.target_log_prob_fn(z)
    H = np.asarray(jax.hessian(neg_lp)(m.qz_start)).astype(np.float64)
    H = 0.5 * (H + H.T)  # symmetrize

    eig, V = np.linalg.eigh(H)
    eig_min_raw = float(eig.min())
    eig_max_raw = float(eig.max())
    n_neg = int(np.sum(eig <= 0.0))

    # Regularize to PD: floor eigenvalues to a fraction of the max eigenvalue.
    floor = eig_floor_frac * eig_max_raw
    eig_floored = np.maximum(eig, floor)
    H_reg = (V * eig_floored) @ V.T
    H_reg = 0.5 * (H_reg + H_reg.T)  # symmetrize again

    chol = np.linalg.cholesky(H_reg)  # raises if not PD
    eig_reg = np.linalg.eigvalsh(H_reg)
    cond_after_floor = float(eig_reg.max() / eig_reg.min())

    out = dict(
        H_reg=H_reg.astype(np.float32),
        H_raw=H.astype(np.float32),
        eig_raw=eig.astype(np.float64),
        eig_min_raw=np.float64(eig_min_raw),
        eig_max_raw=np.float64(eig_max_raw),
        n_negative_eigs=np.int64(n_neg),
        eig_floor=np.float64(floor),
        cond_after_floor=np.float64(cond_after_floor),
    )
    _DATA.mkdir(parents=True, exist_ok=True)
    np.savez(_HESS_CACHE, **out)
    print(f"[laplace_hessian] cached H_reg -> {_HESS_CACHE}", flush=True)
    print(f"[laplace_hessian] eig_min_raw={eig_min_raw:.6e} "
          f"eig_max_raw={eig_max_raw:.6e} n_neg={n_neg} "
          f"cond_after_floor={cond_after_floor:.3e} chol_ok=True", flush=True)
    return out


def momentum_distribution(mode='inv'):
    """Momentum distribution over R^74, loc=zeros.

    mode='inv' (default, CORRECT): covariance = Sigma_hat^-1 via
        MultivariateNormalPrecisionFactorLinearOperator(precision_factor=chol(Sigma_hat),
        precision=Sigma_hat).
    mode='fwd' (WRONG, v11f bug control): covariance = Sigma_hat via
        MultivariateNormalTriL(scale_tril=chol(Sigma_hat)).
    mode='diag': covariance = diag(1/var_i) via
        MultivariateNormalDiag(scale_diag=1/sqrt(diag(Sigma_hat))).
    mode='hess' (Laplace curvature): momentum COVARIANCE = H_reg, the
        regularized Hessian of -log posterior at the v7 MAP. Under the TFP
        preconditioned-HMC convention the optimal mass matrix M = Sigma_post^-1
        ~= H (Laplace precision); momentum p ~ N(0, M), so the momentum
        distribution's covariance must equal H_reg. Built as
        MultivariateNormalTriL(scale_tril=cholesky(H_reg)). H_reg is loaded
        from data/hess_massmatrix.npz if present, else computed and cached.
        This avoids relying on the under-scaled chain-based Sigma_hat.
    """
    if mode == 'hess':
        H_reg = jnp.asarray(laplace_hessian()["H_reg"], dtype=jnp.float32)
        ndim = H_reg.shape[0]
        loc = jnp.zeros(ndim, dtype=jnp.float32)
        chol = jnp.linalg.cholesky(H_reg)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=chol)

    S = sigma_hat()
    ndim = S.shape[0]
    loc = jnp.zeros(ndim, dtype=jnp.float32)
    chol = jnp.linalg.cholesky(S)
    lo = tfp.tf2jax.linalg
    if mode == 'inv':
        return tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
            loc=loc,
            precision_factor=lo.LinearOperatorLowerTriangular(chol),
            precision=lo.LinearOperatorFullMatrix(S),
        )
    elif mode == 'fwd':
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=chol)
    elif mode == 'diag':
        scale_diag = 1.0 / jnp.sqrt(jnp.diag(S))
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
    else:
        raise ValueError(f"unknown mode {mode!r}")


_CACHED_BUILD = None


def _cached_build():
    global _CACHED_BUILD
    if _CACHED_BUILD is None:
        _CACHED_BUILD = build_model()
    return _CACHED_BUILD


def to_physical_mass(samples_np, prob_model=None):
    """Push unconstrained samples through bij.forward, return 6 mass params."""
    if prob_model is None:
        prob_model = _cached_build().prob_model
    arr = jnp.asarray(samples_np)
    physical = prob_model.bij.forward(list(arr.T))
    mass_main = physical[0][0]
    mass_shear = physical[0][1]
    out = {k: np.asarray(mass_main[k]) for k in mass_main.keys()}
    out.update({k: np.asarray(mass_shear[k]) for k in mass_shear.keys()})
    return out


def print_comparison(combined, extra_cols=None):
    """Print a v11f-style comparison table."""
    nuts = to_physical_mass(np.load(_DATA / "nuts_v11f_samples.npz")["samples"])
    extra_cols = extra_cols or {}
    header = (f"  {'param':>10s}    {'ours median ± 1σ':>22s}    "
              f"{'v11f NUTS':>17s}    {'v10 SVI':>17s}    {'paper':>17s}    "
              f"{'ours-paper':>10s}")
    for cn in extra_cols:
        header += f"    {cn:>12s}"
    print(header, flush=True)
    for k, (mu_p, sig_p) in PAPER.items():
        a = combined.get(k)
        if a is None:
            continue
        med = float(np.median(a))
        lo = float(np.percentile(a, 16))
        hi = float(np.percentile(a, 84))
        nm = float(np.median(nuts[k])) if k in nuts else float('nan')
        ns = float(np.std(nuts[k])) if k in nuts else float('nan')
        v10m, v10s = V10[k]
        line = (f"  {k:>10s}    {med:+7.4f} (+{hi-med:.4f}/{lo-med:+.4f})    "
                f"{nm:+7.4f}±{ns:.4f}    {v10m:+7.4f}±{v10s:.4f}    "
                f"{mu_p:+7.4f}±{sig_p:.4f}    {med-mu_p:+8.4f}")
        for cn, cd in extra_cols.items():
            v = cd.get(k, float('nan'))
            line += f"    {v:+12.4f}"
        print(line, flush=True)
