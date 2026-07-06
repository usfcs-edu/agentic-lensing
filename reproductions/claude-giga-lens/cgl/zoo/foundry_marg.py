"""T2 zoo target: the foundry-i 46-dim ridge-marginalized REAL posterior
(DESI-165.4754-06.0423, cutout_F140W, cond~1e14). float64 ONLY.

Thin wrap of cgl.likelihood.build_marg_model in its PARITY configuration --
the product dict below mirrors 01_parity_harness.build_parity_harness's
`build_parity_product` (itself a line-by-line mirror of foundry-i
_hmc_lib_marg's data block), so the zoo log-posterior is the P0-gated one:
logp(qz_refined) = -45840.984005998456 (data/parity_report.json, gates A-E
all 0.0 vs foundry-i).

Init: MAP_MARG_PD qz_refined (the trust-exact PD mode) + HESS_MARG_PD
(Laplace covariance inv(H_reg) -> guard-floored SVI surrogate; |diag(H_raw)|
for the float64-safe diagraw mass matrix, foundry-i 34_fit_marg convention).

Reference: the 8 foundry-i long_diagraw_s*.npz chains (8000 kept draws each,
diagraw-mass PHMC L=16, all started at qz_refined) -- present on disk, wired
lazily. Caveats recorded: single-chain-per-seed, diagonal likelihood,
ess_min ~ 3.5-7.4 per chain (the documented brutal conditioning).
"""
from __future__ import annotations

import numpy as np

from cgl import guards
from cgl.paths import (CUTOUT_F140W, EMPIRICAL_PSF, HESS_MARG_PD, LONG_DIAGRAW,
                       MAP_MARG_PD)
from cgl.zoo.api import InitBundle, LensPosterior, Reference, assert_dtype_env

EXP_TIME = 1197.7  # _hmc_lib_marg.EXP_TIME (v2-class F140W)

MASS_LABELS = ["mass.theta_E", "mass.gamma", "mass.e1", "mass.e2",
               "mass.center_x", "mass.center_y", "shear.gamma1",
               "shear.gamma2"]


def build_parity_product() -> dict:
    """COPIED WITH ATTRIBUTION from 01_parity_harness.build_parity_product
    (the P0 gate configuration; itself mirrors foundry-i _hmc_lib_marg's data
    block lines 227-247). err_map uses the SAME jnp ops on-device so the bits
    match; the central mask is folded in by build_marg_model (err -> 1e10).
    meta deliberately does NOT declare psf_pixel_scale: empirical_psf.npy
    predates the delta_pix-sampling fix and the parity reference is defined
    on that legacy convention."""
    import jax.numpy as jnp
    from astropy.io import fits

    with fits.open(CUTOUT_F140W) as h:
        sci = h["SCI"].data.astype(np.float64)
        wht = h["WHT"].data.astype(np.float64)
    sky = float(np.median(sci))
    img = (sci - sky).astype(np.float64)
    background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
    num_pix = img.shape[0]
    err_map = np.asarray(
        jnp.sqrt(background_rms ** 2 +
                 jnp.clip(jnp.asarray(img, dtype=jnp.float64), 0.0, np.inf)
                 / EXP_TIME))
    yy, xx = np.indices(img.shape)
    r_center = np.sqrt((xx - num_pix // 2) ** 2 + (yy - num_pix // 2) ** 2)
    keep_mask = r_center > 1.5
    psf = np.load(EMPIRICAL_PSF).astype(np.float64)
    meta = dict(
        delta_pix=0.13, num_pix=int(num_pix), supersample=2, exp_time=EXP_TIME,
        sky=sky, background_rms=background_rms,
        source="parity mirror of foundry-i _hmc_lib_marg data block "
               "(cutout_F140W.fits SCI/WHT + empirical_psf.npy)",
    )
    return dict(img=img, err_map=err_map, keep_mask=keep_mask, psf=psf,
                meta=meta)


def build(model=None) -> LensPosterior:
    """Build the T2 target. `model` lets validation reuse an already-built
    build_marg_model namespace (avoids a second multi-minute compile)."""
    assert_dtype_env("float64")
    guards.require_gpu()

    import jax
    import jax.numpy as jnp

    from cgl.likelihood import build_marg_model

    if model is None:
        model = build_marg_model(build_parity_product(), n_max=6, supersample=2,
                                 delta_pix=0.13, exp_time=EXP_TIME)
    ndim = int(model.ndim)
    prior, bij = model.prior, model.bij

    # batch surface wraps the parity-gated jitted closure
    _log_prob_vmap = jax.jit(jax.vmap(model.target_log_prob_fn))

    def log_prob_batch(Z):
        Z = jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float64))
        return _log_prob_vmap(Z)

    def _log_prior_impl(Z):
        z = list(Z.T)
        x = bij.forward(z)
        return prior.log_prob(x) + bij.forward_log_det_jacobian(z)

    _log_prior_jit = jax.jit(_log_prior_impl)

    def log_prior_batch(Z):
        return _log_prior_jit(jnp.atleast_2d(jnp.asarray(Z, dtype=jnp.float64)))

    # NOTE: no independent jitted log_like surface -- the marginalized data
    # term lives inside build_marg_model (frozen at P0). log_like_batch is
    # derived (log_prob - log_prior); the GENUINE decomposition check for
    # gate (v) uses model.marg_internals (eager op-by-op logL_data +
    # log_prior), see 21_validate_zoo.py.

    # ---- init from the refined PD mode + Hessian artifacts -------------------
    ref = np.load(MAP_MARG_PD)
    qz_refined = np.asarray(ref["qz_refined"], dtype=np.float64)
    stored_logp = float(ref["logp"])
    hd = np.load(HESS_MARG_PD)
    H_raw = np.asarray(hd["H_raw"], dtype=np.float64)
    H_reg = np.asarray(hd["H_reg"], dtype=np.float64)
    # Laplace covariance = inv(H_reg), then the Bug-2 eigenvalue floor.
    cov_laplace = np.linalg.solve(H_reg, np.eye(ndim))
    cov_laplace = 0.5 * (cov_laplace + cov_laplace.T)
    cov_reg, chol, n_floored = guards.floor_svi_covariance(cov_laplace)
    # float64-safe diagonal mass source (34_fit_marg 'diagraw' convention).
    hess_diag_raw = np.maximum(np.abs(np.diag(H_raw)), 1.0)

    from cgl.zoo.api import make_prior_z_sampler
    init = InitBundle(
        prior_sample_fn=make_prior_z_sampler(prior, bij, ndim, "float64"),
        map_z=qz_refined,
        svi_loc=qz_refined,
        svi_scale_tril=chol,
        svi_n_floored=n_floored,
        hess_diag_raw=hess_diag_raw,
        notes="map_z/svi_loc = map_marg_pd.npz['qz_refined'] (trust-exact PD "
              "mode); svi cov = inv(hess_marg_pd H_reg) guard-floored "
              f"({n_floored} eigenvalues raised); hess_diag_raw = "
              "max(|diag(H_raw)|, 1) for the diagraw mass matrix "
              "(foundry-i 34_fit_marg.build_momentum).",
    )

    reference = None
    long_paths = [p for p in LONG_DIAGRAW if p.exists()]
    if long_paths:
        l0 = np.load(long_paths[0])
        reference = Reference(
            provenance=("foundry-i 34_fit_marg.py --mode hmc --massmatrix "
                        "diagraw, L=16, burn 2000 / keep 8000, float64, one "
                        "chain per seed s0..s7, ALL started at qz_refined "
                        f"({len(long_paths)} files long_diagraw_s*.npz)"),
            samples_path=[str(p) for p in long_paths],
            samples_key="samples",
            mode_labels=("mode0",),
            mode_weights=np.array([1.0]),
            mode_weights_trusted=False,
            mode_centers=qz_refined[None, :],
            mode_assigner={"method": "mahalanobis"},
            truth={"stored_map_logp": stored_logp,
                   "gamma_median_per_seed": [
                       float(np.load(p)["gamma_median"]) for p in long_paths],
                   "ess_min_per_seed": [
                       float(np.load(p)["ess_min"]) for p in long_paths]},
            caveats=("DIAGONAL-likelihood chains; single chain per seed "
                     "(cross-seed R-hat only); per-chain ess_min ~ 3.5-7.4 "
                     "of 8000 (the documented cond~1e14 pathology) -- treat "
                     "as a mixing-quality yardstick, not a converged "
                     "posterior; adapted_step_size stored per seed."),
        )

    return LensPosterior(
        name="foundry_marg46", tier="T2", dim=ndim, dtype="float64",
        noise_model="diag",
        log_prob_batch=log_prob_batch, log_prior_batch=log_prior_batch,
        labels=list(model.index_labels), mass_labels=MASS_LABELS,
        to_physical=lambda Z: model.to_physical_mass(
            np.atleast_2d(np.asarray(Z, dtype=np.float64))),
        init=init, bijector=bij,
        prior_transform=None,   # impractical: 46-dim cond-1e14 marg posterior
        log_like_x=None,        # is not a nested-sampling track in P2
        reference=reference,
        meta={"parity_logp_qz_refined": -45840.984005998456,
              "stored_map_logp": stored_logp,
              "product": "parity mirror (cutout_F140W + legacy empirical_psf)",
              "model_surface": "cgl.likelihood.build_marg_model (P0-frozen)",
              "batch": "jit(vmap(target_log_prob_fn)); ~57 MB design tensor "
                       "per lane, keep N <= 64 on an L4",
              "marg_internals": "available via cgl.likelihood model; NOT "
                                "carried on the dataclass"},
    )


# handle for validation scripts that need the raw model surface (marg_internals)
def build_raw_model():
    """Rebuild the underlying build_marg_model namespace (validation only)."""
    assert_dtype_env("float64")
    from cgl.likelihood import build_marg_model
    return build_marg_model(build_parity_product(), n_max=6, supersample=2,
                            delta_pix=0.13, exp_time=EXP_TIME)
