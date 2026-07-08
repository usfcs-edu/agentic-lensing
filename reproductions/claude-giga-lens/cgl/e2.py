"""E2 real-data cross-scale unification driver (P1c, the pillar-1 headline).

Refits the real HST system DESI-165.4754-06.0423 with the correlated
marginalized likelihood (cgl.likelihood.build_marg_model, 46-dim, float64,
whiten_fn from cgl.whiten) at three drizzle scales:

    E2a  v3   fine   260^2 @ 0.04"  ss1   whitener_v3.npz      (strict M=20)
    E2b  v3b  binned 130^2 @ 0.08"  ss2   whitener_v3b.npz     (strict M=10)
    E2c  v2d  native  80^2 @ 0.13"  ss2   whitener_v2d_relaxed (relaxed M=10, D3)

Multi-basin start plumbing (foundry-i R8, the fiddliest part): the map_v11_*
artifacts are 74-dim PAPER-scale z-vectors (EPL+Shear, 4 lens-light Sersic,
Sersic + Shapelets(n_max=6) source with 28 EXPLICIT amps). The 46-dim marg
model is the SAME prior with the 28 shapelet amplitudes MARGINALIZED (dropped).
So a 74-dim MAP maps to a 46-dim start by:
  1. forward best_z74 through the paper bijector -> physical pytree x74;
  2. drop the 28 amps from x74[2][1] (keep beta, center_x, center_y);
  3. optionally override the mass+shear sector (build a low-basin FINE start,
     which has no own-scale MAP, from the steep fine MAP + a low-basin
     mass sector — mass/shear/ellipticity are scale-independent arcsec params);
  4. z46 = marg_model.bij.inverse(marg_x).
Every start is VALIDATED (logp finite + rendered-model chi2 sane) before any
A100 time is burned.

MEASURED basin gammas (probe, forward through the paper bijector):
    map_v11_v3cold      gamma=2.578  STEEP  (E2a fine steep seed)
    map_v11_v3b_warm    gamma=2.672  STEEP  (E2b binned steep seed)
    map_v11_v3b_cold2d  gamma=1.369  LOW    (E2b binned low seed; E2a fine low mass override)
    map_v11_v3b_cold    gamma=1.465  LOW
    map_v11_v2d_cold    gamma=1.411  LOW    (E2c native low seed)
    map_v11_v2d_warm    gamma=1.434  LOW
NOTE (deviation from the pre-reg brief): the brief called map_v11_v3b_cold the
"steep" v3b seed, but it measures gamma=1.465 (LOW). The genuinely steep v3b
MAP is map_v11_v3b_warm (gamma=2.672); E2b seeds that as the steep basin.

Sampler: production two-stage re-preconditioned PHMC (the P1b diagnosis recipe),
per basin, with the Laplace (MAP-Hessian) metric as the stage-1 momentum
covariance and the pooled cross-chain stage-1 covariance for stage 2.
"""
from __future__ import annotations

import time
import types

import numpy as np

from cgl.paths import (CUTOUT_V2D, CUTOUT_V3, CUTOUT_V3B, DATA, FOUNDRY_I_DATA,
                       NEARBY_GALAXY_LOC, bootstrap_vendor, load_product)

# --------------------------------------------------------------------------- #
# per-product configuration
# --------------------------------------------------------------------------- #
PRODUCTS = {
    "v3":  dict(cutout=CUTOUT_V3,  whitener="whitener_v3.npz",
                delta_pix=0.04, supersample=1, label="fine 260^2"),
    "v3b": dict(cutout=CUTOUT_V3B, whitener="whitener_v3b.npz",
                delta_pix=0.08, supersample=2, label="binned 130^2"),
    "v2d": dict(cutout=CUTOUT_V2D, whitener="whitener_v2d_relaxed.npz",
                delta_pix=0.13, supersample=2, label="native 80^2 (relaxed)"),
}

# basin seeds: (paper-MAP artifact, mass-override artifact or None)
#   mass_override=None -> use this artifact's own mass sector.
#   mass_override=name -> keep this artifact's light/source, swap in `name`'s
#                         mass+shear (build a missing-basin start).
BASIN_SEEDS = {
    "v3":  dict(steep=("map_v11_v3cold", None),
                low=("map_v11_v3cold", "map_v11_v3b_cold2d")),   # fine low: mass override
    "v3b": dict(steep=("map_v11_v3b_warm", None),
                low=("map_v11_v3b_cold2d", None)),
    "v2d": dict(steep=("map_v11_v2d_cold", "map_v11_v3cold"),    # native steep: mass override
                low=("map_v11_v2d_cold", None)),
}

GAMMA_STEEP_LOW_SPLIT = 1.9   # gamma threshold separating the two basins
EXP_TIME = 1197.7


# --------------------------------------------------------------------------- #
# 74-dim paper prior + bijector (start mapping only; identical to foundry-i
# _hmc_lib.build_model's prior — the distribution the map_v11 best_z live in)
# --------------------------------------------------------------------------- #
def build_paper_bij():
    """Return (prior74, bij74, ndim74) for the 74-dim paper model, built in the
    active jax dtype. Used only to forward map_v11 best_z -> physical."""
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    bootstrap_vendor()
    from gigalens.jax.profiles.light import shapelets

    tfd = tfp.distributions
    tfb = tfp.bijectors
    nb = np.load(NEARBY_GALAXY_LOC)
    NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])

    def _sersic(R_med, R_sig, Ie_med, Ie_sig, cx=0.0, cy=0.0, c_sig=0.05):
        return tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            center_x=tfd.Normal(cx, c_sig), center_y=tfd.Normal(cy, c_sig),
            Ie=tfd.LogNormal(jnp.log(Ie_med), Ie_sig)))

    lens_mass = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
            gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
            e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
            center_x=tfd.Normal(0.0, 0.02), center_y=tfd.Normal(0.0, 0.02))),
        tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05),
                                        gamma2=tfd.Normal(0.0, 0.05)))])
    lens_light = tfd.JointDistributionSequential([
        _sersic(0.4, 0.3, 5.0, 0.5, c_sig=0.02),
        _sersic(2.0, 0.3, 2.0, 0.5, c_sig=0.02),
        _sersic(0.3, 0.3, 1.0, 0.5, cx=NEAR_X, cy=NEAR_Y, c_sig=0.1),
        _sersic(0.6, 0.3, 0.5, 0.5, cx=NEAR_X, cy=NEAR_Y, c_sig=0.1)])
    srcS = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
        Ie=tfd.LogNormal(jnp.log(2.0), 0.5)))
    amp_names = shapelets.Shapelets(n_max=6)._amp_names
    amp_priors = {n: tfd.Normal(0.0, 5.0 / float(jnp.sqrt(i + 1)))
                  for i, n in enumerate(amp_names)}
    srcShp = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
        **amp_priors))
    prior = tfd.JointDistributionSequential(
        [lens_mass, lens_light, tfd.JointDistributionSequential([srcS, srcShp])])
    example = prior.sample(seed=jax.random.PRNGKey(0))
    pack = tfb.pack_sequence_as(example)
    bij = tfb.Chain([prior.experimental_default_event_space_bijector(), pack])
    ndim = int(sum(np.size(np.asarray(v))
                   for v in jax.tree_util.tree_leaves(example)))
    assert ndim == 74, ndim
    return prior, bij, ndim


def _squeeze_tree(x):
    import jax
    return jax.tree_util.tree_map(lambda v: np.asarray(v).reshape(-1)[0], x)


def paper_z_to_marg_z(best_z74, marg_model, mass_z74=None, ie_scale=1.0):
    """Map a 74-dim paper MAP z to a 46-dim marg-model z start.

    Args:
        best_z74: (74,) paper-model z (map_v11 best_z).
        marg_model: cgl.likelihood.build_marg_model output.
        mass_z74: optional (74,) paper z whose mass+shear sector overrides
            best_z74's (build a missing-basin start). None = no override.
        ie_scale: multiply the 5 Sersic Ie by this factor. The paper model
            compares simulate()=SB*conversion_factor to the data, so its Ie are
            in SB/cf units; the marg model compares M_det=SB (cf divided out) to
            the SAME data, so marg Ie = paper Ie * cf (cf = delta_pix^2). Pass
            the marg model's conversion_factor here. Linear amps only; the 28
            shapelet amps are marginalized (units-agnostic).

    Returns (z46 np.float64, phys_report dict).
    """
    import jax
    import jax.numpy as jnp

    prior74, bij74, _ = build_paper_bij()
    dt = jnp.float64
    z = jnp.asarray(np.asarray(best_z74, dtype=np.float64))
    x74 = bij74.forward(list(z[:, None]))          # batch=1 physical pytree
    mass = x74[0][0]
    shear = x74[0][1]
    if mass_z74 is not None:
        zo = jnp.asarray(np.asarray(mass_z74, dtype=np.float64))
        xo = bij74.forward(list(zo[:, None]))
        mass, shear = xo[0][0], xo[0][1]
    # rescale the 5 Sersic Ie into the marg model's amplitude convention
    s = jnp.asarray(ie_scale, dtype=dt)
    LL = [{k: (v * s if k == "Ie" else v) for k, v in x74[1][i].items()}
          for i in range(4)]
    srcS = {k: (v * s if k == "Ie" else v) for k, v in x74[2][0].items()}
    # drop the 28 amps: keep beta/center_x/center_y only
    shp = x74[2][1]
    shp46 = {k: shp[k] for k in ("beta", "center_x", "center_y")}
    marg_x = [[mass, shear], LL, [srcS, shp46]]
    z46_list = marg_model.bij.inverse(marg_x)
    z46 = np.array([float(np.asarray(v).reshape(-1)[0]) for v in z46_list],
                   dtype=np.float64)

    # round-trip check: forward z46 -> physical, compare gamma/theta_E
    xr = marg_model.bij.forward(list(jnp.asarray(z46)[:, None]))
    report = dict(
        gamma=float(np.asarray(xr[0][0]["gamma"]).reshape(-1)[0]),
        theta_E=float(np.asarray(xr[0][0]["theta_E"]).reshape(-1)[0]),
        e1=float(np.asarray(xr[0][0]["e1"]).reshape(-1)[0]),
        e2=float(np.asarray(xr[0][0]["e2"]).reshape(-1)[0]),
        gamma_target=float(np.asarray(mass["gamma"]).reshape(-1)[0]),
        thetaE_target=float(np.asarray(mass["theta_E"]).reshape(-1)[0]),
    )
    report["roundtrip_dgamma"] = abs(report["gamma"] - report["gamma_target"])
    return z46, report


# --------------------------------------------------------------------------- #
# grouped-conv marg model (the XLA-compile fix)
# --------------------------------------------------------------------------- #
# cgl.likelihood.build_marg_model whitens the 28 shapelet design columns with
# ``jax.vmap(whiten_fn, in_axes=2)`` — 28 SEPARATE conv ops. In REVERSE mode
# (the sampler needs grad(logp)), XLA's fusion of the 28 f64 conv-transposes
# LIVELOCKS the compile (100% CPU, GPU idle, 20+ min; measured on L4). The
# diagonal marg (no conv) grad compiles in 14s. Collapsing the 28 per-column
# convs into ONE depthwise (feature_group_count) conv makes the whitened-design
# grad graph O(1) conv-transposes and compiles like the diagonal case. This
# module rebuilds the marg model with that single change; its logp is asserted
# bit-identical to cgl.likelihood.build_marg_model at build time.
def _grouped_whiten_ops(h, sqrt_d_inv, keep_w):
    """Return (whiten_R, whiten_X): residual (H,W)->(n,) and design
    (H,W,C)->(n,C) whiteners, the design one a single depthwise conv."""
    import jax
    import jax.numpy as jnp

    h = jnp.asarray(h)
    sdi = jnp.asarray(sqrt_d_inv)
    kw = jnp.asarray(keep_w)
    kh, kwd = h.shape

    def whiten_R(R):
        v = (R * sdi)[None, None, :, :]
        u = jax.lax.conv_general_dilated(
            v.astype(v.dtype), h[:, :, None, None].astype(v.dtype),
            (1, 1), "SAME", dimension_numbers=("NCHW", "HWIO", "NCHW"))
        return jnp.reshape(u[0, 0] * kw.astype(v.dtype), (-1,))

    def whiten_X(ret):
        # ret: (H, W, C) -> depthwise conv with shared kernel h -> (H*W, C)
        C = ret.shape[-1]
        v = jnp.transpose(ret * sdi[:, :, None], (2, 0, 1))[None]  # (1,C,H,W)
        kern = jnp.broadcast_to(h[:, :, None, None], (kh, kwd, 1, C))
        u = jax.lax.conv_general_dilated(
            v.astype(v.dtype), kern.astype(v.dtype), (1, 1), "SAME",
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
            feature_group_count=C)                                 # (1,C,H,W)
        u = u[0] * kw.astype(v.dtype)[None]                        # (C,H,W)
        return jnp.reshape(u, (C, -1)).T                           # (n, C)

    return whiten_R, whiten_X


def build_marg_model_grouped(product, whitener, *, n_max=6, supersample=2,
                             delta_pix=0.13, exp_time=EXP_TIME,
                             shapelet_sigma0=5.0, mask_err=1e10):
    """cgl.likelihood.build_marg_model with grouped-conv design whitening.

    `whitener`: dict with h (taps), keep_w (eroded mask). Same 46-dim model,
    priors, simulators and marg math as cgl.likelihood; ONLY the design-column
    whitening is restructured (single depthwise conv). Returns the same
    SimpleNamespace surface (target_log_prob_fn, bij, index_labels, ...).
    """
    import jax
    import jax.numpy as jnp

    from cgl import guards, likelihood
    from cgl.marg import marg_loglik
    from cgl.paths import bootstrap_vendor

    guards.require_x64()
    bootstrap_vendor()
    from gigalens.jax.profiles.light import sersic, shapelets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig
    from objax.functional import average_pool_2d

    _FDT = jnp.float64
    img = np.asarray(product["img"], dtype=np.float64)
    err_in = np.asarray(product["err_map"], dtype=np.float64)
    keep_np = np.asarray(product["keep_mask"], dtype=bool)
    psf = np.asarray(product["psf"], dtype=np.float64)
    NUM_PIX = img.shape[0]
    sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=NUM_PIX,
                                 supersample=supersample, kernel=psf)
    Y = jnp.asarray(img, dtype=_FDT)
    keep_mask = jnp.asarray(keep_np)
    masked_err_map = jnp.where(keep_mask, jnp.asarray(err_in, dtype=_FDT),
                               jnp.asarray(mask_err, dtype=_FDT))
    sqrtW = 1.0 / masked_err_map

    nb = np.load(NEARBY_GALAXY_LOC)
    near_xy = (float(nb["arcsec_x"]), float(nb["arcsec_y"]))
    prior = likelihood._build_prior(near_xy[0], near_xy[1])
    import tensorflow_probability.substrates.jax as tfp
    tfb = tfp.bijectors
    example = prior.sample(seed=jax.random.PRNGKey(0))
    pack_bij = tfb.pack_sequence_as(example)
    bij = tfb.Chain([prior.experimental_default_event_space_bijector(), pack_bij])
    ndim = int(sum(jnp.size(v) for v in jax.tree_util.tree_leaves(example)))
    assert ndim == 46, ndim
    index_labels = likelihood._label_index_map(bij, ndim)

    phys_det = PhysicalModel([epl.EPL(50), shear.Shear()],
                             [sersic.SersicEllipse(use_lstsq=False)] * 4,
                             [sersic.SersicEllipse(use_lstsq=False)])
    sim_det = LensSimulator(phys_det, sim_config, bs=1)
    _INV_CF = 1.0 / float(sim_det.conversion_factor)
    phys_shp = PhysicalModel([epl.EPL(50), shear.Shear()], [],
                             [shapelets.Shapelets(n_max=n_max, use_lstsq=True,
                                                  interpolate=False)])
    sim_shp = LensSimulator(phys_shp, sim_config, bs=1)
    depth = (n_max + 1) * (n_max + 2) // 2
    i_idx = jnp.arange(depth, dtype=_FDT)
    Lambda_diag = (i_idx + 1.0) / (shapelet_sigma0 * shapelet_sigma0)

    whiten_R, whiten_X = _grouped_whiten_ops(
        np.asarray(whitener["h"], dtype=np.float64), sqrtW,
        np.asarray(whitener["keep_w"]).astype(np.float64))

    def _design_ret(x_shp):
        lens_params = x_shp[0]
        source_light_params = x_shp[1]
        beta_x, beta_y = sim_shp._beta(lens_params)
        im = jnp.zeros((0, *sim_shp.img_X.shape))
        for lm, p in zip(sim_shp.phys_model.source_light, source_light_params):
            im = jnp.concatenate((im, lm.light(beta_x, beta_y, **p)), axis=0)
        im = jnp.nan_to_num(im)
        im = jnp.transpose(im, (3, 0, 1, 2))
        ret = jax.lax.conv_general_dilated(
            im, sim_shp.kernel, (1, 1), padding="SAME",
            feature_group_count=sim_shp.depth,
            dimension_numbers=("NCHW", "HWOI", "NCHW"))
        if sim_shp.supersample != 1:
            ret = average_pool_2d(ret, size=(sim_shp.supersample,
                                             sim_shp.supersample), padding="SAME")
        ret = jnp.transpose(ret, (0, 2, 3, 1))
        return jnp.squeeze(ret, axis=0)

    def _parts(z):
        z_list = list(z[None, :].T)
        x = bij.forward(z_list)
        M_det = jnp.squeeze(sim_det.simulate(x)) * _INV_CF
        ret = _design_ret([x[0], [x[2][1]]])
        Xw = whiten_X(ret)
        Rw = whiten_R(Y - M_det)
        return z_list, x, M_det, ret, Xw, Rw

    @jax.jit
    def _logpost(z):
        z_list, x, M_det, ret, Xw, Rw = _parts(z)
        logL, a_star, logdetA = marg_loglik(Xw, Rw, Lambda_diag)
        log_prior = prior.log_prob(x) + bij.forward_log_det_jacobian(z_list)
        return jnp.squeeze(logL + log_prior)

    @jax.jit
    def m_det(z):
        x = bij.forward(list(z[None, :].T))
        return jnp.squeeze(sim_det.simulate(x)) * _INV_CF

    @jax.jit
    def model_image(z):
        _, _, M_det, ret, Xw, Rw = _parts(z)
        _, a_star, _ = marg_loglik(Xw, Rw, Lambda_diag)
        return M_det + jnp.sum(ret * a_star[None, None, :], axis=-1)

    @jax.jit
    def shapelet_amps(z):
        _, _, _, _, Xw, Rw = _parts(z)
        _, a_star, _ = marg_loglik(Xw, Rw, Lambda_diag)
        return a_star

    def marg_internals(z):
        z = jnp.asarray(z, dtype=_FDT)
        z_list, x, M_det, ret, Xw, Rw = _parts(z)
        b = Xw.T @ Rw
        A = Xw.T @ Xw + jnp.diag(Lambda_diag)
        chol = jnp.linalg.cholesky(A)
        a_star = jax.scipy.linalg.cho_solve((chol, True), b)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        logL_data = (-0.5 * jnp.sum(Rw ** 2) + 0.5 * jnp.dot(b, a_star)
                     - 0.5 * logdetA)
        log_prior = prior.log_prob(x) + bij.forward_log_det_jacobian(z_list)
        return dict(Rw=np.asarray(Rw), logdetA=float(logdetA),
                    logL_data=float(np.squeeze(logL_data)),
                    log_prior=float(np.squeeze(np.asarray(log_prior))),
                    logpost=float(np.squeeze(np.asarray(logL_data + log_prior))))

    def to_physical_mass(samples_np):
        arr = jnp.asarray(samples_np)
        physical = bij.forward(list(arr.T))
        mm, ms = physical[0][0], physical[0][1]
        out = {k: np.asarray(mm[k]) for k in mm}
        out.update({k: np.asarray(ms[k]) for k in ms})
        return out

    return types.SimpleNamespace(
        target_log_prob_fn=_logpost, prior=prior, bij=bij, sim_config=sim_config,
        ndim=ndim, Y=Y, keep_mask=keep_mask, masked_err_map=masked_err_map,
        index_labels=index_labels, labels=index_labels,
        to_physical_mass=to_physical_mass, to_physical=to_physical_mass,
        m_det=m_det, model_image=model_image, shapelet_amps=shapelet_amps,
        marg_internals=marg_internals, depth=depth, n_max=n_max,
        Lambda_diag=np.asarray(Lambda_diag), near_xy=near_xy)


# --------------------------------------------------------------------------- #
# target builder: correlated marg model + batched logp
# --------------------------------------------------------------------------- #
def build_target(tag, n_max=6, verify_logp=False, whitener_file=None):
    """Build the correlated marg model + batched log-prob for a product tag.

    Uses the grouped-conv design whitening (build_marg_model_grouped) so the
    sampler's grad graph compiles. If verify_logp, assert logp matches the
    reference cgl.likelihood.build_marg_model (vmap-conv) at a prior draw.
    whitener_file overrides PRODUCTS[tag]['whitener'] (e.g. the strict v2d
    whitener for the whitener-vs-prior-pull diagnostic).

    Returns SimpleNamespace(model, batched_lp, whiten_meta, tag, cfg, ...)."""
    import jax
    import jax.numpy as jnp

    cfg = PRODUCTS[tag]
    prod = load_product(cfg["cutout"])
    wname = whitener_file or cfg["whitener"]
    wz = np.load(DATA / wname, allow_pickle=True)
    whitener = dict(h=np.asarray(wz["h"], dtype=np.float64),
                    keep_w=np.asarray(wz["keep_w"]).astype(bool))
    M = int(wz["M"])
    e_op = float(wz["e_op"])

    model = build_marg_model_grouped(
        prod, whitener, n_max=n_max, supersample=cfg["supersample"],
        delta_pix=cfg["delta_pix"], exp_time=EXP_TIME)

    if verify_logp:
        from cgl import likelihood
        from cgl.whiten import make_conv_whitener
        err = np.asarray(prod["err_map"], dtype=np.float64)
        keep = np.asarray(prod["keep_mask"], dtype=bool)
        masked_err = np.where(keep, err, 1e10)
        ref = likelihood.build_marg_model(
            prod, n_max=n_max, supersample=cfg["supersample"],
            delta_pix=cfg["delta_pix"], exp_time=EXP_TIME,
            whiten_fn=make_conv_whitener(
                jnp.asarray(whitener["h"]), jnp.asarray(1.0 / masked_err),
                jnp.asarray(whitener["keep_w"].astype(np.float64))))
        xt = model.prior.sample(seed=jax.random.PRNGKey(3))
        zt = np.asarray([np.asarray(v).reshape(-1)[0]
                         for v in model.bij.inverse(xt)], dtype=np.float64)
        a = float(model.target_log_prob_fn(jnp.asarray(zt)))
        b = float(ref.target_log_prob_fn(jnp.asarray(zt)))
        model._logp_verify = dict(grouped=a, reference=b, absdiff=abs(a - b))

    batched_lp = jax.jit(jax.vmap(model.target_log_prob_fn))
    conversion_factor = float(cfg["delta_pix"]) ** 2   # det(delta_pix*I)
    whiten_meta = dict(whitener=wname, M=M, e_op=e_op,
                       n_keep_w=int(whitener["keep_w"].sum()))
    return types.SimpleNamespace(model=model, batched_lp=batched_lp,
                                 whiten_meta=whiten_meta, tag=tag, cfg=cfg,
                                 keep_w=whitener["keep_w"],
                                 conversion_factor=conversion_factor)


def make_start(tag, basin, target):
    """Build a validated 46-dim start z for (tag, basin).

    `target` is a build_target namespace (needs .model + .conversion_factor)."""
    model = target.model if hasattr(target, "model") else target
    ie_scale = getattr(target, "conversion_factor", 1.0)
    art, over = BASIN_SEEDS[tag][basin]
    bz = np.load(FOUNDRY_I_DATA / f"{art}.npz")["best_z"].astype(np.float64)
    mz = None
    if over is not None:
        mz = np.load(FOUNDRY_I_DATA / f"{over}.npz")["best_z"].astype(np.float64)
    z46, rep = paper_z_to_marg_z(bz, model, mass_z74=mz, ie_scale=ie_scale)
    rep["artifact"] = art
    rep["mass_override"] = over
    rep["ie_scale"] = float(ie_scale)
    return z46, rep


# --------------------------------------------------------------------------- #
# MAP polish (correlated likelihood) + Laplace evidence
# --------------------------------------------------------------------------- #
def map_polish(target, z0, rounds=6, iters=400):
    """L-BFGS MAP polish of the 46-dim correlated marg log-posterior from z0.

    Returns dict(z_map, logp0, logp_map, round_lps, wall_s)."""
    import functools

    import jax
    import jax.numpy as jnp
    import optax

    lp1 = target.model.target_log_prob_fn
    logp0 = float(lp1(jnp.asarray(z0, dtype=jnp.float64)))

    def fun(z):
        return -lp1(z.astype(jnp.float64))

    opt = optax.lbfgs()
    vgf = optax.value_and_grad_from_state(fun)

    @functools.partial(jax.jit, static_argnums=(1,))
    def run(zz, n):
        def step(carry, _):
            z, s = carry
            v, g = vgf(z, state=s)
            u, s = opt.update(g, s, z, value=v, grad=g, value_fn=fun)
            return (optax.apply_updates(z, u), s), v
        (zf, _), vals = jax.lax.scan(step, (zz, opt.init(zz)), length=n)
        return zf, vals

    t0 = time.time()
    z = jnp.asarray(z0, dtype=jnp.float64)
    best_z = np.asarray(z, dtype=np.float64)
    best_lp = logp0
    round_lps = []
    for _ in range(int(rounds)):
        z, vals = run(z, int(iters))
        lp = -float(np.asarray(vals)[-1])
        round_lps.append(lp)
        if np.isfinite(lp) and lp > best_lp:
            best_lp, best_z = lp, np.asarray(z, dtype=np.float64)
        if len(round_lps) > 1 and abs(round_lps[-1] - round_lps[-2]) < 0.2:
            break
    return dict(z_map=best_z, logp0=logp0, logp_map=best_lp,
                round_lps=round_lps, wall_s=time.time() - t0)


def laplace_evidence(target, z_map, eig_floor=1e-3):
    """Laplace log-evidence at z_map + a ROBUST HMC metric covariance.

    H = Hessian of -logpost. The metric covariance uses 1/|eigenvalue| (floored
    at eig_floor) rather than flooring near-zero/negative eigenvalues to a tiny
    value: at a saddle (common on the weak-likelihood products, min_eig down to
    -1e13) the floored form makes cov ~ 1e8 in the negative directions -> the
    leapfrog explodes (R-hat ~ 1e31). |lambda| keeps the metric on the local
    curvature SCALE and PD, so stage 1 stays stable; the two-stage recipe then
    re-estimates the metric from pooled draws. The log-evidence itself uses the
    floored-positive spectrum (a saddle evidence is only a rough basin-mass
    proxy; reported with that caveat).

    Returns (log_evidence, logdet_H, cov, n_floored, min_eig, n_neg)."""
    import jax
    import jax.numpy as jnp

    lp1 = target.model.target_log_prob_fn
    H = -np.asarray(jax.hessian(lp1)(jnp.asarray(z_map, dtype=jnp.float64)),
                    dtype=np.float64)
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    n_neg = int(np.sum(w <= 0))
    ndim = H.shape[0]
    logp_map = float(lp1(jnp.asarray(z_map, dtype=jnp.float64)))
    # evidence: floored-positive spectrum (rough for saddles)
    w_pos = np.where(w < eig_floor, eig_floor, w)
    logdet_H = float(np.sum(np.log(w_pos)))
    log_ev = logp_map + 0.5 * ndim * np.log(2 * np.pi) - 0.5 * logdet_H
    # metric: 1/|lambda|, curvature-scaled and PD
    w_abs = np.maximum(np.abs(w), eig_floor)
    cov = (V * (1.0 / w_abs)) @ V.T
    cov = 0.5 * (cov + cov.T)
    return dict(log_evidence=float(log_ev), logdet_H=logdet_H,
                logp_map=logp_map, cov=cov, n_floored=n_neg,
                min_eig=float(w.min()), n_neg=n_neg)


# --------------------------------------------------------------------------- #
# batched two-stage re-preconditioned PHMC (the P1b production recipe)
# --------------------------------------------------------------------------- #
def _run_phmc(target, loc_cov, start, chains, burn, keep, step_size,
              num_leapfrog, seed, target_accept=0.75):
    """One batched PHMC + dual-averaging run (FIXED leapfrog, the foundry-i
    34_fit_marg recipe for this exact 46-dim correlated marg model — NO
    GradientBasedTrajectoryLengthAdaptation: GBTLA's meta-gradient graph makes
    the XLA compile of the f64 correlated-marg leapfrog livelock at 100% CPU,
    GPU idle, 20+ min and counting; foundry-i's fixed num_leapfrog=16 + DA
    compiles in seconds and delivered the 48x(750+8000) anchor).

    loc_cov = momentum COVARIANCE (posterior cov estimate); start=(chains,ndim).
    """
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from tensorflow_probability.python.internal import unnest

    from cgl import guards

    tfd = tfp.distributions
    tfe = tfp.experimental

    cov_reg, chol, n_floored = guards.floor_svi_covariance(np.asarray(loc_cov))
    ndim = target.model.ndim
    lp = target.batched_lp
    momentum = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros(ndim, dtype=jnp.float64),
        covariance_matrix=jnp.asarray(np.linalg.inv(cov_reg)))
    num_adapt = int(0.8 * burn)
    kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=lp, momentum_distribution=momentum,
        step_size=np.float64(step_size), num_leapfrog_steps=int(num_leapfrog))
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=num_adapt,
        target_accept_prob=np.float64(target_accept))

    def trace_fn(_, pkr):
        ir = pkr.inner_results
        return (pkr.new_step_size,
                jnp.exp(jnp.minimum(ir.log_accept_ratio, 0.0)))

    start = jnp.asarray(np.asarray(start), dtype=jnp.float64)
    t0 = time.time()

    @jax.jit
    def run(start, seed):
        return tfp.mcmc.sample_chain(
            num_results=keep, num_burnin_steps=burn, current_state=start,
            kernel=kernel, trace_fn=trace_fn, seed=seed)

    try:
        draws, trace = run(start, jax.random.PRNGKey(seed))
        step_hist = np.asarray(trace[0], dtype=np.float32)
        accept_hist = np.asarray(trace[1], dtype=np.float32)
    except Exception:
        @jax.jit
        def run2(start, seed):
            return tfp.mcmc.sample_chain(
                num_results=keep, num_burnin_steps=burn, current_state=start,
                kernel=kernel, trace_fn=None, seed=seed)
        draws = run2(start, jax.random.PRNGKey(seed))
        step_hist = accept_hist = np.array([], dtype=np.float32)
    draws = np.asarray(jax.block_until_ready(draws), dtype=np.float64)
    return dict(draws=draws, step_hist=step_hist, accept_hist=accept_hist,
                n_floored=int(n_floored), wall_s=time.time() - t0,
                num_adapt=num_adapt)


def run_staged(target, z_map, metric_cov, chains=24, seed=2,
               stage1_burn=500, stage1_keep=500, burn=500, keep=2000,
               step_size=0.3, num_leapfrog=16, jitter=0.5):
    """Two-stage re-preconditioned PHMC for one basin (fixed-leapfrog PHMC).

    Stage 1: `chains` seeded at z_map + jitter*chol(metric_cov) noise, metric =
    metric_cov (the Laplace/SVI cov). Stage 2: metric re-estimated from pooled
    cross-chain stage-1 draws; warm-start from stage-1 chain ends."""
    import jax
    import jax.numpy as jnp

    from cgl import guards

    cov_reg, chol, _ = guards.floor_svi_covariance(np.asarray(metric_cov))
    key = jax.random.PRNGKey(seed)
    eps = np.asarray(jax.random.normal(key, (chains, target.model.ndim),
                                       dtype=jnp.float64))
    start1 = z_map[None, :] + jitter * (eps @ chol.T)

    s1 = _run_phmc(target, cov_reg, start1, chains, stage1_burn, stage1_keep,
                   step_size, num_leapfrog, seed)
    d1 = s1["draws"]                                   # (T1, C, ndim)
    ess1, rhat1 = diagnostics(d1)
    pooled = d1.reshape(-1, d1.shape[-1])
    loc2 = pooled.mean(axis=0)
    cov2 = np.cov(pooled.T)
    s2 = _run_phmc(target, cov2, d1[-1], chains, burn, keep, step_size,
                   num_leapfrog, seed + 1)
    ess2, rhat2 = diagnostics(s2["draws"])
    out = dict(
        draws=s2["draws"], ess=ess2, rhat=rhat2,
        stage1_rhat_max=float(np.max(rhat1)), stage1_ess_min=float(np.min(ess1)),
        stage2_rhat_max=float(np.max(rhat2)), stage2_ess_min=float(np.min(ess2)),
        stage1_wall_s=s1["wall_s"], stage2_wall_s=s2["wall_s"],
        wall_s=s1["wall_s"] + s2["wall_s"],
        step_final=float(s2["step_hist"][-1]) if s2["step_hist"].size else np.nan,
        accept_mean=float(np.mean(s2["accept_hist"])) if s2["accept_hist"].size
        else np.nan,
        n_floored=s1["n_floored"] + s2["n_floored"], loc2=loc2, cov2=cov2)
    return out


def diagnostics(draws):
    """(T, C, ndim) -> (ess, rhat) per dim (tfp split-chain)."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    s = jnp.asarray(draws)
    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(
        s, independent_chain_ndims=1, split_chains=True), dtype=np.float64)
    ess = np.asarray(tfp.mcmc.effective_sample_size(
        s, cross_chain_dims=1), dtype=np.float64)
    return ess, rhat


def gamma_index(model):
    """z-index of mass.gamma in the marg model's index_labels."""
    return int(list(model.index_labels).index("mass.gamma"))


def theta_e_index(model):
    return int(list(model.index_labels).index("mass.theta_E"))
