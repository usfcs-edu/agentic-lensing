"""Generalized ridge-marginalized (shapelet-amplitude) lens log-posterior.

COPIED WITH ATTRIBUTION from
    /raid/benson/git/agentic-lensing/reproductions/foundry-i/_hmc_lib_marg.py
(foundry-i campaign, ``build_model_marg``: the 46-dim ridge-marginalized
log-posterior for the v2-class HST product of DESI-165.4754-06.0423). The
math is UNCHANGED — see that module's docstring for the full derivation:
28 shapelet amplitudes marginalized analytically under their Normal(0,
5/sqrt(i+1)) priors as Tikhonov regularization Lambda_ii=(i+1)/25; design
matrix = the gigalens lstsq_simulate per-component ``ret`` for a
shapelets-only PhysicalModel (NO conversion_factor on design columns; M_det
multiplied by 1/conversion_factor to match); Cholesky solve, never pinv;
logL = -1/2||Rw||^2 + 1/2 b.a* - 1/2 logdetA; prior + bijector
forward_log_det_jacobian; central-mask pixels via err -> 1e10.

GENERALIZATIONS vs the original (complete list):
  1. Product-dict input (img / err_map / keep_mask / psf / meta — the
     ``cgl.paths.load_product`` layout) instead of hardcoded foundry-i paths
     (cutout_F140W.fits + BackwardProbModel err-map recomputation +
     empirical_psf.npy). The caller owns data/noise/mask construction.
  2. Pluggable whitening: ``whiten_fn(image_hw) -> (n_pix,)`` applied
     IDENTICALLY to the residual and each design column (cgl.whiten builds
     conv whiteners). ``whiten_fn=None`` = the original diagonal sqrt(W)
     path, bit-identical math.
  3. Knobs exposed with v2-class defaults: n_max, supersample, delta_pix,
     exp_time, near_xy (LL2/LL3 prior centers; default reads the foundry-i
     nearby_galaxy_loc.npz artifact), shapelet_sigma0 (ridge scale, 5.0).
  4. Array-level marg math factored to ``cgl.marg.marg_loglik`` (same ops,
     same order — inlines to the identical jaxpr under jit).
  5. Debug/parity surface: m_det(z), model_image(z), marg_internals(z).
  6. Guards: require_x64 at build; assert_psf_sampling when the product meta
     declares its PSF pixel scale; delta_pix/supersample cross-checked
     against meta when present.

Priors, the 46-dim structure (41 nonlinear + 5 LogNormal Sersic Ie), the
EPL(50)+Shear / 4 SersicEllipse / Sersic+Shapelets(n_max=6) model, and the
start-vector helper are ported verbatim for the v2-class product.

Import-order contract (same as the original): GIGALENS_X64=1 must be set in
the environment BEFORE this module (or jax) is imported; x64 is flipped at
import time, never later.
"""
import os as _os
import types as _types

import jax

# float64 must be enabled BEFORE the first jnp array is created (original
# _hmc_lib_marg pattern; see cgl/__init__.py for the campaign contract).
X64 = _os.environ.get("GIGALENS_X64", "0") == "1"
if X64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

from cgl import guards  # noqa: E402
from cgl.marg import marg_loglik  # noqa: E402
from cgl.paths import NEARBY_GALAXY_LOC, bootstrap_vendor  # noqa: E402

bootstrap_vendor()  # must precede any `import gigalens.*`

from gigalens.jax.profiles.light import sersic, shapelets  # noqa: E402
from gigalens.jax.profiles.mass import epl, shear  # noqa: E402
from gigalens.jax.simulator import LensSimulator  # noqa: E402
from gigalens.model import PhysicalModel  # noqa: E402
from gigalens.simulator import SimulatorConfig  # noqa: E402
from objax.functional import average_pool_2d  # noqa: E402

tfd = tfp.distributions
tfb = tfp.bijectors

DEFAULT_EXP_TIME = 1197.7  # v2-class F140W exposure time (original EXP_TIME)
_FDTYPE = jnp.float64 if X64 else jnp.float32
_NP_FDTYPE = np.float64 if X64 else np.float32


# --------------------------------------------------------------------------- #
# priors (ported verbatim from _hmc_lib_marg; near_xy generalized to an arg)
# --------------------------------------------------------------------------- #
def _sersic_prior_ie(R_med, R_sig, Ie_med, Ie_sig, n_lo=0.5, n_hi=8.0,
                     cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    """Sersic prior INCLUDING the LogNormal(Ie) amplitude (use_lstsq=False)."""
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
        Ie=tfd.LogNormal(jnp.log(Ie_med), Ie_sig),
    ))


def _build_prior(near_x, near_y):
    """46-dim prior: 5 Sersic Ie present (LogNormal), shapelet amps ABSENT
    (marginalized). Hyperparameters verbatim from _hmc_lib_marg._build_prior."""
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
        _sersic_prior_ie(0.4, 0.3, 5.0, 0.5, c_sig=0.02),
        _sersic_prior_ie(2.0, 0.3, 2.0, 0.5, c_sig=0.02),
        _sersic_prior_ie(0.3, 0.3, 1.0, 0.5, cx_mean=near_x, cy_mean=near_y,
                         c_sig=0.1),
        _sersic_prior_ie(0.6, 0.3, 0.5, 0.5, cx_mean=near_x, cy_mean=near_y,
                         c_sig=0.1),
    ])
    src_sersic_prior = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
        Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
    ))
    src_shp_prior = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
    ))
    source_light_prior = tfd.JointDistributionSequential(
        [src_sersic_prior, src_shp_prior])
    prior = tfd.JointDistributionSequential(
        [lens_mass_prior, lens_light_prior, source_light_prior])
    return prior


# --------------------------------------------------------------------------- #
# label helpers (ported verbatim)
# --------------------------------------------------------------------------- #
def _flat_named46(x):
    """Flatten bijector.forward output (mass, lens_light, source_light) into a
    list of (label, value) for the 46-dim Ie-bearing parameterization."""
    out = []
    for k in x[0][0]:
        out.append(("mass." + k, float(np.asarray(x[0][0][k]).squeeze())))
    for k in x[0][1]:
        out.append(("shear." + k, float(np.asarray(x[0][1][k]).squeeze())))
    for i, c in enumerate(x[1]):
        for k in c:
            out.append((f"LL{i}." + k, float(np.asarray(c[k]).squeeze())))
    for k in x[2][0]:
        out.append(("srcS." + k, float(np.asarray(x[2][0][k]).squeeze())))
    for k in x[2][1]:
        out.append(("srcShp." + k, float(np.asarray(x[2][1][k]).squeeze())))
    return out


def _label_index_map(bij, ndim):
    """z-index -> human label via single-coordinate perturbation probing."""
    base = np.zeros(ndim, dtype=_NP_FDTYPE)
    b = _flat_named46(bij.forward(list(base[:, None])))
    labels0 = [lab for lab, _ in b]
    bvals = np.array([v for _, v in b])
    labels = []
    for j in range(ndim):
        z = base.copy()
        z[j] = 10.0
        pj = _flat_named46(bij.forward(list(z[:, None])))
        pv = np.array([v for _, v in pj])
        changed = np.where(np.abs(pv - bvals) > 1e-4)[0]
        if len(changed) == 1:
            labels.append(labels0[changed[0]])
        elif len(changed) == 0:
            labels.append(f"z{j}_unmapped")
        else:
            labels.append("|".join(labels0[c] for c in changed))
    return labels


# Reduced (41-dim, no-Ie) labels of map_refined_lstsq64.npz['qz_refined'] order.
_LSTSQ_REDUCED_LABELS = [
    "mass.center_x", "mass.center_y", "mass.e1", "mass.e2", "mass.gamma",
    "mass.theta_E", "shear.gamma1", "shear.gamma2",
    "LL0.R_sersic", "LL0.center_x", "LL0.center_y", "LL0.e1", "LL0.e2", "LL0.n_sersic",
    "LL1.R_sersic", "LL1.center_x", "LL1.center_y", "LL1.e1", "LL1.e2", "LL1.n_sersic",
    "LL2.R_sersic", "LL2.center_x", "LL2.center_y", "LL2.e1", "LL2.e2", "LL2.n_sersic",
    "LL3.R_sersic", "LL3.center_x", "LL3.center_y", "LL3.e1", "LL3.e2", "LL3.n_sersic",
    "srcS.R_sersic", "srcS.center_x", "srcS.center_y", "srcS.e1", "srcS.e2",
    "srcS.n_sersic", "srcShp.beta", "srcShp.center_x", "srcShp.center_y",
]


# --------------------------------------------------------------------------- #
# model builder
# --------------------------------------------------------------------------- #
def build_marg_model(product: dict, *, n_max=6, supersample=2, delta_pix=0.13,
                     exp_time=DEFAULT_EXP_TIME, whiten_fn=None, near_xy=None,
                     shapelet_sigma0=5.0, mask_err=1e10):
    """Ridge-marginalized model for a Stage-A product dict. ndim = 46.

    Args:
        product: dict with img (H,W), err_map (H,W), keep_mask (H,W bool),
            psf (kh,kw), meta (dict) — the ``cgl.paths.load_product`` layout.
            err_map is the UNMASKED per-pixel sigma; masking is applied here
            as err -> mask_err (1e10) outside keep_mask, exactly like the
            original central-mask convention.
        n_max: shapelet order (depth = (n_max+1)(n_max+2)/2 columns; 28 at 6).
        supersample, delta_pix, exp_time: simulator config (v2-class
            defaults). Cross-checked against product meta when present.
        whiten_fn: optional whitener image(H,W) -> (n_pix,); None = the
            original diagonal sqrt(W) path, bit-identical math.
        near_xy: (arcsec_x, arcsec_y) prior centers for LL2/LL3 (the nearby
            galaxy); None reads the foundry-i nearby_galaxy_loc.npz artifact.
        shapelet_sigma0: shapelet amp prior scale sigma_i = sigma0/sqrt(i+1)
            -> Lambda_ii = (i+1)/sigma0^2 (original: 5.0 -> (i+1)/25).
        mask_err: inflated err value inside the mask (original 1e10).

    Returns a SimpleNamespace: target_log_prob_fn(z), bij, prior, ndim,
    labels/index_labels, to_physical/to_physical_mass, shapelet_amps(z),
    m_det(z), model_image(z), marg_internals(z), sim_config, Y,
    masked_err_map, keep_mask, Lambda_diag, design_matrix_source.
    """
    guards.require_x64()  # real-data marg posteriors are float64-only

    meta = product.get("meta") or {}
    if "delta_pix" in meta and abs(float(meta["delta_pix"]) - delta_pix) > 1e-12:
        raise ValueError(
            f"delta_pix={delta_pix} but product meta says {meta['delta_pix']}")
    if "supersample" in meta and int(meta["supersample"]) != int(supersample):
        raise ValueError(
            f"supersample={supersample} but product meta says {meta['supersample']}")
    if "psf_pixel_scale" in meta:
        guards.assert_psf_sampling(float(meta["psf_pixel_scale"]), delta_pix)

    img = np.asarray(product["img"], dtype=_NP_FDTYPE)
    err_in = np.asarray(product["err_map"], dtype=_NP_FDTYPE)
    keep_np = np.asarray(product["keep_mask"], dtype=bool)
    psf = np.asarray(product["psf"], dtype=_NP_FDTYPE)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"expected a square image, got {img.shape}")
    if img.shape != err_in.shape or img.shape != keep_np.shape:
        raise ValueError("img / err_map / keep_mask shapes disagree")

    NUM_PIX = img.shape[0]
    sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=NUM_PIX,
                                 supersample=supersample, kernel=psf)

    Y = jnp.asarray(img, dtype=_FDTYPE)
    keep_mask = jnp.asarray(keep_np)
    INF_ERR = jnp.asarray(mask_err, dtype=_FDTYPE)
    masked_err_map = jnp.where(keep_mask, jnp.asarray(err_in, dtype=_FDTYPE),
                               INF_ERR)

    # ---- prior + bijector (46-dim, 5 Ie present, no shapelet amps) ----------
    if near_xy is None:
        nb = np.load(NEARBY_GALAXY_LOC)
        near_xy = (float(nb["arcsec_x"]), float(nb["arcsec_y"]))
    prior = _build_prior(float(near_xy[0]), float(near_xy[1]))
    example = prior.sample(seed=jax.random.PRNGKey(0))
    pack_bij = tfb.pack_sequence_as(example)
    bij = tfb.Chain([prior.experimental_default_event_space_bijector(), pack_bij])

    ndim = int(sum(jnp.size(v) for v in jax.tree_util.tree_leaves(example)))
    assert ndim == 46, f"expected 46 sampled params (v2-class model), got {ndim}"
    index_labels = _label_index_map(bij, ndim)

    # ---- M_det simulator: lensed source Sersic + 4 lens-light Sersics, all
    #      use_lstsq=False (5 explicit sampled Ie). NO shapelets here. ---------
    phys_det = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)] * 4,
        [sersic.SersicEllipse(use_lstsq=False)],
    )
    sim_det = LensSimulator(phys_det, sim_config, bs=1)
    # simulate() multiplies by conversion_factor, but the lstsq design columns
    # (and thus the solved amps) do NOT include it. Divide it back out so
    # M_det is in the same units as X and the data. (original _INV_CF)
    _INV_CF = 1.0 / float(sim_det.conversion_factor)

    # ---- shapelet design-matrix simulator: shapelets-only source ------------
    phys_shp = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [],
        [shapelets.Shapelets(n_max=n_max, use_lstsq=True, interpolate=False)],
    )
    sim_shp = LensSimulator(phys_shp, sim_config, bs=1)
    depth = (n_max + 1) * (n_max + 2) // 2
    assert sim_shp.depth == depth, (
        f"expected {depth} shapelet columns, got {sim_shp.depth}")

    # Tikhonov precision Lambda_ii = 1/sigma_i^2, sigma_i = sigma0/sqrt(i+1)
    # (original sigma0=5 -> (i+1)/25, the exact 74-dim-model shapelet priors).
    i_idx = jnp.arange(depth, dtype=_FDTYPE)
    Lambda_diag = (i_idx + 1.0) / (shapelet_sigma0 * shapelet_sigma0)

    sqrtW = 1.0 / masked_err_map          # (h, w), ~1e-10 inside the mask
    Wcol = sqrtW[..., None]               # (h, w, 1)

    def _design_ret(x_shp):
        """UNWHITENED design tensor ret (h, w, depth): the lensed, PSF-convolved,
        supersample-average-pooled unit-amplitude shapelet basis images in
        DATA-pixel space. Mirrors gigalens lstsq_simulate's per-component
        ``ret`` (the part BEFORE the pinv solve); NO conversion_factor."""
        lens_params = x_shp[0]
        source_light_params = x_shp[1]  # [shapelet_param_dict]
        beta_x, beta_y = sim_shp._beta(lens_params)
        im = jnp.zeros((0, *sim_shp.img_X.shape))
        for lm, p in zip(sim_shp.phys_model.source_light, source_light_params):
            im = jnp.concatenate((im, lm.light(beta_x, beta_y, **p)), axis=0)
        im = jnp.nan_to_num(im)
        im = jnp.transpose(im, (3, 0, 1, 2))  # bs, n_comp, h, w
        ret = jax.lax.conv_general_dilated(
            im, sim_shp.kernel, (1, 1), padding="SAME",
            feature_group_count=sim_shp.depth,
            dimension_numbers=("NCHW", "HWOI", "NCHW"))
        if sim_shp.supersample != 1:
            ret = average_pool_2d(
                ret, size=(sim_shp.supersample, sim_shp.supersample),
                padding="SAME")
        ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, n_comp
        return jnp.squeeze(ret, axis=0)         # (h, w, depth)

    # Whitening applied identically to R and each design column. The default
    # (whiten_fn=None) keeps the original expressions literally, so the
    # diagonal path is bit-identical to _hmc_lib_marg.
    if whiten_fn is None:
        def _whiten_R(R):
            return jnp.reshape(R * sqrtW, (-1,))

        def _whiten_X(ret):
            return jnp.reshape(ret * Wcol, (-1, depth))
    else:
        def _whiten_R(R):
            return whiten_fn(R)

        def _whiten_X(ret):
            return jax.vmap(whiten_fn, in_axes=2, out_axes=1)(ret)

    def _parts(z):
        """Shared per-eval pipeline: z -> (x, M_det, ret, Xw, Rw)."""
        z_list = list(z[None, :].T)
        x = bij.forward(z_list)  # physical params (46-dim, batch=1)
        M_det = jnp.squeeze(sim_det.simulate(x)) * _INV_CF
        ret = _design_ret([x[0], [x[2][1]]])
        Xw = _whiten_X(ret)
        Rw = _whiten_R(Y - M_det)
        return z_list, x, M_det, ret, Xw, Rw

    @jax.jit
    def _logpost(z):
        z_list, x, M_det, ret, Xw, Rw = _parts(z)
        logL, a_star, logdetA = marg_loglik(Xw, Rw, Lambda_diag)
        log_prior = prior.log_prob(x) + bij.forward_log_det_jacobian(z_list)
        return jnp.squeeze(logL + log_prior)

    @jax.jit
    def shapelet_amps(z):
        _, _, _, _, Xw, Rw = _parts(z)
        _, a_star, _ = marg_loglik(Xw, Rw, Lambda_diag)
        return a_star

    @jax.jit
    def m_det(z):
        z_list = list(z[None, :].T)
        x = bij.forward(z_list)
        return jnp.squeeze(sim_det.simulate(x)) * _INV_CF

    @jax.jit
    def model_image(z):
        """Full model image M_det + sum_i a*_i ret_i (data units, lstsq
        no-conversion-factor convention)."""
        _, _, M_det, ret, Xw, Rw = _parts(z)
        _, a_star, _ = marg_loglik(Xw, Rw, Lambda_diag)
        return M_det + jnp.sum(ret * a_star[None, None, :], axis=-1)

    def marg_internals(z):
        """Debug/parity surface: every intermediate of one log-prob eval, as
        numpy (NOT jitted — eager op-by-op; ULP-level fusion differences from
        the jitted _logpost are possible and expected)."""
        z = jnp.asarray(z, dtype=_FDTYPE)
        z_list, x, M_det, ret, Xw, Rw = _parts(z)
        b = Xw.T @ Rw
        A = Xw.T @ Xw + jnp.diag(Lambda_diag)
        chol = jnp.linalg.cholesky(A)
        a_star = jax.scipy.linalg.cho_solve((chol, True), b)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        logL_data = (-0.5 * jnp.sum(Rw ** 2) + 0.5 * jnp.dot(b, a_star)
                     - 0.5 * logdetA)
        log_prior = prior.log_prob(x) + bij.forward_log_det_jacobian(z_list)
        return dict(
            M_det=np.asarray(M_det), ret=np.asarray(ret), Xw=np.asarray(Xw),
            Rw=np.asarray(Rw), b=np.asarray(b), A=np.asarray(A),
            chol=np.asarray(chol), a_star=np.asarray(a_star),
            logdetA=float(logdetA), logL_data=float(np.squeeze(logL_data)),
            log_prior=float(np.squeeze(np.asarray(log_prior))),
            logpost=float(np.squeeze(np.asarray(logL_data + log_prior))),
        )

    def to_physical_mass(samples_np):
        arr = jnp.asarray(samples_np)
        physical = bij.forward(list(arr.T))
        mass_main = physical[0][0]
        mass_shear = physical[0][1]
        out = {k: np.asarray(mass_main[k]) for k in mass_main.keys()}
        out.update({k: np.asarray(mass_shear[k]) for k in mass_shear.keys()})
        return out

    prob_model = _types.SimpleNamespace(bij=bij, prior=prior,
                                        observed_image=Y, err_map=masked_err_map)

    return _types.SimpleNamespace(
        target_log_prob_fn=_logpost,
        prob_model=prob_model,
        prior=prior,
        bij=bij,
        sim_config=sim_config,
        ndim=ndim,
        data_arr=img,
        Y=Y,
        keep_mask=keep_mask,
        masked_err_map=masked_err_map,
        index_labels=index_labels,
        labels=index_labels,
        to_physical_mass=to_physical_mass,
        to_physical=to_physical_mass,
        shapelet_amps=shapelet_amps,
        m_det=m_det,
        model_image=model_image,
        marg_internals=marg_internals,
        Lambda_diag=np.asarray(Lambda_diag),
        whiten_fn=whiten_fn,
        n_max=n_max,
        depth=depth,
        exp_time=exp_time,
        near_xy=tuple(near_xy),
        meta=meta,
        design_matrix_source=(
            "gigalens lstsq_simulate per-component `ret` for a shapelets-only "
            "PhysicalModel ([EPL,Shear] lenses, [] lens_light, "
            f"[Shapelets(n_max={n_max}, use_lstsq=True)] source): the lensed "
            "(deflected by mass), PSF-convolved (conv_general_dilated, "
            "feature_group_count=depth), supersample average-pooled "
            "unit-amplitude basis images in data-pixel space; NO "
            "conversion_factor on design columns (M_det carries 1/cf instead). "
            "Ported with attribution from foundry-i _hmc_lib_marg."),
    )


# --------------------------------------------------------------------------- #
# start-vector helper (ported verbatim; optional — parity uses map_marg_pd)
# --------------------------------------------------------------------------- #
def qz_start_from_lstsq(model, refined_npz_path):
    """46-vector start: 41 nonlinear z from a refined-lstsq npz ['qz_refined']
    + the 5 lstsq-solved positive Sersic Ie ['amps'][:5] mapped through the
    LogNormal bijector by bisection. Ported verbatim from _hmc_lib_marg's
    build_model_marg start-point block.

    Returns (qz_start_jnp, missing_labels).
    """
    rd = np.load(refined_npz_path)
    qz41 = np.asarray(rd["qz_refined"], dtype=_NP_FDTYPE)
    amps33 = np.asarray(rd["amps"], dtype=_NP_FDTYPE)
    # amps order (gigalens lstsq_simulate): LL0..LL3 Ie, srcS Ie, then shapelets.
    ie_phys = {
        "LL0.Ie": float(amps33[0]), "LL1.Ie": float(amps33[1]),
        "LL2.Ie": float(amps33[2]), "LL3.Ie": float(amps33[3]),
        "srcS.Ie": float(amps33[4]),
    }
    lstsq_lookup = {lab: qz41[i] for i, lab in enumerate(_LSTSQ_REDUCED_LABELS)}

    bij, ndim, index_labels = model.bij, model.ndim, model.index_labels
    qz_start = np.zeros(ndim, dtype=_NP_FDTYPE)
    missing = []
    # 1) nonlinear labels: copy z directly (same 1-D bijector per shared label).
    for j, lab in enumerate(index_labels):
        if lab in lstsq_lookup:
            qz_start[j] = lstsq_lookup[lab]
        elif lab in ie_phys:
            qz_start[j] = 0.0  # placeholder, fixed below
        else:
            missing.append((j, lab))
    # 2) Ie labels: set z so bij.forward(z) gives the solved positive Ie
    #    (monotone LogNormal bijector -> bisection on a wide z range).
    z_vec = qz_start.copy()
    for j, lab in enumerate(index_labels):
        if lab in ie_phys:
            target = ie_phys[lab]
            lo, hi = -50.0, 50.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                zt = z_vec.copy()
                zt[j] = mid
                val = dict(_flat_named46(bij.forward(list(zt[:, None]))))[lab]
                if val < target:
                    lo = mid
                else:
                    hi = mid
            z_vec[j] = 0.5 * (lo + hi)
    return jnp.asarray(z_vec, dtype=_FDTYPE), missing
