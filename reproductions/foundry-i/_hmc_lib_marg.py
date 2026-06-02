"""Exact ridge-regularized Gaussian marginalization model (the principled fix).

PROBLEM with build_model_lstsq (_hmc_lib_lstsq.py): gigalens lstsq_simulate uses
``pinv(X^T W X, rcond=1e-6)`` on a 33-column amplitude design matrix that is
near rank-deficient (the 28 SHAPELET basis columns are the degenerate ones; the
5 Sersic Ie are well-constrained).  pinv truncates tiny singular values, which is
NON-SMOOTH in theta: ||grad(-log_post)|| floors at ~1e5 even in float64, the
reduced Hessian stays indefinite, and no PD mode / good HMC mixing is reachable.

THE FIX (this module, build_model_marg): marginalize ONLY the 28 shapelet amps
analytically, using their Gaussian priors Normal(0, sigma_i), sigma_i=5/sqrt(i+1),
as Tikhonov regularization.  This makes the normal matrix

    A = X^T diag(W) X + Lambda,     Lambda_ii = 1/sigma_i^2 = (i+1)/25

ALWAYS positive-definite, so the amplitude solve is a SMOOTH Cholesky solve (never
pinv).  The 5 Sersic Ie are instead SAMPLED (LogNormal priors -> strictly
positive, non-degenerate).  Sampled dimension:

    41 current-nonlinear (mass 8 + 4 lens-light Sersic shape 24 + src Sersic shape
       6 + src shapelet beta/centers 3)
  +  5 Sersic Ie (LL0..LL3 + srcS, LogNormal)
  = 46.

PER-LOG-PROB-EVAL DERIVATION (all float64, all differentiable):
  Y = masked data; noise precision W = 1/err^2, with the err inflated to 1e10
      inside the central r<=1.5px mask (reuse the existing masking) so W ~ 0 there.
  M_det = deterministic image = (lensed source Sersic + 4 lens-light Sersics) built
      with the 5 SAMPLED Ie (Sersics use_lstsq=False so Ie are explicit params).
  X = 28-column shapelet design matrix: column i = the lensed, PSF-convolved,
      downsampled unit-amplitude shapelet basis image i, in DATA-pixel space, for
      the current nonlinear (mass + beta + source-shapelet center) params.  This is
      exactly the per-component ``ret`` that gigalens lstsq_simulate builds for the
      use_lstsq=True Shapelets component (see design_matrix_source below).
  R = Y - M_det.
  b = X^T (W (.) R).
  A = X^T diag(W) X + Lambda.        (PD)
  a* = cholesky_solve(A, b).         (smooth; NEVER pinv)
  logL = -0.5 * sum(W R^2) + 0.5 * b . a* - 0.5 * logdet(A)
         where logdet(A) = 2 sum(log(diag(chol(A)))).
  The -0.5 logdet(A) is the Gaussian-evidence / Occam term gigalens OMITS; it
  regularizes degenerate configs and removes the indefinite curvature.

  log_posterior = logL
                + prior.log_prob(physical nonlinear params incl the 5 LogNormal Ie)
                + bijector.forward_log_det_jacobian.

SEED, empirical PSF, supersample=2, exp_time, central mask, sky subtraction,
background_rms are IDENTICAL to build_model_lstsq.

This target is SMOOTH (Cholesky solve + slogdet of a PD matrix), so jax.grad has no
pinv floor.
"""
import functools as _ft
import os as _os
import types as _types
from pathlib import Path

import jax

# float64 must be enabled BEFORE the first jnp array is created.  Calling scripts
# set os.environ['GIGALENS_X64']='1' (or pass --x64) BEFORE importing this module.
X64 = _os.environ.get("GIGALENS_X64", "0") == "1"
if X64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402
from astropy.io import fits  # noqa: E402

from gigalens.jax.profiles.light import sersic, shapelets  # noqa: E402
from gigalens.jax.profiles.mass import epl, shear  # noqa: E402
from gigalens.jax.simulator import LensSimulator  # noqa: E402
from gigalens.model import PhysicalModel  # noqa: E402
from gigalens.simulator import SimulatorConfig  # noqa: E402

tfd = tfp.distributions
tfb = tfp.bijectors
_REPRO = Path(__file__).parent
_DATA = _REPRO / "data"
N_MAX = 6
EXP_TIME = 1197.7
_FDTYPE = jnp.float64 if X64 else jnp.float32
INF_ERR = jnp.asarray(1e10, dtype=_FDTYPE)


def _sersic_prior_ie(R_med, R_sig, Ie_med, Ie_sig, n_lo=0.5, n_hi=8.0,
                     cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    """Sersic prior INCLUDING the LogNormal(Ie) amplitude (use_lstsq=False).

    Hyperparameters identical to _hmc_lib._sersic_prior (the full 74-dim model).
    """
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
        Ie=tfd.LogNormal(jnp.log(Ie_med), Ie_sig),
    ))


def _build_prior():
    """46-dim prior: 5 Sersic Ie present (LogNormal), 28 shapelet amps ABSENT
    (marginalized).  Returns (prior, NEAR_X, NEAR_Y)."""
    nb = np.load(_DATA / "nearby_galaxy_loc.npz")
    NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])

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
    # Same Ie hyperparameters as the full 74-dim _hmc_lib model.
    lens_light_prior = tfd.JointDistributionSequential([
        _sersic_prior_ie(0.4, 0.3, 5.0, 0.5, c_sig=0.02),
        _sersic_prior_ie(2.0, 0.3, 2.0, 0.5, c_sig=0.02),
        _sersic_prior_ie(0.3, 0.3, 1.0, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
        _sersic_prior_ie(0.6, 0.3, 0.5, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
    ])
    src_sersic_prior = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
        Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
    ))
    # Shapelet amps MARGINALIZED -> keep only beta + centers in the prior.
    src_shp_prior = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
    ))
    source_light_prior = tfd.JointDistributionSequential(
        [src_sersic_prior, src_shp_prior])
    prior = tfd.JointDistributionSequential(
        [lens_mass_prior, lens_light_prior, source_light_prior])
    return prior, NEAR_X, NEAR_Y


def _flat_named_46(x):
    """Flatten the bijector.forward output (mass, lens_light, source_light) into a
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
    base = np.zeros(ndim, dtype=(np.float64 if X64 else np.float32))
    b = _flat_named_46(bij.forward(list(base[:, None])))
    labels0 = [l for l, _ in b]
    bvals = np.array([v for _, v in b])
    labels = []
    for j in range(ndim):
        z = base.copy()
        z[j] = 10.0
        pj = _flat_named_46(bij.forward(list(z[:, None])))
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


def build_model_marg(x64=False):
    """Ridge-regularized Gaussian-marginalized model.  ndim = 46.

    Returns a SimpleNamespace with:
      target_log_prob_fn(z)  - jit, scalar log_posterior over 46 params.
      ndim                   = 46.
      qz_start               - 46-vector start: 41 nonlinear from
                               data/map_refined_lstsq64.npz['qz_refined'] + the 5
                               lstsq-solved positive Sersic Ie mapped through the
                               LogNormal bijector.
      to_physical_mass(samples) -> dict of 6 mass params.
      prob_model             - SimpleNamespace(bij=...) (bijector access).
      index_labels           - list[str] label per z-index.
      shapelet_amps(z)       - solved 28 marginal-mode shapelet amps a* at z.
      design_matrix_source   - provenance string for X.
    """
    if x64 and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "build_model_marg(x64=True) requires jax_enable_x64. Set GIGALENS_X64=1 "
            "(or pass --x64) BEFORE importing _hmc_lib_marg.")
    if x64 and not X64:
        raise RuntimeError(
            "build_model_marg(x64=True) but GIGALENS_X64 was not set at import; too "
            "late to enable x64. Set os.environ['GIGALENS_X64']='1' before import.")
    np_fdtype = np.float64 if x64 else np.float32
    jnp_fdtype = jnp.float64 if x64 else jnp.float32

    # ---- data / noise / PSF (IDENTICAL to build_model_lstsq) ----------------
    with fits.open(_DATA / "cutout_F140W.fits") as h:
        sci = h["SCI"].data.astype(np_fdtype)
        wht = h["WHT"].data.astype(np_fdtype)
    sky = float(np.median(sci))
    data_arr = (sci - sky).astype(np_fdtype)
    background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
    kernel = np.load(_DATA / "empirical_psf.npy").astype(np_fdtype)
    NUM_PIX = data_arr.shape[0]
    sim_config = SimulatorConfig(delta_pix=0.13, num_pix=NUM_PIX, supersample=2,
                                 kernel=kernel)

    Y = jnp.asarray(data_arr, dtype=jnp_fdtype)

    # err_map computed exactly like BackwardProbModel (from observed image).
    err_map = jnp.sqrt(background_rms ** 2 +
                       jnp.clip(Y, 0.0, np.inf) / EXP_TIME)
    yy, xx = np.indices(data_arr.shape)
    r_center = np.sqrt((xx - NUM_PIX // 2) ** 2 + (yy - NUM_PIX // 2) ** 2)
    keep_mask = jnp.asarray(r_center > 1.5)
    masked_err_map = jnp.where(keep_mask, err_map, INF_ERR)
    W = (1.0 / masked_err_map ** 2).astype(jnp_fdtype)  # noise precision, ~0 in mask

    # ---- prior + bijector (46-dim, 5 Ie present, no shapelet amps) ----------
    prior, NEAR_X, NEAR_Y = _build_prior()
    example = prior.sample(seed=jax.random.PRNGKey(0))
    pack_bij = tfb.pack_sequence_as(example)
    bij = tfb.Chain([prior.experimental_default_event_space_bijector(), pack_bij])

    ndim = int(sum(jnp.size(v) for v in jax.tree_util.tree_leaves(example)))
    assert ndim == 46, f"expected 46 sampled params, got {ndim}"
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
    # (and thus the stored Sersic-Ie amps) do NOT include it.  Divide it back out
    # so M_det is in the same units as X and the data.
    _INV_CF = 1.0 / float(sim_det.conversion_factor)

    # ---- shapelet design-matrix simulator: shapelets-only source, use_lstsq=True.
    #      Its lstsq_simulate builds the per-component ``ret`` of shape (1,h,w,28)
    #      = the lensed, PSF-convolved, downsampled unit-amplitude basis images
    #      in DATA-pixel space.  This IS the 28-column design matrix X.
    phys_shp = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [],
        [shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False)],
    )
    sim_shp = LensSimulator(phys_shp, sim_config, bs=1)
    assert sim_shp.depth == 28, f"expected 28 shapelet columns, got {sim_shp.depth}"

    # Tikhonov precision Lambda_ii = 1/sigma_i^2, sigma_i = 5/sqrt(i+1) (the exact
    # shapelet Normal(0,sigma_i) priors of the full 74-dim model) -> (i+1)/25.
    i_idx = jnp.arange(28, dtype=jnp_fdtype)
    Lambda_diag = (i_idx + 1.0) / 25.0  # (28,)

    Wcol = (1.0 / masked_err_map)[..., None]  # whitening sqrt(W), shape (h,w,1)

    def _design_matrix(x_shp):
        """Return X whitened columns (n_pix, 28): each col = sqrt(W) (.) ret_i.

        Mirrors the gigalens lstsq_simulate ``ret`` construction (the part BEFORE
        the pinv solve) for the shapelets-only phys model.
        """
        lens_params = x_shp[0]
        source_light_params = x_shp[1]  # [shapelet_param_dict]
        beta_x, beta_y = sim_shp._beta(lens_params)
        img = jnp.zeros((0, *sim_shp.img_X.shape))
        for lm, p in zip(sim_shp.phys_model.source_light, source_light_params):
            img = jnp.concatenate(
                (img, lm.light(beta_x, beta_y, **p)), axis=0)
        img = jnp.nan_to_num(img)
        img = jnp.transpose(img, (3, 0, 1, 2))  # bs, n_comp, h, w
        ret = jax.lax.conv_general_dilated(
            img, sim_shp.kernel, (1, 1), padding="SAME",
            feature_group_count=sim_shp.depth,
            dimension_numbers=("NCHW", "HWOI", "NCHW"))
        from objax.functional import average_pool_2d
        from objax.constants import ConvPadding
        ret = average_pool_2d(ret, size=(sim_shp.supersample, sim_shp.supersample),
                              padding="SAME")
        ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, n_comp
        # NOTE: gigalens lstsq_simulate does NOT apply conversion_factor to its
        # design columns (unlike simulate()); the solved amps absorb it.  We must
        # match that convention so the marginal amps == the lstsq amps.
        ret = jnp.squeeze(ret, axis=0)  # (h,w,28)  -- NO conversion_factor
        Xw = jnp.reshape(ret * Wcol, (-1, sim_shp.depth))  # (n_pix, 28)
        return Xw, ret

    @jax.jit
    def _logpost(z):
        z_list = list(z[None, :].T)
        x = bij.forward(z_list)  # physical params (46-dim, batch=1)

        # M_det: drop the batch dim -> (h,w).  Undo simulate()'s conversion_factor
        # to match the lstsq (no-conv-factor) convention used for X and the amps.
        M_det = jnp.squeeze(sim_det.simulate(x)) * _INV_CF

        # X (design matrix) from the shapelet-only phys model: reuse mass + shapelet
        # beta/center; the shapelet param dict has no amps (use_lstsq design).
        x_shp = [x[0], [x[2][1]]]
        Xw, _ = _design_matrix(x_shp)  # Xw: (n_pix, 28) whitened columns

        R = (Y - M_det)                       # residual, (h,w)
        sqrtW = (1.0 / masked_err_map)        # (h,w)
        Rw = jnp.reshape(R * sqrtW, (-1,))    # whitened residual (n_pix,)

        # b = X^T (W (.) R) = (sqrtW X)^T (sqrtW R) = Xw^T Rw
        b = Xw.T @ Rw                          # (28,)
        # A = X^T W X + Lambda = Xw^T Xw + Lambda
        A = Xw.T @ Xw + jnp.diag(Lambda_diag)  # (28,28), PD

        chol = jnp.linalg.cholesky(A)
        astar = jax.scipy.linalg.cho_solve((chol, True), b)  # smooth, no pinv
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

        # logL (drop theta-independent consts): -0.5 sum(W R^2) + 0.5 b.a* - 0.5 logdetA
        logL = -0.5 * jnp.sum(Rw ** 2) + 0.5 * jnp.dot(b, astar) - 0.5 * logdetA

        log_prior = prior.log_prob(x) + bij.forward_log_det_jacobian(z_list)
        return jnp.squeeze(logL + log_prior)

    target_log_prob_fn = _logpost

    @jax.jit
    def shapelet_amps(z):
        z_list = list(z[None, :].T)
        x = bij.forward(z_list)
        M_det = jnp.squeeze(sim_det.simulate(x)) * _INV_CF
        x_shp = [x[0], [x[2][1]]]
        Xw, _ = _design_matrix(x_shp)
        sqrtW = (1.0 / masked_err_map)
        Rw = jnp.reshape((Y - M_det) * sqrtW, (-1,))
        b = Xw.T @ Rw
        A = Xw.T @ Xw + jnp.diag(Lambda_diag)
        chol = jnp.linalg.cholesky(A)
        return jax.scipy.linalg.cho_solve((chol, True), b)

    # ---- start point: 41 nonlinear from refined lstsq + 5 lstsq-solved Ie -----
    rd = np.load(_DATA / "map_refined_lstsq64.npz")
    qz41 = np.asarray(rd["qz_refined"], dtype=np_fdtype)
    amps33 = np.asarray(rd["amps"], dtype=np_fdtype)
    # amps order (gigalens lstsq_simulate): LL0..LL3 Ie, srcS Ie, then 28 shapelets.
    ie_phys = {
        "LL0.Ie": float(amps33[0]), "LL1.Ie": float(amps33[1]),
        "LL2.Ie": float(amps33[2]), "LL3.Ie": float(amps33[3]),
        "srcS.Ie": float(amps33[4]),
    }
    lstsq_lookup = {lab: qz41[i] for i, lab in enumerate(_LSTSQ_REDUCED_LABELS)}

    # Build the 46-vector start in PHYSICAL space, then unconstrain via bij.inverse,
    # so the LogNormal(Ie) bijector maps the positive solved Ie to the right z.
    base = np.zeros(ndim, dtype=np_fdtype)
    x0 = bij.forward(list(base[:, None]))
    phys0 = _flat_named_46(x0)  # list of (label, value) in z-order
    # Recover physical start values per label: nonlinear from lstsq forward(qz41),
    # Ie from solved amps.  We need physical nonlinear values -> forward the lstsq
    # reduced z through THIS bijector restricted to shared labels.  Simpler: set z
    # directly for shared labels (the bijector for a given label is 1-D & shared),
    # then set Ie z by inverting the LogNormal.
    qz_start = np.zeros(ndim, dtype=np_fdtype)
    missing = []
    # 1) nonlinear labels: copy z directly (same 1-D bijector per shared label).
    for j, lab in enumerate(index_labels):
        if lab in lstsq_lookup:
            qz_start[j] = lstsq_lookup[lab]
        elif lab in ie_phys:
            qz_start[j] = 0.0  # placeholder, fixed below
        else:
            missing.append((j, lab))
    # 2) Ie labels: set z so bij.forward(z) gives the solved positive Ie. Because
    #    each Ie is an independent LogNormal coord, perturb only that index and
    #    solve the scalar bijector numerically by a short search on z.
    #    LogNormal default-space bijector here is Exp/affine; invert directly.
    z_vec = qz_start.copy()
    for j, lab in enumerate(index_labels):
        if lab in ie_phys:
            target = ie_phys[lab]
            # scalar root: find z[j] s.t. forward gives target. The map is monotone
            # (LogNormal -> Exp bijector). Bisection on a wide z range.
            lo, hi = -50.0, 50.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                zt = z_vec.copy()
                zt[j] = mid
                val = dict(_flat_named_46(bij.forward(list(zt[:, None]))))[lab]
                if val < target:
                    lo = mid
                else:
                    hi = mid
            z_vec[j] = 0.5 * (lo + hi)
    qz_start = z_vec
    qz_start_j = jnp.asarray(qz_start, dtype=jnp_fdtype)

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
        target_log_prob_fn=target_log_prob_fn,
        prob_model=prob_model,
        prior=prior,
        bij=bij,
        sim_config=sim_config,
        ndim=ndim,
        data_arr=data_arr,
        qz_start=qz_start_j,
        qz_start_nonlinear=qz_start_j,  # alias
        index_labels=index_labels,
        missing_labels=missing,
        to_physical_mass=to_physical_mass,
        shapelet_amps=shapelet_amps,
        Lambda_diag=np.asarray(Lambda_diag),
        design_matrix_source=(
            "gigalens lstsq_simulate per-component `ret` for a shapelets-only "
            "PhysicalModel ([EPL,Shear] lenses, [] lens_light, "
            "[Shapelets(n_max=6, use_lstsq=True)] source): the lensed (deflected by "
            "mass), PSF-convolved (conv_general_dilated, feature_group_count=depth), "
            "supersample=2 average-pooled, conversion-factor-scaled unit-amplitude "
            "basis images in data-pixel space. ret shape (h,w,28) -> X = reshape to "
            "(n_pix,28); the Sersic image (5 sampled Ie) is the fixed M_det part. "
            "This is exactly gigalens's use_lstsq design-matrix split."),
    )
