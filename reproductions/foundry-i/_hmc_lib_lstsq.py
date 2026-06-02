"""lstsq-marginalized model for the foundry-i gigalens reproduction.

The full 74-dim model (_hmc_lib.build_model) samples 33 LINEAR light amplitudes
(4 lens-light Sersic Ie + 1 source Sersic Ie + 28 source shapelet amps). At the
refined MAP those amplitudes leave ~56 near-flat directions in the Hessian of
-log_post, which is why fixed-leapfrog HMC mixes terribly.

build_model_lstsq() profiles-out (marginalizes via least squares per log_prob
call) all 33 linear amplitudes by setting use_lstsq=True on every light profile
(4 lens SersicEllipse, 1 source SersicEllipse, 1 source Shapelets(n_max=6)) and
using a BackwardProbModel. HMC then samples ONLY the ~41 NONLINEAR parameters:

    8  mass        : theta_E, gamma, e1, e2, center_x, center_y, gamma1, gamma2
    24 lens light  : 4 SersicEllipse x {R_sersic, n_sersic, e1, e2, cx, cy}
    6  source Ser. : R_sersic, n_sersic, e1, e2, cx, cy
    3  source shp. : beta, center_x, center_y
    --
    41 total

Everything shared with _hmc_lib.build_model() (data, sky subtraction,
background_rms, EXP_TIME, empirical PSF, supersample=2, central r>1.5px mask) is
kept IDENTICAL so the only change is the amplitude marginalization.

The lstsq solve in gigalens (simulator.lstsq_simulate) is a plain pinv weighted
least-squares -- fully DIFFERENTIABLE (no NNLS / clamp), so HMC gradients flow
cleanly. Amplitudes are therefore UNCONSTRAINED (can be negative); we report how
many are negative rather than forcing positivity (which would break grad).
"""
import functools as _ft
import os as _os
import types as _types
from pathlib import Path

import jax

# ---------------------------------------------------------------------------
# float64 support.  The reduced (amplitude-marginalized) objective is extremely
# stiff (reduced-Hessian cond ~1e10), which EXCEEDS float32's ~7 digits: the
# gradient of -log_post then saturates at a ~1.2e4 float32 noise floor, no PD
# mode is reachable, and HMC freezes.  cond ~1e10 DOES fit float64's ~16 digits,
# and GIGA-Lens 2.0 supports float64 throughout.
#
# jax.config.update('jax_enable_x64', True) MUST run before ANY jnp array is
# created, so we gate it on the GIGALENS_X64 env var read at module import,
# BEFORE the first `import jax.numpy as jnp` use below.  Calling scripts set
# os.environ['GIGALENS_X64'] = '1' (or pass --x64, which sets it) BEFORE
# `import _hmc_lib_lstsq`.  build_model_lstsq(x64=...) then asserts consistency
# and loads the data/PSF/qz_start/mask in the matching dtype.
# ---------------------------------------------------------------------------
X64 = _os.environ.get("GIGALENS_X64", "0") == "1"
if X64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (must follow the x64 config update)
import numpy as np  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402
from astropy.io import fits  # noqa: E402

from gigalens.jax.model import BackwardProbModel  # noqa: E402
from gigalens.jax.profiles.light import sersic, shapelets  # noqa: E402
from gigalens.jax.profiles.mass import epl, shear  # noqa: E402
from gigalens.jax.simulator import LensSimulator  # noqa: E402
from gigalens.model import PhysicalModel  # noqa: E402
from gigalens.simulator import SimulatorConfig  # noqa: E402

tfd = tfp.distributions
_REPRO = Path(__file__).parent
_DATA = _REPRO / "data"
N_MAX = 6
EXP_TIME = 1197.7
# Float dtype used to load data/PSF/mask/qz_start.  float64 when x64 on so the
# whole forward pass (and thus the jax.grad) runs in float64.
_FDTYPE = jnp.float64 if X64 else jnp.float32
INF_ERR = jnp.asarray(1e10, dtype=_FDTYPE)


def _sersic_prior_nolstsq(R_med, R_sig, n_lo=0.5, n_hi=8.0,
                          cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    """Sersic prior WITHOUT the Ie amplitude (use_lstsq=True drops it).

    Hyperparameters for R_sersic/n_sersic/e1/e2/centers are identical to
    _hmc_lib._sersic_prior; only the LogNormal(Ie) term is removed.
    """
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
    ))


def _build_prior():
    """Reduced prior (no light amplitudes). Returns (prior, NEAR_X, NEAR_Y)."""
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
    lens_light_prior = tfd.JointDistributionSequential([
        _sersic_prior_nolstsq(0.4, 0.3, c_sig=0.02),
        _sersic_prior_nolstsq(2.0, 0.3, c_sig=0.02),
        _sersic_prior_nolstsq(0.3, 0.3, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
        _sersic_prior_nolstsq(0.6, 0.3, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
    ])
    src_sersic_prior = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
    ))
    # Shapelets use_lstsq=True -> drop the 28 amp priors; keep beta + centers.
    src_shp_prior = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
    ))
    source_light_prior = tfd.JointDistributionSequential(
        [src_sersic_prior, src_shp_prior])
    prior = tfd.JointDistributionSequential(
        [lens_mass_prior, lens_light_prior, source_light_prior])
    return prior, NEAR_X, NEAR_Y


def _label_index_map(prob_model, ndim):
    """Perturbation-probe z-index -> human label for a prob_model's bijector.

    Returns list `labels` of length ndim (labels[j] is the param at z-index j).
    Robust to bijector reordering. Uses CPU-friendly small perturbations.
    """
    # Match the active float width so the bijector (built from float64 prior
    # distributions under x64) doesn't see a dtype mismatch.
    base = np.zeros(ndim, dtype=(np.float64 if X64 else np.float32))

    def flat_named(x):
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

    b = flat_named(prob_model.bij.forward(list(base[:, None])))
    labels0 = [l for l, _ in b]
    bvals = np.array([v for _, v in b])
    labels = []
    for j in range(ndim):
        z = base.copy()
        z[j] = 10.0
        pj = flat_named(prob_model.bij.forward(list(z[:, None])))
        pv = np.array([v for _, v in pj])
        changed = np.where(np.abs(pv - bvals) > 1e-4)[0]
        if len(changed) == 1:
            labels.append(labels0[changed[0]])
        elif len(changed) == 0:
            labels.append(f"z{j}_unmapped")
        else:
            labels.append("|".join(labels0[c] for c in changed))
    return labels


def build_model_lstsq(x64=False):
    """Reduced (amplitude-marginalized) model.  ndim ~= 41.

    Args:
      x64: if True, REQUIRES that jax_enable_x64 is already on (set via the
        GIGALENS_X64 env var at module import; calling scripts pass --x64 which
        sets it BEFORE importing this module).  Data/PSF/mask/qz_start are then
        loaded as float64 so the entire forward pass and the jax.grad of
        target_log_prob_fn are float64.  cond ~1e10 fits float64 (~16 digits)
        but exceeds float32 (~7), so this is what lets the MAP refinement reach
        a genuine PD reduced mode instead of stalling at the f32 ~1.2e4 floor.

    Returns a SimpleNamespace with:
      target_log_prob_fn(z)  - jit, over the reduced nonlinear params, scalar.
      prob_model             - masked BackwardProbModel.
      phys_model, sim_config, data_arr
      ndim                   - reduced parameter count (41).
      qz_start_nonlinear     - reduced-parameterization start mapped from
                               data/map_refined.npz['qz_refined'].
      reduced_index_labels   - list[str], label per reduced z-index.
      to_physical_mass(samples_np) -> dict of 6 mass params.
      lstsq_amps(z)          - solved 33 linear amplitudes at z (for diagnostics).
    """
    if x64 and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "build_model_lstsq(x64=True) requires jax_enable_x64. Set the env var "
            "GIGALENS_X64=1 (or pass --x64, which does so) BEFORE importing "
            "_hmc_lib_lstsq, so x64 is enabled before any jnp array is created.")
    if x64 and not X64:
        # x64 requested but module imported without GIGALENS_X64 -> too late.
        raise RuntimeError(
            "build_model_lstsq(x64=True) but GIGALENS_X64 was not set at module "
            "import. Set os.environ['GIGALENS_X64']='1' before `import "
            "_hmc_lib_lstsq`.")
    np_fdtype = np.float64 if x64 else np.float32
    jnp_fdtype = jnp.float64 if x64 else jnp.float32

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

    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=True)] * 4,
        [sersic.SersicEllipse(use_lstsq=True),
         shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False)],
    )

    prior, _NEAR_X, _NEAR_Y = _build_prior()

    yy, xx = np.indices(data_arr.shape)
    r_center = np.sqrt((xx - NUM_PIX // 2) ** 2 + (yy - NUM_PIX // 2) ** 2)
    keep_mask = r_center > 1.5
    mask_jax = jnp.asarray(keep_mask)

    prob_model = BackwardProbModel(prior, data_arr, background_rms=background_rms,
                                   exp_time=EXP_TIME)
    # BackwardProbModel precomputes err_map from the OBSERVED image. Inflate it at
    # masked (central) pixels so masked pixels are down-weighted both in the linear
    # lstsq solve AND the likelihood -- consistent masking.
    masked_err_map = jnp.where(mask_jax, prob_model.err_map, INF_ERR)

    @_ft.partial(jax.jit, static_argnums=(0, 1))
    def masked_log_prob(self, simulator, z):
        z = list(z.T)
        x = self.bij.forward(z)
        im_sim = simulator.lstsq_simulate(
            x, self.observed_image, masked_err_map)[0]
        log_like = tfd.Independent(
            tfd.Normal(im_sim, masked_err_map), reinterpreted_batch_ndims=2
        ).log_prob(self.observed_image)
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
        chisq = jnp.mean(
            jnp.where(mask_jax, ((im_sim - self.observed_image) / masked_err_map) ** 2,
                      0.0), axis=(-2, -1))
        return log_like + log_prior, chisq

    prob_model.log_prob = _types.MethodType(masked_log_prob, prob_model)
    lens_sim_bs1 = LensSimulator(phys_model, sim_config, bs=1)

    @jax.jit
    def target_log_prob_fn(z):
        z_batched = z[None, :]
        lp, _ = prob_model.log_prob(lens_sim_bs1, z_batched)
        return jnp.squeeze(lp)

    # ndim from a prior sample.
    s = prior.sample(1, seed=jax.random.PRNGKey(0))
    ndim = int(sum(jnp.size(v) for v in jax.tree_util.tree_leaves(s)))

    # ---- map refined 74-vector into the reduced parameterization ------------
    reduced_labels = _label_index_map(prob_model, ndim)

    rd = np.load(_DATA / "map_refined.npz")
    qz74 = np.asarray(rd["qz_refined"], dtype=np_fdtype)
    # Full-model z-index -> label (verified ordering, hard-coded for robustness;
    # mass + per-Sersic + shapelet blocks). Build by the same kept-param logic.
    full_labels = _FULL_LABELS
    full_lookup = {lab: i for i, lab in enumerate(full_labels)}
    qz_start = np.zeros(ndim, dtype=np_fdtype)
    missing = []
    for j, lab in enumerate(reduced_labels):
        if lab in full_lookup:
            qz_start[j] = qz74[full_lookup[lab]]
        else:
            missing.append((j, lab))
    qz_start_nonlinear = jnp.asarray(qz_start, dtype=jnp_fdtype)

    def to_physical_mass(samples_np):
        arr = jnp.asarray(samples_np)
        physical = prob_model.bij.forward(list(arr.T))
        mass_main = physical[0][0]
        mass_shear = physical[0][1]
        out = {k: np.asarray(mass_main[k]) for k in mass_main.keys()}
        out.update({k: np.asarray(mass_shear[k]) for k in mass_shear.keys()})
        return out

    @jax.jit
    def lstsq_amps(z):
        x = prob_model.bij.forward(list(z[None, :].T))
        _, coeffs = lens_sim_bs1.lstsq_simulate(
            x, prob_model.observed_image, masked_err_map)
        return coeffs

    return _types.SimpleNamespace(
        target_log_prob_fn=target_log_prob_fn,
        prob_model=prob_model,
        phys_model=phys_model,
        sim_config=sim_config,
        ndim=ndim,
        data_arr=data_arr,
        qz_start_nonlinear=qz_start_nonlinear,
        reduced_index_labels=reduced_labels,
        full_index_labels=full_labels,
        missing_labels=missing,
        to_physical_mass=to_physical_mass,
        lstsq_amps=lstsq_amps,
    )


# Full-model z-index -> label (perturbation-probed from _hmc_lib.build_model).
_FULL_LABELS = [
    "mass.center_x", "mass.center_y", "mass.e1", "mass.e2", "mass.gamma",
    "mass.theta_E", "shear.gamma1", "shear.gamma2",
    "LL0.Ie", "LL0.R_sersic", "LL0.center_x", "LL0.center_y", "LL0.e1", "LL0.e2",
    "LL0.n_sersic",
    "LL1.Ie", "LL1.R_sersic", "LL1.center_x", "LL1.center_y", "LL1.e1", "LL1.e2",
    "LL1.n_sersic",
    "LL2.Ie", "LL2.R_sersic", "LL2.center_x", "LL2.center_y", "LL2.e1", "LL2.e2",
    "LL2.n_sersic",
    "LL3.Ie", "LL3.R_sersic", "LL3.center_x", "LL3.center_y", "LL3.e1", "LL3.e2",
    "LL3.n_sersic",
    "srcS.Ie", "srcS.R_sersic", "srcS.center_x", "srcS.center_y", "srcS.e1",
    "srcS.e2", "srcS.n_sersic",
] + [f"srcShp.amp{str(i).zfill(2)}" for i in range(28)] + [
    "srcShp.beta", "srcShp.center_x", "srcShp.center_y",
]
