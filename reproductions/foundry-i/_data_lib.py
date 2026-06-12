"""Shared data + model builders for the paper-scale Foundry-I pipeline (scripts 40-45).

Implements the research-group feedback on the Phase-1 reproduction:
  * tight crop so the lens system fills the frame (80x80 px = 10.4" at 0.13"/px),
  * masks for interloping objects (3 faint galaxies + object in arc A + cores),
  * drizzle-corrected per-pixel noise normalized so source-free sky has
    reduced chi^2 = 1, with ONE error-map definition shared by fit and plots.

The model stack runs on the vendored gigalens-sean library (multinode-2025
branch, ref in vendor/gigalens-sean/VENDORED_REF.txt) -- the group's latest
stable line with shard_map multi-device MAP/SVI/HMC. The masked probability
model mirrors the carousel-branch ProbModel conventions (masked observed_dist,
observed-image Poisson term, masked reduced chi^2).

Everything is float32 (paper mode); the published pipeline ran the same
41-parameter model class at single precision throughput.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent
DATA = REPRO / "data"
VENDOR_SRC = REPRO / "vendor" / "gigalens-sean" / "src"

EXP_TIME = 1197.7          # 3 x 399.23 s, total F140W exposure
DELTA_PIX = 0.13           # arcsec / px (native WFC3/IR scale of the cutout)
SUPERSAMPLE = 2
N_MAX = 6                  # shapelet order, matches the paper

PAPER = dict(theta_E=2.6463, gamma=1.372, e1=0.1091, e2=-0.1320,
             gamma1=0.0657, gamma2=-0.0939)


def bootstrap_vendor():
    """Put the vendored gigalens-sean first on sys.path (shadows any pip install)."""
    p = str(VENDOR_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_v2(data_file: str = "cutout_v2.npz"):
    """Load a Stage-A data product (40_make_cutout_v2 / 40b_make_cutout_fine)."""
    z = np.load(DATA / data_file)
    meta = json.loads(str(z["meta"]))
    return dict(
        img=z["img"].astype(np.float32),
        err_map=z["err_map"].astype(np.float32),
        keep_mask=z["keep_mask"].astype(bool),
        psf=z["psf"].astype(np.float32),
        meta=meta,
    )


def build_prior_and_phys(near_x: float, near_y: float, n_max: int = N_MAX,
                         companion_extra: bool = False):
    """Paper Table-2 priors + the 74-parameter physical model (v9 port).

    EPL + external shear; 2 Sersic for the main lens light + 2 for the nearby
    companion (3 with companion_extra, for the Stage-B flexibility pass);
    Sersic + shapelets(n_max) source. All intensities sampled explicitly
    (use_lstsq=False) with positive LogNormal priors.
    """
    bootstrap_vendor()
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.profiles.light import sersic, shapelets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.model import PhysicalModel

    tfd = tfp.distributions

    n_ll = 5 if companion_extra else 4
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False) for _ in range(n_ll)],
        [sersic.SersicEllipse(use_lstsq=False),
         shapelets.Shapelets(n_max=n_max, use_lstsq=False, interpolate=False)],
    )

    def sersic_prior(R_med, R_sig, Ie_med, Ie_sig, n_lo=0.5, n_hi=8.0,
                     cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
        return tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
            n_sersic=tfd.Uniform(n_lo, n_hi),
            e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
            Ie=tfd.LogNormal(jnp.log(Ie_med), Ie_sig),
        ))

    lens_mass_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
            gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
            e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
            center_x=tfd.Normal(0.0, 0.02), center_y=tfd.Normal(0.0, 0.02),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))),
    ])
    ll_priors = [
        sersic_prior(0.4, 0.3, 5.0, 0.5, c_sig=0.02),
        sersic_prior(2.0, 0.3, 2.0, 0.5, c_sig=0.02),
        sersic_prior(0.3, 0.3, 1.0, 0.5, cx_mean=near_x, cy_mean=near_y, c_sig=0.1),
        sersic_prior(0.6, 0.3, 0.5, 0.5, cx_mean=near_x, cy_mean=near_y, c_sig=0.1),
    ]
    if companion_extra:
        ll_priors.append(
            sersic_prior(0.15, 0.4, 0.3, 0.7, cx_mean=near_x, cy_mean=near_y,
                         c_sig=0.15))
    lens_light_prior = tfd.JointDistributionSequential(ll_priors)
    src_sersic_prior = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
        Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
    ))
    shp_amp_names = shapelets.Shapelets(n_max=n_max)._amp_names
    amp_priors = {name: tfd.Normal(0.0, 5.0 / float(jnp.sqrt(i + 1)))
                  for i, name in enumerate(shp_amp_names)}
    src_shp_prior = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
        **amp_priors,
    ))
    source_light_prior = tfd.JointDistributionSequential([src_sersic_prior, src_shp_prior])
    prior = tfd.JointDistributionSequential(
        [lens_mass_prior, lens_light_prior, source_light_prior])
    return prior, phys_model


def make_masked_prob_model(prior, img, err_map, keep_mask):
    """ForwardProbModel with mask + precomputed error map.

    Mirrors the carousel-branch ProbModel pixel conventions:
      * the likelihood is an Independent Normal over UNMASKED pixels only,
      * the chi^2 statistic returned by log_prob is the masked reduced chi^2
        (mean over unmasked pixels) -- this is the Stage-B gate quantity,
      * the error map is fixed (data-based Poisson + rescaled sky), identical
        in the fit and in every plot.
    """
    bootstrap_vendor()
    import functools
    import jax.numpy as jnp
    from jax import jit
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.model import ForwardProbModel

    tfd = tfp.distributions

    model = ForwardProbModel(prior, img, background_rms=1.0, exp_time=EXP_TIME)
    mask = np.asarray(keep_mask, dtype=bool)
    obs = jnp.asarray(img, dtype=jnp.float32)
    err = jnp.asarray(err_map, dtype=jnp.float32)
    observed_dist = tfd.Independent(
        tfd.Normal(obs[mask], err[mask]), reinterpreted_batch_ndims=1)
    mask_j = jnp.asarray(mask)

    @functools.partial(jit, static_argnums=(0, 1))
    def masked_log_prob(self, simulator, z):
        z = list(z.T)
        x = self.bij.forward(z)
        im_sim = simulator.simulate(x)
        im_sim = im_sim.reshape((-1, *obs.shape))
        log_like = observed_dist.log_prob(im_sim[:, mask_j])
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
        red_chi2 = jnp.mean(
            ((im_sim - obs) / err)[:, mask_j] ** 2, axis=-1)
        return log_like + log_prior, red_chi2

    import types
    model.log_prob = types.MethodType(masked_log_prob, model)
    model.keep_mask = mask
    model.error_map = np.asarray(err_map, dtype=np.float32)
    return model


def build_all(n_max: int = N_MAX, companion_extra: bool = False,
              data_file: str = "cutout_v2.npz"):
    """One-call assembly: (data dict, prior, phys_model, prob_model, sim_config).

    The pixel scale and supersampling come from the data product's metadata
    (v2: 0.13"/px supersample 2; v3 fine skycell: 0.04"/px supersample 1).
    """
    bootstrap_vendor()
    from gigalens.simulator import SimulatorConfig

    d = load_v2(data_file)
    near = d["meta"]["nearby_arcsec"]
    prior, phys = build_prior_and_phys(near[0], near[1], n_max=n_max,
                                       companion_extra=companion_extra)
    prob = make_masked_prob_model(prior, d["img"], d["err_map"], d["keep_mask"])
    sim_config = SimulatorConfig(
        delta_pix=float(d["meta"].get("delta_pix", DELTA_PIX)),
        num_pix=d["img"].shape[0],
        supersample=int(d["meta"].get("supersample", SUPERSAMPLE)),
        kernel=d["psf"])
    return d, prior, phys, prob, sim_config


def mass_params_from_z(prob_model, z_array):
    """Map unconstrained z (n, 74) -> dict of the 6 mass parameters (numpy)."""
    import jax.numpy as jnp
    physical = prob_model.bij.forward(list(jnp.asarray(z_array).T))
    main, sh = physical[0][0], physical[0][1]
    return {k: np.asarray(main[k]) for k in
            ("theta_E", "gamma", "e1", "e2", "center_x", "center_y")} | {
            "gamma1": np.asarray(sh["gamma1"]), "gamma2": np.asarray(sh["gamma2"])}
