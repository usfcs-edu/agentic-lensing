"""04 - PSF ablation: re-fit with the paper's 15x15 Gaussian PSF (FWHM 0.6") on the
SAME public DESI Legacy g-band data, to show that the ~17% theta_E undershoot in 02
is driven by the broad Legacy coadd PSF (FWHM ~1.35"), not by the model/data.

This is a deconvolution-style test: the data still has the 1.35" Legacy seeing baked
in, so feeding the model a 0.6" PSF over-sharpens and should push theta_E and the
recovered structure.  It quantifies the PSF sensitivity.

Run (A16 index 6):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=6 /raid/benson/.venvs/gigalens/bin/python 04_psf_ablation.py
"""
from pathlib import Path
import time

import jax
import jax.experimental.shard_map  # noqa: F401
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).parent / ".jax_cache"))

import numpy as np
import optax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import sie, shear

tfd = tfp.distributions
REPRO = Path(__file__).parent
DATA = REPRO / "data"

obs = np.load(DATA / "cikota_g_image.npy").astype(np.float32)
meta = np.load(DATA / "cikota_g_meta.npz")
delta_pix = float(meta["delta_pix"]); num_pix = int(meta["num_pix"])
background_rms = float(meta["background_rms"]); exp_time = float(meta["exp_time"])

# paper PSF: 15x15 Gaussian, FWHM 0.6" -> sigma = 0.6/2.355/delta_pix pixels
KSZ = 15
fwhm_arcsec = 0.6
sigma_pix = fwhm_arcsec / 2.3548 / delta_pix
yy, xx = np.mgrid[:KSZ, :KSZ]
c = (KSZ - 1) / 2.0
psf = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma_pix ** 2)).astype(np.float32)
psf /= psf.sum()
print(f"Paper-style Gaussian PSF {psf.shape}, FWHM {fwhm_arcsec}\" (sigma {sigma_pix:.2f} px)")

# identical priors / model to 02
lens_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(theta_E=tfd.LogNormal(jnp.log(2.0), 0.25),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05))),
    tfd.JointDistributionNamed(dict(theta_E=tfd.LogNormal(jnp.log(0.25), 0.3),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(1.836, 0.15), center_y=tfd.Normal(-1.563, 0.15))),
    tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))),
])
lens_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
        n_sersic=tfd.Uniform(1.0, 5.0), e1=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3), center_x=tfd.Normal(0.0, 0.05),
        center_y=tfd.Normal(0.0, 0.05), Ie=tfd.LogNormal(jnp.log(25.0), 0.3))),
    tfd.JointDistributionNamed(dict(R_sersic=tfd.LogNormal(jnp.log(0.5), 0.15),
        n_sersic=tfd.Uniform(1.0, 5.0), center_x=tfd.Normal(1.836, 0.15),
        center_y=tfd.Normal(-1.563, 0.15), Ie=tfd.LogNormal(jnp.log(25.0), 0.3))),
])
source_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
        n_sersic=tfd.Uniform(0.5, 4.0), e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5), center_x=tfd.Normal(0.0, 0.25),
        center_y=tfd.Normal(0.0, 0.25), Ie=tfd.LogNormal(jnp.log(150.0), 0.5))),
])
prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])

phys_model = PhysicalModel(
    lenses=[sie.SIE(), sie.SIE(), shear.Shear()],
    lens_light=[sersic.SersicEllipse(use_lstsq=False), sersic.Sersic(use_lstsq=False)],
    source_light=[sersic.SersicEllipse(use_lstsq=False)],
)
sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=2, kernel=psf)
prob_model = ForwardProbModel(prior, obs, background_rms=background_rms, exp_time=exp_time)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

print("=== MAP (paper 0.6\" PSF) ===")
t0 = time.time()
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
map_est, map_lp, map_chisq = model_seq.MAP(opt, n_samples=500, num_steps=1500, seed=0)
print(f"MAP done {time.time()-t0:.1f}s chisq={float(map_chisq):.4f}")
np.save(DATA / "map_estimate_psf06.npy", np.asarray(map_est))

x = prob_model.bij.forward(list(jnp.array(map_est).T))
def sq(v): return float(np.asarray(v).squeeze())
tE = sq(x[0][0]["theta_E"])
c_kms = 299792.458; D_s = 1690.6; D_ls = 1026.1; a2r = np.pi/180/3600
sig = c_kms * np.sqrt(tE * a2r * D_s / (4*np.pi*D_ls))
print(f"\n0.6\" PSF MAP:  theta_E = {tE:.3f}\"  sigma_SIE = {sig:.0f} km/s  [paper 2.520, 379]")
print(f"1.35\" PSF (02): theta_E = 2.103\"  sigma_SIE = 347 km/s")
print("(both on the SAME 1.35\"-seeing Legacy data; the paper used a true 0.6\" MUSE image)")
