"""02 - GIGA-Lens MAP + SVI fit of the Cikota+2023 Einstein cross on public DESI
Legacy g-band imaging.

Model (Cikota et al. 2023, Table 2/3; 31 parameters):
  MASS:
    L1  SIE   : theta_E, e1, e2, center_x, center_y
    L2  SIE   : theta_E, e1, e2, center_x, center_y   (foreground galaxy near image C)
    Shear     : gamma1, gamma2
  LIGHT:
    L1  SersicEllipse  (elliptical)
    L2  Sersic         (spherical)
    Source SersicEllipse (elliptical)

Priors follow the paper's Table 2 (these are the GIGA-Lens demo defaults specialized
to this system: theta_E ~ exp(N(ln2,0.25)), shear ~ N(0,0.05), etc.).  We sample the
Sersic Ie amplitudes (use_lstsq=False) so the prob model is ForwardProbModel.

Pinned to ONE GPU (CUDA_VISIBLE_DEVICES) so the gigalens shard_map runs single-device;
NEVER use the gigalens HMC() helper (pmaps over all devices, hangs).

Run (A16 index 6):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=6 /raid/benson/.venvs/gigalens/bin/python 02_fit_map_svi.py
"""
import os
import time
from pathlib import Path

import jax
# gigalens inference.py references jax.experimental.shard_map.shard_map but never
# imports the submodule; in JAX 0.6.2 it is not auto-attached, so import it here to
# populate the jax.experimental namespace.
import jax.experimental.shard_map  # noqa: F401
jax.config.update("jax_compilation_cache_dir",
                  str(Path(__file__).parent / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

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

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
obs = np.load(DATA / "cikota_g_image.npy").astype(np.float32)
psf = np.load(DATA / "cikota_g_psf.npy").astype(np.float32)
meta = np.load(DATA / "cikota_g_meta.npz")
delta_pix = float(meta["delta_pix"])
num_pix = int(meta["num_pix"])
background_rms = float(meta["background_rms"])
exp_time = float(meta["exp_time"])
print(f"image {obs.shape} delta_pix {delta_pix} bg_rms {background_rms:.3f} exp_time {exp_time}")

# --------------------------------------------------------------------------- #
# priors (paper Table 2).  Image x is +East = -RA; the paper centers L1 near the
# image origin and L2 at (x,y)=(1.836,-1.563)".  Mass and light centers share priors.
# We follow Table 2 means: L1 light/mass center ~ N(0,0.05); source ~ N(0,0.25);
# L2 prior centered on its Table-3 position.
# --------------------------------------------------------------------------- #
lens_prior = tfd.JointDistributionSequential([
    # L1 SIE
    tfd.JointDistributionNamed(dict(
        theta_E=tfd.LogNormal(jnp.log(2.0), 0.25),
        e1=tfd.Normal(0.0, 0.1),
        e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05),
        center_y=tfd.Normal(0.0, 0.05),
    )),
    # L2 SIE (foreground galaxy near image C; prior centered on Table-3 position)
    tfd.JointDistributionNamed(dict(
        theta_E=tfd.LogNormal(jnp.log(0.25), 0.3),
        e1=tfd.Normal(0.0, 0.1),
        e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(1.836, 0.15),
        center_y=tfd.Normal(-1.563, 0.15),
    )),
    # External shear
    tfd.JointDistributionNamed(dict(
        gamma1=tfd.Normal(0.0, 0.05),
        gamma2=tfd.Normal(0.0, 0.05),
    )),
])

lens_light_prior = tfd.JointDistributionSequential([
    # L1 elliptical Sersic
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
        n_sersic=tfd.Uniform(1.0, 5.0),
        e1=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
        center_x=tfd.Normal(0.0, 0.05),
        center_y=tfd.Normal(0.0, 0.05),
        Ie=tfd.LogNormal(jnp.log(25.0), 0.3),
    )),
    # L2 spherical Sersic (no ellipticity; center prior on Table-3 position)
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.15),
        n_sersic=tfd.Uniform(1.0, 5.0),
        center_x=tfd.Normal(1.836, 0.15),
        center_y=tfd.Normal(-1.563, 0.15),
        Ie=tfd.LogNormal(jnp.log(25.0), 0.3),
    )),
])

source_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
        n_sersic=tfd.Uniform(0.5, 4.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.25),
        center_y=tfd.Normal(0.0, 0.25),
        Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
    )),
])

prior = tfd.JointDistributionSequential(
    [lens_prior, lens_light_prior, source_light_prior]
)

# --------------------------------------------------------------------------- #
# physical model + simulator
# --------------------------------------------------------------------------- #
phys_model = PhysicalModel(
    lenses=[sie.SIE(), sie.SIE(), shear.Shear()],
    lens_light=[sersic.SersicEllipse(use_lstsq=False),
                sersic.Sersic(use_lstsq=False)],
    source_light=[sersic.SersicEllipse(use_lstsq=False)],
)
sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix,
                             supersample=2, kernel=psf)
prob_model = ForwardProbModel(prior, obs, background_rms=background_rms, exp_time=exp_time)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# --------------------------------------------------------------------------- #
# MAP (reuse saved estimate if present to avoid recomputing)
# --------------------------------------------------------------------------- #
MAP_FILE = DATA / "map_estimate.npy"
if MAP_FILE.exists() and os.environ.get("REUSE_MAP", "1") == "1":
    map_est = jnp.array(np.load(MAP_FILE))
    print(f"=== MAP (reused {MAP_FILE.name}, shape {map_est.shape}) ===")
else:
    print("=== MAP ===")
    t0 = time.time()
    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    map_est, map_lp, map_chisq = model_seq.MAP(opt, n_samples=500, num_steps=1500, seed=0)
    print(f"MAP done in {time.time()-t0:.1f}s  chisq={float(map_chisq):.4f}  logp={float(map_lp):.2f}")
    np.save(MAP_FILE, np.asarray(map_est))

def sq(v):
    return float(np.asarray(v).squeeze())

# decode MAP physical params (map_est is (1,31); bij.forward gives batched dicts)
x_map = prob_model.bij.forward(list(jnp.array(map_est).T))
print("L1 SIE  :", {k: sq(v) for k, v in x_map[0][0].items()})
print("L2 SIE  :", {k: sq(v) for k, v in x_map[0][1].items()})
print("Shear   :", {k: sq(v) for k, v in x_map[0][2].items()})
print("L1 light:", {k: sq(v) for k, v in x_map[1][0].items()})
print("L2 light:", {k: sq(v) for k, v in x_map[1][1].items()})
print("Src     :", {k: sq(v) for k, v in x_map[2][0].items()})

# --------------------------------------------------------------------------- #
# SVI (variational posterior around the MAP)
# --------------------------------------------------------------------------- #
print("=== SVI ===")
t0 = time.time()
opt = optax.adabelief(1e-3, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(jnp.array(map_est), opt, n_vi=500, num_steps=2000, seed=1)
print(f"SVI done in {time.time()-t0:.1f}s  final -ELBO={float(loss_hist[-1]):.2f}")

# sample the variational posterior, push to physical params
n_post = 4000
z_post = qz.sample(n_post, seed=jax.random.PRNGKey(7))
x_post = prob_model.bij.forward(list(z_post.T))
post = {
    "L1_theta_E": np.asarray(x_post[0][0]["theta_E"]),
    "L1_e1": np.asarray(x_post[0][0]["e1"]),
    "L1_e2": np.asarray(x_post[0][0]["e2"]),
    "L1_x": np.asarray(x_post[0][0]["center_x"]),
    "L1_y": np.asarray(x_post[0][0]["center_y"]),
    "L2_theta_E": np.asarray(x_post[0][1]["theta_E"]),
    "L2_x": np.asarray(x_post[0][1]["center_x"]),
    "L2_y": np.asarray(x_post[0][1]["center_y"]),
    "gamma1": np.asarray(x_post[0][2]["gamma1"]),
    "gamma2": np.asarray(x_post[0][2]["gamma2"]),
}
np.savez(DATA / "svi_posterior.npz",
         qz_loc=np.asarray(qz.loc), qz_cov=np.asarray(qz.covariance()),
         loss_hist=np.asarray(loss_hist), **post)
print(f"Saved data/map_estimate.npy, data/svi_posterior.npz")

# --------------------------------------------------------------------------- #
# headline comparison vs paper
# --------------------------------------------------------------------------- #
def med_lo_hi(a):
    return np.median(a), np.percentile(a, 16), np.percentile(a, 84)

print("\n================ RESULTS vs Cikota+2023 ================")
m, lo, hi = med_lo_hi(post["L1_theta_E"])
print(f"L1 theta_E = {m:.3f} (+{hi-m:.3f}/-{m-lo:.3f})   paper 2.520 +0.032/-0.031")
m, lo, hi = med_lo_hi(post["L1_e1"])
print(f"L1 e1      = {m:.3f} (+{hi-m:.3f}/-{m-lo:.3f})   paper -0.365 +/-0.009")
m, lo, hi = med_lo_hi(post["L1_e2"])
print(f"L1 e2      = {m:.3f} (+{hi-m:.3f}/-{m-lo:.3f})   paper -0.486 +/-0.011")
m, lo, hi = med_lo_hi(post["gamma1"])
print(f"gamma1_ext = {m:.4f} (+{hi-m:.4f}/-{m-lo:.4f})   paper -0.008 +/-0.006")
m, lo, hi = med_lo_hi(post["gamma2"])
print(f"gamma2_ext = {m:.4f} (+{hi-m:.4f}/-{m-lo:.4f})   paper -0.038 +/-0.006")
m, lo, hi = med_lo_hi(post["L2_theta_E"])
print(f"L2 theta_E = {m:.3f} (+{hi-m:.3f}/-{m-lo:.3f})   paper 0.261 +0.028/-0.027")
print("========================================================")
