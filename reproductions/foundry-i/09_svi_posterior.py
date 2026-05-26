"""SVI-as-posterior: sample 10k draws from the SVI Gaussian, push through the
bijector to physical space, and report posterior medians ± 1σ for mass params.

Paper text (Huang et al. 2025a, p. 20): the SVI surrogate Gaussian is a "good
approximation" to the HMC posterior for the mass parameters. This script
substitutes the SVI variational distribution for the full HMC posterior to
unblock Phase-1 v3 reproduction while the HMC JIT bottleneck is investigated.
"""
from pathlib import Path

import jax
import jax.experimental.shard_map  # noqa: F401 — gigalens 2.0 / JAX 0.6 compat
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.model import BackwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions

REPRO = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i")
DATA = REPRO / "data"

# === Reconstruct v2 model (priors are part of the bijector wiring) ===
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7
kernel = np.load("/raid/benson/lensing-repos/gigalens/src/gigalens/assets/psf.npy").astype(np.float32)
kernel /= kernel.sum()
N_MAX = 6
sim_config = SimulatorConfig(delta_pix=0.13, num_pix=128, supersample=2, kernel=kernel)
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=True), sersic.SersicEllipse(use_lstsq=True)],
    [sersic.SersicEllipse(use_lstsq=True), shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False)],
)

def sersic_lstsq_prior(R_med=1.0, R_sig=0.15, n_lo=0.5, n_hi=10.0, c_sig=0.05):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(0.0, c_sig),
        center_y=tfd.Normal(0.0, c_sig),
    ))

lens_mass_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
        gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
    )),
    tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))),
])
lens_light_prior = tfd.JointDistributionSequential([
    sersic_lstsq_prior(R_med=0.5, R_sig=0.3, n_hi=10.0, c_sig=0.10),
    sersic_lstsq_prior(R_med=2.0, R_sig=0.3, n_hi=10.0, c_sig=0.10),
])
src_sersic_prior = tfd.JointDistributionNamed(dict(
    R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3),
    n_sersic=tfd.Uniform(0.5, 6.0),
    e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
    e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
))
src_shp_prior = tfd.JointDistributionNamed(dict(
    beta=tfd.LogNormal(jnp.log(0.1), 0.1),
    center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
))
source_light_prior = tfd.JointDistributionSequential([src_sersic_prior, src_shp_prior])
prior = tfd.JointDistributionSequential([lens_mass_prior, lens_light_prior, source_light_prior])
prob_model = BackwardProbModel(prior, data_arr, background_rms=background_rms, exp_time=EXP_TIME)

# === Reconstruct qz from saved SVI mean+cov ===
svi = np.load(DATA / "svi_result.npz")
qz_mean = jnp.asarray(svi["mean"])
qz_cov = jnp.asarray(svi["cov"])
scale_tril = jnp.linalg.cholesky(qz_cov)
qz = tfd.MultivariateNormalTriL(loc=qz_mean, scale_tril=scale_tril)
print(f"Loaded SVI: mean {qz_mean.shape}, cov {qz_cov.shape}")
print(f"qz diag stds (z-space): min={float(jnp.min(jnp.sqrt(jnp.diag(qz_cov)))):.4e}, "
      f"max={float(jnp.max(jnp.sqrt(jnp.diag(qz_cov)))):.4e}")

# === Sample and transform ===
N_SAMP = 10000
print(f"\nSampling {N_SAMP} draws from qz and applying bijector forward...")
key = jax.random.PRNGKey(42)
z = qz.sample(N_SAMP, seed=key)  # shape (N_SAMP, 29) in unconstrained z-space
print(f"z shape: {z.shape}")

# bij.forward expects list of arrays (one per param slot, batch on axis 0)
physical_samples = prob_model.bij.forward(list(jnp.asarray(z).T))
mass_main = physical_samples[0][0]
mass_shear = physical_samples[0][1]
combined = {**{k: np.asarray(mass_main[k]) for k in mass_main.keys()},
            **{k: np.asarray(mass_shear[k]) for k in mass_shear.keys()}}
print(f"Per-param sample shapes: {dict((k, v.shape) for k, v in list(combined.items())[:2])} ...")

# Save physical-space mass samples for downstream corner-plot, etc.
np.savez(DATA / "svi_posterior_mass.npz", **combined)
print(f"Saved {DATA / 'svi_posterior_mass.npz'}")

paper = dict(
    theta_E=(2.6463, 0.0017),
    gamma=(1.372, 0.023),
    e1=(0.1091, 0.0020),
    e2=(-0.1320, 0.0020),
    gamma1=(0.0657, 0.0024),
    gamma2=(-0.0939, 0.0022),
)
print("\nSVI posterior summary (10k samples, mass params only):")
print(f"  {'param':>10s}   {'v3 median':>10s}  ({'+1σ':>7s}/{'-1σ':>7s})    "
      f"{'paper (HMC)':>17s}     {'Δ_med':>8s}    {'|Δ|/σ_paper':>10s}    {'σ_v3/σ_paper':>12s}")
for k, (mu_p, sig_p) in paper.items():
    arr = combined.get(k)
    if arr is None:
        continue
    med = float(np.median(arr))
    lo = float(np.percentile(arr, 16))
    hi = float(np.percentile(arr, 84))
    sig_v3 = 0.5 * (hi - lo)
    print(f"  {k:>10s}   {med:+7.4f}  ({hi-med:+7.4f}/{lo-med:+7.4f})    "
          f"{mu_p:+7.4f}±{sig_p:.4f}    {med-mu_p:+7.4f}    {abs(med-mu_p)/sig_p:8.2f}σ    "
          f"{sig_v3/sig_p:10.2f}×")
print("\nDone.")
