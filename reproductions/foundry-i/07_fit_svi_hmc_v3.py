"""v3 GIGA-Lens posterior: SVI initialized from v2 MAP, then HMC.

Replicates the three-stage GIGA-Lens inference (multi-start MAP → SVI → HMC).
v3 reuses the v2 model (29 free non-linear params + lstsq amps).
"""
import time
from pathlib import Path
import gigalens

import jax
import jax.experimental.shard_map  # noqa: F401
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions

REPRO = Path(__file__).parent
DATA = REPRO / "data"

# === Data ===
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

kernel = np.load(Path(gigalens.__file__).parent / "assets" / "psf.npy").astype(np.float32)
kernel /= kernel.sum()

# === v2 model exactly as 05_fit_map_v2.py ===
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
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# === Load v2 MAP best params as SVI starting point ===
v2 = np.load(DATA / "map_v2_result.npz")
start = jnp.asarray(v2["best_params"])[None, :]   # shape (1, 29)
print(f"SVI start shape: {start.shape}, v2 best log_p={float(v2['best_lp']):.2f}")

# === SVI ===
N_VI = 200
SVI_STEPS = 600
print(f"\nSVI: n_vi={N_VI}, num_steps={SVI_STEPS}")
opt_svi = optax.adabelief(1e-4, b1=0.95, b2=0.99)
t0 = time.time()
qz, loss_hist = model_seq.SVI(start, opt_svi, n_vi=N_VI, num_steps=SVI_STEPS, seed=0)
elapsed = time.time() - t0
print(f"SVI done in {elapsed:.1f} s ({elapsed/60:.2f} min)")

# Save loss history
np.save(DATA / "svi_loss_hist.npy", np.asarray(loss_hist))
print(f"Final -ELBO mean (last 50): {float(jnp.mean(loss_hist[-50:])):.2f}")

# Variational mean (the SVI point estimate)
qz_mean = np.asarray(qz.mean())
qz_cov  = np.asarray(qz.covariance())
np.savez(DATA / "svi_result.npz", mean=qz_mean, cov=qz_cov)
print(f"qz mean shape: {qz_mean.shape}, cov shape: {qz_cov.shape}")

# === HMC ===
N_HMC = 20            # n_hmc chains
N_BURN = 200
N_KEEP = 500
print(f"\nHMC: n_hmc={N_HMC}, burnin={N_BURN}, results={N_KEEP}")
t0 = time.time()
samples = model_seq.HMC(qz, n_hmc=N_HMC, num_burnin_steps=N_BURN, num_results=N_KEEP, seed=0)
elapsed = time.time() - t0
print(f"HMC done in {elapsed:.1f} s ({elapsed/60:.2f} min)")

samples_np = np.asarray(samples)
print(f"Samples shape: {samples_np.shape}")
np.save(DATA / "hmc_samples.npy", samples_np)

# === Posterior summary in physical space ===
# samples shape (num_devices, num_results, n_chains_per_device, n_dim) typically;
# flatten to (-1, n_dim) for downstream analysis
flat = samples_np.reshape(-1, samples_np.shape[-1])
print(f"Flat samples shape: {flat.shape}")

# Forward-transform via the bijector to physical params
# Note: bij.forward expects a list of arrays (one per param slot, batch first)
arr = jnp.asarray(flat)
physical_samples = prob_model.bij.forward(list(arr.T))  # input arrays shape (n_samples,)

# Indices that point to the mass params in the nested structure:
# physical_samples[0][0] = main lens mass dict  (theta_E, gamma, e1, e2, center_x, center_y)
# physical_samples[0][1] = shear dict          (gamma1, gamma2)
mass_main = physical_samples[0][0]
mass_shear = physical_samples[0][1]

print("\nPosterior summary (mass params):")
paper = dict(
    theta_E=(2.6463, 0.0017),
    gamma=(1.372, 0.023),
    e1=(0.1091, 0.0020),
    e2=(-0.1320, 0.0020),
    gamma1=(0.0657, 0.0024),
    gamma2=(-0.0939, 0.0022),
)
combined = {**{k: np.asarray(mass_main[k]) for k in mass_main.keys()},
            **{k: np.asarray(mass_shear[k]) for k in mass_shear.keys()}}
print(f"  {'param':>10s}  {'v3 median ± 1σ':>20s}   {'paper':>15s}   {'Δ_med':>8s}   {'#σ_paper':>10s}")
for k, (mu_p, sig_p) in paper.items():
    arr = combined.get(k)
    if arr is None:
        continue
    med = float(np.median(arr))
    lo = float(np.percentile(arr, 16))
    hi = float(np.percentile(arr, 84))
    sig_v3 = 0.5 * (hi - lo)
    print(f"  {k:>10s}   {med:+7.4f} (+{hi-med:.4f}/{lo-med:.4f})  {mu_p:+7.4f}±{sig_p:.4f}   {med-mu_p:+7.4f}   {(med-mu_p)/sig_p:+8.2f}σ")

print("\nDone.")
