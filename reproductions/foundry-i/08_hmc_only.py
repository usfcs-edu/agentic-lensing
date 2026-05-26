"""HMC-only resume of v3: load saved SVI result, run HMC with progress bar."""
import time
from pathlib import Path
import gigalens

import jax
import jax.experimental.shard_map  # noqa: F401
import jax.numpy as jnp
import numpy as np
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

# === v2 model exactly as before ===
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

# === Reconstruct qz from saved SVI mean+cov ===
svi = np.load(DATA / "svi_result.npz")
qz_mean = jnp.asarray(svi["mean"])
qz_cov = jnp.asarray(svi["cov"])
# tfd.MultivariateNormalTriL requires lower-triangular scale_tril (Cholesky of covariance)
scale_tril = jnp.linalg.cholesky(qz_cov)
qz = tfd.MultivariateNormalTriL(loc=qz_mean, scale_tril=scale_tril)
print(f"Loaded SVI: qz_mean shape {qz_mean.shape}, qz_cov shape {qz_cov.shape}")
print(f"qz diag stds: min={float(jnp.min(jnp.sqrt(jnp.diag(qz_cov)))):.4e}, max={float(jnp.max(jnp.sqrt(jnp.diag(qz_cov)))):.4e}")

# === HMC with progress bar enabled, reduced JIT footprint ===
# Cutting max_leapfrog_steps from 30 (default) to 10 shrinks the dynamic-leapfrog
# branch fan-out and dramatically speeds the JIT compile. Also shortening burnin.
N_HMC = 20
N_BURN = 100
N_KEEP = 300
MAX_LEAPFROG = 10
print(f"\nHMC: n_hmc={N_HMC}, burnin={N_BURN}, results={N_KEEP}, "
      f"max_leapfrog={MAX_LEAPFROG}, pbar_interval=10")
t0 = time.time()
samples = model_seq.HMC(qz, n_hmc=N_HMC, num_burnin_steps=N_BURN, num_results=N_KEEP,
                       max_leapfrog_steps=MAX_LEAPFROG, seed=0, pbar_interval=10)
elapsed = time.time() - t0
print(f"\nHMC done in {elapsed:.1f} s ({elapsed/60:.2f} min)")

samples_np = np.asarray(samples)
print(f"Samples raw shape: {samples_np.shape}")
np.save(DATA / "hmc_samples.npy", samples_np)

# Flatten (drop chain/device structure) for posterior analysis
flat = samples_np.reshape(-1, samples_np.shape[-1])
print(f"Flat samples shape: {flat.shape}")

# Bijector forward to physical
arr = jnp.asarray(flat)
physical_samples = prob_model.bij.forward(list(arr.T))

mass_main = physical_samples[0][0]
mass_shear = physical_samples[0][1]
combined = {**{k: np.asarray(mass_main[k]) for k in mass_main.keys()},
            **{k: np.asarray(mass_shear[k]) for k in mass_shear.keys()}}

paper = dict(
    theta_E=(2.6463, 0.0017),
    gamma=(1.372, 0.023),
    e1=(0.1091, 0.0020),
    e2=(-0.1320, 0.0020),
    gamma1=(0.0657, 0.0024),
    gamma2=(-0.0939, 0.0022),
)
print("\nPosterior summary (mass params):")
print(f"  {'param':>10s}    {'v3 median':>10s}  ({'+1σ':>+7s}/{'-1σ':>+7s})    {'paper':>15s}    {'Δ_med':>+8s}   #σ_paper")
for k, (mu_p, sig_p) in paper.items():
    arr = combined.get(k)
    if arr is None:
        continue
    med = float(np.median(arr))
    lo = float(np.percentile(arr, 16))
    hi = float(np.percentile(arr, 84))
    print(f"  {k:>10s}    {med:+7.4f}  ({hi-med:+7.4f}/{lo-med:+7.4f})    {mu_p:+7.4f}±{sig_p:.4f}    {med-mu_p:+7.4f}   {(med-mu_p)/sig_p:+6.2f}σ")

# Save physical-space samples for downstream plotting/corner
np.savez(DATA / "hmc_physical_mass.npz", **combined)
print("\nDone.")
