"""v10: SVI from v9 best paper-mode chain, with empirical PSF.

Combines v8 (SVI from paper-mode start) with v9 (empirical PSF). Should produce
the cleanest reproduction posterior so far.
"""
import time
from pathlib import Path

import jax
import jax.experimental.shard_map  # noqa: F401
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions
REPRO = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i")
DATA = REPRO / "data"

with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

# ===== Empirical PSF =====
kernel = np.load(DATA / "empirical_psf.npy").astype(np.float32)
print(f"PSF: shape={kernel.shape} (empirical)")

nb = np.load(DATA / "nearby_galaxy_loc.npz")
NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])

NUM_PIX = data_arr.shape[0]
sim_config = SimulatorConfig(delta_pix=0.13, num_pix=NUM_PIX, supersample=2, kernel=kernel)
N_MAX = 6
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False)] * 4,
    [sersic.SersicEllipse(use_lstsq=False),
     shapelets.Shapelets(n_max=N_MAX, use_lstsq=False, interpolate=False)],
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
    tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))),
])
lens_light_prior = tfd.JointDistributionSequential([
    sersic_prior(0.4, 0.3, 5.0, 0.5, c_sig=0.02),
    sersic_prior(2.0, 0.3, 2.0, 0.5, c_sig=0.02),
    sersic_prior(0.3, 0.3, 1.0, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
    sersic_prior(0.6, 0.3, 0.5, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
])
src_sersic_prior = tfd.JointDistributionNamed(dict(
    R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
    e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
    e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
    Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
))
shp_amp_names = shapelets.Shapelets(n_max=N_MAX)._amp_names
amp_priors = {name: tfd.Normal(0.0, 5.0 / float(jnp.sqrt(i + 1)))
              for i, name in enumerate(shp_amp_names)}
src_shp_prior = tfd.JointDistributionNamed(dict(
    beta=tfd.LogNormal(jnp.log(0.1), 0.1),
    center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
    **amp_priors,
))
source_light_prior = tfd.JointDistributionSequential([src_sersic_prior, src_shp_prior])
prior = tfd.JointDistributionSequential([lens_mass_prior, lens_light_prior, source_light_prior])

yy, xx = np.indices(data_arr.shape)
r_center = np.sqrt((xx - NUM_PIX // 2) ** 2 + (yy - NUM_PIX // 2) ** 2)
keep_mask = r_center > 1.5
prob_model = ForwardProbModel(prior, data_arr, background_rms=background_rms, exp_time=EXP_TIME)

import functools as _ft, types as _types
from jax import jit as _jit
INF_ERR = jnp.float32(1e10)
mask_jax = jnp.asarray(keep_mask)

@_ft.partial(_jit, static_argnums=(0, 1))
def masked_log_prob(self, simulator, z):
    z = list(z.T)
    x = self.bij.forward(z)
    im_sim = simulator.simulate(x)
    err_map = jnp.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
    err_map = jnp.where(mask_jax, err_map, INF_ERR)
    log_like = tfd.Independent(
        tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
    ).log_prob(self.observed_image)
    log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
    chisq = jnp.mean(jnp.where(mask_jax, ((im_sim - self.observed_image) / err_map) ** 2, 0.0),
                     axis=(-2, -1))
    return log_like + log_prior, chisq

prob_model.log_prob = _types.MethodType(masked_log_prob, prob_model)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)
print("Model: 74 params, empirical PSF, central mask")

# ----- Load v9 paper-mode best as SVI starting point -----
pm = np.load(DATA / "map_v9_paper_mode.npz")
start = jnp.asarray(pm["best_params"])[None, :]
print(f"\nv9 paper-mode start: log_p={float(pm['best_lp']):.1f}, chi2={float(pm['best_chi']):.3f}")

# ----- SVI -----
N_VI = 200
SVI_STEPS = 500
LR = 1e-4
INIT_SCALES = 1e-4
print(f"\nSVI v10: n_vi={N_VI}, num_steps={SVI_STEPS}, lr={LR}, init_scales={INIT_SCALES}")
opt = optax.adabelief(LR, b1=0.95, b2=0.99)
t0 = time.time()
qz, loss_hist = model_seq.SVI(start, opt, n_vi=N_VI, num_steps=SVI_STEPS,
                              init_scales=INIT_SCALES, seed=0)
elapsed = time.time() - t0
print(f"SVI done in {elapsed:.1f} s ({elapsed/60:.2f} min)")
print(f"-ELBO trajectory: start={float(loss_hist[0]):.1f} -> end={float(loss_hist[-1]):.1f}")

qz_mean = np.asarray(qz.mean())
qz_cov = np.asarray(qz.covariance())
np.savez(DATA / "svi_v10_paper_mode_empirical.npz",
         mean=qz_mean, cov=qz_cov, loss_hist=np.asarray(loss_hist))

# ----- 10k posterior draws -----
N_SAMP = 10000
print(f"\nSampling {N_SAMP} from qz...")
key = jax.random.PRNGKey(42)
z_samples = qz.sample(N_SAMP, seed=key)
physical_samples = prob_model.bij.forward(list(jnp.asarray(z_samples).T))
mass_main = physical_samples[0][0]
mass_shear = physical_samples[0][1]
combined = {**{k: np.asarray(mass_main[k]) for k in mass_main.keys()},
            **{k: np.asarray(mass_shear[k]) for k in mass_shear.keys()}}
np.savez(DATA / "svi_v10_posterior_mass.npz", **combined)

paper = dict(
    theta_E=(2.6463, 0.0017), gamma=(1.372, 0.023),
    e1=(0.1091, 0.0020), e2=(-0.1320, 0.0020),
    gamma1=(0.0657, 0.0024), gamma2=(-0.0939, 0.0022),
)
print("\nv10 SVI posterior (paper-mode + empirical PSF, 10k samples):")
print(f"  {'param':>10s}    {'v10 median':>10s}  ({'+1σ':>7s}/{'-1σ':>7s})    "
      f"{'paper (HMC)':>17s}    {'Δ_med':>8s}    % off    sign?")
for k, (mu_p, sig_p) in paper.items():
    arr = combined.get(k)
    if arr is None: continue
    med = float(np.median(arr))
    lo = float(np.percentile(arr, 16))
    hi = float(np.percentile(arr, 84))
    pct = 100 * abs(med - mu_p) / abs(mu_p) if mu_p != 0 else 0.0
    sign_ok = "✓" if (med * mu_p > 0 or abs(med) < 1e-4 or abs(mu_p) < 1e-4) else "✗"
    print(f"  {k:>10s}    {med:+7.4f}  ({hi-med:+7.4f}/{lo-med:+7.4f})    "
          f"{mu_p:+7.4f}±{sig_p:.4f}    {med-mu_p:+7.4f}    {pct:5.1f}%    {sign_ok}")
print("\nDone.")
