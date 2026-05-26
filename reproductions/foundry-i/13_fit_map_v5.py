"""v5 GIGA-Lens MAP fit: positive amplitudes + tight constraints.

Changes from v4:
  - All lens-light Sersics use_lstsq=False with LogNormal Ie priors -> POSITIVE
    flux only (no negative-amplitude cancellations).
  - Drop shapelets for the source. Source = single Sersic with LogNormal Ie.
  - Tight prior on main-lens-mass center: N(0, 0.02") (was 0.05").
  - Tight prior on main-lens-light Sersic centers: N(0, 0.02") (forces the two
    main-lens components to effectively share a center).
  - ForwardProbModel (no lstsq).
  - Keep central 1.5-px mask + nearby galaxy components at detected location.
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
from gigalens.jax.model import ForwardProbModel
from gigalens.jax.profiles.light import sersic
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

# Nearby galaxy location from v2 source detection
nb = np.load(DATA / "nearby_galaxy_loc.npz")
NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])
print(f"Nearby galaxy prior center: ({NEAR_X:+.2f}, {NEAR_Y:+.2f}) arcsec")

NUM_PIX = data_arr.shape[0]
DELTA_PIX = 0.13
SUPERSAMPLE = 2
sim_config = SimulatorConfig(delta_pix=DELTA_PIX, num_pix=NUM_PIX, supersample=SUPERSAMPLE, kernel=kernel)

# === Physical model: all use_lstsq=False -> positive Ie sampled ===
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [
        sersic.SersicEllipse(use_lstsq=False),   # main lens compact
        sersic.SersicEllipse(use_lstsq=False),   # main lens extended
        sersic.SersicEllipse(use_lstsq=False),   # nearby galaxy compact
        sersic.SersicEllipse(use_lstsq=False),   # nearby galaxy extended
    ],
    [
        sersic.SersicEllipse(use_lstsq=False),   # source
    ],
)

# === Priors ===
def sersic_prior(R_med, R_sig, Ie_med, Ie_sig, n_lo=0.5, n_hi=8.0,
                 cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig),
        center_y=tfd.Normal(cy_mean, c_sig),
        Ie=tfd.LogNormal(jnp.log(Ie_med), Ie_sig),   # POSITIVE only
    ))

# Mass: tighter center prior 0.02"
lens_mass_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
        gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.02),   # was 0.05 in v4
        center_y=tfd.Normal(0.0, 0.02),
    )),
    tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05),
                                     gamma2=tfd.Normal(0.0, 0.05))),
])

lens_light_prior = tfd.JointDistributionSequential([
    # main lens: two Sersics, both pinned tightly at lens center (0.02" prior)
    sersic_prior(R_med=0.4, R_sig=0.3, Ie_med=5.0, Ie_sig=0.5, c_sig=0.02),
    sersic_prior(R_med=2.0, R_sig=0.3, Ie_med=2.0, Ie_sig=0.5, c_sig=0.02),
    # nearby galaxy: two Sersics at detected location, tighter prior (0.1")
    sersic_prior(R_med=0.3, R_sig=0.3, Ie_med=1.0, Ie_sig=0.5,
                 cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
    sersic_prior(R_med=0.6, R_sig=0.3, Ie_med=0.5, Ie_sig=0.5,
                 cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
])

source_light_prior = tfd.JointDistributionSequential([
    sersic_prior(R_med=0.5, R_sig=0.3, Ie_med=2.0, Ie_sig=0.5, c_sig=0.1),
])

prior = tfd.JointDistributionSequential([lens_mass_prior, lens_light_prior, source_light_prior])

s = prior.sample(1, seed=jax.random.PRNGKey(0))
n_free = sum(jnp.size(v) for v in jax.tree_util.tree_leaves(s))
print(f"Free params: {n_free}")

# === ForwardProbModel + central mask via err inflation ===
yy, xx = np.indices(data_arr.shape)
r_center = np.sqrt((xx - NUM_PIX // 2) ** 2 + (yy - NUM_PIX // 2) ** 2)
keep_mask = r_center > 1.5
n_masked = int(np.sum(~keep_mask))
print(f"Central mask: {n_masked} px")

prob_model = ForwardProbModel(prior, data_arr, background_rms=background_rms, exp_time=EXP_TIME)

# ForwardProbModel computes err inside log_prob from im_sim. We need to mask too.
# Patch log_prob to inflate err at masked pixels.
import functools as _ft
from jax import jit as _jit
from gigalens.jax import simulator as _sim

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

# Monkey-patch
import types
prob_model.log_prob = types.MethodType(masked_log_prob, prob_model)
print("ForwardProbModel built with masked log_prob")

model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# === MAP ===
N_SAMPLES = 20
NUM_STEPS = 400
print(f"\nMAP v5: n_samples={N_SAMPLES}, num_steps={NUM_STEPS}")
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
t0 = time.time()
map_all = model_seq.MAP(opt, n_samples=N_SAMPLES, num_steps=NUM_STEPS, seed=0, output_type="best")
elapsed = time.time() - t0
print(f"MAP done in {elapsed:.1f} s ({elapsed/60:.2f} min)")

best_params, best_lp, best_chi = map_all
best_params_np = np.asarray(best_params).reshape(-1)
best_lp_np = float(np.asarray(best_lp).squeeze())
best_chi_np = float(np.asarray(best_chi).squeeze())
print(f"Best log_p={best_lp_np:.2f}, chi2={best_chi_np:.4f}")
np.savez(DATA / "map_v5_result.npz", best_params=best_params_np, best_lp=best_lp_np, best_chi=best_chi_np)
print(f"Saved {DATA / 'map_v5_result.npz'}")

physical = prob_model.bij.forward(list(best_params_np[:, None]))
mass_main = physical[0][0]
mass_shear = physical[0][1]

# Inspect Sersic Ie's — should all be positive
print("\nLens-light Ie values (should all be > 0):")
for i, comp in enumerate(physical[1]):
    Ie = float(jnp.asarray(comp["Ie"]).squeeze())
    cx = float(jnp.asarray(comp["center_x"]).squeeze())
    cy = float(jnp.asarray(comp["center_y"]).squeeze())
    R = float(jnp.asarray(comp["R_sersic"]).squeeze())
    print(f"  lens_light[{i}]: Ie={Ie:+.3f} (R={R:.2f}\", center=({cx:+.2f},{cy:+.2f}))")
for i, comp in enumerate(physical[2]):
    Ie = float(jnp.asarray(comp["Ie"]).squeeze())
    print(f"  source_light[{i}]: Ie={Ie:+.3f}")

print("\nBest-fit mass:")
for k in mass_main.keys():
    print(f"  {k:>10s} = {float(jnp.asarray(mass_main[k]).squeeze()):+.4f}")
print("  -- shear --")
for k in mass_shear.keys():
    print(f"  {k:>10s} = {float(jnp.asarray(mass_shear[k]).squeeze()):+.4f}")

paper = dict(
    theta_E=(2.6463, 0.0017), gamma=(1.372, 0.023),
    e1=(0.1091, 0.0020), e2=(-0.1320, 0.0020),
    gamma1=(0.0657, 0.0024), gamma2=(-0.0939, 0.0022),
)
fit = {**{k: float(jnp.asarray(mass_main[k]).squeeze()) for k in mass_main.keys()},
       **{k: float(jnp.asarray(mass_shear[k]).squeeze()) for k in mass_shear.keys()}}
print("\nv5 vs Huang 2025a Table 3:")
print(f"  {'param':>10s}    {'v5 fit':>8s}     {'paper':>15s}    {'Δ_abs':>8s}")
for k, (mu, sig) in paper.items():
    fv = fit.get(k, np.nan)
    print(f"  {k:>10s}     {fv:+7.4f}     {mu:+7.4f}±{sig:.4f}    {fv-mu:+7.4f}")
print("\nDone.")
