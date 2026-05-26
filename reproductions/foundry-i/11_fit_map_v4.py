"""v4 GIGA-Lens MAP fit for Foundry I demo system.

Adds to v2:
  - Two Sersic light components for the nearby galaxy (detected in v2 residual at
    (-2.34", -2.86") offset from main lens).
  - Central 1.5-px circular mask at main lens (mimics paper's 2.5 px @ 0.065")
    to avoid the EPL inner-critical-curve singularity (γ<2 case).

Now 41 free non-linear params: 8 (mass) + 6×4 (lens light) + 6 (source Sersic) + 3
(source shapelets) — matches Huang 2025a Table 2.
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
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions

REPRO = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i")
DATA = REPRO / "data"

# === Data ===
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

kernel = np.load("/raid/benson/lensing-repos/gigalens/src/gigalens/assets/psf.npy").astype(np.float32)
kernel /= kernel.sum()

# === Nearby galaxy location (from v2 residual source detection) ===
nb = np.load(DATA / "nearby_galaxy_loc.npz")
NEAR_X = float(nb["arcsec_x"])  # arcsec offset from main lens center
NEAR_Y = float(nb["arcsec_y"])
print(f"Nearby galaxy prior center: ({NEAR_X:+.2f}, {NEAR_Y:+.2f}) arcsec")

# === Central mask: radius 1.5 px on main lens (~0.2") ===
NUM_PIX = data_arr.shape[0]
DELTA_PIX = 0.13
SUPERSAMPLE = 2
yy, xx = np.indices(data_arr.shape)
r_center = np.sqrt((xx - NUM_PIX // 2) ** 2 + (yy - NUM_PIX // 2) ** 2)
keep_mask = r_center > 1.5
n_masked = int(np.sum(~keep_mask))
print(f"Central mask: r < 1.5 px → masking {n_masked} pixels at lens center")

# === Simulator ===
sim_config = SimulatorConfig(delta_pix=DELTA_PIX, num_pix=NUM_PIX, supersample=SUPERSAMPLE, kernel=kernel)

# === Physical model: 4 lens-light Sersics + (Sersic + Shapelets) source ===
N_MAX = 6
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [
        sersic.SersicEllipse(use_lstsq=True),   # main lens, compact
        sersic.SersicEllipse(use_lstsq=True),   # main lens, extended
        sersic.SersicEllipse(use_lstsq=True),   # nearby galaxy, compact
        sersic.SersicEllipse(use_lstsq=True),   # nearby galaxy, extended
    ],
    [
        sersic.SersicEllipse(use_lstsq=True),
        shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False),
    ],
)

# === Priors ===
def sersic_lstsq_prior(R_med, R_sig, n_lo=0.5, n_hi=10.0,
                       cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig),
        center_y=tfd.Normal(cy_mean, c_sig),
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
    sersic_lstsq_prior(R_med=0.5, R_sig=0.3, c_sig=0.10),                            # main 1
    sersic_lstsq_prior(R_med=2.0, R_sig=0.3, c_sig=0.10),                            # main 2
    sersic_lstsq_prior(R_med=0.3, R_sig=0.3, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.3),  # nearby 1
    sersic_lstsq_prior(R_med=0.6, R_sig=0.3, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.3),  # nearby 2
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

# === Build prob_model and apply mask via err_map inflation ===
prob_model = BackwardProbModel(prior, data_arr, background_rms=background_rms, exp_time=EXP_TIME)
err_masked = jnp.where(keep_mask, prob_model.err_map, jnp.float32(1e10))
prob_model.err_map = err_masked
prob_model.observed_dist = tfd.Independent(
    tfd.Normal(prob_model.observed_image, err_masked), reinterpreted_batch_ndims=2
)
print(f"BackwardProbModel built; err_map masked at {n_masked} pixels")

model_seq = ModellingSequence(phys_model, prob_model, sim_config)

s = prior.sample(1, seed=jax.random.PRNGKey(0))
n_free = sum(jnp.size(v) for v in jax.tree_util.tree_leaves(s))
print(f"Free non-linear params: {n_free} (target: 41)")

# === MAP ===
N_SAMPLES = 20
NUM_STEPS = 350
print(f"\nMAP v4: n_samples={N_SAMPLES}, num_steps={NUM_STEPS}")
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
np.savez(DATA / "map_v4_result.npz", best_params=best_params_np, best_lp=best_lp_np, best_chi=best_chi_np)
print(f"Saved {DATA / 'map_v4_result.npz'}")

physical = prob_model.bij.forward(list(best_params_np[:, None]))
mass_main = physical[0][0]
mass_shear = physical[0][1]
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
print("\nv4 vs Huang 2025a Table 3:")
print(f"  {'param':>10s}    {'v4 fit':>8s}     {'paper':>15s}    {'Δ_abs':>8s}")
for k, (mu, sig) in paper.items():
    fv = fit.get(k, np.nan)
    print(f"  {k:>10s}     {fv:+7.4f}     {mu:+7.4f}±{sig:.4f}    {fv-mu:+7.4f}")
print("\nDone.")
