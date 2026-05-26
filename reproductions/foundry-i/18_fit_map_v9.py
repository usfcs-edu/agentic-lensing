"""v9: same v6 model + priors, but PSF = empirical (from 17_build_empirical_psf.py).

If the PSF mismatch was responsible for our γ_EPL being too high (2.2 instead of
paper's 1.37), v9 should find γ converging to paper-like values without needing
multi-start hunts.
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
REPRO = Path(__file__).parent
DATA = REPRO / "data"

# ----- data -----
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

# ===== Empirical PSF (the v9 change) =====
kernel = np.load(DATA / "empirical_psf.npy").astype(np.float32)
print(f"PSF: shape={kernel.shape} (empirical, supersampled to 0.065\")")
assert abs(kernel.sum() - 1.0) < 1e-4

nb = np.load(DATA / "nearby_galaxy_loc.npz")
NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])

NUM_PIX = data_arr.shape[0]
DELTA_PIX = 0.13
SUPERSAMPLE = 2
sim_config = SimulatorConfig(delta_pix=DELTA_PIX, num_pix=NUM_PIX, supersample=SUPERSAMPLE, kernel=kernel)
N_MAX = 6
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False), sersic.SersicEllipse(use_lstsq=False),
     sersic.SersicEllipse(use_lstsq=False), sersic.SersicEllipse(use_lstsq=False)],
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

# ----- 200-chain MAP -----
N_SAMPLES = 200
NUM_STEPS = 400
print(f"\nMAP v9 (empirical PSF): n_samples={N_SAMPLES}, num_steps={NUM_STEPS}")
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
t0 = time.time()
all_params, all_lp, all_chi = model_seq.MAP(opt, n_samples=N_SAMPLES, num_steps=NUM_STEPS,
                                             seed=0, output_type="all")
elapsed = time.time() - t0
print(f"MAP done in {elapsed:.1f} s ({elapsed/60:.2f} min)")

final_params = np.asarray(all_params[:, -1, :])
final_lp = np.asarray(all_lp[:, -1])
final_chi = np.asarray(all_chi[:, -1])
print(f"log_p stats: min={final_lp.min():.1f}, median={np.median(final_lp):.1f}, max={final_lp.max():.1f}")
print(f"chi2 stats: min={final_chi.min():.3f}, median={np.median(final_chi):.3f}, max={final_chi.max():.3f}")

arr = jnp.asarray(final_params)
physical = prob_model.bij.forward(list(arr.T))
mass_main = physical[0][0]
mass_shear = physical[0][1]
theta_E = np.asarray(mass_main["theta_E"])
gamma_v = np.asarray(mass_main["gamma"])
e1 = np.asarray(mass_main["e1"])
e2 = np.asarray(mass_main["e2"])
g1 = np.asarray(mass_shear["gamma1"])
g2 = np.asarray(mass_shear["gamma2"])

order = np.argsort(-final_lp)
print(f"\nTop 20 chains by log_p:")
print(f"  {'rk':>3s} {'log_p':>10s} {'chi2':>8s} {'theta_E':>8s} {'gamma':>7s} "
      f"{'e1':>7s} {'e2':>7s} {'g_ext1':>7s} {'g_ext2':>7s}")
for rk, i in enumerate(order[:20]):
    print(f"  {rk+1:>3d} {final_lp[i]:>10.1f} {final_chi[i]:>8.3f} "
          f"{theta_E[i]:>+8.4f} {gamma_v[i]:>+7.3f} {e1[i]:>+7.4f} {e2[i]:>+7.4f} "
          f"{g1[i]:>+7.4f} {g2[i]:>+7.4f}")

paper_mode_mask = (e1 > 0) & (e2 < 0) & (g1 > 0) & (g2 < 0)
n_paper_mode = int(np.sum(paper_mode_mask))
print(f"\n{n_paper_mode}/{N_SAMPLES} chains are 'paper-mode' (e1>0, e2<0, g1>0, g2<0)")

# How many chains have gamma in paper's region (γ < 1.7)?
n_low_gamma = int(np.sum(gamma_v < 1.7))
print(f"{n_low_gamma}/{N_SAMPLES} chains have gamma < 1.7 (paper has gamma=1.37)")
print(f"  gamma stats: min={gamma_v.min():.3f}, median={np.median(gamma_v):.3f}, max={gamma_v.max():.3f}")

# Save all
np.savez(DATA / "map_v9_all_chains.npz", final_params=final_params,
         final_lp=final_lp, final_chi=final_chi)

# Identify the best paper-mode chain
if n_paper_mode > 0:
    pm_idx = np.where(paper_mode_mask)[0]
    pm_best = pm_idx[np.argmax(final_lp[pm_idx])]
    print(f"\nBest paper-mode chain (v9, empirical PSF): idx={pm_best}, "
          f"log_p={final_lp[pm_best]:.1f}, chi2={final_chi[pm_best]:.3f}")
    print(f"  theta_E={theta_E[pm_best]:+.4f} (paper +2.6463)")
    print(f"  gamma  ={gamma_v[pm_best]:+.4f} (paper +1.372)")
    print(f"  e1, e2 = {e1[pm_best]:+.4f}, {e2[pm_best]:+.4f} (paper +0.109, -0.132)")
    print(f"  g_ext  = {g1[pm_best]:+.4f}, {g2[pm_best]:+.4f} (paper +0.066, -0.094)")
    np.savez(DATA / "map_v9_paper_mode.npz",
             best_params=final_params[pm_best],
             best_lp=final_lp[pm_best], best_chi=final_chi[pm_best])

# Headline: best overall
best = order[0]
print(f"\nGLOBAL BEST chain (v9, empirical PSF): idx={best}, "
      f"log_p={final_lp[best]:.1f}, chi2={final_chi[best]:.3f}")
print(f"  theta_E={theta_E[best]:+.4f} (paper +2.6463)")
print(f"  gamma  ={gamma_v[best]:+.4f} (paper +1.372)")
print(f"  e1, e2 = {e1[best]:+.4f}, {e2[best]:+.4f} (paper +0.109, -0.132)")
print(f"  g_ext  = {g1[best]:+.4f}, {g2[best]:+.4f} (paper +0.066, -0.094)")
np.savez(DATA / "map_v9_best.npz",
         best_params=final_params[best], best_lp=final_lp[best], best_chi=final_chi[best])

print("\nDone.")
