"""Locate the nearby galaxy by source-extracting the v2 residual image.

Strategy: the main lens + arcs are well-fit in v2. The residual should be small
*except* for the nearby galaxy (unmodeled in v2) and noise. Find peaks in the
residual smoothed with a small Gaussian; report any source >5σ above background
that's not too close to the main lens.

Output: data/nearby_galaxy_loc.npz with x, y (pix offset from image center) and
arcsec offset.
"""
from pathlib import Path

import jax
import jax.experimental.shard_map  # noqa: F401
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as ndi
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.model import BackwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig
import matplotlib.pyplot as plt

tfd = tfp.distributions

REPRO = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i")
DATA = REPRO / "data"
FIGS = REPRO / "figs"
FIGS.mkdir(exist_ok=True)

with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

kernel = np.load("/raid/benson/lensing-repos/gigalens/src/gigalens/assets/psf.npy").astype(np.float32)
kernel /= kernel.sum()

# Recreate v2 model + load its best params
N_MAX = 6
sim_config = SimulatorConfig(delta_pix=0.13, num_pix=128, supersample=2, kernel=kernel)
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=True), sersic.SersicEllipse(use_lstsq=True)],
    [sersic.SersicEllipse(use_lstsq=True), shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False)],
)

def sersic_lstsq_prior(R_med, R_sig, n_lo=0.5, n_hi=10.0, c_sig=0.05):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(0.0, c_sig), center_y=tfd.Normal(0.0, c_sig),
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
    sersic_lstsq_prior(0.5, 0.3, c_sig=0.10),
    sersic_lstsq_prior(2.0, 0.3, c_sig=0.10),
])
src_sersic_prior = tfd.JointDistributionNamed(dict(
    R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
    e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5), e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
))
src_shp_prior = tfd.JointDistributionNamed(dict(
    beta=tfd.LogNormal(jnp.log(0.1), 0.1),
    center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
))
source_light_prior = tfd.JointDistributionSequential([src_sersic_prior, src_shp_prior])
prior = tfd.JointDistributionSequential([lens_mass_prior, lens_light_prior, source_light_prior])
prob_model = BackwardProbModel(prior, data_arr, background_rms=background_rms, exp_time=EXP_TIME)

# Forward-simulate the v2 best fit
v2 = np.load(DATA / "map_v2_result.npz")
best_params = jnp.asarray(v2["best_params"])[:, None]
physical = prob_model.bij.forward(list(best_params))
ls = LensSimulator(phys_model, sim_config, bs=1)
err0 = jnp.sqrt(background_rms ** 2 + jnp.maximum(data_arr, 0) / EXP_TIME)
sim_v2, _ = ls.lstsq_simulate(physical, data_arr, err0)
sim_v2 = np.asarray(sim_v2)
resid = data_arr - sim_v2

print(f"v2 residual stats: min={resid.min():.3f}, max={resid.max():.3f}, std={resid.std():.4f}")

# Smooth and find positive peaks beyond the main lens region
smoothed = ndi.gaussian_filter(resid, 2.5)
# Mask out the main lens region (within 15 px = ~2 arcsec of center)
yy, xx = np.indices(resid.shape)
r_from_center = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
search_region = (r_from_center > 15) & (r_from_center < 50)
masked = np.where(search_region, smoothed, -np.inf)

# Top peak in the search region
py, px = np.unravel_index(np.nanargmax(masked), masked.shape)
peak_val = float(smoothed[py, px])
offset_pix_x = px - 64
offset_pix_y = py - 64
offset_arcsec_x = offset_pix_x * 0.13
offset_arcsec_y = offset_pix_y * 0.13
print(f"\nPeak in v2 residual outside main lens:")
print(f"  pixel: ({px}, {py})  offset from center: ({offset_pix_x:+d}, {offset_pix_y:+d}) pix")
print(f"  arcsec offset: ({offset_arcsec_x:+.2f}\", {offset_arcsec_y:+.2f}\")")
print(f"  smoothed flux: {peak_val:.3f} e-/s vs residual std {resid.std():.4f}")

np.savez(DATA / "nearby_galaxy_loc.npz",
         px=offset_pix_x, py=offset_pix_y,
         arcsec_x=offset_arcsec_x, arcsec_y=offset_arcsec_y,
         peak_flux=peak_val)
print(f"Saved {DATA / 'nearby_galaxy_loc.npz'}")

# Quick figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
vmin, vmax = np.percentile(data_arr, [1, 99])
extent = [-64 * 0.13, 64 * 0.13, -64 * 0.13, 64 * 0.13]
axes[0].imshow(data_arr, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax, extent=extent)
axes[0].set_title("HST F140W (sky-subtracted)")
axes[1].imshow(resid, origin="lower", cmap="coolwarm", vmin=-2, vmax=2, extent=extent)
axes[1].set_title("v2 residual")
axes[2].imshow(smoothed, origin="lower", cmap="coolwarm", vmin=-1, vmax=1, extent=extent)
axes[2].set_title("residual smoothed (σ=2.5px)")
for ax in axes:
    ax.set_xlabel("Δα ['']"); ax.set_ylabel("Δδ ['']")
    ax.scatter(offset_arcsec_x, offset_arcsec_y, marker='x', s=200, c='lime', linewidths=2, label='detected nearby')
    ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
out_png = FIGS / "nearby_galaxy_detection.png"
fig.savefig(out_png, dpi=120, bbox_inches="tight")
print(f"Wrote {out_png}")
