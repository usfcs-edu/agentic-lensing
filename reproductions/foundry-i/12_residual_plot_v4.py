"""v4 residual figure (4 panels: data / model / residual / chi)."""
from pathlib import Path
import gigalens

import jax
import jax.experimental.shard_map  # noqa: F401
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.model import BackwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7
kernel = np.load(Path(gigalens.__file__).parent / "assets" / "psf.npy").astype(np.float32)
kernel /= kernel.sum()

nb = np.load(DATA / "nearby_galaxy_loc.npz")
NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])

N_MAX = 6
sim_config = SimulatorConfig(delta_pix=0.13, num_pix=128, supersample=2, kernel=kernel)
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=True), sersic.SersicEllipse(use_lstsq=True),
     sersic.SersicEllipse(use_lstsq=True), sersic.SersicEllipse(use_lstsq=True)],
    [sersic.SersicEllipse(use_lstsq=True),
     shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False)],
)

def sersic_lstsq_prior(R_med, R_sig, n_lo=0.5, n_hi=10.0, cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
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
    sersic_lstsq_prior(0.3, 0.3, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.3),
    sersic_lstsq_prior(0.6, 0.3, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.3),
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

# Reapply central mask (same as v4 fit)
yy, xx = np.indices(data_arr.shape)
r_center = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
keep_mask = r_center > 1.5
err_masked = jnp.where(keep_mask, prob_model.err_map, jnp.float32(1e10))

res = np.load(DATA / "map_v4_result.npz")
best_params = jnp.asarray(res["best_params"])
print(f"v4 best params shape {best_params.shape}, log_p={res['best_lp']:.2f}, chi2={res['best_chi']:.4f}")

physical = prob_model.bij.forward(list(best_params[:, None]))
ls = LensSimulator(phys_model, sim_config, bs=1)
sim_best, coeffs = ls.lstsq_simulate(physical, data_arr, err_masked)
sim_best = np.asarray(sim_best)
coeffs_np = np.asarray(coeffs)
print(f"lstsq coeffs: {coeffs_np.shape}, "
      f"min={coeffs_np.min():.2f}, max={coeffs_np.max():.2f}, "
      f"#negative={int(np.sum(coeffs_np < 0))}/{coeffs_np.size}")

err_map = np.sqrt(background_rms ** 2 + np.maximum(sim_best, 0) / EXP_TIME)
# In residual stats, exclude masked pixels
resid = sim_best - data_arr
chi_map = np.where(keep_mask, resid / err_map, 0.0)
red_chi2 = float(np.mean(chi_map[keep_mask] ** 2))
print(f"Reduced χ² (un-masked pixels only): {red_chi2:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
vmin, vmax = np.percentile(data_arr, [1, 99.5])
extent = [-64 * 0.13, 64 * 0.13, -64 * 0.13, 64 * 0.13]
panels = [
    (data_arr, "HST F140W (sky-subtracted)", dict(cmap="gray_r", vmin=vmin, vmax=vmax)),
    (sim_best, "GIGA-Lens v4 MAP (41 params)", dict(cmap="gray_r", vmin=vmin, vmax=vmax)),
    (resid, "Residual (data − model)", dict(cmap="coolwarm", vmin=-2, vmax=2)),
    (chi_map, f"χ (red. χ² = {red_chi2:.2f})", dict(cmap="coolwarm", vmin=-5, vmax=5)),
]
for ax, (arr, title, kw) in zip(axes, panels):
    im = ax.imshow(arr, origin="lower", extent=extent, **kw)
    ax.set_title(title)
    ax.set_xlabel("Δα [arcsec]"); ax.set_ylabel("Δδ [arcsec]")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # mark nearby galaxy location
    ax.plot(NEAR_X, NEAR_Y, marker='o', mfc='none', mec='lime', ms=15, mew=1.5)
    # mark central mask
    ax.add_patch(plt.Circle((0, 0), 1.5 * 0.13, fill=False, color='cyan', lw=1.5))

fig.suptitle("DESI-165.4754−06.0423 v4 (4 lens-light Sersics + Sersic+Shapelets src, central mask, 41 params)", y=1.02)
fig.tight_layout()
out_png = FIGS / "map_v4_residual.png"
fig.savefig(out_png, dpi=120, bbox_inches="tight")
print(f"Wrote {out_png}")
