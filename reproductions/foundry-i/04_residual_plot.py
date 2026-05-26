"""Generate data / model / residual / chi figure for the v1 Foundry I MAP fit.

Reads:
  data/cutout_F140W.fits
  data/map_result.npz  (raw unconstrained params + best lp/chi)
Writes:
  figs/map_residual.png
"""
import sys
from pathlib import Path
import gigalens

import jax
import jax.experimental.shard_map  # noqa: F401 — gigalens 2.0 / JAX 0.6.2 compat
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.model import ForwardProbModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

# Load cutout and noise model
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

# Load PSF
kernel = np.load(Path(gigalens.__file__).parent / "assets" / "psf.npy").astype(np.float32)
kernel /= kernel.sum()

# Rebuild model exactly as in 03_fit_map.py
sim_config = SimulatorConfig(delta_pix=0.13, num_pix=128, supersample=2, kernel=kernel)
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False)],
    [sersic.SersicEllipse(use_lstsq=False)],
)
lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
                gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
                e1=tfd.Normal(0.0, 0.1),
                e2=tfd.Normal(0.0, 0.1),
                center_x=tfd.Normal(0.0, 0.05),
                center_y=tfd.Normal(0.0, 0.05),
            )
        ),
        tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.3),
                n_sersic=tfd.Uniform(0.5, 8.0),
                e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
                e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
                center_x=tfd.Normal(0.0, 0.1),
                center_y=tfd.Normal(0.0, 0.1),
                Ie=tfd.LogNormal(jnp.log(5.0), 0.5),
            )
        )
    ]
)
source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                n_sersic=tfd.Uniform(0.5, 6.0),
                e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
                center_x=tfd.Normal(0.0, 0.25),
                center_y=tfd.Normal(0.0, 0.25),
                Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
            )
        )
    ]
)
prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])
prob_model = ForwardProbModel(prior, data_arr, background_rms=background_rms, exp_time=EXP_TIME)

# Load MAP result
res = np.load(DATA / "map_result.npz")
best_params = res["best_params"]
print(f"Best params shape: {best_params.shape}; log_p={res['best_lp']:.2f}, chi2={res['best_chi']:.2f}")

# Bijector forward to physical params
physical = prob_model.bij.forward(list(jnp.asarray(best_params)[:, None]))

# Simulate at best fit
ls = LensSimulator(phys_model, sim_config, bs=1)
sim_best = np.asarray(ls.simulate(physical))
print(f"Forward sim done; range [{sim_best.min():.3f}, {sim_best.max():.3f}]")

# Error map and residual
err_map = np.sqrt(background_rms ** 2 + np.maximum(sim_best, 0) / EXP_TIME)
resid = sim_best - data_arr
chi_map = resid / err_map
red_chi2 = float(np.nanmean(chi_map ** 2))
print(f"Reduced χ² = {red_chi2:.3f}")

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
vmin, vmax = np.percentile(data_arr, [1, 99.5])
extent = [-64 * 0.13, 64 * 0.13, -64 * 0.13, 64 * 0.13]  # arcsec from center

im0 = axes[0].imshow(data_arr, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax, extent=extent)
axes[0].set_title("HST F140W (sky-subtracted)")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(sim_best, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax, extent=extent)
axes[1].set_title("GIGA-Lens v1 MAP model")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(resid, origin="lower", cmap="coolwarm", vmin=-3, vmax=3, extent=extent)
axes[2].set_title("Residual (data − model)")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

im3 = axes[3].imshow(chi_map, origin="lower", cmap="coolwarm", vmin=-5, vmax=5, extent=extent)
axes[3].set_title(f"χ (red. χ² = {red_chi2:.2f})")
plt.colorbar(im3, ax=axes[3], fraction=0.046)

for ax in axes:
    ax.set_xlabel("Δα [arcsec]")
    ax.set_ylabel("Δδ [arcsec]")

fig.suptitle("DESI-165.4754−06.0423 (Foundry I demo) — v1 simplified GIGA-Lens MAP fit", y=1.02)
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
out_png = FIGS / "map_residual.png"
fig.savefig(out_png, dpi=120, bbox_inches="tight")
print(f"Wrote {out_png}")
