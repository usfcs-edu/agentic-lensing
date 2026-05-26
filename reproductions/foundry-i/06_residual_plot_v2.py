"""Generate residual figure for v2 Foundry I MAP fit."""
from pathlib import Path

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

REPRO = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i")
DATA = REPRO / "data"
FIGS = REPRO / "figs"

with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

kernel = np.load("/raid/benson/lensing-repos/gigalens/src/gigalens/assets/psf.npy").astype(np.float32)
kernel /= kernel.sum()

# Reconstruct v2 model
N_MAX = 6
sim_config = SimulatorConfig(delta_pix=0.13, num_pix=128, supersample=2, kernel=kernel)
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=True), sersic.SersicEllipse(use_lstsq=True)],
    [sersic.SersicEllipse(use_lstsq=True), shapelets.Shapelets(n_max=N_MAX, use_lstsq=True, interpolate=False)],
)

def sersic_lstsq_prior(R_med=1.0, R_sig=0.15, n_lo=0.5, n_hi=10.0, c_sig=0.05):
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
            n_sersic=tfd.Uniform(n_lo, n_hi),
            e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            center_x=tfd.Normal(0.0, c_sig),
            center_y=tfd.Normal(0.0, c_sig),
        )
    )

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

res = np.load(DATA / "map_v2_result.npz")
best_params = res["best_params"]
print(f"v2 best params shape={best_params.shape}, log_p={res['best_lp']:.2f}, chi2={res['best_chi']:.4f}")

physical = prob_model.bij.forward(list(jnp.asarray(best_params)[:, None]))
ls = LensSimulator(phys_model, sim_config, bs=1)

# Initial sigma-scaled err map (data-only) for lstsq amplitude solve.
err0 = jnp.sqrt(background_rms ** 2 + jnp.maximum(data_arr, 0) / EXP_TIME)
sim_best, coeffs = ls.lstsq_simulate(physical, data_arr, err0)
sim_best = np.asarray(sim_best)
print(f"lstsq amplitudes shape: {np.asarray(coeffs).shape}")

# Now compute proper chi with sim-based err map
err_map = np.sqrt(background_rms ** 2 + np.maximum(sim_best, 0) / EXP_TIME)
resid = sim_best - data_arr
chi_map = resid / err_map
red_chi2 = float(np.nanmean(chi_map ** 2))
print(f"Forward sim done; range [{sim_best.min():.3f}, {sim_best.max():.3f}]; red. χ² = {red_chi2:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
vmin, vmax = np.percentile(data_arr, [1, 99.5])
extent = [-64 * 0.13, 64 * 0.13, -64 * 0.13, 64 * 0.13]
panels = [
    (data_arr, "HST F140W (sky-subtracted)", dict(cmap="gray_r", vmin=vmin, vmax=vmax)),
    (sim_best, "GIGA-Lens v2 MAP model", dict(cmap="gray_r", vmin=vmin, vmax=vmax)),
    (resid, "Residual (data − model)", dict(cmap="coolwarm", vmin=-3, vmax=3)),
    (chi_map, f"χ  (red. χ² = {red_chi2:.2f})", dict(cmap="coolwarm", vmin=-5, vmax=5)),
]
for ax, (arr, title, kw) in zip(axes, panels):
    im = ax.imshow(arr, origin="lower", extent=extent, **kw)
    ax.set_title(title)
    ax.set_xlabel("Δα [arcsec]"); ax.set_ylabel("Δδ [arcsec]")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("DESI-165.4754−06.0423 — v2 GIGA-Lens MAP (2 lens Sersics + Sersic+Shapelets source, lstsq amps)", y=1.02)
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
out_png = FIGS / "map_v2_residual.png"
fig.savefig(out_png, dpi=120, bbox_inches="tight")
print(f"Wrote {out_png}")
