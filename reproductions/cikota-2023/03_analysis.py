"""03 - Residual + critical-curve figure and derived quantities (sigma_SIE,
magnifications) from the MAP/SVI fit of the Cikota+2023 Einstein cross.

Reads data/map_estimate.npy + data/svi_posterior.npz (from 02), rebuilds the
GIGA-Lens model, and produces:
  figs/03_residual_critcurve.png  : data | model | reduced residual | source plane
                                     with critical curve (image plane) + caustic.
  prints theta_E -> sigma_SIE (SIS relation, paper cosmology + redshifts),
  total magnification, and the posterior table vs the paper.

Run (A16 index 6):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=6 /raid/benson/.venvs/gigalens/bin/python 03_analysis.py
"""
from pathlib import Path

import jax
import jax.experimental.shard_map  # noqa: F401
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).parent / ".jax_cache"))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.model import ForwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import sie, shear

tfd = tfp.distributions
REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

# --------------------------------------------------------------------------- #
# data + model (must match 02 exactly)
# --------------------------------------------------------------------------- #
obs = np.load(DATA / "cikota_g_image.npy").astype(np.float32)
psf = np.load(DATA / "cikota_g_psf.npy").astype(np.float32)
meta = np.load(DATA / "cikota_g_meta.npz")
delta_pix = float(meta["delta_pix"]); num_pix = int(meta["num_pix"])
background_rms = float(meta["background_rms"]); exp_time = float(meta["exp_time"])

# rebuild prior (identical to 02; only used to construct the bijector)
import importlib.util
spec = importlib.util.spec_from_file_location("fit02", REPRO / "02_fit_map_svi.py")
# Instead of importing 02 (which runs the fit), reconstruct the prior inline:
lens_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(theta_E=tfd.LogNormal(jnp.log(2.0), 0.25),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05))),
    tfd.JointDistributionNamed(dict(theta_E=tfd.LogNormal(jnp.log(0.25), 0.3),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(1.836, 0.15), center_y=tfd.Normal(-1.563, 0.15))),
    tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))),
])
lens_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
        n_sersic=tfd.Uniform(1.0, 5.0), e1=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3), center_x=tfd.Normal(0.0, 0.05),
        center_y=tfd.Normal(0.0, 0.05), Ie=tfd.LogNormal(jnp.log(25.0), 0.3))),
    tfd.JointDistributionNamed(dict(R_sersic=tfd.LogNormal(jnp.log(0.5), 0.15),
        n_sersic=tfd.Uniform(1.0, 5.0), center_x=tfd.Normal(1.836, 0.15),
        center_y=tfd.Normal(-1.563, 0.15), Ie=tfd.LogNormal(jnp.log(25.0), 0.3))),
])
source_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
        n_sersic=tfd.Uniform(0.5, 4.0), e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5), center_x=tfd.Normal(0.0, 0.25),
        center_y=tfd.Normal(0.0, 0.25), Ie=tfd.LogNormal(jnp.log(150.0), 0.5))),
])
prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])

phys_model = PhysicalModel(
    lenses=[sie.SIE(), sie.SIE(), shear.Shear()],
    lens_light=[sersic.SersicEllipse(use_lstsq=False), sersic.Sersic(use_lstsq=False)],
    source_light=[sersic.SersicEllipse(use_lstsq=False)],
)
sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=2, kernel=psf)
prob_model = ForwardProbModel(prior, obs, background_rms=background_rms, exp_time=exp_time)

# --------------------------------------------------------------------------- #
# MAP decode -> physical param list-of-list-of-dicts
# --------------------------------------------------------------------------- #
map_est = jnp.array(np.load(DATA / "map_estimate.npy"))  # (1,31)
x = prob_model.bij.forward(list(map_est.T))
def sq(v): return float(np.asarray(v).squeeze())
x_scalar = [[{k: sq(v) for k, v in d.items()} for d in grp] for grp in x]
print("L1 SIE :", x_scalar[0][0])
print("L2 SIE :", x_scalar[0][1])
print("Shear  :", x_scalar[0][2])

# --------------------------------------------------------------------------- #
# simulate model image (bs=1)
# --------------------------------------------------------------------------- #
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
model_img = np.asarray(lens_sim.simulate(x)).squeeze()
err_map = np.sqrt(background_rms ** 2 + np.clip(model_img, 0, None) / exp_time)
resid = (obs - model_img) / err_map
chisq = float(np.mean(resid ** 2))
print(f"reduced chi^2 (per pixel) = {chisq:.3f}")

# --------------------------------------------------------------------------- #
# critical curve / caustic from the magnification map on a fine grid
# --------------------------------------------------------------------------- #
# build a fine coordinate grid in arcsec, centered (gigalens convention: origin at
# image center, x increasing with pixel column).
N = 400
half = num_pix * delta_pix / 2.0
xs = np.linspace(-half, half, N)
X, Y = np.meshgrid(xs, xs)
Xj, Yj = jnp.array(X), jnp.array(Y)

# total deflection from the three mass profiles (2 SIE + shear)
ax_tot = jnp.zeros_like(Xj); ay_tot = jnp.zeros_like(Xj)
for prof, pars in zip(phys_model.lenses, x[0]):
    fx, fy = prof.deriv(Xj, Yj, **pars)
    ax_tot = ax_tot + fx.squeeze(); ay_tot = ay_tot + fy.squeeze()

# magnification = 1/det(A), A = I - d(alpha)/d(theta).  Numerical jacobian.
dx = xs[1] - xs[0]
axx = np.gradient(np.asarray(ax_tot), dx, axis=1)
axy = np.gradient(np.asarray(ax_tot), dx, axis=0)
ayx = np.gradient(np.asarray(ay_tot), dx, axis=1)
ayy = np.gradient(np.asarray(ay_tot), dx, axis=0)
A11 = 1 - axx; A22 = 1 - ayy; A12 = -axy; A21 = -ayx
detA = A11 * A22 - A12 * A21
inv_mag = detA  # 1/mu
# caustic: map critical points (detA=0) back to source plane via beta = theta - alpha
src_x = X - np.asarray(ax_tot)
src_y = Y - np.asarray(ay_tot)

# --------------------------------------------------------------------------- #
# sigma_SIE from theta_E via SIS relation (paper cosmology + redshifts)
# theta_E = 4 pi (sigma/c)^2 D_ls/D_s   ->   sigma = c sqrt(theta_E D_s/(4 pi D_ls))
# paper gives D_s=1690.6 Mpc, D_L1-s=1026.1 Mpc, c=299792.458 km/s
# --------------------------------------------------------------------------- #
post = np.load(DATA / "svi_posterior.npz")
thetaE_samps = post["L1_theta_E"]
c_kms = 299792.458
D_s = 1690.6; D_ls = 1026.1
arcsec2rad = np.pi / 180.0 / 3600.0
def sigma_of_thetaE(tE):
    tE_rad = tE * arcsec2rad
    return c_kms * np.sqrt(tE_rad * D_s / (4 * np.pi * D_ls))
sig = sigma_of_thetaE(thetaE_samps)
tE_med = np.median(thetaE_samps); tE_lo = np.percentile(thetaE_samps,16); tE_hi = np.percentile(thetaE_samps,84)
sig_med = np.median(sig); sig_lo = np.percentile(sig,16); sig_hi = np.percentile(sig,84)
print(f"\ntheta_E = {tE_med:.3f} (+{tE_hi-tE_med:.3f}/-{tE_med-tE_lo:.3f})\"  [paper 2.520]")
print(f"sigma_SIE = {sig_med:.1f} (+{sig_hi-sig_med:.1f}/-{sig_med-sig_lo:.1f}) km/s  [paper 379]")
# sanity: paper's own theta_E=2.520 through our formula
print(f"  check: paper theta_E=2.520 -> sigma = {sigma_of_thetaE(2.520):.1f} km/s (paper quotes 379)")

# --------------------------------------------------------------------------- #
# total magnification of the source (sum |mu| over the four images) — estimate by
# the flux ratio model_source_lensed / source_unlensed.
# --------------------------------------------------------------------------- #
# The simulator unpacks params as [lens, source] when lens_light is empty (see
# gigalens.jax.simulator.simulate).  Build a source-only PhysicalModel and pass
# 2 param groups [lens_params, source_params].  no_deflection=True gives the
# unlensed source on the SAME grid -> flux ratio = total magnification.
src_only_phys = PhysicalModel(lenses=phys_model.lenses, lens_light=[],
                              source_light=phys_model.source_light)
sim_src = LensSimulator(src_only_phys, sim_config, bs=1)
lensed_src = np.asarray(sim_src.simulate([x[0], x[2]])).squeeze()
# Unlensed source: render the same source Sersic on the image grid as an
# (undeflected) lens_light profile -> conserves the intrinsic source flux.
nolens_phys = PhysicalModel(lenses=[], lens_light=phys_model.source_light, source_light=[])
sim_nolens = LensSimulator(nolens_phys, sim_config, bs=1)
unlensed = np.asarray(sim_nolens.simulate([[], [x[2][0]], []])).squeeze()
total_mag = lensed_src.sum() / unlensed.sum()
print(f"total magnification (flux ratio) = {total_mag:.2f}  [paper 10.47]")

# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
extent = [-half, half, -half, half]
vmax = np.percentile(obs, 99.5)

im0 = ax[0].imshow(obs, origin="lower", cmap="cubehelix", vmin=-2*background_rms, vmax=vmax, extent=extent)
ax[0].set_title("DESI Legacy g-band (data)")
plt.colorbar(im0, ax=ax[0], fraction=0.046)

im1 = ax[1].imshow(model_img, origin="lower", cmap="cubehelix", vmin=-2*background_rms, vmax=vmax, extent=extent)
# overlay critical curve (image plane)
ax[1].contour(X, Y, inv_mag, levels=[0.0], colors="cyan", linewidths=1.2)
ax[1].set_title("GIGA-Lens model + critical curve")
plt.colorbar(im1, ax=ax[1], fraction=0.046)

im2 = ax[2].imshow(resid, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4, extent=extent)
ax[2].set_title(f"reduced residual (chi$^2$/px={chisq:.2f})")
plt.colorbar(im2, ax=ax[2], fraction=0.046)

# source plane: unlensed source model + caustic
im3 = ax[3].imshow(unlensed, origin="lower", cmap="cubehelix", extent=extent)
# caustic: plot detA=0 contour mapped to source plane as a scatter of critical points
cs = ax[3].contour(X, Y, inv_mag, levels=[0.0], colors="none")
for seg in cs.allsegs[0]:
    # map each critical point to source plane via beta = theta - alpha (interp)
    from scipy.interpolate import RegularGridInterpolator
    ix = RegularGridInterpolator((xs, xs), src_x.T, bounds_error=False, fill_value=None)
    iy = RegularGridInterpolator((xs, xs), src_y.T, bounds_error=False, fill_value=None)
    bx = ix(seg); by = iy(seg)
    ax[3].plot(bx, by, color="orange", lw=1.0)
ax[3].set_xlim(-3, 3); ax[3].set_ylim(-3, 3)
ax[3].set_title("unlensed source + caustic")
plt.colorbar(im3, ax=ax[3], fraction=0.046)

for a in ax:
    a.set_xlabel("x [arcsec]")
ax[0].set_ylabel("y [arcsec]")
fig.suptitle(f"Cikota+2023 DESI-253.2534+26.8843 — GIGA-Lens on public DESI Legacy g  "
             f"(theta_E={tE_med:.2f}\" vs 2.52, sigma={sig_med:.0f} vs 379 km/s, mu={total_mag:.1f} vs 10.47)")
fig.tight_layout()
fig.savefig(FIGS / "03_residual_critcurve.png", dpi=120, bbox_inches="tight")
print(f"\nWrote {FIGS / '03_residual_critcurve.png'}")

np.savez(DATA / "derived_quantities.npz",
         theta_E_med=tE_med, theta_E_lo=tE_lo, theta_E_hi=tE_hi,
         sigma_SIE_med=sig_med, sigma_SIE_lo=sig_lo, sigma_SIE_hi=sig_hi,
         total_magnification=total_mag, reduced_chisq=chisq,
         D_s=D_s, D_ls=D_ls)
print(f"Wrote {DATA / 'derived_quantities.npz'}")
