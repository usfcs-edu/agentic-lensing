"""Set up the lenstronomy multi-plane mass model for the Carousel Lens and validate it
against the published best-fit (Sheu et al. 2024, Table 2).

Model: two elliptical power-law deflectors (EPL/PEMD), both at z_l=0.49, centered at
La and Ld, + external SHEAR. Five source planes at z_s = 0.962, 0.962, 1.166, 1.432, 1.432.

lenstronomy parameterizes EPL by theta_E defined per source plane. The paper quotes
theta_E w.r.t. z_s=1.432. With multi_plane=True, lenstronomy scales deflections by the
lensing-efficiency ratio beta between planes internally, so a single set of physical
convergence-normalizations applies. We validate: (a) reduced deflection -> theta_E at
z=1.432 recovers 13.03"; (b) projected mass M(<theta_E) ~ 4.78e13 Msun; (c) critical
curves / image multiplicities are sane.

Outputs: data/model_setup.npz (the full kwargs + grids), figs/forward_model.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

# ---------------------------------------------------------------- cosmology / redshifts
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
Z_L = 0.49
Z_SOURCES = [0.962, 0.962, 1.166, 1.432, 1.432]
Z_REF = 1.432  # theta_E quoted w.r.t. this plane

# ---------------------------------------------------------------- published Table 2
# All angular measures "North of East"; PA negative. lenstronomy EPL uses e1,e2 from q,phi.
# Center offsets: paper centers profile a at La, profile b at Ld. We place La at origin and
# Ld at its measured offset. From Fig 1, Ld sits ~south/east of La; we adopt a small offset
# (the exact La-Ld separation is set during the fit below; for the forward check we use the
# published axis ratios/PA and let theta_E define the scale).
from lenstronomy.Util.param_util import phi_q2_ellipticity

# convert PA (deg, North of East) -> lenstronomy phi (rad). lenstronomy phi measured from
# +x axis (East) CCW; "North of East" with North up means phi = PA in standard orientation.
def ell(q, pa_deg):
    phi = np.deg2rad(pa_deg)
    return phi_q2_ellipticity(phi, q)

e1_a, e2_a = ell(0.87, -45.0)
e1_b, e2_b = ell(0.69, -38.0)

# external shear
g_ext, phi_ext = 0.11, np.deg2rad(9.0)
gamma1_ext = g_ext * np.cos(2 * phi_ext)
gamma2_ext = g_ext * np.sin(2 * phi_ext)

# La-Ld separation: from the HST image, Ld is the bright member ~north of La near the
# arc-3 region. We use a representative offset; refined in fit scripts. Units: arcsec.
LD_DX, LD_DY = 6.0, 9.0   # placeholder offset (refined in 05/06)

THETA_E_A_REF = 13.03  # at z_ref=1.432
THETA_E_B_REF = 0.99
GAMMA_A = 1.67
GAMMA_B = 2.12

# ---------------------------------------------------------------- multi-plane LensModel
lens_model_list = ["EPL", "EPL", "SHEAR"]
lens_redshift_list = [Z_L, Z_L, Z_L]

lensModel_ref = LensModel(
    lens_model_list=lens_model_list,
    z_lens=Z_L,
    z_source=Z_REF,
    multi_plane=False,  # single deflector plane (both profiles at z_l) -> efficient
    cosmo=COSMO,
)

kwargs_lens_ref = [
    {"theta_E": THETA_E_A_REF, "gamma": GAMMA_A, "e1": e1_a, "e2": e2_a,
     "center_x": 0.0, "center_y": 0.0},
    {"theta_E": THETA_E_B_REF, "gamma": GAMMA_B, "e1": e1_b, "e2": e2_b,
     "center_x": LD_DX, "center_y": LD_DY},
    {"gamma1": gamma1_ext, "gamma2": gamma2_ext, "ra_0": 0, "dec_0": 0},
]

# ---------------------------------------------------------------- validate theta_E
# theta_E is the radius of the TANGENTIAL critical curve. Find it along the major axis as
# the outer radius where the inverse magnification (1/mu) changes sign. (The convergence
# is singular at r->0, so a pixel-grid mean-kappa is numerically unreliable near center;
# the critical-curve / deflection definition is the robust one.)
grid = np.linspace(-25, 25, 1001)
xx, yy = np.meshgrid(grid, grid)
r = np.hypot(xx, yy)

def tangential_theta_E(lm, kw):
    """Outer tangential critical-curve radius along the x-axis."""
    x = np.linspace(0.05, 24, 4000)
    y = np.zeros_like(x)
    inv_mu = 1.0 / lm.magnification(x, y, kw)
    flips = np.where(np.diff(np.sign(inv_mu)))[0]
    if len(flips) == 0:
        return np.nan
    i0 = flips[-1]  # outer (tangential) crossing
    return float(np.interp(0.0, [inv_mu[i0], inv_mu[i0 + 1]], [x[i0], x[i0 + 1]]))

# primary profile alone (paper quotes per-profile theta_E)
lensModel_a = LensModel(["EPL"], z_lens=Z_L, z_source=Z_REF, cosmo=COSMO)
kwargs_a = [kwargs_lens_ref[0]]
theta_E_a_meas = tangential_theta_E(lensModel_a, kwargs_a)
theta_E_eff = tangential_theta_E(lensModel_ref, kwargs_lens_ref)
print(f"Tangential theta_E, primary EPL alone @ z=1.432: {theta_E_a_meas:.2f}\" "
      f"(paper a = {THETA_E_A_REF}\")")
print(f"Tangential theta_E, full system (2 EPL + shear): {theta_E_eff:.2f}\"")

# ---------------------------------------------------------------- projected mass within theta_E
# For a circular lens the mean convergence within the Einstein radius is exactly 1, so
# M(<theta_E) = Sigma_crit * pi * (D_d * theta_E)^2. This is the standard relation the
# paper uses. We compute it analytically and cross-check via the deflection
# (alpha(theta_E) = theta_E for a circular EPL <=> enclosed mean kappa = 1).
Dd = COSMO.angular_diameter_distance(Z_L)
Ds = COSMO.angular_diameter_distance(Z_REF)
Dds = COSMO.angular_diameter_distance_z1z2(Z_L, Z_REF)
sigma_crit = (const.c ** 2 / (4 * np.pi * const.G) * Ds / (Dd * Dds)).to(u.Msun / u.Mpc ** 2)
arcsec_to_Mpc = (Dd * (1 * u.arcsec).to(u.rad).value).to(u.Mpc)
R_E = (arcsec_to_Mpc * THETA_E_A_REF)
mass_a = (np.pi * sigma_crit * R_E ** 2).to(u.Msun)
# deflection cross-check: mean kappa within theta_E = alpha_circ(theta_E)/theta_E
ax, ay = lensModel_a.alpha(THETA_E_A_REF, 0.0, kwargs_a)
mean_kappa_E = float(np.hypot(ax, ay)) / THETA_E_A_REF
print(f"Sigma_crit = {sigma_crit:.3e}")
print(f"R(theta_E_a) = {R_E.to(u.kpc):.1f}")
print(f"mean kappa within theta_E (deflection) = {mean_kappa_E:.3f}  (expect ~1)")
print(f"M(<theta_E_a={THETA_E_A_REF}\") = {mass_a.value:.3e} Msun "
      f"(paper 4.78e13; ratio {mass_a.value/4.78e13:.3f})")

# ---------------------------------------------------------------- image multiplicity check
solver = LensEquationSolver(lensModel_ref)
# put a source just inside the caustic
beta_x, beta_y = 0.3, 0.2
img_x, img_y = solver.image_position_from_source(
    beta_x, beta_y, kwargs_lens_ref, min_distance=0.05, search_window=40,
    precision_limit=1e-8, num_iter_max=100)
print(f"Test source (0.3,0.2) -> {len(img_x)} images")

# ---------------------------------------------------------------- figures
kappa = lensModel_ref.kappa(xx.ravel(), yy.ravel(), kwargs_lens_ref).reshape(xx.shape)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
im0 = axes[0].imshow(np.log10(kappa + 1e-3), origin="lower",
                     extent=[grid.min(), grid.max(), grid.min(), grid.max()], cmap="viridis")
axes[0].set_title("log10 convergence (2 EPL + shear)")
axes[0].add_patch(plt.Circle((0, 0), THETA_E_A_REF, fill=False, color="r", lw=1.5,
                             label=f"theta_E={THETA_E_A_REF}\""))
axes[0].legend(loc="upper right")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# inverse magnification (critical curves where it = 0)
inv_mu = 1.0 / lensModel_ref.magnification(xx.ravel(), yy.ravel(), kwargs_lens_ref).reshape(xx.shape)
axes[1].imshow(np.sign(inv_mu), origin="lower",
               extent=[grid.min(), grid.max(), grid.min(), grid.max()], cmap="coolwarm")
axes[1].contour(xx, yy, inv_mu, levels=[0], colors="k", linewidths=1.5)
axes[1].add_patch(plt.Circle((0, 0), theta_E_a_meas, fill=False, color="lime", lw=1.2, ls="--"))
axes[1].set_title(f"critical curves; primary theta_E={theta_E_a_meas:.1f}\"")

# magnification map
mag = lensModel_ref.magnification(xx.ravel(), yy.ravel(), kwargs_lens_ref).reshape(xx.shape)
im2 = axes[2].imshow(np.clip(mag, -30, 30), origin="lower",
                     extent=[grid.min(), grid.max(), grid.min(), grid.max()], cmap="RdBu_r")
axes[2].set_title("magnification (z=1.432)")
axes[2].scatter(img_x, img_y, c="lime", s=30, marker="x")
plt.colorbar(im2, ax=axes[2], fraction=0.046)
fig.tight_layout()
fig.savefig(FIGS / "forward_model.png", dpi=120, bbox_inches="tight")
print(f"Wrote {FIGS / 'forward_model.png'}")

np.savez(DATA / "model_setup.npz",
         lens_model_list=lens_model_list,
         z_l=Z_L, z_sources=Z_SOURCES, z_ref=Z_REF,
         theta_E_a=THETA_E_A_REF, theta_E_b=THETA_E_B_REF,
         gamma_a=GAMMA_A, gamma_b=GAMMA_B,
         e1_a=e1_a, e2_a=e2_a, e1_b=e1_b, e2_b=e2_b,
         gamma1_ext=gamma1_ext, gamma2_ext=gamma2_ext,
         ld_dx=LD_DX, ld_dy=LD_DY,
         theta_E_eff=theta_E_eff, theta_E_a_meas=theta_E_a_meas,
         sigma_crit=sigma_crit.value, arcsec_to_Mpc=arcsec_to_Mpc.value,
         mass_a=mass_a.to(u.Msun).value)
print(f"Saved data/model_setup.npz")
