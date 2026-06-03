"""Predict the multi-image configuration of the 5 confirmed sources with the published
best-fit model, and overlay on the HST F140W data to confirm the model reproduces the
observed arcs. Also locate La and Ld in the image and set the La-Ld offset used by the fit.

For each source plane we solve the lens equation for a source position chosen to land the
images on the observed arcs, and report image multiplicities (double / naked-cusp-triple /
Einstein-cross-quad / fold-quad) matching the paper's description per family.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import median_filter, maximum_filter

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

setup = np.load(DATA / "model_setup.npz", allow_pickle=True)
meta = np.load(DATA / "cutout_meta.npz")
data = meta["data"]
scale = float(meta["scale"])
median_bg = float(meta["median_bg"])
std_bg = float(meta["std_bg"])
ny, nx = data.shape

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
Z_L = float(setup["z_l"])
Z_REF = float(setup["z_ref"])

# ---------------------------------------------------------------- locate La and Ld in image
# La = central BCG (brightest pixel near center). Ld = second-brightest member that, per the
# paper, perturbs arc family 3 (north of the lens). Locate via local maxima on the data.
mfull = median_filter(np.nan_to_num(data, nan=median_bg), size=35)
sm = np.nan_to_num(data, nan=median_bg)
# La: brightest pixel within 2"
yy, xx = np.mgrid[0:ny, 0:nx]
c0x, c0y = nx / 2.0, ny / 2.0
r0 = np.hypot(xx - c0x, yy - c0y) * scale
laMask = r0 < 2.0
iy, ix = np.unravel_index(np.argmax(np.where(laMask, sm, -np.inf)), sm.shape)
la_x, la_y = float(ix), float(iy)
print(f"La pixel ({la_x:.1f},{la_y:.1f})")

# Detect bright members: local maxima of lightly-smoothed image, brightness-ranked.
sm_light = median_filter(sm, size=3)
peaks = (maximum_filter(sm_light, size=7) == sm_light) & (sm_light > median_bg + 8 * std_bg)
py, px = np.where(peaks)
pr = np.hypot(px - la_x, py - la_y) * scale
sel = (pr > 2.5) & (pr < 14.0)
py, px, pr = py[sel], px[sel], pr[sel]
order = np.argsort(sm_light[py, px])[::-1]
py, px = py[order], px[order]
pr = pr[order]
print(f"Brightest members within 2.5-14\" of La ({len(px)} found; E,N offset from La):")
for i in range(min(8, len(px))):
    print(f"  ({(px[i]-la_x)*scale:+.1f}, {(py[i]-la_y)*scale:+.1f})  r={pr[i]:.1f}\"")

# Adopt Ld as the brightest member in the N quadrant (perturbs arc 3, north of lens).
ld_candidates = [(px[i], py[i]) for i in range(len(px))
                 if (py[i] - la_y) > 0]  # north of La
if ld_candidates:
    ld_x, ld_y = ld_candidates[0]
elif len(px):
    ld_x, ld_y = px[0], py[0]
else:
    ld_x, ld_y = la_x + 47, la_y + 70  # fallback ~6,9 arcsec
LD_DX = (ld_x - la_x) * scale
LD_DY = (ld_y - la_y) * scale
print(f"Adopted Ld at offset ({LD_DX:+.1f}, {LD_DY:+.1f})\" from La")

# ---------------------------------------------------------------- build per-plane lens models
def kwargs_lens(theta_E_a=13.03, gamma_a=1.67, theta_E_b=0.99, gamma_b=2.12,
                ld_dx=LD_DX, ld_dy=LD_DY):
    return [
        {"theta_E": theta_E_a, "gamma": gamma_a,
         "e1": float(setup["e1_a"]), "e2": float(setup["e2_a"]),
         "center_x": 0.0, "center_y": 0.0},
        {"theta_E": theta_E_b, "gamma": gamma_b,
         "e1": float(setup["e1_b"]), "e2": float(setup["e2_b"]),
         "center_x": ld_dx, "center_y": ld_dy},
        {"gamma1": float(setup["gamma1_ext"]), "gamma2": float(setup["gamma2_ext"]),
         "ra_0": 0, "dec_0": 0},
    ]

# theta_E scales as sqrt(Dls/Ds) between source planes; reference is z=1.432.
def theta_E_at(z, theta_E_ref):
    Ds = COSMO.angular_diameter_distance(z).value
    Dds = COSMO.angular_diameter_distance_z1z2(Z_L, z).value
    Ds_ref = COSMO.angular_diameter_distance(Z_REF).value
    Dds_ref = COSMO.angular_diameter_distance_z1z2(Z_L, Z_REF).value
    return theta_E_ref * np.sqrt((Dds / Ds) / (Dds_ref / Ds_ref))

source_planes = {
    "1 (z=0.962, double)":   (0.962, (2.0, -2.5)),
    "3 (z=1.166, cusp)":     (1.166, (0.5, 5.0)),
    "4 (z=1.432, quad)":     (1.432, (-1.0, 1.5)),
    "5 (z=1.432, fold)":     (1.432, (2.5, -1.5)),
}

stretch = np.arcsinh((data - median_bg) / (3 * std_bg))
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(stretch, origin="lower", cmap="magma",
          vmin=np.percentile(stretch, 5), vmax=np.percentile(stretch, 99.8),
          extent=[(0 - la_x) * scale, (nx - la_x) * scale,
                  (0 - la_y) * scale, (ny - la_y) * scale])
ax.add_patch(plt.Circle((0, 0), 13.03, fill=False, color="lime", ls="--", lw=1))
ax.plot(0, 0, "c+", ms=14, mew=2, label="La")
ax.plot(LD_DX, LD_DY, "y+", ms=14, mew=2, label="Ld")

colors = {"1 (z=0.962, double)": "cyan", "3 (z=1.166, cusp)": "magenta",
          "4 (z=1.432, quad)": "orange", "5 (z=1.432, fold)": "lime"}
results = {}
for name, (z, (sx, sy)) in source_planes.items():
    tE = theta_E_at(z, 13.03)
    tEb = theta_E_at(z, 0.99)
    lm = LensModel(["EPL", "EPL", "SHEAR"], z_lens=Z_L, z_source=z, cosmo=COSMO)
    kw = kwargs_lens(theta_E_a=tE, theta_E_b=tEb)
    solver = LensEquationSolver(lm)
    ix_, iy_ = solver.image_position_from_source(
        sx, sy, kw, min_distance=0.1, search_window=44, precision_limit=1e-8,
        num_iter_max=200)
    results[name] = (z, tE, len(ix_))
    ax.scatter(ix_, iy_, s=80, facecolors="none", edgecolors=colors[name], lw=2,
               label=f"{name}: {len(ix_)} img (tE={tE:.1f}\")")
    print(f"{name}: theta_E(z={z})={tE:.2f}\", {len(ix_)} images at "
          + ", ".join(f"({a:+.1f},{b:+.1f})" for a, b in zip(ix_, iy_)))

ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.set_xlim(-23, 23); ax.set_ylim(-23, 23)
ax.set_xlabel("dx East [\"]"); ax.set_ylabel("dy North [\"]")
ax.set_title("Published best-fit model: predicted image configs over HST F140W")
fig.tight_layout()
fig.savefig(FIGS / "predicted_images.png", dpi=130, bbox_inches="tight")
print(f"Wrote {FIGS / 'predicted_images.png'}")

np.savez(DATA / "lens_geometry.npz", la_x=la_x, la_y=la_y, ld_dx=LD_DX, ld_dy=LD_DY,
         scale=scale)
print(f"Saved data/lens_geometry.npz")
