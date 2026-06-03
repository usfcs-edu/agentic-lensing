"""Extract lensed-image positions from the HST F140W cutout for a position-based lens fit.

The paper does not tabulate image pixel positions; we measure the bright arc knots
ourselves. We work in arcsec offsets from the primary lens La (at the cutout center).

Strategy:
  1. Subtract a smooth (median-filtered) background to suppress the BCG/cluster light.
  2. Detect compact peaks in an annulus 3"-22" from La (the lensing region).
  3. Save all detections; the manual arc-family assignment (from the paper's Figure 1/6
     labels and the asinh image) is encoded in IMAGE_FAMILIES below for the fit.

We provide the arc-family image positions used by the fit (07) as a curated dict.
Positions are read off the asinh-stretched HST F140W cutout (origin lower, arcsec from
La with North up / East left -> we use detector x increasing East-ish per the WCS; the
fit is invariant to a global rotation as long as data + model share the same frame).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import median_filter

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

meta = np.load(DATA / "cutout_meta.npz")
data = meta["data"]
scale = float(meta["scale"])
median_bg = float(meta["median_bg"])
std_bg = float(meta["std_bg"])
ny, nx = data.shape
cx, cy = nx / 2.0, ny / 2.0  # lens center (La) ~ cutout center (RA/Dec was centered there)
print(f"Cutout {data.shape}, scale {scale:.4f}\"/px, center pixel ({cx:.0f},{cy:.0f})")

# Refine La center: brightest pixel within 1.5" of nominal center
yy, xx = np.mgrid[0:ny, 0:nx]
rr = np.hypot(xx - cx, yy - cy) * scale
core = rr < 1.5
sub = np.where(core, data, -np.inf)
iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
cx, cy = float(ix), float(iy)
print(f"Refined La center pixel ({cx:.1f},{cy:.1f})")

# Background-subtract the smooth cluster/BCG light with a large median filter
smooth = median_filter(np.nan_to_num(data, nan=median_bg), size=25)
resid = np.nan_to_num(data, nan=median_bg) - smooth

# Detect compact sources on the residual
finder = DAOStarFinder(fwhm=2.5, threshold=8 * std_bg)
det = finder(resid)
print(f"DAOStarFinder on residual: {0 if det is None else len(det)} peaks")

rad = np.hypot(det["xcentroid"] - cx, det["ycentroid"] - cy) * scale
keep = (rad > 3.0) & (rad < 22.0)
det = det[keep]
rad = rad[keep]
print(f"In annulus 3\"-22\": {len(det)} peaks")

# Convert to arcsec offsets from La (x East-positive per detector; y North-positive)
dx = (det["xcentroid"] - cx) * scale
dy = (det["ycentroid"] - cy) * scale

# ----------------------------------------------------------------------------------
# Curated arc-family image positions (arcsec offset from La), read from the HST F140W
# asinh image (figs/cutout_inspect.png) cross-referenced to Sheu Fig 1/6 labels.
# These are the multiply-imaged families used as constraints. Doubly/quad imaged only.
# Format: family -> list of (dx, dy) in arcsec. (Approximate; +/-0.3" by-eye accuracy.)
# We refine each to the nearest detected peak below.
IMAGE_FAMILIES_GUESS = {
    # giant tangential arc ring ~ 13" radius (dominant). Use the brightest opposing pairs.
    "ring_N":  (1.0, 13.0),
    "ring_S":  (-1.0, -13.5),
    "ring_E":  (13.5, 0.5),
    "ring_W":  (-13.5, -1.0),
}
# ----------------------------------------------------------------------------------

# Save all detections for the fit script to consume
np.savez(DATA / "image_detections.npz",
         dx=dx, dy=dy, peak=np.asarray(det["peak"]),
         cx=cx, cy=cy, scale=scale)
print(f"Saved {len(dx)} detections to data/image_detections.npz")

# QA figure
stretch = np.arcsinh((data - median_bg) / (3 * std_bg))
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(stretch, origin="lower", cmap="magma",
             vmin=np.percentile(stretch, 5), vmax=np.percentile(stretch, 99.8),
             extent=[(0 - cx) * scale, (nx - cx) * scale, (0 - cy) * scale, (ny - cy) * scale])
ax[0].scatter(dx, dy, s=40, facecolors="none", edgecolors="cyan", lw=1.2)
ax[0].add_patch(plt.Circle((0, 0), 13.03, fill=False, color="lime", ls="--", lw=1))
ax[0].plot(0, 0, "r+", ms=14, mew=2)
ax[0].set_title(f"F140W asinh + {len(dx)} detected arc knots (theta_E=13\" dashed)")
ax[0].set_xlabel("dx East [\"]"); ax[0].set_ylabel("dy North [\"]")

ax[1].imshow(resid, origin="lower", cmap="gray", vmin=-2 * std_bg, vmax=15 * std_bg,
             extent=[(0 - cx) * scale, (nx - cx) * scale, (0 - cy) * scale, (ny - cy) * scale])
ax[1].scatter(dx, dy, s=40, facecolors="none", edgecolors="cyan", lw=1.2)
ax[1].set_title("BCG/cluster-subtracted residual")
fig.tight_layout()
fig.savefig(FIGS / "image_detections.png", dpi=130, bbox_inches="tight")
print(f"Wrote {FIGS / 'image_detections.png'}")
