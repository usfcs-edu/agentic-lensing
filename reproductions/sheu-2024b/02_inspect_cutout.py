"""Locate the Carousel Lens in the HST F140W mosaic and build a modeling cutout.

Sheu et al. model on the F140W image (0.070"/px). We use the combined HAP drizzled
product hst_16773_c7_wfc3_ir_f140w_ienhc7_drz.fits, find the lens center (La/Lb/Lc
near RA=90.9854, Dec=-35.9683) via WCS, and cut out a ~40" box (the full lensing
configuration spans ~26"+, theta_E=13"). Saves cutout + pixel scale + WCS for later.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"
FIGS.mkdir(exist_ok=True)

RA, DEC = 90.9854, -35.9683
CUTOUT_ARCSEC = 44.0  # box half-size handled below; full lensing config + margin

DRZ = DATA / "mastDownload/HST/hst_16773_c7_wfc3_ir_f140w_ienhc7/hst_16773_c7_wfc3_ir_f140w_ienhc7_drz.fits"
print(f"Opening {DRZ}")
with fits.open(DRZ) as hdul:
    for i, h in enumerate(hdul):
        print(f"  [{i}] {h.name} {getattr(h,'shape',None)} {h.header.get('EXTNAME','')}")
    sci_idx = next(i for i, h in enumerate(hdul)
                   if h.header.get("EXTNAME", "").upper() == "SCI")
    sci = hdul[sci_idx].data.astype(np.float32)
    hdr = hdul[sci_idx].header
    wcs = WCS(hdr)

print(f"SCI shape {sci.shape}")
# Pixel scale
try:
    scale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
except Exception:
    scale = abs(hdr.get("CD1_1", hdr.get("CDELT1", 0))) * 3600.0
print(f"Pixel scale: {scale:.4f} arcsec/px (paper quotes 0.070 for F140W)")

center = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
x_c, y_c = wcs.world_to_pixel(center)
print(f"Lens center at pixel ({x_c:.1f}, {y_c:.1f}) in mosaic of shape {sci.shape}")

npix = int(round(CUTOUT_ARCSEC / scale))
print(f"Cutout size {npix} px = {npix*scale:.1f} arcsec")
cut = Cutout2D(sci, position=center, size=(npix, npix), wcs=wcs, mode="trim")
data = np.asarray(cut.data, dtype=np.float32)
print(f"Cutout shape {data.shape}")

# Background
mean_bg, median_bg, std_bg = sigma_clipped_stats(data, sigma=3.0)
print(f"Background median={median_bg:.4f} std={std_bg:.4f}")

# Save cutout FITS (with WCS) and a numpy bundle
cut_hdr = cut.wcs.to_header()
cut_hdr["PIXSCALE"] = (scale, "arcsec/px")
cut_hdr["BKG_MED"] = median_bg
cut_hdr["BKG_STD"] = std_bg
fits.writeto(DATA / "f140w_cutout.fits", data, cut_hdr, overwrite=True)
np.savez(DATA / "cutout_meta.npz",
         data=data, scale=scale, median_bg=median_bg, std_bg=std_bg,
         x_center_full=float(x_c), y_center_full=float(y_c),
         ra=RA, dec=DEC, npix=npix)
print(f"Saved data/f140w_cutout.fits and data/cutout_meta.npz")

# QA figure
vmin = median_bg - 1 * std_bg
vmax = median_bg + 30 * std_bg
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
ax[0].set_title(f"F140W cutout {data.shape} ({data.shape[0]*scale:.0f}\")")
# asinh stretch for arcs
stretch = np.arcsinh((data - median_bg) / (3 * std_bg))
ax[1].imshow(stretch, origin="lower", cmap="magma",
             vmin=np.percentile(stretch, 5), vmax=np.percentile(stretch, 99.8))
ax[1].set_title("asinh stretch (arcs visible)")
# mark center
for a in ax:
    a.plot(data.shape[1] / 2, data.shape[0] / 2, "c+", ms=12, mew=2)
fig.tight_layout()
fig.savefig(FIGS / "cutout_inspect.png", dpi=120, bbox_inches="tight")
print(f"Wrote {FIGS / 'cutout_inspect.png'}")
