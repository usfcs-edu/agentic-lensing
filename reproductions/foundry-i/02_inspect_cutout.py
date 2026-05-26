"""Inspect drizzled HST F140W frames; produce a 128x128 cutout at 0.065"/px centered on
DESI-165.4754-06.0423 per Huang 2025a Foundry I.

Inputs: drz.fits products under data/mastDownload/HST/
Output: data/cutout_F140W.fits, figs/cutout_preview.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

DATA = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i/data")
FIGS = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i/figs")
TARGET = SkyCoord(ra=165.4754 * u.deg, dec=-6.0423 * u.deg, frame="icrs")
CUTOUT_PIX = 128
PIXEL_SCALE_ARCSEC = 0.065

candidates = sorted(DATA.rglob("hst_15867_65_wfc3_ir_f140w_ie5065_drz.fits"))
if not candidates:
    raise SystemExit("No HAP combined drz product found")
DRZ = candidates[0]
print("Using", DRZ)

with fits.open(DRZ) as hdul:
    print("\nHDU list:")
    hdul.info()
    sci_idx = next(i for i, h in enumerate(hdul) if h.header.get("EXTNAME", "").upper() == "SCI")
    wht_idx = next(i for i, h in enumerate(hdul) if h.header.get("EXTNAME", "").upper() == "WHT")
    sci = hdul[sci_idx].data.astype(np.float64)
    wht = hdul[wht_idx].data.astype(np.float64)
    hdr_sci = hdul[sci_idx].header
    wcs = WCS(hdr_sci)

print(f"\nSCI shape: {sci.shape}, dtype {sci.dtype}")
print(f"WCS pixel scale: {wcs.proj_plane_pixel_scales()}")
print(f"Exposure time (EXPTIME): {hdr_sci.get('EXPTIME', 'n/a')} s")
print(f"PHOTFLAM={hdr_sci.get('PHOTFLAM', 'n/a')}, PHOTZPT={hdr_sci.get('PHOTZPT', 'n/a')}")

cutout = Cutout2D(sci, position=TARGET, size=CUTOUT_PIX * u.pixel, wcs=wcs, mode="strict")
err_cutout = 1.0 / np.sqrt(np.where(wht > 0, wht, np.nan))  # rms ~ 1/sqrt(weight)
wht_cutout = Cutout2D(wht, position=TARGET, size=CUTOUT_PIX * u.pixel, wcs=wcs, mode="strict")
print(f"\nCutout shape: {cutout.data.shape}, center px: {cutout.input_position_cutout}")
print(f"Cutout flux range: [{np.nanmin(cutout.data):.4f}, {np.nanmax(cutout.data):.4f}] e-/s")
print(f"Median weight: {np.nanmedian(wht_cutout.data):.4f}")

# Write a single-HDU FITS with SCI + WHT extensions for downstream modeling
new_hdul = fits.HDUList(
    [
        fits.PrimaryHDU(),
        fits.ImageHDU(cutout.data.astype(np.float32), header=cutout.wcs.to_header(), name="SCI"),
        fits.ImageHDU(wht_cutout.data.astype(np.float32), header=wht_cutout.wcs.to_header(), name="WHT"),
    ]
)
out_fits = DATA / "cutout_F140W.fits"
new_hdul.writeto(out_fits, overwrite=True)
print(f"\nWrote {out_fits}")

# Quick visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
vmin, vmax = np.nanpercentile(cutout.data, [1, 99.5])
axes[0].imshow(cutout.data, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
axes[0].set_title(f"F140W SCI ({CUTOUT_PIX}x{CUTOUT_PIX})")
axes[0].set_xlabel("x [pix]")
axes[0].set_ylabel("y [pix]")
axes[1].imshow(np.arcsinh(cutout.data / np.nanmedian(np.abs(cutout.data))), origin="lower", cmap="magma")
axes[1].set_title("asinh stretch")
axes[2].imshow(wht_cutout.data, origin="lower", cmap="viridis")
axes[2].set_title("WHT")
for ax in axes:
    ax.scatter(CUTOUT_PIX / 2, CUTOUT_PIX / 2, marker="+", s=120, color="red", linewidths=1)
FIGS.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(FIGS / "cutout_preview.png", dpi=120)
print(f"Wrote {FIGS / 'cutout_preview.png'}")
