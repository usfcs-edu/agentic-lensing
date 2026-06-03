"""Build an empirical PSF for HST WFC3/F140W from field stars in the GO-16773 mosaic.

Mirrors Sheu et al.'s approach ("stacking nearby stars in the HST exposure") and the
foundry-i/17 EPSFBuilder recipe. Operates on the full combined F140W mosaic
(hst_16773_c7_wfc3_ir_f140w_ienhc7_drz.fits, 0.128"/px). Saves data/empirical_psf.npy.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder, extract_stars

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

DRZ = DATA / "mastDownload/HST/hst_16773_c7_wfc3_ir_f140w_ienhc7/hst_16773_c7_wfc3_ir_f140w_ienhc7_drz.fits"
RA, DEC = 90.9854, -35.9683

with fits.open(DRZ) as hdul:
    sci_idx = next(i for i, h in enumerate(hdul)
                   if h.header.get("EXTNAME", "").upper() == "SCI")
    sci = hdul[sci_idx].data.astype(np.float32)
    wcs = WCS(hdul[sci_idx].header)
print(f"SCI shape {sci.shape}")

x_lens, y_lens = wcs.world_to_pixel(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg))
print(f"Lens at pixel ({x_lens:.0f}, {y_lens:.0f})")

mean_bg, median_bg, std_bg = sigma_clipped_stats(sci, sigma=3.0)
print(f"Background median={median_bg:.4f} std={std_bg:.4f}")

# Detect bright point-like sources. WFC3/IR F140W FWHM ~ 1.5 px at 0.128"/px.
finder = DAOStarFinder(fwhm=2.0, threshold=40.0 * std_bg, sharplo=0.4, sharphi=0.9,
                       roundlo=-0.4, roundhi=0.4)
sources = finder(sci - median_bg)
print(f"DAOStarFinder: {0 if sources is None else len(sources)} candidates")
if sources is None or len(sources) == 0:
    raise SystemExit("No sources detected.")

# Exclude the lens/cluster region (within ~25" = ~195 px) and image edges.
r_lens = np.hypot(sources["xcentroid"] - x_lens, sources["ycentroid"] - y_lens)
sources = sources[r_lens > 200]
PAD = 20
ok = ((sources["xcentroid"] > PAD) & (sources["xcentroid"] < sci.shape[1] - PAD)
      & (sources["ycentroid"] > PAD) & (sources["ycentroid"] < sci.shape[0] - PAD))
sources = sources[ok]
sources.sort("peak")
sources.reverse()
# keep moderately bright but unsaturated stars
sources = sources[:40]
print(f"After lens/edge filter, top {len(sources)} by peak")

stars_tbl = Table()
stars_tbl["x"] = sources["xcentroid"]
stars_tbl["y"] = sources["ycentroid"]

nddata = NDData(data=sci - median_bg)
STAR_SIZE = 21
stars = extract_stars(nddata, stars_tbl, size=STAR_SIZE)
print(f"Extracted {len(stars)} star patches")

epsf_builder = EPSFBuilder(oversampling=2, maxiters=10, progress_bar=False,
                           smoothing_kernel="quartic", recentering_maxiters=20)
epsf, fitted_stars = epsf_builder(stars)
psf = np.asarray(epsf.data, dtype=np.float32)
print(f"EPSF (oversampled 2x) shape {psf.shape}")

# Resample back to detector sampling for lenstronomy (which uses a same-pixel-scale PSF
# unless supersampling is configured). Take every-2nd sample -> detector grid kernel.
psf_det = psf[::2, ::2].copy()
psf_det = np.clip(psf_det, 0, None)
psf_det = psf_det / psf_det.sum()
# crop to odd size centered
n = psf_det.shape[0]
if n % 2 == 0:
    psf_det = psf_det[:-1, :-1]
print(f"Detector-sampling PSF shape {psf_det.shape}, sum {psf_det.sum():.4f}")

np.save(DATA / "empirical_psf.npy", psf_det)
np.save(DATA / "empirical_psf_oversampled2.npy", np.clip(psf, 0, None) / np.clip(psf, 0, None).sum())
print(f"Saved data/empirical_psf.npy")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
med_star = np.median(np.stack([np.asarray(s.data) for s in fitted_stars[:12]]), axis=0)
axes[0].imshow(med_star, origin="lower", cmap="magma")
axes[0].set_title(f"Median of {min(12,len(fitted_stars))} stars (raw)")
axes[1].imshow(np.clip(psf, 0, None) ** 0.4, origin="lower", cmap="magma")
axes[1].set_title(f"EPSF oversampled {psf.shape}")
axes[2].imshow(psf_det ** 0.4, origin="lower", cmap="magma")
axes[2].set_title(f"Detector PSF {psf_det.shape}")
for a in axes:
    a.axis("off")
fig.suptitle(f"Empirical F140W PSF from {len(fitted_stars)} field stars")
fig.tight_layout()
fig.savefig(FIGS / "psf.png", dpi=120, bbox_inches="tight")
print(f"Wrote {FIGS / 'psf.png'}")
