"""Empirical PSF at the HAP fine-skycell scale (0.04"/px), for the Track-A rebuild.

Same recipe as 17_build_empirical_psf.py (DAOStarFinder -> isolated bright
stars -> photutils EPSFBuilder) but run directly on the 0.04"/px skycell
mosaic, so the kernel is natively sampled at the modeling scale
(SimulatorConfig supersample=1, kernel scale == data scale).

Output: data/empirical_psf_04.npy (51x51 @ 0.04" ~ 2.0" wide) + QA figure.
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder, extract_stars

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

SKYCELL = next(DATA.rglob("hst_skycell-p1184x13y11_wfc3_ir_f140w_all_drz.fits"))
print(f"Using {SKYCELL}")
with fits.open(SKYCELL) as h:
    sci = h["SCI"].data.astype(np.float32)
print(f"SCI shape: {sci.shape} @ 0.04\"/px")

mean_bg, median_bg, std_bg = sigma_clipped_stats(sci, sigma=3.0, maxiters=8)
print(f"Background: median={median_bg:.4f}, std={std_bg:.5f}")

# WFC3/IR PSF FWHM ~ 0.14" = 3.5 px at 0.04"
finder = DAOStarFinder(fwhm=3.5, threshold=40.0 * std_bg, sharplo=0.2,
                       sharphi=1.0, roundlo=-0.4, roundhi=0.4)
sources = finder(sci - median_bg)
print(f"DAOStarFinder: {len(sources) if sources is not None else 0} sources")
assert sources is not None and len(sources) >= 3, "not enough PSF stars"

xc = np.asarray(sources["x_centroid"], dtype=float)
yc = np.asarray(sources["y_centroid"], dtype=float)
flux = np.asarray(sources["flux"], dtype=float)

# isolation + edge filters
SIZE = 51
half = SIZE // 2 + 6
keep = []
for i in range(len(xc)):
    if not (half < xc[i] < sci.shape[1] - half and half < yc[i] < sci.shape[0] - half):
        continue
    d = np.hypot(xc - xc[i], yc - yc[i])
    d[i] = np.inf
    if d.min() < 30:           # 1.2" isolation
        continue
    keep.append(i)
keep = sorted(keep, key=lambda i: -flux[i])[:25]
print(f"isolated star candidates: {len(keep)}")
assert len(keep) >= 3

tbl = Table(dict(x=xc[keep], y=yc[keep]))
nddata = NDData(data=sci - median_bg)
stars = extract_stars(nddata, tbl, size=SIZE + 8)

builder = EPSFBuilder(oversampling=1, maxiters=12, progress_bar=False)
epsf, fitted = builder(stars)
psf = epsf.data.astype(np.float32)
print(f"EPSF raw shape: {psf.shape}")

# center-crop to SIZE and normalize
cy, cx = np.array(psf.shape) // 2
h2 = SIZE // 2
psf = psf[cy - h2:cy + h2 + 1, cx - h2:cx + h2 + 1]
psf = np.clip(psf, 0, None)
psf /= psf.sum()
print(f"final kernel: {psf.shape}, sum={psf.sum():.6f}, "
      f"FWHM~{2.355*np.sqrt((psf > psf.max()/2).sum()/np.pi)*0.04:.3f} arcsec")

np.save(DATA / "empirical_psf_04.npy", psf)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(np.arcsinh(psf / psf.max() * 1e3), origin="lower", cmap="magma")
ax[0].set_title(f"empirical PSF @0.04\" ({SIZE}x{SIZE}, {len(keep)} stars)")
prof = psf[h2, :]
ax[1].semilogy(np.arange(SIZE) - h2, np.maximum(prof, 1e-7))
ax[1].set_title("central row profile")
fig.tight_layout()
fig.savefig(FIGS / "empirical_psf_04.png", dpi=120)
print("wrote data/empirical_psf_04.npy, figs/empirical_psf_04.png")
