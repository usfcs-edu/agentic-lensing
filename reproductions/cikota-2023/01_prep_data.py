"""01 - Prepare the public DESI Legacy Surveys g-band cutout of the Einstein cross
DESI-253.2534+26.8843 (Cikota et al. 2023, arXiv:2307.12470) for GIGA-Lens modeling.

The paper modeled a MUSE-derived g-band image at ~0.6" seeing.  We do NOT have the
proprietary VLT/MUSE cube, so we substitute the PUBLIC DESI Legacy Surveys DR10
g-band coadd (the same survey the system was discovered in, Huang et al. 2021),
downloaded from the public cutout server in 00 / by curl:

  https://www.legacysurvey.org/viewer/fits-cutout?ra=253.2534&dec=26.8843&\
      layer=ls-dr10&pixscale=0.262&bands=grz&size=120
  https://www.legacysurvey.org/viewer/coadd-psf/?ra=...&layer=ls-dr10&bands=g

This is a HARDER reproduction than the paper: the Legacy g-band coadd PSF FWHM is
~1.35" (vs the paper's 0.6" MUSE PSF), so the four images of the cross are partially
blended in g.  We model the g band, with the actual Legacy g coadd PSF, on a tight
field centered on the lens.

Outputs (data/):
  cikota_g_image.npy        cropped g-band science image (float32, ADU/nmgy)
  cikota_g_psf.npy          Legacy g-band coadd PSF, normalized, odd-sized kernel
  cikota_g_meta.npz         delta_pix, num_pix, background_rms, exp_time, crop bbox
  figs/01_data_inspect.png  QA figure
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"
FIGS.mkdir(exist_ok=True)

DELTA_PIX = 0.262  # arcsec/pix (Legacy DR10 native)
NUM_PIX = 60       # tight ~15.7" field centered on the lens

# --------------------------------------------------------------------------- #
# load grz cube, keep g band (band 0)
# --------------------------------------------------------------------------- #
cube = fits.getdata(DATA / "legacy_dr10_grz_120.fits").astype(np.float32)  # (3,120,120)
g_full = cube[0]
ny, nx = g_full.shape
# lens center is at CRVAL pixel = (59.5, 59.5) for the 120px cutout
cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0  # 59.5, 59.5
half = NUM_PIX // 2
y0, y1 = int(round(cy)) - half, int(round(cy)) - half + NUM_PIX
x0, x1 = int(round(cx)) - half, int(round(cx)) - half + NUM_PIX
g = np.ascontiguousarray(g_full[y0:y1, x0:x1])
print(f"Cropped g-band to {g.shape} bbox y[{y0}:{y1}] x[{x0}:{x1}]")

# --------------------------------------------------------------------------- #
# background statistics (sky already subtracted by Legacy pipeline; estimate rms)
# --------------------------------------------------------------------------- #
mean_bg, median_bg, std_bg = sigma_clipped_stats(g, sigma=3.0)
print(f"g-band sky: mean={mean_bg:.5g} median={median_bg:.5g} rms={std_bg:.5g}")
# Re-zero residual sky and rescale to a convenient surface-brightness unit.
# GIGA-Lens expects the image and background_rms in the SAME units; the absolute
# scale is arbitrary (Sersic Ie priors are exp(N(ln 25,...)) so a *25x scaling
# brings the bright source images to ~tens of ADU, matching the paper's prior).
SCALE = 1000.0  # nmgy -> "milli-nmgy" so peak source pixels are O(10-30)
g = (g - median_bg) * SCALE
background_rms = float(std_bg * SCALE)
print(f"After rescale (x{SCALE:g}): peak={g.max():.2f} background_rms={background_rms:.3f}")

# Effective exposure time controls the Poisson term err = sqrt(rms^2 + I/exp_time).
# Legacy g coadds are deep; pick exp_time so the Poisson term is comparable to the
# sky term for the brightest source pixels (keeps the error map well behaved).
EXP_TIME = 100.0

# --------------------------------------------------------------------------- #
# PSF: actual Legacy g-band coadd PSF (preferred over the paper's 0.6" Gaussian
# because it matches THIS data).
# --------------------------------------------------------------------------- #
psf = fits.getdata(DATA / "legacy_psf_g.fits").astype(np.float32)
# ensure odd size and normalized
if psf.shape[0] % 2 == 0:
    psf = psf[:-1, :-1]
psf = np.clip(psf, 0, None)
psf /= psf.sum()
# crop to a tight odd kernel (~5" => 21 px) to speed convolution
KSZ = 25
if psf.shape[0] > KSZ:
    c = psf.shape[0] // 2
    h = KSZ // 2
    psf = psf[c - h:c + h + 1, c - h:c + h + 1]
    psf /= psf.sum()
peak = psf.max()
fwhm_pix = 2 * np.sqrt((psf > peak / 2).sum() / np.pi)
print(f"Legacy g PSF: shape {psf.shape} sum {psf.sum():.4f} "
      f"FWHM~{fwhm_pix:.2f}px = {fwhm_pix * DELTA_PIX:.3f}\" (paper used 0.6\")")

# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
np.save(DATA / "cikota_g_image.npy", g.astype(np.float32))
np.save(DATA / "cikota_g_psf.npy", psf.astype(np.float32))
np.savez(DATA / "cikota_g_meta.npz",
         delta_pix=DELTA_PIX, num_pix=NUM_PIX,
         background_rms=background_rms, exp_time=EXP_TIME,
         scale=SCALE, bbox=np.array([y0, y1, x0, x1]),
         psf_fwhm_arcsec=fwhm_pix * DELTA_PIX)
print(f"Saved data/cikota_g_image.npy, cikota_g_psf.npy, cikota_g_meta.npz")

# --------------------------------------------------------------------------- #
# QA figure
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
vmax = np.percentile(g, 99.5)
im0 = ax[0].imshow(g, origin="lower", cmap="cubehelix", vmin=-2 * background_rms, vmax=vmax)
ax[0].set_title(f"DESI Legacy DR10 g-band ({NUM_PIX}px, {NUM_PIX*DELTA_PIX:.1f}\")")
plt.colorbar(im0, ax=ax[0], fraction=0.046)
# overlay paper image positions A,B,C,D and L2 relative to lens (arcsec -> pix)
# Table 1: A(+2.21,+1.22) B(-2.49,-1.02) C(+0.94,-1.77) D(-1.12,+1.46)
# RA increases to East (left), so x_pix = center - dRA/delta ; y_pix = center + dDec/delta
ctr = NUM_PIX / 2.0 - 0.5
labels = dict(A=(2.21, 1.22), B=(-2.49, -1.02), C=(0.94, -1.77), D=(-1.12, 1.46))
for nm, (dra, ddec) in labels.items():
    px = ctr - dra / DELTA_PIX
    py = ctr + ddec / DELTA_PIX
    ax[0].plot(px, py, "+", color="cyan", ms=10, mew=1.5)
    ax[0].text(px + 1, py + 1, nm, color="cyan", fontsize=10)
# L2 from Table 3 mass center (1.836, -1.563) arcsec
l2x = ctr - 1.836 / DELTA_PIX
l2y = ctr + (-1.563) / DELTA_PIX
ax[0].plot(l2x, l2y, "x", color="orange", ms=9, mew=1.5)
ax[0].text(l2x + 1, l2y - 2, "L2", color="orange", fontsize=10)

im1 = ax[1].imshow(np.log10(psf / psf.max() + 1e-4), origin="lower", cmap="magma")
ax[1].set_title(f"Legacy g coadd PSF\nFWHM~{fwhm_pix*DELTA_PIX:.2f}\"")
plt.colorbar(im1, ax=ax[1], fraction=0.046)

err = np.sqrt(background_rms ** 2 + np.clip(g, 0, None) / EXP_TIME)
im2 = ax[2].imshow(g / err, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
ax[2].set_title("S/N map")
plt.colorbar(im2, ax=ax[2], fraction=0.046)
fig.suptitle("Cikota+2023 Einstein cross DESI-253.2534+26.8843 — public DESI Legacy g-band")
fig.tight_layout()
fig.savefig(FIGS / "01_data_inspect.png", dpi=120, bbox_inches="tight")
print(f"Wrote {FIGS / '01_data_inspect.png'}")
