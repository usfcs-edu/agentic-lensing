"""Build an empirical PSF for HST WFC3/F140W from field stars in the GO-15867 exposure.

Pipeline (mirrors Huang 2025a's described approach):
  1. Open the full HAP drizzled product (1523x1528) — not the 128x128 cutout.
  2. Detect sources with DAOStarFinder.
  3. Filter for star-like candidates: bright, isolated, near-PSF FWHM.
  4. Stack with photutils EPSFBuilder, supersampled by 2x → 0.065"/px effective.
  5. Save as data/empirical_psf.npy; also save QA figure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder, extract_stars
from astropy.table import Table

REPRO = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i")
DATA = REPRO / "data"
FIGS = REPRO / "figs"
FIGS.mkdir(exist_ok=True)

DRZ = next(DATA.rglob("hst_15867_65_wfc3_ir_f140w_ie5065_drz.fits"))
print(f"Using {DRZ}")
with fits.open(DRZ) as hdul:
    sci_idx = next(i for i, h in enumerate(hdul) if h.header.get("EXTNAME", "").upper() == "SCI")
    sci = hdul[sci_idx].data.astype(np.float32)
print(f"SCI shape: {sci.shape}")

# Background stats
mean_bg, median_bg, std_bg = sigma_clipped_stats(sci, sigma=3.0)
print(f"Background: mean={mean_bg:.3f}, median={median_bg:.3f}, std={std_bg:.4f}")

# Detect bright sources. Use FWHM ~3 px (HST/F140W native), threshold 50σ.
finder = DAOStarFinder(fwhm=3.0, threshold=50.0 * std_bg, sharplo=0.2, sharphi=1.0,
                       roundlo=-0.5, roundhi=0.5)
sources = finder(sci - median_bg)
print(f"DAOStarFinder: {len(sources) if sources else 0} sources at >50σ")
if sources is None or len(sources) == 0:
    raise SystemExit("No sources detected — try lower threshold.")

# Filter: exclude near image edges and the lens target region; keep top 30 by peak
mask_lens = ((sources['xcentroid'] - 760) ** 2 + (sources['ycentroid'] - 764) ** 2) < 200 ** 2
sources = sources[~mask_lens]
PAD = 25
clean_edges = ((sources['xcentroid'] > PAD) & (sources['xcentroid'] < sci.shape[1] - PAD)
               & (sources['ycentroid'] > PAD) & (sources['ycentroid'] < sci.shape[0] - PAD))
sources = sources[clean_edges]
sources.sort('peak')
sources.reverse()
sources = sources[:30]
print(f"After edge+lens-region filter, top 30 by peak: {len(sources)}")
print(sources['xcentroid', 'ycentroid', 'peak', 'sharpness', 'roundness1'][:10])

# Build star table for EPSFBuilder
stars_tbl = Table()
stars_tbl['x'] = sources['xcentroid']
stars_tbl['y'] = sources['ycentroid']

# Background-subtract image
nddata = NDData(data=sci - median_bg)

# Extract 25x25 patches around each star
STAR_SIZE = 25  # px
stars = extract_stars(nddata, stars_tbl, size=STAR_SIZE)
print(f"Extracted {len(stars)} star patches")

# Build effective PSF with oversampling=2 → 0.065" effective grid
epsf_builder = EPSFBuilder(oversampling=2, maxiters=8, progress_bar=False,
                            smoothing_kernel='quartic', recentering_maxiters=20)
epsf, fitted_stars = epsf_builder(stars)
print(f"EPSF shape: {epsf.data.shape} (oversampled 2x → 0.065\"/px effective)")

# Crop EPSF to a tight kernel ~ 1.7" wide (~13 native px = ~26 supersampled px)
psf = np.asarray(epsf.data, dtype=np.float32)
ny, nx = psf.shape
target = 13 * 2 + 1  # 27 supersampled px so kernel is ~1.75" wide
if nx > target:
    ix0 = (nx - target) // 2
    iy0 = (ny - target) // 2
    psf = psf[iy0:iy0 + target, ix0:ix0 + target]
# Ensure positive and normalize
psf = np.clip(psf, 0, None)
psf = psf / psf.sum()
print(f"Cropped/normalized PSF: shape {psf.shape}, sum {psf.sum():.4f}, "
      f"FWHM ~ {2.355 * np.sqrt(((psf>psf.max()/2)).sum()/np.pi) / 2:.2f} supersampled px")

# Save
np.save(DATA / "empirical_psf.npy", psf.astype(np.float32))
print(f"Saved {DATA / 'empirical_psf.npy'}")

# QA figure: empirical PSF vs TinyTim
tinytim = np.load("/raid/benson/lensing-repos/gigalens/src/gigalens/assets/psf.npy").astype(np.float32)
tinytim_norm = tinytim / tinytim.sum()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
# Star postage stamps preview
stack = np.median(np.stack([s.compute_residual_image(epsf) for s in fitted_stars[:9]], axis=0), axis=0)
axes[0].imshow(np.median(np.stack([np.asarray(s.data) for s in fitted_stars[:9]], axis=0), axis=0),
               origin="lower", cmap="magma")
axes[0].set_title(f"Median of top 9 stars (raw)")
axes[1].imshow(psf, origin="lower", cmap="magma")
axes[1].set_title(f"Empirical PSF (this work)\nshape={psf.shape}")
axes[2].imshow(tinytim_norm, origin="lower", cmap="magma")
axes[2].set_title(f"TinyTim F140W (gigalens default)\nshape={tinytim_norm.shape}")
diff_size = min(psf.shape[0], tinytim_norm.shape[0])
def center_crop(a, n):
    h, w = a.shape
    return a[(h - n) // 2:(h + n) // 2, (w - n) // 2:(w + n) // 2]
p_c = center_crop(psf, diff_size)
t_c = center_crop(tinytim_norm, diff_size)
diff = p_c / p_c.max() - t_c / t_c.max()
axes[3].imshow(diff, origin="lower", cmap="coolwarm", vmin=-0.3, vmax=0.3)
axes[3].set_title("(empirical − tinytim) peak-normalized")
for ax in axes:
    ax.axis("off")
fig.suptitle(f"Empirical PSF from {len(fitted_stars)} fitted F140W field stars vs TinyTim model")
fig.tight_layout()
fig.savefig(FIGS / "psf_comparison.png", dpi=120, bbox_inches="tight")
print(f"Wrote {FIGS / 'psf_comparison.png'}")
