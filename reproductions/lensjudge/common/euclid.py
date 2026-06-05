"""Euclid Q1 cutout loader + multi-view renderer for LensJudge.

Euclid Q1 "Strong Lensing Discovery Engine" cutouts (Zenodo 15025832): per object a
multi-extension FITS with VIS + NIR Y/J/H (each FLUX/PSF/RMS), all resampled to a common
0.1"/px grid (300x300 = 30" FoV) -- ~13x finer than the 0.262"/px DESI grz cutouts. VIS is
the sharp broad-optical luminance band where arcs are clearest; the NIR bands add color
(old red lens galaxy vs blue lensed source). This renders the same *kinds* of views the
DESI grader uses (color context + sharp luminance + tight zoom), adapted to Euclid bands.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

from lensjudge.common.render import png_b64  # noqa: F401  (re-exported for callers)

EUCLID_ROOT = Path(__file__).resolve().parents[2] / "euclid-q1" / "data"
SUBSETS = ("lens", "unsuccess", "group", "recenter")
PIXSCALE = 0.1  # arcsec/px (CD1_1 = 2.7778e-05 deg)
BANDS = ("VIS_FLUX", "NIR_Y_FLUX", "NIR_J_FLUX", "NIR_H_FLUX")


def obj_dir(id_str: str) -> Path | None:
    for s in SUBSETS:
        d = EUCLID_ROOT / s / id_str
        if (d / f"{id_str}.fits").exists():
            return d
    return None


def load_euclid(id_str: str) -> dict | None:
    """Return {band: 2D float array} for the downloadable bands, or None."""
    d = obj_dir(id_str)
    if d is None:
        return None
    out = {}
    with fits.open(d / f"{id_str}.fits") as h:
        for b in BANDS:
            try:
                out[b] = np.nan_to_num(np.asarray(h[b].data, np.float32))
            except KeyError:
                pass
    return out or None


def _crop(a: np.ndarray, arcsec: float) -> np.ndarray:
    n = a.shape[0]; c = n // 2
    half = min(int(round(arcsec / PIXSCALE / 2)), c)
    return a[c - half:c + half, c - half:c + half]


def _stretch(a: np.ndarray, a_soft: float = 0.1, lo: float = 40.0, hi: float = 99.6) -> np.ndarray:
    """Background-subtracted, percentile-clipped asinh stretch -> [0,1]."""
    a = a.astype(np.float64)
    a = a - np.nanpercentile(a, lo)
    top = np.nanpercentile(a, hi)
    if top <= 0:
        top = a.max() if a.max() > 0 else 1.0
    a = np.clip(a / top, 0, 1)
    return np.arcsinh(a / a_soft) / np.arcsinh(1.0 / a_soft)


def _to_img(arr, px: int = 400) -> Image.Image:
    a = np.clip(np.asarray(arr) * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(a[::-1], "RGB" if a.ndim == 3 else "L")
    return im.resize((px, px), Image.NEAREST)


def rgb_view(bands: dict, arcsec: float = 16.0, a_soft: float = 0.15, px: int = 400) -> Image.Image:
    """Euclid color: B=VIS (optical), G=NIR_J, R=NIR_H, falling back to VIS if a band missing."""
    def ch(name):
        return _stretch(_crop(bands.get(name, bands["VIS_FLUX"]), arcsec), a_soft)
    rgb = np.dstack([ch("NIR_H_FLUX"), ch("NIR_J_FLUX"), _stretch(_crop(bands["VIS_FLUX"], arcsec), a_soft)])
    return _to_img(rgb, px)


def vis_view(bands: dict, arcsec: float = 10.0, a_soft: float = 0.07, px: int = 400) -> Image.Image:
    return _to_img(_stretch(_crop(bands["VIS_FLUX"], arcsec), a_soft), px)


def _azimuthal_median_model(img: np.ndarray) -> np.ndarray:
    """Smooth, circularly-symmetric lens-light model = median flux in each 1px radial ring."""
    n = img.shape[0]; c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c).astype(int)
    model = np.zeros_like(img, dtype=float)
    flat_r = r.ravel(); flat_i = img.ravel()
    for rr in range(r.max() + 1):
        m = flat_r == rr
        if m.any():
            model.ravel()[m] = np.median(flat_i[m])
    return model


def vis_sub_view(bands: dict, arcsec: float = 8.0, a_soft: float = 0.2, px: int = 400) -> Image.Image:
    """VIS with the smooth (azimuthally-symmetric) lens galaxy subtracted -> rings/arcs pop.

    Light Gaussian smoothing + a high clip floor suppress per-pixel noise so the residual
    arc/ring (a coherent multi-pixel feature) dominates over background speckle.
    """
    from scipy.ndimage import gaussian_filter
    vis = _crop(bands["VIS_FLUX"], arcsec).astype(float)
    res = vis - _azimuthal_median_model(vis)
    res = gaussian_filter(res, sigma=0.8)
    res = np.clip(res - np.nanpercentile(res, 75.0), 0, None)  # keep only positive residual
    return _to_img(_stretch(res, a_soft, lo=0.0, hi=99.3), px)


_RENDERERS = {
    "full": lambda b, px: rgb_view(b, arcsec=16.0, a_soft=0.15, px=px),
    "zoom": lambda b, px: rgb_view(b, arcsec=6.0, a_soft=0.12, px=px),
    "vis": lambda b, px: vis_view(b, arcsec=10.0, a_soft=0.07, px=px),
    "vis_zoom": lambda b, px: vis_view(b, arcsec=5.0, a_soft=0.04, px=px),
    "vis_sub": lambda b, px: vis_sub_view(b, arcsec=10.0, a_soft=0.08, px=px),
}
VIEWS = ("full", "vis", "zoom", "vis_zoom", "vis_sub")


def render_euclid_views(bands: dict, views=VIEWS, px: int = 400) -> dict:
    out = {}
    for v in views:
        try:
            out[v] = _RENDERERS[v](bands, px)
        except Exception:
            continue
    return out
