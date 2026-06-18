"""common/hsc.py — HSC-SSP PDR3 cutout loader + multi-view renderer for LensJudge.

The tier-2 high-resolution analog of common/euclid.py, one rung coarser: HSC coadds are
0.168"/px (~0.6" seeing) vs Euclid's 0.1" and DESI's ~1.3". Loads grizy coadd cutouts via
common/hsc_fetch (PDR3 das_cutout service) and renders the same *kinds* of views the DESI/Euclid
graders use — color context (i/r/g), a sharp i-band luminance, a tight zoom, and a lens-light
subtracted residual where thin arcs/rings pop. Reuses euclid.py's scale-independent stretch /
RGB-image / azimuthal-median lens-light helpers; only the crop (pixel scale) differs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

# scale-independent render helpers (reused verbatim from the Euclid renderer)
from lensjudge.common.euclid import _azimuthal_median_model, _stretch, _to_img
from lensjudge.common import hsc_fetch

PIXSCALE = hsc_fetch.PIXSCALE  # 0.168 arcsec/px
BANDS = ("g", "r", "i", "z", "y")


def load_hsc(ra, dec, **kw) -> Optional[dict]:
    """Return ``{band_letter: 2D float array}`` for the (ra, dec), or None (no coverage/creds)."""
    return hsc_fetch.fetch_hsc_cutout(ra, dec, **kw)


def _crop(a: np.ndarray, arcsec: float) -> np.ndarray:
    n = a.shape[0]; c = n // 2
    half = min(int(round(arcsec / PIXSCALE / 2)), c)
    return a[c - half:c + half, c - half:c + half]


def _lum(bands: dict) -> np.ndarray:
    return bands.get("i", bands.get("r"))


def rgb_view(bands: dict, arcsec: float = 16.0, a_soft: float = 0.15, px: int = 400):
    """HSC color: R=i, G=r, B=g (old red lens galaxy red/orange, lensed source blue)."""
    def ch(name):
        return _stretch(_crop(bands.get(name, _lum(bands)), arcsec), a_soft)
    rgb = np.dstack([ch("i"), ch("r"), ch("g")])
    return _to_img(rgb, px)


def lum_view(bands: dict, arcsec: float = 10.0, a_soft: float = 0.07, px: int = 400):
    return _to_img(_stretch(_crop(_lum(bands), arcsec), a_soft), px)


def lum_sub_view(bands: dict, arcsec: float = 10.0, px: int = 400, lim: float = 5.0, **kw):
    """i-band SIGNED lens-light residual: chi=(i-band - azimuthal-median model)/noise on a
    diverging RdBu scale (red = unmodelled excess / arc / counter-image, blue = over-subtraction,
    fixed +/-5 sigma). Sign is preserved (no positive-only clip), so over-subtraction stays visible."""
    from lensjudge.common import render
    base = _crop(_lum(bands), arcsec).astype(float)
    res = base - _azimuthal_median_model(base)
    chi = res / render._robust_sigma(res)
    return Image.fromarray(render._diverging_rgb(chi, lim)).resize((px, px), Image.NEAREST)


_RENDERERS = {
    "full": lambda b, px: rgb_view(b, arcsec=16.0, a_soft=0.15, px=px),
    "zoom": lambda b, px: rgb_view(b, arcsec=6.0, a_soft=0.12, px=px),
    "lum": lambda b, px: lum_view(b, arcsec=10.0, a_soft=0.07, px=px),
    "lum_zoom": lambda b, px: lum_view(b, arcsec=5.0, a_soft=0.04, px=px),
    "lum_sub": lambda b, px: lum_sub_view(b, arcsec=10.0, px=px),
}
VIEWS = ("full", "lum", "zoom", "lum_zoom", "lum_sub")


def render_hsc_views(bands: dict, views=VIEWS, px: int = 400) -> dict:
    out = {}
    for v in views:
        try:
            out[v] = _RENDERERS[v](bands, px)
        except Exception:
            continue
    return out
