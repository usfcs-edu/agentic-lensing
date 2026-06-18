"""FITS grz cube -> multi-view Lupton-RGB PNGs for the vision tool.

The Agent SDK has no image-in-prompt, so the only way the model sees pixels is an
image content block returned by a tool. This module turns a (3,101,101) grz cube
into several PNG views that give the model the affordances a human grader uses:

  full        — Lupton-RGB of the whole 101x101 cutout (matches the group's viewer)
  zoom        — 2.5x center crop (the 1-5" lens/source region)
  residual    — per-band minus a Gaussian-smoothed model, re-Lupton'd (exposes
                low-surface-brightness arcs hiding under the lens-galaxy light)
  highcontrast— stronger arcsinh stretch (faint-feature boost)

Render params (Q=8, stretch=0.5, z/r/g -> RGB, vertical flip, NEAREST upsample)
are identical to reproductions/inchausti-2025/16_build_inspection_viewer.py so a
cutout renders the same here as in the team's inspection tool.
"""
from __future__ import annotations

import base64
import io as _io
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
from astropy.visualization import make_lupton_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter

from lensjudge import config

VIEWS = ("full", "zoom", "residual", "highcontrast")

# Signed-residual display limit: chi = (data-model)/sigma is rendered on a fixed +/-5 sigma
# diverging scale so the gallery and the grader are comparable object-to-object (no per-object
# auto-scaling). See residual() / _diverging_rgb().
RESID_SIGMA_LIM = 5.0


def load_cube(path: str | Path) -> np.ndarray | None:
    """Load a (3,101,101) float32 grz cube; return None if shape is wrong."""
    try:
        with fits.open(path) as hdul:
            cube = np.asarray(hdul[0].data, dtype=np.float32)
    except Exception:
        return None
    if cube is None or cube.shape != config.CUTOUT_SHAPE:
        return None
    return np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)


def _img_from_rgb(rgb: np.ndarray, px: int = config.RENDER_PX) -> Image.Image:
    # match the team's viewer: vertical flip, NEAREST upsample for legibility
    return Image.fromarray(rgb[::-1, :, :]).resize((px, px), Image.NEAREST)


def lupton(cube: np.ndarray, q: float = config.LUPTON_Q,
           stretch: float = config.LUPTON_STRETCH, px: int = config.RENDER_PX) -> Image.Image:
    rgb = make_lupton_rgb(cube[2], cube[1], cube[0], Q=q, stretch=stretch)
    return _img_from_rgb(rgb, px)


def zoom(cube: np.ndarray, factor: float = 2.5, **kw) -> Image.Image:
    n = cube.shape[1]
    half = max(8, int(round(n / (2 * factor))))
    c = n // 2
    sub = cube[:, c - half:c + half + 1, c - half:c + half + 1]
    return lupton(sub, **kw)


# ---------------------------------------------------------------------------
# Honest residual: chi = (data - smooth elliptical-galaxy model) / sigma, SIGNED.
#
# The smooth deflector light is modelled per band as a sigma-clipped MEDIAN over elliptical
# annuli (the galaxy's own center/eps/PA): fast (~13 ms/band), always converges, leaves no
# isotropic-blur "butterfly" artifact, and -- being a median -- a tangential arc filling a
# minority of an annulus is rejected from the model and SURVIVES in the residual. The model
# is built in the seeing-convolved data (no deconvolution -> automatically PSF-matched, which
# matters since the cutouts carry no PSF). An OPT-IN photutils floating-isophote engine
# (engine="isophote") is higher fidelity on high-S/N images, but it converges <~15% on faint
# 101px grz cutouts (and is ~10-50x slower), so it is NOT the default here; when chosen it
# falls back to the median per band. The residual is divided by a robust per-band noise sigma
# (mad_std) and shown on a fixed +/-5 sigma RdBu diverging scale (red = unmodelled excess /
# candidate arc, blue = over-subtraction). Nothing is re-floored or clipped, so over-
# subtraction stays visible. See residual_legacy() for the deprecated (sign-destroying) high-pass.
# ---------------------------------------------------------------------------

# RdBu diverging knots (blue <- white -> red) for t in [-1, 0, +1]; 0 maps to white.
_RDBU_KNOTS = np.array([[33, 102, 172], [247, 247, 247], [178, 24, 43]], dtype=float)


def _estimate_shape(cube: np.ndarray) -> tuple[float, float, float, float]:
    """Center (x0,y0), ellipticity eps=1-b/a, and PA theta (rad) of the smooth galaxy from
    sigma-clipped second moments of the r+z luminance. Deterministic, sub-millisecond."""
    n = cube.shape[1]
    lum = (cube[1] + cube[2]).astype(float)
    bkg = float(np.ma.median(sigma_clip(lum, sigma=3.0, maxiters=3)))
    sig = float(mad_std(lum, ignore_nan=True)) or 1.0
    w = np.where(lum - bkg > 2.0 * sig, lum - bkg, 0.0)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return (n / 2.0, n / 2.0, 0.0, 0.0)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    x0 = float((w * xx).sum() / tot)
    y0 = float((w * yy).sum() / tot)
    dx, dy = xx - x0, yy - y0
    Ixx = (w * dx * dx).sum() / tot
    Iyy = (w * dy * dy).sum() / tot
    Ixy = (w * dx * dy).sum() / tot
    tr, det = Ixx + Iyy, Ixx * Iyy - Ixy * Ixy
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    a = max(tr / 2.0 + disc, 1e-6) ** 0.5
    b = max(tr / 2.0 - disc, 1e-6) ** 0.5
    eps = float(np.clip(1.0 - b / a, 0.0, 0.9))
    theta = float(0.5 * np.arctan2(2.0 * Ixy, Ixx - Iyy))
    return (x0, y0, eps, theta)


def _isophote_model(band: np.ndarray, x0: float, y0: float, eps: float,
                    theta: float) -> tuple[np.ndarray | None, bool]:
    """photutils elliptical-isophote model of the smooth light (PSF-matched by construction).
    Returns (model, True) or (None, False) if the fit raises or fails to converge."""
    try:
        from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
        n = band.shape[0]
        geom = EllipseGeometry(x0=x0, y0=y0, sma=max(6.0, n / 10.0),
                               eps=float(np.clip(eps, 0.05, 0.9)), pa=theta)
        iso = Ellipse(band.astype(float), geom).fit_image(
            integrmode="median", sclip=3.0, nclip=3, step=0.1, maxsma=0.7 * (n / 2.0))
        if iso is None or len(iso) < 3:
            return None, False
        model = build_ellipse_model(band.shape, iso)
        if not np.isfinite(model).all() or float(np.std(model)) == 0.0:
            return None, False
        return model.astype(float), True
    except Exception:
        return None, False


def _elliptical_median_model(band: np.ndarray, x0: float, y0: float, eps: float,
                             theta: float) -> np.ndarray:
    """Guaranteed-stable fallback: sigma-clipped median in elliptical annuli, interpolated
    back to a smooth 2D model. Carries the galaxy's eps/PA so it leaves no butterfly artifact."""
    n = band.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    dx, dy = xx - x0, yy - y0
    ct, st = np.cos(theta), np.sin(theta)
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    q = max(1.0 - eps, 0.1)
    r_ell = np.sqrt(xr * xr + (yr / q) ** 2)
    rmax = float(r_ell.max())
    nb = max(int(rmax) + 1, 4)
    edges = np.linspace(0.0, rmax + 1e-6, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(r_ell.ravel(), edges) - 1, 0, nb - 1)
    flat = band.astype(float).ravel()
    vals = np.full(nb, np.nan)
    for bn in range(nb):
        m = idx == bn
        if m.any():
            vals[bn] = float(np.ma.median(sigma_clip(flat[m], sigma=3.0, maxiters=2)))
    good = np.isfinite(vals)
    if good.sum() >= 2:
        vals = np.interp(centers, centers[good], vals[good])
    else:
        vals = np.nan_to_num(vals, nan=float(np.median(band)))
    return np.interp(r_ell.ravel(), centers, vals).reshape(band.shape).astype(float)


def _robust_sigma(res_band: np.ndarray) -> float:
    """Per-band noise sigma: mad_std of the 3-sigma-clipped residual (rejects arcs+core, so it
    measures sky), cross-checked against the frame border; take the larger if they disagree >1.5x."""
    s_res = float(mad_std(sigma_clip(res_band, sigma=3.0, maxiters=3), ignore_nan=True))
    n = res_band.shape[0]
    b = max(8, n // 12)
    border = np.concatenate([res_band[:b].ravel(), res_band[-b:].ravel(),
                             res_band[b:-b, :b].ravel(), res_band[b:-b, -b:].ravel()])
    s_bord = float(mad_std(border, ignore_nan=True))
    cand = [s for s in (s_res, s_bord) if np.isfinite(s) and s > 0]
    if not cand:
        return 1.0
    if len(cand) == 2 and max(cand) > 1.5 * min(cand):
        return max(cand)
    return max(s_res if (np.isfinite(s_res) and s_res > 0) else max(cand), 1e-6)


def _model_band(band: np.ndarray, shape: tuple[float, float, float, float],
                engine: str = "median") -> np.ndarray:
    """Smooth-light model. engine='median' (default): robust elliptical sigma-clipped median
    annuli -- always converges, ~13 ms/band. engine='isophote': photutils floating isophotes
    (higher fidelity on high-S/N images) with the median model as the per-band fallback; it
    converges <~15% on faint 101px grz cutouts, so 'median' is the default there."""
    if engine == "isophote":
        model, ok = _isophote_model(band, *shape)
        if ok:
            return model
    return _elliptical_median_model(band, *shape)


def _chi_band(band: np.ndarray, shape: tuple[float, float, float, float],
              engine: str = "median") -> np.ndarray:
    res = band.astype(float) - _model_band(band, shape, engine)
    return res / _robust_sigma(res)


def _diverging_rgb(chi: np.ndarray, lim: float = RESID_SIGMA_LIM) -> np.ndarray:
    """Signed chi -> RdBu RGB (uint8). asinh-compressed within +/-lim (linear inside ~+/-1.5
    sigma), 0 -> white. Red = positive (excess/arc), blue = negative (over-subtraction)."""
    a = 1.5
    t = np.arcsinh(np.clip(chi, -lim, lim) / a) / np.arcsinh(lim / a)   # [-1, 1]
    pos = np.clip(t + 1.0, 0.0, 2.0)                                    # [0, 2]
    lo = np.clip(np.floor(pos).astype(int), 0, 1)
    frac = (pos - lo)[..., None]
    rgb = (1.0 - frac) * _RDBU_KNOTS[lo] + frac * _RDBU_KNOTS[lo + 1]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def residual(cube: np.ndarray, lim: float = RESID_SIGMA_LIM, px: int = config.RENDER_PX,
             engine: str = "median", **kw) -> Image.Image:
    """Honest signed residual: per-band chi=(data-elliptical model)/sigma rendered as a
    g | r | z RdBu diverging montage. Red = unmodelled excess (candidate arc), blue =
    over-subtraction. Cross-band coherence (a real arc shows in g AND r) is the key
    real-vs-artifact discriminant, which is why the bands are shown separately."""
    shape = _estimate_shape(cube)
    tiles = [_diverging_rgb(_chi_band(cube[b], shape, engine), lim) for b in range(cube.shape[0])]
    sep = np.full((tiles[0].shape[0], 2, 3), 255, np.uint8)            # thin white divider
    montage = np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], axis=1)
    h, w = montage.shape[:2]
    return Image.fromarray(montage[::-1, :, :]).resize(
        (int(round(px * w / h)), px), Image.NEAREST)                   # flip + NEAREST, keep 3:1


def residual_chi_luminance(cube: np.ndarray, lim: float = RESID_SIGMA_LIM,
                           px: int = config.RENDER_PX, engine: str = "median", **kw) -> Image.Image:
    """Single square RdBu chi tile of the r+z luminance -- for gallery composites that need a
    1:1 panel. The g|r|z montage from residual() is the primary (grader) view."""
    lum = (cube[1] + cube[2]).astype(float)
    chi = _chi_band(lum, _estimate_shape(cube), engine)
    return _img_from_rgb(_diverging_rgb(chi, lim), px)


def residual_legacy(cube: np.ndarray, sigma: float = 3.0, **kw) -> Image.Image:
    """DEPRECATED pre-2026 residual: per-band Gaussian high-pass, re-floored to non-negative
    (DESTROYS sign), Lupton asinh. Not honest (hides over-subtraction, butterfly artifact on
    ellipticals, no noise units). Kept only to reproduce frozen runs; not in default VIEWS."""
    res = np.empty_like(cube)
    for b in range(3):
        smooth = gaussian_filter(cube[b], sigma=sigma)
        res[b] = cube[b] - smooth
    res -= res.min()
    return lupton(res, stretch=max(0.1, config.LUPTON_STRETCH * 0.4), **kw)


def highcontrast(cube: np.ndarray, **kw) -> Image.Image:
    return lupton(cube, q=20.0, stretch=0.1, **kw)


_RENDERERS = {"full": lupton, "zoom": zoom, "residual": residual, "highcontrast": highcontrast}


def render_views(cube: np.ndarray, views=VIEWS, px: int = config.RENDER_PX) -> dict[str, Image.Image]:
    out = {}
    for v in views:
        try:
            out[v] = _RENDERERS[v](cube, px=px)
        except Exception:
            continue
    return out


def png_bytes(img: Image.Image) -> bytes:
    bio = _io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def png_b64(img: Image.Image) -> str:
    return base64.b64encode(png_bytes(img)).decode("ascii")


def save_png(img: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path
