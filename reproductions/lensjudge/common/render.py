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
from astropy.visualization import make_lupton_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter

from lensjudge import config

VIEWS = ("full", "zoom", "residual", "highcontrast")


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


def residual(cube: np.ndarray, sigma: float = 3.0, **kw) -> Image.Image:
    """Subtract a per-band Gaussian-smoothed model to surface compact/arc features."""
    res = np.empty_like(cube)
    for b in range(3):
        smooth = gaussian_filter(cube[b], sigma=sigma)
        res[b] = cube[b] - smooth
    # re-zero the floor so make_lupton_rgb's asinh behaves
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
