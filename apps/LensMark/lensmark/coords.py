"""Coordinate transforms - pure functions, no PIL, mirrored one-to-one in ``frontend/src/coords.ts``.

Canonical geometry in a LensMark file is **normalized** ``(u, v)``: ``u`` in [0, 1] left -> right and
``v`` in [0, 1] top -> bottom of the displayed image (origin at the top-left corner). All physical
sizes are in **arcsec**. Pixels are derived, never stored, so the same annotation renders at any size.

Conventions
-----------
* display px      x = u * W                 y = v * H
* pixel scale     ps = cutout_arcsec / W     (arcsec per display pixel; the cutout width is the reference)
* sky offsets     dx = (u - 0.5) * W * ps   (+ right)      dy = (0.5 - v) * H * ps   (+ up)
                  dE = -dx if east_left else dx           dN = dy if north_up else -dy
* polar           r = hypot(dE, dN)          PA = atan2(dE, dN) in degrees, from North (0) through
                  East (90) - the same convention as reproductions/lensjudge/golden/annotate.py and the
                  JWST pipeline (East LEFT on screen when east_left is true).
* FITS / DS9      1-based, y increasing upward:  x_fits = u * W + 0.5
                  y_fits = (1 - v) * H + 0.5 if array_origin == "upper" (PNG row 0 is the array's LAST
                  row, i.e. the PNG was flipped for display - what lensjudge's render.py does) else
                  v * H + 0.5 (PNG rows are array rows).
* screen angle    0 = +x (right), 90 = up, counter-clockwise (the 21_annotate.py arrow convention).
"""
from __future__ import annotations

import math
from typing import Literal

ArrayOrigin = Literal["upper", "lower"]


def uv_to_px(u: float, v: float, W: int, H: int) -> tuple[float, float]:
    return u * W, v * H


def px_to_uv(x: float, y: float, W: int, H: int) -> tuple[float, float]:
    return x / W, y / H


def pixel_scale(W: int, cutout_arcsec: float) -> float:
    """arcsec per display pixel."""
    return cutout_arcsec / W


def arcsec_to_px(a: float, W: int, cutout_arcsec: float) -> float:
    return a * W / cutout_arcsec


def px_to_arcsec(p: float, W: int, cutout_arcsec: float) -> float:
    return p * cutout_arcsec / W


def uv_to_fits(u: float, v: float, W: int, H: int, array_origin: ArrayOrigin = "upper") -> tuple[float, float]:
    x = u * W + 0.5
    y = (1.0 - v) * H + 0.5 if array_origin == "upper" else v * H + 0.5
    return x, y


def fits_to_uv(x: float, y: float, W: int, H: int, array_origin: ArrayOrigin = "upper") -> tuple[float, float]:
    u = (x - 0.5) / W
    v = 1.0 - (y - 0.5) / H if array_origin == "upper" else (y - 0.5) / H
    return u, v


def uv_to_dEdN(u: float, v: float, W: int, H: int, cutout_arcsec: float,
               north_up: bool = True, east_left: bool = True) -> tuple[float, float]:
    ps = pixel_scale(W, cutout_arcsec)
    dx = (u - 0.5) * W * ps
    dy = (0.5 - v) * H * ps
    dE = -dx if east_left else dx
    dN = dy if north_up else -dy
    return dE, dN


def dEdN_to_uv(dE: float, dN: float, W: int, H: int, cutout_arcsec: float,
               north_up: bool = True, east_left: bool = True) -> tuple[float, float]:
    ps = pixel_scale(W, cutout_arcsec)
    dx = -dE if east_left else dE
    dy = dN if north_up else -dN
    u = dx / (W * ps) + 0.5
    v = 0.5 - dy / (H * ps)
    return u, v


def dEdN_to_rpa(dE: float, dN: float) -> tuple[float, float]:
    """(r arcsec, PA deg from North through East in [0, 360))."""
    r = math.hypot(dE, dN)
    pa = math.degrees(math.atan2(dE, dN)) % 360.0
    return r, pa


def rpa_to_dEdN(r: float, pa_deg: float) -> tuple[float, float]:
    p = math.radians(pa_deg)
    return r * math.sin(p), r * math.cos(p)


def uv_to_rpa(u: float, v: float, W: int, H: int, cutout_arcsec: float, cu: float = 0.5, cv: float = 0.5,
              north_up: bool = True, east_left: bool = True) -> tuple[float, float]:
    """Polar coordinates of (u, v) about an arbitrary centre (cu, cv) (default: image centre)."""
    dE, dN = uv_to_dEdN(u, v, W, H, cutout_arcsec, north_up, east_left)
    cE, cN = uv_to_dEdN(cu, cv, W, H, cutout_arcsec, north_up, east_left)
    return dEdN_to_rpa(dE - cE, dN - cN)


def screen_angle_deg(x0: float, y0: float, x1: float, y1: float) -> float:
    """Angle of the vector (x0,y0)->(x1,y1) in screen convention: 0 = right, 90 = up, CCW, [0, 360)."""
    return math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 360.0


def dist_uv(a: tuple[float, float] | list[float], b: tuple[float, float] | list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist_arcsec(a, b, W: int, H: int, cutout_arcsec: float) -> float:
    """Angular separation between two (u, v) points."""
    ps = pixel_scale(W, cutout_arcsec)
    return math.hypot((a[0] - b[0]) * W * ps, (a[1] - b[1]) * H * ps)
