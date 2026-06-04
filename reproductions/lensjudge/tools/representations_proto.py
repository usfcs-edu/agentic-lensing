"""Tier-2 representation features — runs in .venvs/lens (photutils / skimage / sep).

Cleaner implementations of three representations whose Tier-1 (scipy-only) versions
were weak, to test whether better algorithms lift the easy-regime / gold AUC:
  R1  photutils.isophote.Ellipse lens-light model -> residual flux fraction (vs the
      Tier-1 azimuthal-median subtraction)
  R5  skimage.filters.frangi ridge "arcness" (vs the Tier-1 scipy Hessian)
  R6  sep.extract detection geometry -> count of TANGENTIALLY-oriented sources in the
      1-5" annulus (a true arc's major axis is ~perpendicular to the radius vector)

Kept OUT of the SDK process (heavy deps); invoked as a subprocess like quicklens_proto:
  /home/benson/.venvs/lens/bin/python representations_proto.py <cube1.fits> [cube2 ...]
prints one JSON line per cube ({"path":..., feature:...}).
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")
PIXSCALE = 0.262


def _load(path):
    with fits.open(path) as h:
        return np.nan_to_num(np.asarray(h[0].data, dtype=float))


def _centroid(img):
    n = img.shape[0]; c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    w = np.clip(img, 0, None) * (np.hypot(xx - c, yy - c) <= 0.35 * n)
    t = w.sum()
    if t <= 0:
        return c, c
    return float((w * yy).sum() / t), float((w * xx).sum() / t)


def _annulus(n, cy, cx, r0=3, r1=10):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - cx, yy - cy)
    return (r > r0) & (r <= r1), r


def iso_residual(img, cy, cx):
    """photutils isophote lens-light model subtracted -> residual flux fraction."""
    try:
        from photutils.isophote import Ellipse, EllipseGeometry
        from photutils.isophote import build_ellipse_model
        g = EllipseGeometry(x0=cx, y0=cy, sma=8.0, eps=0.2, pa=0.0)
        iso = Ellipse(img, geometry=g).fit_image(maxsma=img.shape[0] / 2, maxit=30)
        if len(iso) == 0:
            return None
        model = build_ellipse_model(img.shape, iso)
        res = img - model
        ann, _ = _annulus(img.shape[0], cy, cx)
        tot = np.abs(img[ann]).sum() + 1e-9
        return float(np.abs(res[ann]).sum() / tot), res
    except Exception:
        return None


def frangi_arcness(res, cy, cx):
    try:
        from skimage.filters import frangi
        r = res - res.min()
        f = frangi(r, sigmas=range(1, 4), black_ridges=False)
        ann, _ = _annulus(res.shape[0], cy, cx)
        return float(f[ann].max())
    except Exception:
        return None


def sep_geometry(img, cy, cx):
    """SEP detection geometry: count tangentially-oriented sources at 1-5\"."""
    try:
        import sep
        data = np.ascontiguousarray(img.astype(np.float32))
        bkg = sep.Background(data)
        obj = sep.extract(data - bkg.back(), 1.5, err=bkg.globalrms, minarea=4,
                          deblend_cont=0.005)
        if len(obj) == 0:
            return 0, 0, 0.0
        n_tang = 0; max_ellip = 0.0
        for o in obj:
            dx, dy = o["x"] - cx, o["y"] - cy
            r = np.hypot(dx, dy) * PIXSCALE
            if not (0.8 < r < 5.0):
                continue
            ellip = 1.0 - o["b"] / (o["a"] + 1e-9)
            max_ellip = max(max_ellip, float(ellip))
            radial_pa = np.arctan2(dy, dx)
            tang_pa = radial_pa + np.pi / 2
            d = abs(((o["theta"] - tang_pa + np.pi / 2) % np.pi) - np.pi / 2)
            if ellip > 0.3 and d < np.radians(30):
                n_tang += 1
        return int(len(obj)), int(n_tang), float(max_ellip)
    except Exception:
        return None


def tier2_features(cube):
    img = cube[1]                       # r-band
    cy, cx = _centroid(img)
    out = {}
    iso = iso_residual(img, cy, cx)
    if iso is not None:
        out["iso_residual_flux_fraction"], res = iso
        fa = frangi_arcness(res, cy, cx)
        if fa is not None:
            out["frangi_arcness"] = fa
    sg = sep_geometry(img, cy, cx)
    if sg is not None:
        out["sep_n_sources"], out["sep_n_tangential"], out["sep_max_ellip"] = sg
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(json.dumps({"path": p, **tier2_features(_load(p))}))
        except Exception as e:
            print(json.dumps({"path": p, "error": str(e)}))
