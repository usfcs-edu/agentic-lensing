"""Engineered lens-image representations (Tier-1: scipy/numpy/PIL, in-process).

Instead of having the model inspect the raw Lupton-RGB, these transforms make the
lensing signal explicit and emit scalar lensing-features that can be scored as
standalone discriminators (eval/run_representations.py) before any model spend:

  R1 lens-light subtraction   azimuthal-median galaxy model removed -> arc residual
  R2 polar/tangential (r,theta) a true tangential arc becomes a horizontal bar
  R3 180-degree symmetry resid. I - rot180(I): symmetric galaxy cancels, arc survives
  R4 blue-excess color isolation suppress the red lens, light up the blue source
  R5 DoG + Hessian "arcness"   Frangi-like ridge map: elongated curves, not blobs
  R6 per-band grayscale        faint blue arcs hidden under r/z in the composite

Heavy variants (photutils isophote, skimage frangi, SEP) live in
tools/representations_proto.py (.venvs/lens, subprocess) — never imported here, so the
SDK process stays dependency-light. ``compute_features(cube) -> dict`` returns every
scalar; the render_* functions return model-facing PIL images. All annulus-referenced
features are computed over a trial Einstein-radius bank (the cheap path avoids the 45 s
GIGA-Lens fit) and reported at the max-response radius.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from lensjudge import config
from lensjudge.common import render as _render

PIXSCALE = config.PIXSCALE
THETA_E_BANK_PX = tuple(round(a / PIXSCALE) for a in (0.8, 1.2, 1.6, 2.2, 3.0))  # ~3..11 px


# --------------------------------------------------------------------------- centroid
def centroid(cube: np.ndarray) -> tuple[float, float, float]:
    """Flux-weighted lens centroid (r-band, central region); geometric fallback.
    Returns (cy, cx, confidence in [0,1])."""
    img = np.clip(cube[1], 0, None)
    n = img.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    central = np.hypot(xx - c, yy - c) <= 0.35 * n
    w = img * central
    tot = float(w.sum())
    if tot <= 0:
        return c, c, 0.0
    cy = float((w * yy).sum() / tot)
    cx = float((w * xx).sum() / tot)
    off = np.hypot(cx - c, cy - c) / (0.5 * n)
    return cy, cx, float(max(0.0, 1.0 - off))


def _robust_sigma(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))) + 1e-9)


# --------------------------------------------------------------------------- R1 lens-light sub
def _azimuthal_median_model(img: np.ndarray, cy: float, cx: float) -> np.ndarray:
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    rint = np.clip(np.hypot(xx - cx, yy - cy).round().astype(int), 0, 2 * n)
    nb = int(rint.max()) + 1
    med = np.zeros(nb, dtype=float)
    flat_r, flat_v = rint.ravel(), img.ravel().astype(float)
    for rad in range(nb):
        m = flat_v[flat_r == rad]
        if m.size:
            med[rad] = np.median(m)
    return med[rint]


def lenssub(cube: np.ndarray, cy: float, cx: float) -> np.ndarray:
    """Per-band azimuthal-median (symmetric-galaxy) subtraction -> (3,N,N) residual."""
    res = np.empty(cube.shape, dtype=float)
    for b in range(3):
        res[b] = cube[b].astype(float) - _azimuthal_median_model(cube[b], cy, cx)
    return res


# --------------------------------------------------------------------------- R3 symmetry
def symmetry_residual(img: np.ndarray, cy: float, cx: float) -> np.ndarray:
    n = img.shape[0]
    c = (n - 1) / 2.0
    centered = ndi.shift(img.astype(float), (c - cy, c - cx), order=1, mode="constant")
    return centered - centered[::-1, ::-1]


# --------------------------------------------------------------------------- R2 polar
def to_polar(img: np.ndarray, cy: float, cx: float, nr: int = 64, nth: int = 120,
             rmax: float | None = None):
    n = img.shape[0]
    rmax = rmax if rmax is not None else 0.5 * n
    rs = np.linspace(0, rmax, nr)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    R, TH = np.meshgrid(rs, th, indexing="ij")
    X, Y = cx + R * np.cos(TH), cy + R * np.sin(TH)
    polar = ndi.map_coordinates(img.astype(float), [Y, X], order=1, mode="constant")
    return polar, rs, th


def _tangential_metrics(residual_band: np.ndarray, cy, cx, te_px):
    """In the theta_E annulus of the polar residual: tangential extent (deg),
    tangentiality (azimuthal/radial), and a counter-image parity proxy."""
    polar, rs, th = to_polar(residual_band, cy, cx)
    nth = polar.shape[1]
    lo, hi = 0.6 * te_px, 1.7 * te_px
    band = (rs >= lo) & (rs <= hi)
    if band.sum() < 2:
        return 0.0, 0.0, 0.0
    prof = polar[band].max(axis=0)              # per-theta peak flux in annulus
    sig = _robust_sigma(polar[band])
    above = prof > 3 * sig
    if not above.any():
        return 0.0, 0.0, 0.0
    # longest contiguous theta run (circular)
    ext = np.concatenate([above, above])
    best = run = 0
    for v in ext:
        run = run + 1 if v else 0
        best = max(best, run)
    best = min(best, nth)
    tang_deg = best * 360.0 / nth
    # tangentiality: azimuthal span vs radial span of the brightest theta column
    jmax = int(np.argmax(prof))
    col = polar[:, jmax]
    rad_span = (col > 3 * sig).sum() * (rs[1] - rs[0]) if (col > 3 * sig).any() else 1.0
    az_span = best * (2 * np.pi * rs[band].mean() / nth)
    tangentiality = float(az_span / (rad_span + 1e-6))
    # counter-image parity: corr of annulus profile with its 180-deg shift
    sh = np.roll(prof, nth // 2)
    pc = float(np.corrcoef(prof, sh)[0, 1]) if prof.std() > 0 else 0.0
    return float(tang_deg), tangentiality, max(0.0, pc)


# --------------------------------------------------------------------------- R4 blue excess
def blue_excess_map(cube: np.ndarray) -> np.ndarray:
    def nrm(b):
        b = b.astype(float) - np.median(b)
        return b / (np.percentile(np.abs(b), 99) + 1e-9)
    return nrm(cube[0]) - nrm(cube[2])          # g - z: + where bluer than the red lens


def _blue_metrics(cube, cy, cx, te_px):
    be = blue_excess_map(cube)
    n = be.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - cx, yy - cy)
    core = r <= 0.5 * te_px
    annulus = (r > 0.6 * te_px) & (r <= 1.7 * te_px)
    if annulus.sum() == 0 or core.sum() == 0:
        return 0.0, 0.0
    excess = float(be[annulus].mean() - be[core].mean())
    # fraction of the theta_E ring that is blue (full Einstein ring -> high)
    polar, rs, th = to_polar(be, cy, cx)
    band = (rs >= 0.6 * te_px) & (rs <= 1.7 * te_px)
    ring = polar[band].mean(axis=0) if band.any() else np.zeros(polar.shape[1])
    ring_blue_frac = float((ring > _robust_sigma(be)).mean())
    return excess, ring_blue_frac


# --------------------------------------------------------------------------- R5 arcness
def arcness_map(img: np.ndarray, scales=(1.0, 1.6, 2.6)) -> np.ndarray:
    """Frangi-like multiscale Hessian ridge for bright curvilinear (arc) structures."""
    img = img.astype(float)
    best = np.zeros_like(img)
    for s in scales:
        sm = ndi.gaussian_filter(img, s)
        Hxx = ndi.gaussian_filter(sm, s, order=(0, 2))
        Hyy = ndi.gaussian_filter(sm, s, order=(2, 0))
        Hxy = ndi.gaussian_filter(sm, s, order=(1, 1))
        disc = np.sqrt(np.maximum((Hxx - Hyy) ** 2 + 4 * Hxy ** 2, 0)) / 2.0
        tr = (Hxx + Hyy) / 2.0
        l1, l2 = tr + disc, tr - disc
        lam1 = np.where(np.abs(l1) >= np.abs(l2), l1, l2)   # larger |.|
        lam2 = np.where(np.abs(l1) >= np.abs(l2), l2, l1)   # smaller |.|
        Rb = np.abs(lam2) / (np.abs(lam1) + 1e-9)
        S = np.sqrt(l1 ** 2 + l2 ** 2)
        cval = 0.5 * float(S.max()) if S.max() > 0 else 1.0
        v = np.exp(-Rb ** 2 / 0.5) * (1 - np.exp(-S ** 2 / (2 * cval ** 2)))
        v[lam1 > 0] = 0                          # bright ridges only (lam1 < 0)
        best = np.maximum(best, v * s)
    return best


def _arcness_metrics(residual_band, cy, cx, te_px):
    amap = arcness_map(residual_band)
    n = amap.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - cx, yy - cy)
    annulus = (r > 0.5 * te_px) & (r <= 1.8 * te_px)
    if annulus.sum() == 0 or amap.max() <= 0:
        return 0.0, 0.0
    score = float(amap[annulus].max())
    # point_vs_arc: ridge response concentrated (point) vs spread (arc) -> spread is good
    thr = 0.3 * amap.max()
    hot = (amap > thr) & annulus
    spread = float(hot.sum())                   # # ridge pixels; arcs span more
    return score, spread


# --------------------------------------------------------------------------- features
def compute_features(cube: np.ndarray) -> dict:
    """All Tier-1 scalar lensing-features for one (3,N,N) grz cube."""
    cube = np.nan_to_num(np.asarray(cube, dtype=float))
    cy, cx, conf = centroid(cube)
    res = lenssub(cube, cy, cx)
    res_r = res[1]
    # residual flux fraction in a generic 1-2" annulus
    n = cube.shape[1]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - cx, yy - cy)
    ann = (r > 3) & (r <= 10)
    tot = float(np.abs(cube[1][ann]).sum()) + 1e-9
    residual_flux_fraction = float(np.abs(res_r[ann]).sum() / tot)
    # symmetry
    sym = symmetry_residual(cube[1], cy, cx)
    asym = float(np.abs(sym[ann]).sum() / tot)
    # blue-excess map: couple the geometric transforms to BLUE flux, so tangential /
    # counter-image / arcness features fire on a blue lensed arc, not on residual noise.
    bmap = np.clip(blue_excess_map(cube), 0, None)
    # over the theta_E bank: take the max-response radius for each metric
    tang, tangentiality, parity, arcness, point_arc, blue_exc, ring_blue, arc_rad = (
        0., 0., 0., 0., 0., -9., 0., 0.)
    b_tang, b_parity, b_arcness = 0., 0., 0.
    for te in THETA_E_BANK_PX:
        t, ty, pc = _tangential_metrics(res_r, cy, cx, te)
        a, sp = _arcness_metrics(res_r, cy, cx, te)
        be, rb = _blue_metrics(cube, cy, cx, te)
        bt, _, bpc = _tangential_metrics(bmap, cy, cx, te)
        ba, _ = _arcness_metrics(bmap, cy, cx, te)
        if t > tang:
            tang, arc_rad = t, te * PIXSCALE
        tangentiality = max(tangentiality, ty)
        parity = max(parity, pc)
        arcness = max(arcness, a)
        point_arc = max(point_arc, sp)
        if be > blue_exc:
            blue_exc = be
        ring_blue = max(ring_blue, rb)
        b_tang = max(b_tang, bt); b_parity = max(b_parity, bpc); b_arcness = max(b_arcness, ba)
    # per-band core/annulus ratios (R6 scalar part)
    core = r <= 4
    band_ratio = {}
    for i, bn in enumerate("grz"):
        cflux = float(np.clip(cube[i][core], 0, None).sum()) + 1e-9
        aflux = float(np.clip(cube[i][ann], 0, None).sum())
        band_ratio[f"annulus_core_ratio_{bn}"] = round(aflux / cflux, 4)
    return {
        "centroid_confidence": round(conf, 3),
        "residual_flux_fraction": round(residual_flux_fraction, 4),
        "asymmetry_index": round(asym, 4),
        "tangential_extent_deg": round(tang, 1),
        "tangentiality": round(tangentiality, 3),
        "counterimage_parity": round(parity, 3),
        "arc_radius_arcsec": round(arc_rad, 2),
        "arcness_score": round(arcness, 4),
        "arc_spread_px": round(point_arc, 1),
        "blue_excess_at_thetaE": round(blue_exc, 4),
        "ring_blue_fraction": round(ring_blue, 3),
        "blue_tangential_extent_deg": round(b_tang, 1),
        "blue_counterimage_parity": round(b_parity, 3),
        "blue_arcness_score": round(b_arcness, 4),
        **band_ratio,
    }


# --------------------------------------------------------------------------- renderers
def _gray_png(arr: np.ndarray, px: int = config.RENDER_PX) -> Image.Image:
    a = arr.astype(float)
    a = a - np.percentile(a, 1)
    a = np.clip(a, 0, None)
    hi = np.percentile(a, 99.5)
    a = np.arcsinh(a / (hi + 1e-9) * 10.0) / np.arcsinh(10.0)
    g = np.clip(a * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(g[::-1, :]).resize((px, px), Image.NEAREST)


def _signed_png(arr: np.ndarray, px: int = config.RENDER_PX) -> Image.Image:
    """Diverging map for signed residuals: blue<0, black~0, red>0."""
    a = arr.astype(float)
    s = np.percentile(np.abs(a), 99) + 1e-9
    t = np.clip(a / s, -1, 1)
    r = np.clip(t, 0, 1); b = np.clip(-t, 0, 1)
    rgb = np.stack([r, np.zeros_like(t), b], axis=-1)
    g = (rgb * 255).astype(np.uint8)
    return Image.fromarray(g[::-1, :, :]).resize((px, px), Image.NEAREST)


def render_lenssub(cube, **kw):
    cy, cx, _ = centroid(cube)
    res = lenssub(cube, cy, cx)
    res = res - res.min()
    return _render.lupton(res, stretch=0.2)


def render_polar(cube, **kw):
    cy, cx, _ = centroid(cube)
    res_r = lenssub(cube, cy, cx)[1]
    polar, rs, th = to_polar(res_r, cy, cx)
    return _gray_png(polar)            # rows=radius, cols=theta; an arc = horizontal bar


def render_symmetry(cube, **kw):
    cy, cx, _ = centroid(cube)
    return _signed_png(symmetry_residual(cube[1], cy, cx))


def render_coloriso(cube, **kw):
    return _signed_png(blue_excess_map(cube))


def render_arcness(cube, **kw):
    cy, cx, _ = centroid(cube)
    return _gray_png(arcness_map(lenssub(cube, cy, cx)[1]))


def _gray_band(cube, i, **kw):
    return _gray_png(cube[i])


REPR_RENDERERS = {
    "lenssub": render_lenssub,
    "polar": render_polar,
    "symmetry": render_symmetry,
    "color_iso": render_coloriso,
    "arcness": render_arcness,
    "gray_g": lambda c, **k: _gray_band(c, 0),
    "gray_r": lambda c, **k: _gray_band(c, 1),
    "gray_z": lambda c, **k: _gray_band(c, 2),
}

REPR_VIEW_DESC = {
    "lenssub": "lens galaxy removed (azimuthal-median): low-surface-brightness arcs/"
               "counter-images stand out on a flat background.",
    "polar": "polar (r,theta) of the residual: RADIUS is the vertical axis, ANGLE the "
             "horizontal. A real tangential arc is a HORIZONTAL bar at fixed radius "
             "spanning a wide angle; a spiral arm/companion is not.",
    "symmetry": "I - rot180(I) about the lens center: a symmetric elliptical/star cancels "
                "(black); an off-center arc or a counter-image pair lights up (red/blue).",
    "color_iso": "blue-excess map (g-z): the red lens is suppressed; blue lensed-source "
                 "flux is bright. A blue arc/ring around the centre is the signal.",
    "arcness": "Frangi ridge 'arcness' of the residual: bright where flux forms thin "
               "ELONGATED curves (arcs), dark on round blobs/points/stars.",
    "gray_g": "g-band only (bluest): lensed sources are brightest here, lens galaxy faint.",
    "gray_r": "r-band only.",
    "gray_z": "z-band only (reddest): dominated by the lens galaxy.",
}


def render_view(cube: np.ndarray, name: str, px: int = config.RENDER_PX) -> Image.Image | None:
    fn = REPR_RENDERERS.get(name)
    return fn(cube, px=px) if fn else None
