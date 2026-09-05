#!/usr/bin/env python3
"""Generate the AstroMark reference cutout — a synthetic strong-lens field.

    python make_reference_cutout.py [--out reference/] [--seed 20260903]

Why synthetic: the nine Hubble cutouts the project has been annotating are STScI-embargoed, so no
render derived from them can ship with a public standard. This cutout carries no embargo, is
reproducible from a seed, and can be distributed with the spec forever as the canonical worked
example.

Why THIS field: a notation is only tested by the cases that strain it. The scene deliberately
contains every feature class the lens profile must be able to express, including the three that
today's notation cannot say at all:

  deflector            elliptical, de Vaucouleurs, with a DUST LANE crossing it
  second deflector     a smaller elliptical inside the field -> "lens mass, not a lensed image"
  satellite            bound to the deflector, sitting near the ring -> never masked
  main arc             tangentially stretched, blue, at theta_E, with a bright KNOT in it
  counter-image        opposite the arc but FARTHER from the deflector than it (the case that
                       gets wrongly rejected on radius)
  spiral-arm mimic     a face-on spiral in the field whose arm looks like an arc -> the negative
                       feature, the one a notation must be able to mark as NOT lensing
  field galaxies       ordinary, to be masked
  bright field galaxy  close enough that its light reaches the system -> modelled, not masked
  star                 with diffraction spikes

Output is a plain RGB PNG at 403x403 (16.0 arcsec across, 0.03970 arcsec/px) plus a JSON sidecar
recording the true positions of every component, so the example annotation can be authored against
ground truth rather than by eye.

Deterministic: same seed -> byte-identical PNG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

W = H = 403
CUTOUT_ARCSEC = 16.0
PIXEL_SCALE = CUTOUT_ARCSEC / W          # 0.0397 "/px
THETA_E_ARCSEC = 1.45                     # the truth the annotation should recover

# Deflector centre in normalized (u, v) — u right, v down, origin top-left.
DEFLECTOR_UV = (0.500, 0.510)


def as_px(uv: tuple[float, float]) -> tuple[float, float]:
    return uv[0] * W, uv[1] * H


def arcsec(px: float) -> float:
    return px * PIXEL_SCALE


def px(a: float) -> float:
    return a / PIXEL_SCALE


def _grids() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    return x + 0.5, y + 0.5


def sersic(x, y, cx, cy, flux, r_e_px, n, q=1.0, pa_deg=0.0):
    """Elliptical Sersic profile. pa measured from +x axis, counter-clockwise on screen."""
    t = np.deg2rad(pa_deg)
    dx, dy = x - cx, y - cy
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    r = np.sqrt(xr**2 + (yr / max(q, 1e-3)) ** 2)
    b_n = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    prof = np.exp(-b_n * ((r / max(r_e_px, 1e-3)) ** (1.0 / n) - 1.0))
    return flux * prof


def gaussian_blob(x, y, cx, cy, flux, sigma_px, q=1.0, pa_deg=0.0):
    t = np.deg2rad(pa_deg)
    dx, dy = x - cx, y - cy
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    r2 = xr**2 + (yr / max(q, 1e-3)) ** 2
    return flux * np.exp(-0.5 * r2 / sigma_px**2)


def tangential_arc(x, y, cx, cy, flux, r_px, width_px, pa_deg, span_deg):
    """A tangentially stretched arc: a Gaussian ridge in radius, apodised in azimuth.

    This is the shape lensing actually makes — curvature centred on the deflector — which is why
    the notation's 'tangential' idea has physical meaning.
    """
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    th = np.degrees(np.arctan2(dy, dx))
    dth = (th - pa_deg + 180.0) % 360.0 - 180.0
    radial = np.exp(-0.5 * ((r - r_px) / width_px) ** 2)
    # cos^2 apodisation across the span, zero outside it
    frac = np.clip(np.abs(dth) / (span_deg / 2.0), 0.0, 1.0)
    azim = np.cos(frac * np.pi / 2.0) ** 2
    return flux * radial * azim


def spiral_arm(x, y, cx, cy, flux, r0_px, pitch, width_px, pa_deg, span_deg, turns=1.0):
    """A logarithmic spiral arm — the classic arc mimic.

    Unlike a lensed arc its centre of curvature drifts, and it is attached to its own galaxy
    rather than wrapped around a foreground mass. Both facts are the reasons an expert rejects it,
    and both are visible here.
    """
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    th = np.degrees(np.arctan2(dy, dx))
    out = np.zeros_like(x)
    n_steps = 240
    for i in range(n_steps):
        f = i / (n_steps - 1)
        ang = pa_deg + (f - 0.5) * span_deg * turns
        rr = r0_px * np.exp(pitch * np.deg2rad(ang - pa_deg))
        dth = (th - ang + 180.0) % 360.0 - 180.0
        seg = np.exp(-0.5 * ((r - rr) / width_px) ** 2) * np.exp(-0.5 * (dth / 3.0) ** 2)
        out = np.maximum(out, seg)
    return flux * out


def dust_lane(x, y, cx, cy, depth, length_px, width_px, pa_deg):
    """A multiplicative absorption band across the deflector."""
    t = np.deg2rad(pa_deg)
    dx, dy = x - cx, y - cy
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    band = np.exp(-0.5 * (yr / width_px) ** 2) * np.exp(-0.5 * (xr / length_px) ** 2)
    return 1.0 - depth * band


def star(x, y, cx, cy, flux, sigma_px, spike_len_px, spike_w_px):
    core = gaussian_blob(x, y, cx, cy, flux, sigma_px)
    dx, dy = x - cx, y - cy
    horiz = np.exp(-0.5 * (dy / spike_w_px) ** 2) * np.exp(-0.5 * (dx / spike_len_px) ** 2)
    vert = np.exp(-0.5 * (dx / spike_w_px) ** 2) * np.exp(-0.5 * (dy / spike_len_px) ** 2)
    return core + flux * 0.16 * (horiz + vert)


# (name, role it plays in the annotation, u, v) — written to the sidecar as ground truth.
TRUTH: list[dict] = []


def note(name: str, role: str, cx: float, cy: float, **extra) -> None:
    TRUTH.append(dict(name=name, role=role, u=round(cx / W, 4), v=round(cy / H, 4), **extra))


def build(seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    x, y = _grids()
    r_chan = np.zeros((H, W))
    g_chan = np.zeros((H, W))
    b_chan = np.zeros((H, W))

    dcx, dcy = as_px(DEFLECTOR_UV)
    theta_e_px = px(THETA_E_ARCSEC)

    # --- the deflector: a red early-type, de Vaucouleurs ---------------------------------------
    defl = sersic(x, y, dcx, dcy, 0.40, px(0.62), n=3.2, q=0.70, pa_deg=28.0)
    r_chan += 1.00 * defl
    g_chan += 0.62 * defl
    b_chan += 0.30 * defl
    note("deflector", "deflector", dcx, dcy, theta_e_arcsec=THETA_E_ARCSEC)

    # a dust lane crossing it — the feature that makes a real lens look like a disk galaxy
    lane = dust_lane(x, y, dcx, dcy, depth=0.58, length_px=px(0.92), width_px=px(0.098), pa_deg=28.0)
    r_chan *= lane
    g_chan *= lane
    b_chan *= lane
    note("dust lane", "dust_lane", dcx, dcy, pa_deg=28.0)

    # --- second deflector: smaller elliptical, still lens mass ----------------------------------
    s2x, s2y = dcx - px(2.35), dcy + px(1.30)
    d2 = sersic(x, y, s2x, s2y, 0.34, px(0.52), n=3.0, q=0.80, pa_deg=-15.0)
    r_chan += 1.00 * d2
    g_chan += 0.60 * d2
    b_chan += 0.31 * d2
    note("second deflector", "second_deflector", s2x, s2y)

    # --- satellite bound to the deflector, sitting near the ring --------------------------------
    satx = dcx + px(2.02) * np.cos(np.deg2rad(191.0))
    saty = dcy + px(2.02) * np.sin(np.deg2rad(191.0))
    sat = sersic(x, y, satx, saty, 0.24, px(0.22), n=2.0, q=0.9)
    r_chan += 1.00 * sat
    g_chan += 0.63 * sat
    b_chan += 0.34 * sat
    note("satellite", "satellite", satx, saty)

    # --- the main arc: tangential, blue, at theta_E, with a knot ---------------------------------
    arc = tangential_arc(x, y, dcx, dcy, 0.90, theta_e_px, px(0.155), pa_deg=118.0, span_deg=116.0)
    r_chan += 0.34 * arc
    g_chan += 0.72 * arc
    b_chan += 1.00 * arc
    ax = dcx + theta_e_px * np.cos(np.deg2rad(118.0))
    ay = dcy + theta_e_px * np.sin(np.deg2rad(118.0))
    note("main arc", "arc", ax, ay, source="S1")

    knx = dcx + theta_e_px * np.cos(np.deg2rad(96.0))
    kny = dcy + theta_e_px * np.sin(np.deg2rad(96.0))
    knot = gaussian_blob(x, y, knx, kny, 0.40, px(0.12))
    r_chan += 0.36 * knot
    g_chan += 0.74 * knot
    b_chan += 1.00 * knot
    note("arc knot", "knot", knx, kny, source="S1")

    # --- counter-image: opposite the arc, but FARTHER out than it --------------------------------
    # 1.72" vs theta_E 1.45" — lensing allows this, and a rule that rejects on radius gets it wrong.
    ci_r = px(1.72)
    cix = dcx + ci_r * np.cos(np.deg2rad(118.0 - 180.0))
    ciy = dcy + ci_r * np.sin(np.deg2rad(118.0 - 180.0))
    ci = gaussian_blob(x, y, cix, ciy, 0.38, px(0.16), q=0.62, pa_deg=28.0)
    r_chan += 0.34 * ci
    g_chan += 0.72 * ci
    b_chan += 1.00 * ci     # same colour as the arc: the same-source test passes
    note("counter-image", "counter_image", cix, ciy, source="S1",
         radius_arcsec=1.72, note="farther from the deflector than the arc")

    # --- the mimic: a face-on spiral whose arm reads as an arc -----------------------------------
    spx, spy = dcx + px(4.35), dcy - px(3.95)
    bulge = sersic(x, y, spx, spy, 0.20, px(0.22), n=2.0, q=0.94)
    disk = sersic(x, y, spx, spy, 0.085, px(0.95), n=1.0, q=0.88, pa_deg=35.0)
    r_chan += 0.85 * bulge + 0.62 * disk
    g_chan += 0.74 * bulge + 0.64 * disk
    b_chan += 0.60 * bulge + 0.70 * disk
    arm = spiral_arm(x, y, spx, spy, 0.20, px(0.34), pitch=0.30,
                     width_px=px(0.105), pa_deg=205.0, span_deg=240.0, turns=1.0)
    arm += spiral_arm(x, y, spx, spy, 0.185, px(0.34), pitch=0.30,
                      width_px=px(0.105), pa_deg=25.0, span_deg=240.0, turns=1.0)
    r_chan += 0.42 * arm
    g_chan += 0.66 * arm
    b_chan += 0.88 * arm
    amx = spx + px(0.60) * np.cos(np.deg2rad(150.0))
    amy = spy + px(0.60) * np.sin(np.deg2rad(150.0))
    note("spiral arm (arc mimic)", "NEGATIVE:spiral_arm", amx, amy,
         note="curvature centred on its own bulge, not on the deflector")

    # --- field galaxies: ordinary, to be masked --------------------------------------------------
    field = [
        (dcx - px(5.10), dcy - px(4.30), 0.20, 0.42, 0.75, -40.0, "field galaxy 1"),
        (dcx + px(5.60), dcy + px(4.55), 0.16, 0.36, 0.60, 20.0, "field galaxy 2"),
        (dcx - px(4.05), dcy + px(5.35), 0.13, 0.30, 0.85, 70.0, "field galaxy 3"),
        (dcx + px(1.10), dcy + px(6.10), 0.11, 0.26, 0.70, -10.0, "field galaxy 4"),
    ]
    for fx, fy, flux, re_a, q, pa, name in field:
        gal = sersic(x, y, fx, fy, flux, px(re_a), n=1.5, q=q, pa_deg=pa)
        r_chan += 0.88 * gal
        g_chan += 0.70 * gal
        b_chan += 0.58 * gal
        note(name, "field_galaxy", fx, fy, mask_radius_arcsec=round(re_a * 2.6, 2))

    # a bright close one: its light reaches the system, so it is modelled rather than cut out
    bfx, bfy = dcx - px(3.05), dcy - px(2.65)
    bright = sersic(x, y, bfx, bfy, 0.17, px(0.44), n=1.8, q=0.66, pa_deg=55.0)
    r_chan += 0.94 * bright
    g_chan += 0.74 * bright
    b_chan += 0.56 * bright
    note("nearby galaxy", "field_galaxy:model", bfx, bfy, mask_radius_arcsec=1.15,
         treatment="model", note="light reaches the lens system")

    # --- a star with diffraction spikes ----------------------------------------------------------
    stx, sty = dcx + px(3.40), dcy + px(2.05)
    st = star(x, y, stx, sty, 0.85, px(0.11), px(1.25), px(0.045))
    r_chan += 0.95 * st
    g_chan += 0.95 * st
    b_chan += 1.00 * st
    note("star", "star", stx, sty, mask_radius_arcsec=0.55)

    # --- sky: gradient, noise, seeing -------------------------------------------------------------
    stack = np.stack([r_chan, g_chan, b_chan])
    # a mild PSF so nothing is a hard edge
    stack = np.stack([gaussian_blur(c, 1.15) for c in stack])

    grad = 0.006 * ((x / W) * 0.6 + (1.0 - y / H) * 0.4)
    stack += grad[None, :, :]
    stack += rng.normal(0.0, 0.0042, size=stack.shape)

    # asinh stretch — what an astronomer actually looks at
    q_stretch = 7.0
    stack = np.arcsinh(np.clip(stack, 0.0, None) * q_stretch) / np.arcsinh(q_stretch)
    stack = np.clip(stack * 1.02, 0.0, 1.0)

    rgb = (stack.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return rgb, TRUTH


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with numpy only (no scipy dependency)."""
    rad = max(1, int(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((0, 0), (rad, rad)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    pad = np.pad(out, ((rad, rad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, pad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "reference")
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rgb, truth = build(args.seed)
    img = Image.fromarray(rgb, mode="RGB")
    dest = args.out / "astromark-ref.png"
    # strip metadata so the bytes are a pure function of the seed
    clean = Image.frombytes("RGB", img.size, img.tobytes())
    clean.save(dest, optimize=False)

    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    side = {
        "file": dest.name,
        "sha256": sha,
        "seed": args.seed,
        "width": W, "height": H,
        "cutout_arcsec": CUTOUT_ARCSEC,
        "pixel_scale_arcsec": round(PIXEL_SCALE, 8),
        "north_up": True, "east_left": True,
        "theta_e_true_arcsec": THETA_E_ARCSEC,
        "deflector_uv": list(DEFLECTOR_UV),
        "embargo": None,
        "note": ("Synthetic. Free of any embargo; may ship with the standard. Positions below are "
                 "ground truth, so an example annotation can be authored against them."),
        "components": truth,
    }
    (args.out / "astromark-ref.truth.json").write_text(
        json.dumps(side, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {dest} ({W}x{H}, sha256 {sha[:16]}…)")
    print(f"wrote {args.out / 'astromark-ref.truth.json'} — {len(truth)} components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
