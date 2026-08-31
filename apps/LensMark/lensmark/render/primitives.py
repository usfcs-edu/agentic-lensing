"""Pure drawing primitives for the canonical renderer (PIL only, deterministic).

Every function takes pixel quantities at the caller's (supersampled) scale and, where it draws, an
``ImageDraw``; nothing here reads a LensMark file or a style table. Ported helpers cite their origin:

* ``text_size`` / ``halo_text`` / ``dashed_polyline`` / ``place_label`` / ``clamp_label`` / ``wrap_parts``
  from reproductions/lensjudge/golden/annotate.py (:273, :281, :352, :303, :339, :554), generalised from
  the 240-px panel (``PX``) to an arbitrary canvas ``(W, H)``;
* ``arrow_geometry`` / ``dotted_circle`` / ``approach_angle`` from
  ~/sync/research/jwst-strong-lens-search/scripts/21_annotate.py (:54, :76, :84) - the tip stops ``tip_gap``
  short of the feature, the barbs become a filled triangle, dots are placed by arc length with the phase
  at screen angle 0 (the rightmost point) walking clockwise on screen.

Fonts come only from ``lensmark/render/fonts`` (``load_font``) - never a system font: the fallback chain in
annotate.py:251-268 is exactly the drift hole a golden image must not have.

Screen conventions: image coordinates (x right, y down). "Clockwise on screen" is increasing PIL angle.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Sequence

from PIL import ImageDraw, ImageFont

from .. import config

Point = tuple[float, float]
Box = tuple[float, float, float, float]          # x0, y0, x1, y1
RGBA = tuple[int, int, int, int]

CORNERS: tuple[str, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")


# ----------------------------------------------------------------------------- colours / vectors
def hex_to_rgba(s: str, alpha: int = 255) -> RGBA:
    """``#RRGGBB`` or ``#RRGGBBAA`` -> (r, g, b, a)."""
    h = s.strip().lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha
    if len(h) == 8:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
    raise ValueError(f"bad colour {s!r}")


def unit(dx: float, dy: float, fallback: Point = (0.0, -1.0)) -> Point:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else fallback


# ----------------------------------------------------------------------------- fonts / text
@lru_cache(maxsize=None)
def load_font(name: str, size_px: float) -> ImageFont.FreeTypeFont:
    """A font from ``lensmark/render/fonts`` only, by path (no system-font fallback)."""
    path = config.FONT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"font {name!r} is not vendored in {config.FONT_DIR}")
    return ImageFont.truetype(str(path), size=max(1.0, float(size_px)))


def text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[float, float]:
    """(width, line height) of ``text`` - the height is the font's ascent+descent so every label of one
    size gets the same box (annotate.py:273 measured the glyph bbox; a line box is stable across strings)."""
    ascent, descent = font.getmetrics()
    return float(font.getlength(text)), float(ascent + descent)


def halo_text(draw: ImageDraw.ImageDraw, xy: Point, text: str, font: ImageFont.FreeTypeFont, fill: RGBA,
              halo_fill: Optional[RGBA] = None, halo_w: float = 0.0, anchor: str = "mm") -> None:
    """Anchored text with a dark halo (annotate.py:281 ``_text``, stroke width generalised)."""
    kw = {}
    if halo_fill is not None and halo_w > 0:
        kw = {"stroke_width": int(round(halo_w)), "stroke_fill": halo_fill}
    draw.text(xy, text, font=font, fill=fill, anchor=anchor, **kw)


RichPart = tuple[str, float, float]      # (text, size multiplier, baseline shift as a multiple of size; + = down)


def theta_parts(text: str) -> list[RichPart]:
    """Split a θ_E label into rich parts: ``θ`` + subscript ``E`` + the rest; plain text otherwise."""
    for prefix in ("θ_E", "theta_E", "θE"):
        if text.startswith(prefix):
            return [("θ", 1.0, 0.0), ("E", 0.62, 0.28), (text[len(prefix):], 1.0, 0.0)]
    return [(text, 1.0, 0.0)]


def rich_text_size(parts: Sequence[RichPart], font_name: str, size: float) -> tuple[float, float]:
    """(total width, line height of the base font) of a rich-text run."""
    w = 0.0
    for t, mult, _ in parts:
        if t:
            w += text_size(t, load_font(font_name, size * mult))[0]
    return w, text_size("", load_font(font_name, size))[1]


def rich_halo_text(draw: ImageDraw.ImageDraw, center: Point, parts: Sequence[RichPart], font_name: str, size: float,
                   fill: RGBA, halo_fill: Optional[RGBA] = None, halo_w: float = 0.0) -> None:
    """Rich text (per-part size / baseline shift) centred on ``center`` like an ``mm``-anchored string."""
    base = load_font(font_name, size)
    ascent, descent = base.getmetrics()
    total_w, _ = rich_text_size(parts, font_name, size)
    x = center[0] - total_w / 2.0
    baseline = center[1] + (ascent - descent) / 2.0
    for t, mult, shift in parts:
        if not t:
            continue
        f = load_font(font_name, size * mult)
        halo_text(draw, (x, baseline + shift * size), t, f, fill, halo_fill, halo_w, anchor="ls")
        x += text_size(t, f)[0]


def wrap_parts(parts: Sequence[str], font: ImageFont.FreeTypeFont, max_w: float, sep: str = " · ") -> list[str]:
    """Greedily join ``parts`` with ``sep`` into rows no wider than ``max_w`` px (annotate.py:554). A single
    part wider than a row is split on ';' and then on spaces; nothing is ever dropped."""
    def width(s: str) -> float:
        return text_size(s, font)[0]

    def pieces(part: str) -> list[str]:
        if width(part) <= max_w:
            return [part]
        out: list[str] = []
        for chunk in part.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk = chunk if chunk == part.strip() else chunk + ";"
            if width(chunk) <= max_w:
                out.append(chunk)
                continue
            words, cur = chunk.split(), ""
            for wd in words:
                cand = (cur + " " + wd).strip()
                if cur and width(cand) > max_w:
                    out.append(cur)
                    cur = wd
                else:
                    cur = cand
            if cur:
                out.append(cur)
        if out and out[-1].endswith(";"):
            out[-1] = out[-1][:-1]
        return out or [part]

    rows: list[str] = []
    cur = ""
    for part in parts:
        for piece in pieces(str(part)):
            cand = piece if not cur else cur + sep + piece
            if cur and width(cand) > max_w:
                rows.append(cur)
                cur = piece
            else:
                cur = cand
    if cur:
        rows.append(cur)
    return rows


# ----------------------------------------------------------------------------- strokes
def circle_points(cx: float, cy: float, r: float, step_px: float = 1.5, start_deg: float = 0.0) -> list[Point]:
    """Closed polyline around a circle, starting at screen angle ``start_deg`` (0 = rightmost point) and
    walking clockwise on screen; chords are at most ~``step_px`` long."""
    n = max(24, int(math.ceil(2.0 * math.pi * max(r, 0.5) / step_px)))
    a0 = math.radians(start_deg)
    return [(cx + r * math.cos(a0 + 2.0 * math.pi * k / n), cy + r * math.sin(a0 + 2.0 * math.pi * k / n))
            for k in range(n + 1)]


def stroke_polyline(draw: ImageDraw.ImageDraw, pts: Sequence[Point], fill: RGBA, width: float) -> None:
    if len(pts) >= 2:
        draw.line([tuple(p) for p in pts], fill=fill, width=max(1, int(round(width))), joint="curve")


def dashed_polyline(draw: ImageDraw.ImageDraw, pts: Sequence[Point], fill: RGBA, width: float,
                    dash: float, gap: float) -> None:
    """``dash`` px on / ``gap`` px off along the polyline by arc length, starting "on" at ``pts[0]``
    (annotate.py:352; each on-run is drawn as one joined polyline instead of per-chord segments)."""
    if len(pts) < 2 or dash <= 0:
        return
    on, left = True, float(dash)
    run: list[Point] = [tuple(pts[0])]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        t = 0.0
        while seg - t > 1e-9:
            step = min(left, seg - t)
            b = (x0 + (x1 - x0) * (t + step) / seg, y0 + (y1 - y0) * (t + step) / seg)
            if on:
                run.append(b)
            t += step
            left -= step
            if left <= 1e-9:
                if on:
                    stroke_polyline(draw, run, fill, width)
                    run = []
                else:
                    run = [b]
                on = not on
                left = float(dash if on else gap)
    if on:
        stroke_polyline(draw, run, fill, width)


def dotted_polyline(draw: ImageDraw.ImageDraw, pts: Sequence[Point], fill: RGBA, dot_r: float, spacing: float,
                    phase: float = 0.0) -> int:
    """Filled dots every ``spacing`` px of arc length along the polyline, the first at arc length ``phase``
    (21_annotate.py:76 placed dots by angle; arc length keeps the look size-independent). Returns the count."""
    if len(pts) < 2 or spacing <= 0:
        return 0
    next_at, s, n = float(phase), 0.0, 0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        while next_at <= s + seg + 1e-9:
            f = (next_at - s) / seg
            x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=fill)
            n += 1
            next_at += spacing
        s += seg
    return n


def dashed_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill: RGBA, width: float,
                  dash: float, gap: float) -> int:
    """Dashed circle, phase at screen angle 0 walking clockwise. The dash+gap period is scaled so an integer
    number of periods closes the circle (no truncated dash at the seam). Returns the number of dashes."""
    if r <= 0 or dash <= 0:
        return 0
    L = 2.0 * math.pi * r
    period = dash + gap
    n = max(1, int(round(L / period)))
    k = L / (n * period)
    dashed_polyline(draw, circle_points(cx, cy, r), fill, width, dash * k, gap * k)
    return n


def dot_count(r: float, spacing: float) -> int:
    return max(1, int(round(2.0 * math.pi * r / spacing))) if r > 0 and spacing > 0 else 0


def dotted_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill: RGBA, dot_r: float,
                  spacing: float) -> int:
    """Dots of radius ``dot_r`` with centre spacing ``spacing`` (by arc length, rounded to an integer count so
    the ring closes evenly), the first at screen angle 0 walking clockwise. Returns the dot count."""
    n = dot_count(r, spacing)
    for k in range(n):
        a = 2.0 * math.pi * k / n
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=fill)
    return n


# ----------------------------------------------------------------------------- arrows
def arrow_geometry(tail: Point, head: Point, tip_gap: float, head_len: float, head_w: float
                   ) -> tuple[Optional[tuple[Point, Point]], list[Point], Point]:
    """(shaft (a, b) or None, filled head triangle, apex). The apex sits ``tip_gap`` short of ``head`` along
    the tail->head direction (21_annotate.py:54: the feature stays visible); the shaft stops at the head's
    base. A very short arrow keeps its head and drops the shaft."""
    d = unit(head[0] - tail[0], head[1] - tail[1], fallback=(1.0, 0.0))
    nx, ny = -d[1], d[0]
    apex = (head[0] - d[0] * tip_gap, head[1] - d[1] * tip_gap)
    base = (apex[0] - d[0] * head_len, apex[1] - d[1] * head_len)
    poly = [apex, (base[0] + nx * head_w / 2.0, base[1] + ny * head_w / 2.0),
            (base[0] - nx * head_w / 2.0, base[1] - ny * head_w / 2.0)]
    along = (base[0] - tail[0]) * d[0] + (base[1] - tail[1]) * d[1]
    shaft = (tail, base) if along > 0.5 else None
    return shaft, poly, apex


def draw_arrow(draw: ImageDraw.ImageDraw, tail: Point, head: Point, fill: RGBA, line_w: float, tip_gap: float,
               head_len: float, head_w: float) -> Point:
    """Shaft + filled triangular head; returns the apex."""
    shaft, poly, apex = arrow_geometry(tail, head, tip_gap, head_len, head_w)
    if shaft is not None:
        # overlap the shaft slightly into the head so there is no seam after downsampling
        d = unit(head[0] - tail[0], head[1] - tail[1], fallback=(1.0, 0.0))
        b = (shaft[1][0] + d[0] * line_w * 0.5, shaft[1][1] + d[1] * line_w * 0.5)
        draw.line([shaft[0], b], fill=fill, width=max(1, int(round(line_w))))
    draw.polygon(poly, fill=fill)
    return apex


def approach_angle(dx: float, dy: float, other: Optional[Point] = None) -> float:
    """Screen angle (0 = right, 90 = up, counter-clockwise, degrees in [0, 360)) from which an arrow should
    approach a feature at screen offset ``(dx, dy)`` from the image centre (or from ``other``): the outward
    direction, so the shaft never crosses the feature (21_annotate.py:84). Degenerate -> 225 (lower-left)."""
    sx, sy = dx, dy
    if other is not None:
        sx, sy = sx - other[0], sy - other[1]
    if abs(sx) < 1e-6 and abs(sy) < 1e-6:
        return 225.0
    return math.degrees(math.atan2(-sy, sx)) % 360.0


# ----------------------------------------------------------------------------- label placement
def rect_support(dirx: float, diry: float, w: float, h: float) -> float:
    """Half-extent of a w×h axis-aligned box along the unit direction (dirx, diry)."""
    return abs(dirx) * w / 2.0 + abs(diry) * h / 2.0


def box_of(center: Point, w: float, h: float) -> Box:
    return (center[0] - w / 2.0, center[1] - h / 2.0, center[0] + w / 2.0, center[1] + h / 2.0)


def box_inside(box: Box, W: float, H: float, margin: float = 0.0) -> bool:
    return box[0] >= margin and box[1] >= margin and box[2] <= W - margin and box[3] <= H - margin


def clamp_center(xy: Point, w: float, h: float, W: float, H: float, margin: float = 3.0) -> Point:
    """Centre of a w×h box moved the minimum distance so the box lies inside the canvas (annotate.py:339).
    A box wider/taller than the canvas is centred."""
    x = min(max(xy[0], margin + w / 2.0), W - margin - w / 2.0) if w + 2 * margin <= W else W / 2.0
    y = min(max(xy[1], margin + h / 2.0), H - margin - h / 2.0) if h + 2 * margin <= H else H / 2.0
    return (x, y)


def place_label(xy: Point, w: float, h: float, W: float, H: float, margin: float = 3.0,
                placed: Optional[list[Box]] = None, origin: Optional[Point] = None,
                record: bool = True) -> Optional[Point]:
    """A clash-free centre for a w×h label near ``xy`` inside the canvas, or None when every candidate
    overlaps a box in ``placed`` (annotate.py:303): the clamped origin, then radial nudges from ``origin``
    (default: canvas centre) outward then inward, then sideways. Nudge steps scale with the label height."""
    if placed is None:
        return clamp_center(xy, w, h, W, H, margin)
    ox, oy = origin if origin is not None else (W / 2.0, H / 2.0)
    ux, uy = unit(xy[0] - ox, xy[1] - oy)
    step = max(h, 1.0)
    radial = (0.0, 0.8 * step, 1.6 * step, 2.4 * step, -0.8 * step, -1.6 * step, -2.4 * step, -3.2 * step)
    side = (1.0 * step, -1.0 * step, 2.0 * step, -2.0 * step)

    def clash(cx: float, cy: float) -> bool:
        return any(abs(cx - (bx0 + bx1) / 2) < (w + (bx1 - bx0)) / 2.0 + 2 and
                   abs(cy - (by0 + by1) / 2) < (h + (by1 - by0)) / 2.0 + 1
                   for bx0, by0, bx1, by1 in placed)

    cands = [(xy[0] + ux * d, xy[1] + uy * d) for d in radial]
    cands += [(xy[0] - uy * s + ux * d, xy[1] + ux * s + uy * d) for s in side for d in (0.0, -0.8 * step, -1.6 * step)]
    for cx, cy in cands:
        c = clamp_center((cx, cy), w, h, W, H, margin)
        if not clash(*c):
            if record:
                placed.append(box_of(c, w, h))
            return c
    return None


def clamp_label(xy: Point, w: float, h: float, W: float, H: float, margin: float = 3.0,
                placed: Optional[list[Box]] = None, origin: Optional[Point] = None) -> Point:
    """``place_label`` falling back to the clamped origin (recorded) - a label is always drawn."""
    pos = place_label(xy, w, h, W, H, margin, placed, origin)
    if pos is not None:
        return pos
    c = clamp_center(xy, w, h, W, H, margin)
    if placed is not None:
        placed.append(box_of(c, w, h))
    return c


# ----------------------------------------------------------------------------- legend
def legend_corner(anchors_uv: Sequence[Point]) -> str:
    """The corner whose quadrant holds the fewest anchor points (normalized u, v); ties -> the first of
    top_left, top_right, bottom_left, bottom_right."""
    counts = {c: 0 for c in CORNERS}
    for u, v in anchors_uv:
        key = ("top" if v < 0.5 else "bottom") + "_" + ("left" if u < 0.5 else "right")
        counts[key] += 1
    return min(CORNERS, key=lambda c: (counts[c], CORNERS.index(c)))


def legend_plate_box(corner: str, plate_w: float, plate_h: float, W: float, H: float, inset: float) -> Box:
    x0 = inset if corner.endswith("left") else W - inset - plate_w
    y0 = inset if corner.startswith("top") else H - inset - plate_h
    return (x0, y0, x0 + plate_w, y0 + plate_h)
