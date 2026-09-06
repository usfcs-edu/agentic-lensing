#!/usr/bin/env python3
"""Prototype renderer for the AstroMark notation proposals.

    python render_proposals.py [--content content.json] [--base reference/astromark-ref.png]
                               [--out reference/] [--arms N1 N2 N3 N4 R-CURRENT R-CAMPAIGN NULL]

Renders ONE neutral content file (content.json) through SIX notations plus a null arm, so a
side-by-side sheet varies the notation and nothing else. This is a comparison instrument, not a
LensMark replacement: it implements each proposal far enough to be judged and measured, and no
further.

THE SUBSTRATE — shared by every arm, so the comparison has one variable

  * Canonical output is 2x the cutout (806 px from a 403 px cutout). This is forced, not chosen:
    at 403 px the bound-ring dots, theta_E dots and leader hairlines all land below the ~1.25
    output-pixel floor where a stroke aliases into a grey smear.
  * Overlay drawn at SS=3 supersample and LANCZOS-downsampled.
  * KEEP-CLEAR: no mark may cover the pixels it refers to. Pointers stop short; circles ride the
    boundary; polygons are outline-only.
  * CASING, not halo: every stroke is drawn twice — a casing pass in a contrasting ink at moderate
    alpha, then the stroke. Legible over both black sky and a saturated core without the blocky
    look of a hard black halo.
  * METRIC vs PRESENTATIONAL: every drawn scalar is tagged. Emphasis multiplies PRESENTATIONAL
    scalars only, so a theta_E ring can shout without its radius moving one pixel.
  * Colour is REDUNDANT. Role is carried by glyph shape, polarity by stroke texture. The palette is
    Okabe-Ito, chosen for colour-vision safety, but nothing depends on it — see cvd.py.
  * TERRITORY RULE: no ink inside the image rectangle that is not about a specific location in it.
    The key lives in a caption band below the image. The scale bar is the one exception, because
    its meaning requires physical adjacency to the pixels it measures.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SS = 3                      # overlay supersample
UPSCALE = 2                 # canonical render is 2x the cutout
LABEL_FLOOR_PX = 11.0       # a string smaller than this cannot be read

# Okabe-Ito: colour-blind-safe qualitative palette. Colour REINFORCES; it never carries alone.
OKABE = {
    "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
}
# role family -> ink (redundant with shape and texture)
INK = {
    "lens_mass": OKABE["green"],
    "lensed": OKABE["skyblue"],
    "obstruction": OKABE["purple"],
    "field": OKABE["vermillion"],
    "measure": OKABE["yellow"],
    "model": OKABE["orange"],
    "neutral": "#FFFFFF",
}
FAMILY = {
    "deflector": "lens_mass", "second_deflector": "lens_mass", "satellite": "lens_mass",
    "arc": "lensed", "knot": "lensed", "counter_image": "lensed", "lensed_image": "lensed",
    "lensed_light": "lensed",
    "dust_lane": "obstruction",
    "field_galaxy": "field", "star": "field",
    "einstein_ring": "measure",
    "lens_light": "lens_mass",
}
# PRESENTATIONAL multipliers. Nothing here touches a measured radius or a vertex.
EMPH = {
    "muted":  {"stroke": 0.75, "glyph": 0.85, "label": 0.85, "casing": 1.0, "dot": 0.80, "alpha": 0.55},
    "normal": {"stroke": 1.00, "glyph": 1.00, "label": 1.00, "casing": 1.0, "dot": 1.00, "alpha": 1.00},
    "key":    {"stroke": 1.60, "glyph": 1.25, "label": 1.08, "casing": 1.4, "dot": 1.40, "alpha": 1.00},
}
S = {   # fractions of min(W, H) of the OUTPUT image
    "stroke": 0.0040, "glyph_stroke": 0.0038, "circle": 0.0036, "hairline": 0.0022,
    "poly": 0.0028, "bracket": 0.0030, "casing_add": 0.0022,
    "glyph": 0.026, "tip_gap": 0.010,
    # Texture periods, chosen so that two textures of the SAME KIND on the same mark family differ
    # by at least a full octave (see cvd.py texture_test). Dots on circles: ring 0.011, star 0.024.
    # Dashes on circles: artifact 0.024, galaxy 0.050. Bound rings deliberately share the ring's dot
    # texture — what identifies them is the caliper joining them, not a third dot period, and a
    # third dot could not clear both the octave rule above and the rasterisation floor below.
    "ring_dot": 0.0022,                       # period (2+3)*r = 0.011
    "dash": 0.030, "gap": 0.020,              # galaxy mask, period 0.050
    "artifact_dash": 0.010, "artifact_gap": 0.014,   # period 0.024
    "star_dot": 0.0032, "star_gap": 5.5,      # period (2+5.5)*r = 0.024
    "label": 0.023, "label_alt": 0.019, "label_small": 0.017, "label_offset": 0.014,
}


def hexrgb(h: str, a: float = 1.0) -> tuple[int, int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(round(255 * a)))


def unit(dx, dy):
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


class Ctx:
    """Drawing context: supersampled layers, uv->px, arcsec->px, emphasis-aware scalars."""

    def __init__(self, base: Image.Image, cutout_arcsec: float, caption_h: int = 0):
        self.bw, self.bh = base.width * UPSCALE, base.height * UPSCALE
        self.base = base.resize((self.bw, self.bh), Image.NEAREST)   # never interpolate astronomy
        self.caption_h = caption_h
        self.cutout_arcsec = cutout_arcsec
        self.m = min(self.bw, self.bh) * SS
        self.layers = {k: Image.new("RGBA", (self.bw * SS, self.bh * SS), (0, 0, 0, 0))
                       for k in ("labels", "muted", "normal", "key", "top")}
        self.d = {k: ImageDraw.Draw(v) for k, v in self.layers.items()}
        self._fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self.warnings: list[str] = []
        self.placed: list[tuple[float, float, float, float]] = []   # label keep-out boxes
        self.demoted: list[tuple[int, str, str]] = []               # index -> caption text

    # -- conversions ---------------------------------------------------------------------------
    def f(self, frac: float) -> float:
        return frac * self.m

    def uv(self, u: float, v: float) -> tuple[float, float]:
        return u * self.bw * SS, v * self.bh * SS

    def asec(self, a: float) -> float:
        """arcsec -> supersampled px. METRIC: never scaled by emphasis."""
        return a / self.cutout_arcsec * self.bw * SS

    def font(self, size_px: float, bold: bool = False) -> ImageFont.FreeTypeFont:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        key = (name, int(round(size_px)))
        if key not in self._fonts:
            root = Path(__file__).resolve().parents[4] / "apps/LensMark/lensmark/render/fonts"
            self._fonts[key] = ImageFont.truetype(str(root / name), key[1])
        return self._fonts[key]

    def layer(self, emphasis: str) -> ImageDraw.ImageDraw:
        return self.d[{"muted": "muted", "key": "key"}.get(emphasis, "normal")]

    # -- cased primitives ----------------------------------------------------------------------
    def line(self, dr, p0, p1, w, ink, casing="#000000", casing_a=0.62):
        dr.line([p0, p1], fill=hexrgb(casing, casing_a),
                width=max(1, int(round(w + 2 * self.f(S["casing_add"])))))
        dr.line([p0, p1], fill=hexrgb(ink), width=max(1, int(round(w))))

    def poly(self, dr, pts, w, ink, fill=None, casing="#000000", casing_a=0.62, closed=True):
        seq = list(pts) + ([pts[0]] if closed and len(pts) > 2 else [])
        if fill:
            dr.polygon(list(pts), fill=fill)
        dr.line(seq, fill=hexrgb(casing, casing_a),
                width=max(1, int(round(w + 2 * self.f(S["casing_add"])))), joint="curve")
        dr.line(seq, fill=hexrgb(ink), width=max(1, int(round(w))), joint="curve")

    def filled(self, dr, pts, ink, casing="#000000", casing_a=0.62):
        dr.line(list(pts) + [pts[0]], fill=hexrgb(casing, casing_a),
                width=max(1, int(round(2 * self.f(S["casing_add"])))), joint="curve")
        dr.polygon(list(pts), fill=hexrgb(ink))

    def circle_pts(self, cx, cy, r, n=None):
        n = n or max(48, int(2 * math.pi * r / (1.5 * SS)))
        return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
                for i in range(n)]

    def stroked_circle(self, dr, cx, cy, r, w, ink, pattern="solid", dash=None, gap=None,
                       dot_r=None, dot_gap=3.0):
        """pattern: solid | dash | dot | dashdot. Phase closes exactly (no truncated dash)."""
        if pattern == "dot":
            step = (2.0 + dot_gap) * dot_r
            n = max(6, int(round(2 * math.pi * r / step)))
            for i in range(n):
                a = 2 * math.pi * i / n
                x, y = cx + r * math.cos(a), cy + r * math.sin(a)
                dr.ellipse([x - dot_r - self.f(S["casing_add"]) * 0.5,
                            y - dot_r - self.f(S["casing_add"]) * 0.5,
                            x + dot_r + self.f(S["casing_add"]) * 0.5,
                            y + dot_r + self.f(S["casing_add"]) * 0.5],
                           fill=hexrgb("#000000", 0.55))
                dr.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=hexrgb(ink))
            return
        if pattern == "solid":
            self.poly(dr, self.circle_pts(cx, cy, r), w, ink)
            return
        period = (dash + gap) if pattern == "dash" else (dash + gap + dash * 0.32 + gap)
        L = 2 * math.pi * r
        n = max(3, int(round(L / period)))
        k = L / (n * period)
        for i in range(n):
            a0 = 2 * math.pi * (i * period * k) / L
            segs = [(0.0, dash)] if pattern == "dash" else [(0.0, dash), (dash + gap, dash * 0.32)]
            for off, ln in segs:
                s0 = a0 + 2 * math.pi * (off * k) / L
                s1 = a0 + 2 * math.pi * ((off + ln) * k) / L
                steps = max(2, int(abs(s1 - s0) * r / (1.5 * SS)))
                pts = [(cx + r * math.cos(s0 + (s1 - s0) * j / steps),
                        cy + r * math.sin(s0 + (s1 - s0) * j / steps)) for j in range(steps + 1)]
                self.poly(dr, pts, w, ink, closed=False)

    # -- text ----------------------------------------------------------------------------------
    def text(self, dr, xy, s, size_px, ink, bold=False, anchor="mm"):
        fnt = self.font(size_px, bold)
        dr.text(xy, s, font=fnt, fill=hexrgb("#000000", 0.70), anchor=anchor,
                stroke_width=max(1, int(round(self.f(S["casing_add"]) * 0.9))),
                stroke_fill=hexrgb("#000000", 0.70))
        dr.text(xy, s, font=fnt, fill=hexrgb(ink), anchor=anchor)

    def text_box(self, s, size_px, bold=False):
        f = self.font(size_px, bold)
        b = f.getbbox(s)
        return b[2] - b[0], b[3] - b[1]

    def place_label(self, anchor_xy, dirv, text, size_px, keepout):
        if not (text or "").strip():
            return None
        """Greedy placement: preferred side, then perpendiculars, then demote.

        Returns (x, y) or None when the label must demote to an index digit — deterministic
        failure rather than an overlapping mess.
        """
        w, h = self.text_box(text, size_px)
        off = self.f(S["label_offset"])
        dirs = [dirv, (-dirv[1], dirv[0]), (dirv[1], -dirv[0]), (-dirv[0], -dirv[1])]
        for ang in (45, -45, 135, -135):
            t = math.radians(ang)
            dirs.append((dirv[0] * math.cos(t) - dirv[1] * math.sin(t),
                         dirv[0] * math.sin(t) + dirv[1] * math.cos(t)))
        cands = []
        for scale in (1.0, 1.9, 2.8):
            for d in dirs:
                dist = off * scale + abs(d[0]) * w / 2 + abs(d[1]) * h / 2
                cands.append((anchor_xy[0] + d[0] * dist, anchor_xy[1] + d[1] * dist))
        for cx, cy in cands:
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if box[0] < 4 * SS or box[1] < 4 * SS or box[2] > self.bw * SS - 4 * SS or box[3] > self.bh * SS - 4 * SS:
                continue
            if any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3])
                   for o in self.placed):
                continue
            if any(_hits(box, ko) for ko in keepout):
                continue
            self.placed.append(box)
            return cx, cy
        return None

    # -- compose --------------------------------------------------------------------------------
    def compose(self) -> Image.Image:
        over = Image.new("RGBA", (self.bw * SS, self.bh * SS), (0, 0, 0, 0))
        for name in ("labels", "muted", "normal", "key", "top"):
            lay = self.layers[name]
            a = EMPH.get(name, {}).get("alpha", 1.0)
            if a < 1.0:                       # per-LAYER alpha: per-stroke would double-composite
                r, g, b, al = lay.split()
                lay = Image.merge("RGBA", (r, g, b, al.point(lambda v: int(v * a))))
            over = Image.alpha_composite(over, lay)
        over = over.convert("RGBa").resize((self.bw, self.bh), Image.LANCZOS).convert("RGBA")
        out = Image.alpha_composite(self.base.convert("RGBA"), over).convert("RGB")
        if not self.caption_h:
            return out
        canvas = Image.new("RGB", (self.bw, self.bh + self.caption_h), (8, 8, 10))
        canvas.paste(out, (0, 0))
        return canvas


# ==================================================================================================
# Scene: resolve content.json into drawable geometry, once, shared by every arm.
# ==================================================================================================

class Scene:
    def __init__(self, content: dict):
        self.c = content
        self.marks = content["marks"]
        self.by_id = {m["id"]: m for m in self.marks}
        self.sys = content["system"]

    def head(self, m):
        g = m["geometry"]
        if g["kind"] == "vector":
            return tuple(g["head"])
        if g["kind"] == "point":
            return tuple(g["at"])
        if g["kind"] == "circle":
            if "center_ref" in g:
                return self.head(self.by_id[g["center_ref"]])
            return tuple(g["center"])
        pts = g["points"]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def deflector_uv(self):
        for m in self.marks:
            if m["role"] == "deflector":
                return self.head(m)
        return (0.5, 0.5)

    def lens_mass_uv(self):
        return [self.head(m) for m in self.marks
                if m["role"] in ("deflector", "second_deflector", "satellite")]

    def theta_e(self):
        for m in self.marks:
            if m["role"] == "einstein_ring" and m.get("bound") == "nominal":
                return m["geometry"]["radius_arcsec"]
        return None


def emph_of(m):
    return m.get("emphasis", "normal")


def mul(m, key):
    return EMPH[emph_of(m)][key]


def ink_of(m):
    if m["role"] == "field_galaxy" and m.get("treatment") == "model":
        return INK["model"]
    return INK[FAMILY.get(m["role"], "neutral")]


# ==================================================================================================
# Terminators (N1) — the sign inventory lives in the SHAPE at the business end.
# ==================================================================================================

def _tri(ctx, dr, apex, d, g, w, ink, filled=True, notch=False):
    n = (-d[1], d[0])
    base = (apex[0] - d[0] * g, apex[1] - d[1] * g)
    p1 = (base[0] + n[0] * g * 0.42, base[1] + n[1] * g * 0.42)
    p2 = (base[0] - n[0] * g * 0.42, base[1] - n[1] * g * 0.42)
    if notch:
        mid = (base[0] + d[0] * g * 0.34, base[1] + d[1] * g * 0.34)
        pts = [apex, p1, mid, p2]
    else:
        pts = [apex, p1, p2]
    if filled:
        ctx.filled(dr, pts, ink)
    else:
        ctx.poly(dr, pts, w, ink)


def _chevron(ctx, dr, apex, d, g, w, ink, double=False, curved=False):
    n = (-d[1], d[0])
    for k in ((0.0,) if not double else (0.0, 0.42)):
        a = (apex[0] - d[0] * g * k, apex[1] - d[1] * g * k)
        b = (a[0] - d[0] * g * 0.62, a[1] - d[1] * g * 0.62)
        p1 = (b[0] + n[0] * g * 0.46, b[1] + n[1] * g * 0.46)
        p2 = (b[0] - n[0] * g * 0.46, b[1] - n[1] * g * 0.46)
        if curved:
            mid1 = (a[0] - d[0] * g * 0.20 + n[0] * g * 0.26, a[1] - d[1] * g * 0.20 + n[1] * g * 0.26)
            mid2 = (a[0] - d[0] * g * 0.20 - n[0] * g * 0.26, a[1] - d[1] * g * 0.20 - n[1] * g * 0.26)
            ctx.poly(dr, [p1, mid1, a, mid2, p2], w, ink, closed=False)
        else:
            ctx.poly(dr, [p1, a, p2], w, ink, closed=False)


def _bar(ctx, dr, apex, d, g, w, ink, double=True):
    n = (-d[1], d[0])
    for k in ((0.0, 0.30) if double else (0.0,)):
        a = (apex[0] - d[0] * g * k, apex[1] - d[1] * g * k)
        ctx.poly(dr, [(a[0] + n[0] * g * 0.46, a[1] + n[1] * g * 0.46),
                      (a[0] - n[0] * g * 0.46, a[1] - n[1] * g * 0.46)], w, ink, closed=False)


def _ring_term(ctx, dr, apex, g, w, ink):
    r = g * 0.30
    ctx.poly(dr, ctx.circle_pts(apex[0], apex[1], r), w, ink)


def _strike(ctx, dr, apex, d, g, w, ink):
    """Cross-bar through a terminator: the redundant second channel on a negative mark."""
    n = (-d[1], d[0])
    c = (apex[0] - d[0] * g * 0.34, apex[1] - d[1] * g * 0.34)
    a = (c[0] + n[0] * g * 0.62 - d[0] * g * 0.30, c[1] + n[1] * g * 0.62 - d[1] * g * 0.30)
    b = (c[0] - n[0] * g * 0.62 + d[0] * g * 0.30, c[1] - n[1] * g * 0.62 + d[1] * g * 0.30)
    ctx.poly(dr, [a, b], w * 1.1, ink, closed=False)


TERMINATOR = {
    "deflector":        lambda c, d, a, v, g, w, i: _tri(c, d, a, v, g, w, i, filled=True),
    "second_deflector": lambda c, d, a, v, g, w, i: _tri(c, d, a, v, g, w, i, filled=True, notch=True),
    "satellite":        lambda c, d, a, v, g, w, i: _tri(c, d, a, v, g * 0.68, w, i, filled=True),
    "arc":              lambda c, d, a, v, g, w, i: _chevron(c, d, a, v, g, w, i, curved=True),
    "lensed_image":     lambda c, d, a, v, g, w, i: _chevron(c, d, a, v, g, w, i),
    "counter_image":    lambda c, d, a, v, g, w, i: _chevron(c, d, a, v, g, w, i, double=True),
    "knot":             lambda c, d, a, v, g, w, i: _ring_term(c, d, a, g, w, i),
    "dust_lane":        lambda c, d, a, v, g, w, i: _bar(c, d, a, v, g, w, i, double=True),
}


# ==================================================================================================
# Shared substrate: measured circles and region polygons. Identical in every arm, by design —
# the arms differ in how DESIGNATING marks are drawn, not in how measurements are drawn.
# ==================================================================================================

def dashed_poly(ctx: Ctx, dr, pts, w, ink, on_frac=0.020, off_frac=0.012):
    """Long-dash outline: the lens-light segmentation is deliberately tolerant, and a dashed
    boundary says 'about here' where a solid one would claim precision the isophote does not have."""
    on, off = ctx.f(on_frac), ctx.f(off_frac)
    seq = list(pts) + [pts[0]]
    carry = 0.0
    for a, b in zip(seq, seq[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        d = unit(b[0] - a[0], b[1] - a[1])
        t = -carry
        while t < seg:
            s0, s1 = max(t, 0.0), min(t + on, seg)
            if s1 > s0:
                ctx.poly(dr, [(a[0] + d[0] * s0, a[1] + d[1] * s0),
                              (a[0] + d[0] * s1, a[1] + d[1] * s1)], w, ink, closed=False)
            t += on + off
        carry = (carry + seg) % (on + off)


def draw_measures(ctx: Ctx, sc: Scene, thin: float = 1.0):
    for m in sc.marks:
        g = m["geometry"]
        e = emph_of(m)
        dr = ctx.layer(e)
        ink = ink_of(m)
        if g["kind"] == "circle":
            cu, cv = sc.head(m)
            cx, cy = ctx.uv(cu, cv)
            r = ctx.asec(g["radius_arcsec"])                       # METRIC — never scaled
            w = ctx.f(S["circle"]) * mul(m, "stroke") * thin
            if m["role"] == "einstein_ring":
                bound = m.get("bound", "nominal")
                # Same dot texture for nominal and bound rings, on purpose. A third, finer dot
                # period cannot be an octave from the other two AND clear the aliasing floor, and
                # the bracket is already identified structurally by the caliper joining the pair.
                dot = ctx.f(S["ring_dot"]) * mul(m, "dot")
                dot = min(dot, 0.08 * r)                           # guard: dots must not bias r
                ctx.stroked_circle(dr, cx, cy, r, w, ink if bound == "nominal" else "#BFBFBF",
                                   pattern="dot", dot_r=dot, dot_gap=3.0)
            elif m["role"] == "star":
                ctx.stroked_circle(dr, cx, cy, r, w, ink, pattern="dot",
                                   dot_r=ctx.f(S["star_dot"]) * mul(m, "dot"), dot_gap=S["star_gap"])
            elif m.get("treatment") == "model":
                ctx.stroked_circle(dr, cx, cy, r, w, ink, pattern="dashdot",
                                   dash=ctx.f(S["dash"]), gap=ctx.f(S["gap"]))
            elif m["role"] == "artifact":
                ctx.stroked_circle(dr, cx, cy, r, w, ink, pattern="dash",
                                   dash=ctx.f(S["artifact_dash"]), gap=ctx.f(S["artifact_gap"]))
            else:
                ctx.stroked_circle(dr, cx, cy, r, w, ink, pattern="dash",
                                   dash=ctx.f(S["dash"]), gap=ctx.f(S["gap"]))
        elif g["kind"] == "polygon":
            pts = [ctx.uv(*p) for p in g["points"]]                # METRIC — vertices never scaled
            w = ctx.f(S["poly"]) * mul(m, "stroke") * thin
            if m["role"] == "lens_light":
                dashed_poly(ctx, dr, pts, w, ink)
            else:
                ctx.poly(dr, pts, w, ink)


def draw_caliper(ctx: Ctx, sc: Scene):
    """Two radial ticks joining the lower and upper bound rings: what says 'this is a bracket'."""
    lo = hi = None
    for m in sc.marks:
        if m["role"] == "einstein_ring":
            if m.get("bound") == "lower":
                lo = m
            elif m.get("bound") == "upper":
                hi = m
    if not (lo and hi):
        return
    cu, cv = sc.head(lo)
    cx, cy = ctx.uv(cu, cv)
    r0, r1 = ctx.asec(lo["geometry"]["radius_arcsec"]), ctx.asec(hi["geometry"]["radius_arcsec"])
    dr = ctx.d["normal"]
    for ang in (45.0, 225.0):
        a = math.radians(ang)
        ctx.poly(dr, [(cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
                      (cx + r1 * math.cos(a), cy + r1 * math.sin(a))],
                 ctx.f(S["hairline"]), "#BFBFBF", closed=False)


def corner_brackets(ctx: Ctx, xy, half):
    """The attention mark, used at emphasis=key only. A SQUARE — never a circle, because circles
    are already spent on measurement and a new one reads as another measured radius."""
    dr = ctx.d["key"]
    x, y = xy
    arm = half * 0.42
    w = ctx.f(S["bracket"])
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cx, cy = x + sx * half, y + sy * half
        ctx.poly(dr, [(cx, cy), (cx - sx * arm, cy)], w, "#FFFFFF", closed=False)
        ctx.poly(dr, [(cx, cy), (cx, cy - sy * arm)], w, "#FFFFFF", closed=False)


def keepout_boxes(ctx: Ctx, sc: Scene):
    """Labels must avoid marks AND the annulus of every measured circle: a label lying on a ring
    is read as naming a point on that ring."""
    out = []
    for m in sc.marks:
        g = m["geometry"]
        if g["kind"] == "circle":
            cx, cy = ctx.uv(*sc.head(m))
            r = ctx.asec(g["radius_arcsec"])
            out.append(("ann", cx, cy, r, ctx.f(0.012)))
        elif g["kind"] == "vector":
            hx, hy = ctx.uv(*g["head"])
            p = ctx.f(S["glyph"])
            out.append(("box", hx - p, hy - p, hx + p, hy + p))
    return out


def _hits(box, ko) -> bool:
    """box is (x0,y0,x1,y1). ko is ('box', ...) or ('ann', cx, cy, r, pad)."""
    if ko[0] == "box":
        _, x0, y0, x1, y1 = ko
        return not (box[2] < x0 or box[0] > x1 or box[3] < y0 or box[1] > y1)
    _, cx, cy, r, pad = ko
    dx = max(box[0] - cx, 0.0, cx - box[2])
    dy = max(box[1] - cy, 0.0, cy - box[3])
    dmin = math.hypot(dx, dy)
    dmax = max(math.hypot(cx - x, cy - y)
               for x in (box[0], box[2]) for y in (box[1], box[3]))
    return dmin <= r + pad and dmax >= r - pad


def label_for(m):
    if m.get("label") == "":          # explicitly blanked (the polarity triad) vs simply absent
        return "", "", None
    lbl = m.get("label") or m["role"].replace("_", " ")
    if m.get("polarity") == "negative":
        return lbl, "NOT " + lbl.replace("_", " "), (m.get("alternative") or "").replace("_", " ")
    if m.get("polarity") == "ambiguous":
        return lbl, lbl + " ?", (m.get("alternative") or "").replace("_", " ")
    return lbl, lbl, None


DESIGNATING = ("deflector", "second_deflector", "satellite", "arc", "knot",
               "counter_image", "lensed_image", "dust_lane")


def demote(ctx: Ctx, anchor, dirv, text, ink, size=None):
    """No placement found: draw an index digit at the mark and move the text to the caption."""
    if not (text or "").strip():
        return
    n = len(ctx.demoted) + 1
    size = size or ctx.f(S["label"])
    d = dirv if (dirv[0] or dirv[1]) else (0.0, -1.0)
    at = (anchor[0] + d[0] * ctx.f(0.020), anchor[1] + d[1] * ctx.f(0.020))
    ctx.text(ctx.d["labels"], at, str(n), size, ink, bold=True)
    ctx.demoted.append((n, text, ink))
    ctx.warnings.append(f"label demoted to index {n}: {text}")


def _shaft_pattern(ctx, dr, p0, p1, w, ink, polarity):
    if polarity == "positive":
        ctx.poly(dr, [p0, p1], w, ink, closed=False)
        return
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    d = unit(p1[0] - p0[0], p1[1] - p0[1])
    if polarity == "negative":
        on, off = ctx.f(0.020), ctx.f(0.010)
    else:
        on, off = ctx.f(0.0024), ctx.f(0.0055)
    t = 0.0
    while t < L:
        a = (p0[0] + d[0] * t, p0[1] + d[1] * t)
        b = (p0[0] + d[0] * min(t + on, L), p0[1] + d[1] * min(t + on, L))
        ctx.poly(dr, [a, b], w, ink, closed=False)
        t += on + off


# ==================================================================================================
# ARM N1 — "Terminator alphabet": keep the pointer, move all semantic weight into the shape at the
# business end, thin everything else. Polarity is the shaft texture, reinforced by a strike.
# ==================================================================================================

def arm_N1(ctx: Ctx, sc: Scene):
    draw_measures(ctx, sc)
    draw_caliper(ctx, sc)
    keep = keepout_boxes(ctx, sc)
    # emphasis=key places its label first: the mark the author called out must not lose the
    # placement race to an incidental one and demote to a bare index.
    for m in sorted(sc.marks, key=lambda x: 0 if emph_of(x) == "key" else 1):
        if m["role"] not in DESIGNATING or m["geometry"]["kind"] != "vector":
            continue
        g, e = m["geometry"], emph_of(m)
        dr, ink = ctx.layer(e), ink_of(m)
        tail, head = ctx.uv(*g["tail"]), ctx.uv(*g["head"])
        d = unit(head[0] - tail[0], head[1] - tail[1])
        gap = ctx.f(S["tip_gap"])
        apex = (head[0] - d[0] * gap, head[1] - d[1] * gap)
        gl = ctx.f(S["glyph"]) * mul(m, "glyph")
        w = ctx.f(S["stroke"]) * mul(m, "stroke")
        base = (apex[0] - d[0] * gl, apex[1] - d[1] * gl)
        _shaft_pattern(ctx, dr, tail, base, w, ink, m.get("polarity", "positive"))
        fn = TERMINATOR.get(m["role"])
        if fn:
            fn(ctx, dr, apex, d, gl, ctx.f(S["glyph_stroke"]) * mul(m, "stroke"), ink)
        if m.get("polarity") in ("negative", "ambiguous"):
            _strike(ctx, dr, apex, d, gl, ctx.f(S["glyph_stroke"]), ink)
        if e == "key":
            corner_brackets(ctx, apex, gl * 1.5)
        _, disp, alt = label_for(m)
        size = ctx.f(S["label"]) * mul(m, "label")
        pos = ctx.place_label(tail, (-d[0], -d[1]), disp, size, keep)
        if pos:
            ctx.text(ctx.d["labels"], pos, disp, size, ink, bold=(e == "key"))
            if alt:
                pre = "≠ " if m["polarity"] == "negative" else "? "
                ctx.text(ctx.d["labels"], (pos[0], pos[1] + size * 1.05), pre + alt,
                         ctx.f(S["label_alt"]), ink)
        else:
            demote(ctx, tail, (-d[0], -d[1]), disp + (f"  ≠ {alt}" if alt else ""), ink, size)


# ==================================================================================================
# ARM N2 — "Bertin ledger": each visual variable carries exactly one semantic dimension.
# The move that distinguishes it: ORIENTATION is derived from the physics — a tangential tick means
# lensed light (which really is tangentially stretched), a radial tick means lens mass.
# ==================================================================================================

def arm_N2(ctx: Ctx, sc: Scene):
    draw_measures(ctx, sc)
    draw_caliper(ctx, sc)
    keep = keepout_boxes(ctx, sc)
    masses = sc.lens_mass_uv()
    for m in sc.marks:
        if m["role"] not in DESIGNATING or m["geometry"]["kind"] != "vector":
            continue
        e = emph_of(m)
        dr, ink = ctx.layer(e), ink_of(m)
        hu, hv = m["geometry"]["head"]
        hx, hy = ctx.uv(hu, hv)
        # Orientation reference. For lensed light and field objects this is the nearest lens-mass
        # item, which is what makes "tangential" mean the physics. For a lens-mass mark the radius
        # vector to itself is degenerate, so it is measured from the PRIMARY deflector instead, and
        # the primary deflector itself falls back to a fixed vertical tick — the documented
        # degenerate case, handled rather than left to produce a zero vector.
        if m["role"] in ("deflector", "second_deflector", "satellite"):
            ref = sc.deflector_uv()
        else:
            ref = min(masses, key=lambda p: (p[0] - hu) ** 2 + (p[1] - hv) ** 2)
        if math.hypot(hu - ref[0], hv - ref[1]) < 1e-4:
            rad = (0.0, -1.0)
        else:
            rad = unit(hu - ref[0], hv - ref[1])
        tang = (-rad[1], rad[0])
        role = m["role"]
        if role in ("deflector", "second_deflector", "satellite"):
            vec, reps, frac = rad, 1, 1.0
        elif role == "counter_image":
            vec, reps, frac = tang, 2, 1.0
        elif role == "knot":
            vec, reps, frac = tang, 1, 0.5
        elif role == "dust_lane":
            vec, reps, frac = rad, 1, 1.0
        else:
            vec, reps, frac = tang, 1, 1.0
        tl = ctx.f(S["glyph"]) * mul(m, "glyph") * frac
        w = ctx.f(S["glyph_stroke"]) * mul(m, "stroke")
        off = ctx.f(0.016)
        ax, ay = hx + rad[0] * off, hy + rad[1] * off      # offset OUTWARD so a tangential tick
        for k in range(reps):                              # never lies along the arc it marks
            s = (k - (reps - 1) / 2) * ctx.f(0.010)
            c = (ax + rad[0] * s, ay + rad[1] * s)
            p0 = (c[0] - vec[0] * tl / 2, c[1] - vec[1] * tl / 2)
            p1 = (c[0] + vec[0] * tl / 2, c[1] + vec[1] * tl / 2)
            if role == "dust_lane":     # radial with a centre gap: absorption breaks the light
                gp = tl * 0.22
                ctx.poly(dr, [p0, (c[0] - vec[0] * gp, c[1] - vec[1] * gp)], w, ink, closed=False)
                ctx.poly(dr, [(c[0] + vec[0] * gp, c[1] + vec[1] * gp), p1], w, ink, closed=False)
            else:
                _shaft_pattern(ctx, dr, p0, p1, w, ink, m.get("polarity", "positive"))
        if m.get("polarity") == "negative":
            ctx.poly(dr, [(ax - tang[0] * tl * .6 - rad[0] * tl * .4,
                           ay - tang[1] * tl * .6 - rad[1] * tl * .4),
                          (ax + tang[0] * tl * .6 + rad[0] * tl * .4,
                           ay + tang[1] * tl * .6 + rad[1] * tl * .4)], w, ink, closed=False)
        if e == "key":
            corner_brackets(ctx, (hx, hy), tl * 1.3)
        _, disp, alt = label_for(m)
        size = ctx.f(S["label"]) * mul(m, "label")
        anchor = (ax + rad[0] * ctx.f(0.030), ay + rad[1] * ctx.f(0.030))
        pos = ctx.place_label(anchor, rad, disp, size, keep)
        if pos:
            ctx.poly(ctx.d["labels"], [(ax + rad[0] * tl * 0.6, ay + rad[1] * tl * 0.6), pos],
                     ctx.f(S["hairline"]), ink, closed=False)   # leader hairline
            ctx.text(ctx.d["labels"], pos, disp, size, ink, bold=(e == "key"))
            if alt:
                pre = "≠ " if m["polarity"] == "negative" else "? "
                ctx.text(ctx.d["labels"], (pos[0], pos[1] + size * 1.05), pre + alt,
                         ctx.f(S["label_alt"]), ink)
        else:
            demote(ctx, anchor, rad, disp + (f"  ≠ {alt}" if alt else ""), ink, size)


# ==================================================================================================
# ARM N3 — "Station model": one compact badge per object, slots carrying role / polarity / source /
# treatment. One fixation per object instead of three. Lowest ink of the four.
# ==================================================================================================

IDEOGRAM = {
    "deflector": "ring_dot", "second_deflector": "ring_tick", "satellite": "ring_tether",
    "arc": "arc120", "lensed_image": "arc60", "counter_image": "arc_pair", "knot": "square",
    "dust_lane": "triple_bar", "field_galaxy": "open_circle", "star": "asterisk",
}


def _ideogram(ctx, dr, c, s, w, ink, kind):
    x, y = c
    if kind in ("ring_dot", "ring_tick", "ring_tether"):
        ctx.poly(dr, ctx.circle_pts(x, y, s * 0.34), w, ink)
        if kind == "ring_dot":
            dr.ellipse([x - s * .10, y - s * .10, x + s * .10, y + s * .10], fill=hexrgb(ink))
        elif kind == "ring_tick":
            ctx.poly(dr, [(x + s * .34, y), (x + s * .60, y)], w, ink, closed=False)
        else:
            ctx.poly(dr, [(x, y + s * .34), (x, y + s * .68)], w, ink, closed=False)
    elif kind in ("arc120", "arc60", "arc_pair"):
        spans = {"arc120": [(150, 390)], "arc60": [(200, 340)], "arc_pair": [(150, 250), (290, 390)]}
        for a0, a1 in spans[kind]:
            n = 14
            pts = [(x + s * .40 * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                    y + s * .40 * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
            ctx.poly(dr, pts, w, ink, closed=False)
    elif kind == "square":
        r = s * .18
        ctx.filled(dr, [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)], ink)
    elif kind == "triple_bar":
        for k in (-1, 0, 1):
            ctx.poly(dr, [(x - s * .34, y + k * s * .20), (x + s * .34, y + k * s * .20)],
                     w, ink, closed=False)
    elif kind == "open_circle":
        ctx.poly(dr, ctx.circle_pts(x, y, s * 0.32), w, ink)
    elif kind == "asterisk":
        for a in (90, 210, 330):
            t = math.radians(a)
            ctx.poly(dr, [(x - s * .34 * math.cos(t), y - s * .34 * math.sin(t)),
                          (x + s * .34 * math.cos(t), y + s * .34 * math.sin(t))],
                     w, ink, closed=False)


def arm_N3(ctx: Ctx, sc: Scene):
    draw_measures(ctx, sc)
    draw_caliper(ctx, sc)
    keep = keepout_boxes(ctx, sc)
    cx0, cy0 = ctx.uv(0.5, 0.5)
    for m in sc.marks:
        g = m["geometry"]
        badge_roles = DESIGNATING + ("field_galaxy", "star")
        if m["role"] not in badge_roles or g["kind"] == "polygon":
            continue
        # a badge appears only when it carries something the circle style does not
        if g["kind"] == "circle" and m.get("treatment") != "model":
            continue
        e = emph_of(m)
        dr, ink = ctx.layer(e), ink_of(m)
        hu, hv = sc.head(m)
        hx, hy = ctx.uv(hu, hv)
        d = unit(hx - cx0, hy - cy0)
        if d == (0.0, 0.0):
            d = (0.0, -1.0)
        s = ctx.f(0.042) * mul(m, "glyph")
        bx, by = hx + d[0] * (s * 1.6), hy + d[1] * (s * 1.6)
        w = ctx.f(S["glyph_stroke"]) * mul(m, "stroke")
        ctx.poly(dr, [(hx + d[0] * ctx.f(0.012), hy + d[1] * ctx.f(0.012)),
                      (bx - d[0] * s * .55, by - d[1] * s * .55)],
                 ctx.f(S["hairline"]), ink, closed=False)                 # stem
        _ideogram(ctx, dr, (bx, by), s, w, ink, IDEOGRAM.get(m["role"], "open_circle"))
        pol = m.get("polarity", "positive")
        if pol == "ambiguous":                                            # slot 2: one stroke
            dashed_poly(ctx, dr, [(bx - s * .34, by + s * .46), (bx + s * .34, by + s * .46)],
                        w * .8, ink, on_frac=0.004, off_frac=0.004)
        elif pol == "negative":
            ctx.poly(dr, [(bx - s * .42, by + s * .34), (bx + s * .42, by - s * .34)],
                     w, ink, closed=False)
        if m.get("source"):                                               # slot 3: source index
            ctx.text(dr, (bx - s * .62, by - s * .52), m["source"][-1],
                     ctx.f(S["label_small"]), ink)
        tr = {"mask": "×", "model": "f"}.get(m.get("treatment"))
        if tr:                                                            # slot 4: treatment
            ctx.text(dr, (bx + s * .62, by - s * .52), tr, ctx.f(S["label_small"]), ink)
        if e == "key":
            corner_brackets(ctx, (bx, by), s * 0.95)
        _, disp, alt = label_for(m)
        size = ctx.f(S["label_alt"]) * mul(m, "label")
        pos = ctx.place_label((bx + d[0] * s * .8, by + d[1] * s * .8), d, disp, size, keep)
        if pos:
            ctx.text(ctx.d["labels"], pos, disp, size, ink, bold=(e == "key"))
            if alt:
                ctx.text(ctx.d["labels"], (pos[0], pos[1] + size * 1.05),
                         ("≠ " if m["polarity"] == "negative" else "? ") + alt,
                         ctx.f(S["label_small"]), ink)
        else:
            demote(ctx, (bx, by), d, disp + (f"  ≠ {alt}" if alt else ""), ink, size)


# ==================================================================================================
# ARM N4 — "Evidence graph": annotate the ARGUMENT, not the objects. The primitive is a link.
# A positive panel has an unbroken chord; a negative panel's chord is struck. The non-lens becomes
# visible at thumbnail size, which is the defect none of the current outputs fix.
# ==================================================================================================

def arm_N4(ctx: Ctx, sc: Scene):
    draw_measures(ctx, sc, thin=0.85)
    draw_caliper(ctx, sc)
    keep = keepout_boxes(ctx, sc)
    hair = ctx.f(S["hairline"])
    du, dv = sc.deflector_uv()
    dcx, dcy = ctx.uv(du, dv)
    theta = sc.theta_e()

    anchors = {}
    for m in sc.marks:
        if m["role"] in DESIGNATING and m["geometry"]["kind"] == "vector":
            hx, hy = ctx.uv(*m["geometry"]["head"])
            anchors[m["id"]] = (hx, hy, m)
            ctx.poly(ctx.layer(emph_of(m)), ctx.circle_pts(hx, hy, ctx.f(0.010)),
                     hair, ink_of(m))                                   # the only mark that touches

    dr = ctx.d["normal"]
    dr.ellipse([dcx - ctx.f(.006), dcy - ctx.f(.006), dcx + ctx.f(.006), dcy + ctx.f(.006)],
               fill=hexrgb(INK["lens_mass"]))
    for mid, (hx, hy, m) in anchors.items():                             # deflector stems: mass
        if m["role"] not in ("deflector", "second_deflector", "satellite"):
            continue
        d = unit(dcx - hx, dcy - hy)
        p0 = (hx + d[0] * ctx.f(0.012), hy + d[1] * ctx.f(0.012))
        ctx.poly(dr, [p0, (dcx, dcy)], hair, INK["lens_mass"], closed=False)
        if m["role"] == "satellite":                                     # tether: bound
            mx, my = (p0[0] + dcx) / 2, (p0[1] + dcy) / 2
            n = (-d[1], d[0])
            ctx.poly(dr, [(mx + n[0] * ctx.f(.010), my + n[1] * ctx.f(.010)),
                          (mx - n[0] * ctx.f(.010), my - n[1] * ctx.f(.010))],
                     hair, INK["lens_mass"], closed=False)

    # source chords: one hairline arc per source, concentric with the ring, tying its images
    by_src: dict[str, list] = {}
    for mid, (hx, hy, m) in anchors.items():
        if m["role"] in ("arc", "knot", "counter_image", "lensed_image") and m.get("source"):
            by_src.setdefault(m["source"], []).append((hx, hy, m))
    if theta:
        r_ch = ctx.asec(theta) * 1.18
        for src, members in by_src.items():
            angs = sorted(math.degrees(math.atan2(y - dcy, x - dcx)) % 360.0 for x, y, _ in members)
            neg = any(mm.get("polarity") == "negative" for _, _, mm in members)
            for a0, a1 in zip(angs, angs[1:]):
                if a1 - a0 > 200:
                    continue
                n = max(8, int((a1 - a0) / 3))
                pts = [(dcx + r_ch * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                        dcy + r_ch * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
                       for i in range(n + 1)]
                ctx.poly(dr, pts, hair * 1.25, INK["lensed"], closed=False)
            for x, y, _ in members:                                      # radial ties down to anchors
                d = unit(x - dcx, y - dcy)
                ctx.poly(dr, [(dcx + r_ch * d[0], dcy + r_ch * d[1]),
                              (x - d[0] * ctx.f(0.011), y - d[1] * ctx.f(0.011))],
                         hair, INK["lensed"], closed=False)
            mid_a = math.radians(0.5 * (angs[0] + angs[-1]))
            ctx.text(ctx.d["labels"],
                     (dcx + r_ch * 1.30 * math.cos(mid_a), dcy + r_ch * 1.30 * math.sin(mid_a)),
                     src, ctx.f(S["label_alt"]), INK["lensed"])

    # negative marks: an anchor whose chord to the system is STRUCK
    for mid, (hx, hy, m) in anchors.items():
        if m.get("polarity") != "negative":
            continue
        d = unit(dcx - hx, dcy - hy)
        p1 = (hx + d[0] * ctx.f(0.10), hy + d[1] * ctx.f(0.10))
        ctx.poly(dr, [(hx + d[0] * ctx.f(0.012), hy + d[1] * ctx.f(0.012)), p1],
                 hair, INK["lensed"], closed=False)
        mx, my = (hx + p1[0]) / 2, (hy + p1[1]) / 2
        t = math.radians(60)
        n = (d[0] * math.cos(t) - d[1] * math.sin(t), d[0] * math.sin(t) + d[1] * math.cos(t))
        L = ctx.f(0.024)
        ctx.poly(dr, [(mx - n[0] * L, my - n[1] * L), (mx + n[0] * L, my + n[1] * L)],
                 hair * 1.6, INK["lensed"], closed=False)
        alt = (m.get("alternative") or "").replace("_", " ")
        pos = ctx.place_label((mx, my), (-d[0], -d[1]), alt, ctx.f(S["label_alt"]), keep)
        if pos:
            ctx.text(ctx.d["labels"], pos, alt, ctx.f(S["label_alt"]), INK["lensed"])

    for mid, (hx, hy, m) in anchors.items():
        if m["role"] == "dust_lane":
            _ideogram(ctx, dr, (hx, hy), ctx.f(0.036), ctx.f(S["glyph_stroke"]),
                      INK["obstruction"], "triple_bar")
        if emph_of(m) == "key":
            corner_brackets(ctx, (hx, hy), ctx.f(0.030))


# ==================================================================================================
# Reference arms — the two things the candidates must beat, implemented at their own constants.
# ==================================================================================================

LM_PALETTE = ["#00E000", "#FF00FF", "#00E5FF", "#E8E000", "#FFFFFF", "#FF9500", "#BFBFBF"]


def arm_reference_current(ctx: Ctx, sc: Scene):
    """Today's LensMark: heavy solid arrows, colour carries role, bold labels, on-image legend."""
    Sc = {"line_w": 0.0060, "head_len": 0.045, "head_w": 0.030, "tip_gap": 0.012, "label": 0.038}
    dr = ctx.d["normal"]
    for m in sc.marks:                                     # masks and rings, LensMark constants
        g = m["geometry"]
        if g["kind"] != "circle":
            continue
        cx, cy = ctx.uv(*sc.head(m))
        r = ctx.asec(g["radius_arcsec"])
        if m["role"] == "einstein_ring":
            if m.get("bound") != "nominal":
                continue                                   # today's format cannot express bounds
            ctx.stroked_circle(dr, cx, cy, r, ctx.f(0.0055), "#F2F2F2",
                               pattern="dot", dot_r=ctx.f(0.0022), dot_gap=3.0)
        elif m["role"] == "star":
            ctx.stroked_circle(dr, cx, cy, r, ctx.f(0.0055), "#FF6B6B",
                               pattern="dot", dot_r=ctx.f(0.0030), dot_gap=3.0)
        else:
            ctx.stroked_circle(dr, cx, cy, r, ctx.f(0.0055), "#FF6B6B",
                               pattern="dash", dash=ctx.f(0.030), gap=ctx.f(0.020))
    legend = []
    ci = 0
    for m in sc.marks:
        if m["role"] not in DESIGNATING or m["geometry"]["kind"] != "vector":
            continue
        colour = "#00E000" if m["role"] == "deflector" else LM_PALETTE[1 + (ci % 6)]
        if m["role"] != "deflector":
            ci += 1
        tail, head = ctx.uv(*m["geometry"]["tail"]), ctx.uv(*m["geometry"]["head"])
        d = unit(head[0] - tail[0], head[1] - tail[1])
        apex = (head[0] - d[0] * ctx.f(Sc["tip_gap"]), head[1] - d[1] * ctx.f(Sc["tip_gap"]))
        base = (apex[0] - d[0] * ctx.f(Sc["head_len"]), apex[1] - d[1] * ctx.f(Sc["head_len"]))
        n = (-d[1], d[0])
        hw = ctx.f(Sc["head_w"]) / 2
        dr.line([tail, base], fill=hexrgb(colour), width=int(round(ctx.f(Sc["line_w"]))))
        dr.polygon([apex, (base[0] + n[0] * hw, base[1] + n[1] * hw),
                    (base[0] - n[0] * hw, base[1] - n[1] * hw)], fill=hexrgb(colour))
        lbl = label_for(m)[1]
        size = ctx.f(Sc["label"])
        f = ctx.font(size, bold=True)
        ctx.d["labels"].text(tail, lbl, font=f, fill=hexrgb(colour), anchor="mm",
                             stroke_width=2 * SS, stroke_fill=hexrgb("#000000", 0.80))
        legend.append((colour, lbl))
    # the on-image legend plate — measured at 3.7-6.6% of the panel in the real renders
    size = ctx.f(0.030)
    pad = ctx.f(0.012)
    wmax = max(ctx.text_box("→ " + t, size)[0] for _, t in legend)
    rowh = size * 1.3
    ph, pw = len(legend) * rowh + 2 * pad, wmax + 2 * pad
    x0, y0 = 2 * pad, ctx.bh * SS - ph - 2 * pad
    ctx.d["top"].rectangle([x0, y0, x0 + pw, y0 + ph], fill=hexrgb("#000000", 0.69))
    for i, (colour, t) in enumerate(legend):
        ctx.d["top"].text((x0 + pad, y0 + pad + rowh * (i + 0.5)), "→ " + t,
                          font=ctx.font(size), fill=hexrgb(colour), anchor="lm")


def arm_reference_campaign(ctx: Ctx, sc: Scene):
    """The agent campaign the room liked: heavy arrows, NUMBERED masks, key below the image."""
    dr = ctx.d["normal"]
    n = 0
    for m in sc.marks:
        g = m["geometry"]
        if g["kind"] != "circle":
            continue
        cx, cy = ctx.uv(*sc.head(m))
        r = ctx.asec(g["radius_arcsec"])
        if m["role"] == "einstein_ring":
            if m.get("bound") != "nominal":
                continue
            ctx.stroked_circle(dr, cx, cy, r, ctx.f(0.0030), "#C9B6FF", pattern="solid")
            continue
        pat = "dot" if m["role"] == "star" else "dash"
        ctx.stroked_circle(dr, cx, cy, r, ctx.f(0.0030), "#FFFFFF", pattern=pat,
                           dash=ctx.f(0.014), gap=ctx.f(0.011),
                           dot_r=ctx.f(0.0022), dot_gap=2.2)
        n += 1
        ctx.d["labels"].text((cx + r * 0.72, cy + r * 0.72), str(n), font=ctx.font(ctx.f(0.026)),
                             fill=hexrgb("#FFFFFF"), anchor="mm")
    ci = 0
    for m in sc.marks:
        if m["role"] not in DESIGNATING or m["geometry"]["kind"] != "vector":
            continue
        colour = ["#FF3B30", "#00E000", "#E8E000", "#FFFFFF", "#00E5FF", "#FF00FF",
                  "#FF9500"][ci % 7]
        ci += 1
        tail, head = ctx.uv(*m["geometry"]["tail"]), ctx.uv(*m["geometry"]["head"])
        d = unit(head[0] - tail[0], head[1] - tail[1])
        apex = (head[0] - d[0] * ctx.f(0.009), head[1] - d[1] * ctx.f(0.009))
        base = (apex[0] - d[0] * ctx.f(0.036), apex[1] - d[1] * ctx.f(0.036))
        nn = (-d[1], d[0])
        hw = ctx.f(0.026) / 2
        dr.line([tail, base], fill=hexrgb(colour), width=int(round(ctx.f(0.0048))))
        dr.polygon([apex, (base[0] + nn[0] * hw, base[1] + nn[1] * hw),
                    (base[0] - nn[0] * hw, base[1] - nn[1] * hw)], fill=hexrgb(colour))


def nice_scale(fov: float) -> float:
    """Largest round bar that stays under ~30% of the field."""
    for v in (10.0, 5.0, 2.0, 1.0, 0.5, 0.2):
        if v <= 0.32 * fov:
            return v
    return round(0.3 * fov, 2)


def scale_bar(ctx: Ctx, arcsec: float | None = None):
    """The one mark allowed inside the image rectangle that is not about a specific location:
    a length comparison needs physical adjacency to the pixels being compared."""
    arcsec = arcsec if arcsec is not None else nice_scale(ctx.cutout_arcsec)
    L = ctx.asec(arcsec)
    pad = ctx.f(0.030)
    x1, y1 = ctx.bw * SS - pad, ctx.bh * SS - pad
    x0 = x1 - L
    ctx.poly(ctx.d["top"], [(x0, y1), (x1, y1)], ctx.f(0.0035), "#FFFFFF", closed=False)
    ctx.text(ctx.d["top"], ((x0 + x1) / 2, y1 - ctx.f(0.022)), f'{arcsec:g}"',
             ctx.f(0.024), "#FFFFFF")


# ==================================================================================================
# The caption band — the key lives here, below the image, costing zero image pixels.
# ==================================================================================================

def draw_caption(canvas: Image.Image, ctx: Ctx, sc: Scene, arm: str, key_rows: list[tuple[str, str]]):
    """Lay the band out as a sequence of wrapped rows. The band is sized by caption_rows() in a
    first pass, so this never has to guess whether the content fits."""
    d = ImageDraw.Draw(canvas)
    W = canvas.width
    pad = int(W * 0.018)
    fs = max(11.0, W * 0.0165)
    root = Path(__file__).resolve().parents[4] / "apps/LensMark/lensmark/render/fonts"
    fb = ImageFont.truetype(str(root / "DejaVuSans-Bold.ttf"), int(fs))
    fr = ImageFont.truetype(str(root / "DejaVuSans.ttf"), int(fs * 0.92))

    y = ctx.bh + pad
    for text, font, fill in caption_rows(ctx, sc, arm, key_rows, fr, fb, W - 2 * pad):
        d.text((pad, y), text, font=font, fill=fill)
        y += (fs * 1.22) if font is fb else (fs * 1.16)


def _wrap(text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.getbbox(t)[2] <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def caption_rows(ctx, sc, arm, key_rows, fr, fb, max_w):
    """The band's content, as (text, font, colour) rows. Used to size the band and to draw it."""
    sysd = sc.sys
    th = sysd.get("theta_e") or {}
    ident = Path(sc.c.get("image", {}).get("file", "astromark-ref")).stem
    rows = [(f"{ident}  ·  arm {arm}", fb, (255, 255, 255))]

    bits = [sysd.get("verdict", "?")]
    if sysd.get("grade"):
        bits.append(f'grade {sysd["grade"]}')
    if th.get("value_arcsec"):
        b = f'θ_E ≈ {th["value_arcsec"]:.2f}"'
        if th.get("lower_arcsec") and th.get("upper_arcsec"):
            b += f' [{th["lower_arcsec"]:.2f}–{th["upper_arcsec"]:.2f}]'
        if th.get("method"):
            b += f' ({th["method"]})'
        bits.append(b)
    if sysd.get("counter_image"):
        bits.append("counter-image " + sysd["counter_image"].replace("_", " "))
    if sysd.get("n_images"):
        bits.append(f'{sysd["n_images"]} image' + ("s" if sysd["n_images"] != 1 else ""))
    for ln in _wrap("  ·  ".join(bits), fr, max_w):
        rows.append((ln, fr, (210, 210, 214)))

    if sysd.get("hard_case"):
        for ln in _wrap("hard cases: " + ", ".join(h.replace("_", " ") for h in sysd["hard_case"]),
                        fr, max_w):
            rows.append((ln, fr, (210, 210, 214)))
    if ctx.demoted:
        for ln in _wrap("  ".join(f"{n} {t}" for n, t, _ in ctx.demoted), fr, max_w):
            rows.append((ln, fr, (235, 235, 240)))
    if key_rows:
        for ln in _wrap("   ·   ".join(f"{g} {t}" for g, t in key_rows), fr, max_w):
            rows.append((ln, fr, (190, 190, 196)))
    ps = ctx.cutout_arcsec / ctx.bw * UPSCALE
    rows.append((f'scale: {ctx.cutout_arcsec:g}″ across, {ps:.4f}″/px, shown at {UPSCALE}×',
                 fr, (140, 140, 146)))
    return rows


ARMS = {
    "N1": ("Terminator alphabet", arm_N1,
           [("filled triangle:", "lens mass"), ("open crescent:", "lensed light"),
            ("double bar:", "obstruction"), ("solid:", "positive"),
            ("dashed + struck:", "negative"), ("dotted + hollow:", "ambiguous")]),
    "N2": ("Bertin ledger", arm_N2,
           [("radial tick", "lens mass"), ("tangential tick", "lensed light"),
            ("double tangential", "counter-image"), ("solid/dash/dot", "positive/negative/ambiguous")]),
    "N3": ("Station model", arm_N3,
           [("ring+dot:", "deflector"), ("arc:", "lensed"), ("letter:", "source index"),
            ("underline:", "ambiguous"), ("strike:", "negative"), ("x / f:", "mask / model")]),
    "N4": ("Evidence graph", arm_N4,
           [("open ring:", "cited feature"), ("chord:", "images of one source"),
            ("stem to dot:", "lens mass"), ("struck chord:", "refuted")]),
    "R-CURRENT": ("Reference: LensMark today", arm_reference_current, []),
    "R-CAMPAIGN": ("Reference: agent campaign", arm_reference_campaign, []),
    "NULL": ("Null: no annotation", lambda ctx, sc: None, []),
}


def render_arm(arm: str, content: dict, base: Image.Image, caption: bool = True) -> Image.Image:
    title, fn, key = ARMS[arm]
    sc = Scene(content)
    # two passes: draw first so the demotion list is known, then size the band to what must fit
    probe = Ctx(base, content["image"]["cutout_arcsec"], caption_h=0)
    fn(probe, sc)
    W_out = base.width * UPSCALE
    fs = max(11.0, W_out * 0.0165)
    root = Path(__file__).resolve().parents[4] / "apps/LensMark/lensmark/render/fonts"
    fb = ImageFont.truetype(str(root / "DejaVuSans-Bold.ttf"), int(fs))
    fr = ImageFont.truetype(str(root / "DejaVuSans.ttf"), int(fs * 0.92))
    pad = int(W_out * 0.018)
    n_rows = len(caption_rows(probe, sc, title, key, fr, fb, W_out - 2 * pad))
    cap_h = int(fs * 1.22 * n_rows + 2 * pad) if caption else 0
    ctx = Ctx(base, content["image"]["cutout_arcsec"], caption_h=cap_h)
    fn(ctx, sc)
    if arm != "NULL":
        scale_bar(ctx)
    canvas = ctx.compose()
    if caption:
        draw_caption(canvas, ctx, sc, title, key)
    if ctx.warnings:
        print(f"  [{arm}] {len(ctx.warnings)} layout warning(s): {'; '.join(ctx.warnings[:4])}")
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    ap.add_argument("--content", type=Path, default=here / "content.json")
    ap.add_argument("--base", type=Path, default=here / "reference/astromark-ref.png")
    ap.add_argument("--out", type=Path, default=here / "reference")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    content = json.loads(args.content.read_text(encoding="utf-8"))
    base = Image.open(args.base).convert("RGB")
    args.out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        if arm not in ARMS:
            print(f"unknown arm {arm!r}; choose from {list(ARMS)}")
            return 2
        img = render_arm(arm, content, base, caption=not args.no_caption)
        dest = args.out / f"{args.prefix}{arm.lower()}.png"
        img.save(dest, optimize=False)
        print(f"wrote {dest}  {img.width}x{img.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
