#!/usr/bin/env python3
"""golden/annotate.py — draw the panel's explanation ONTO the composite.

A finished evidence-first run stores, per item, the advocate's LOCATED evidence items
(panel, radius, PA span), each critic's alternative with its location box, and the
arbitrator's rulings (`golden/schemas_panel.py`, rebuilt from the votes parquet by
`golden/records.py`). This module paints those records over the kit JPEG so a reader can
see WHERE the model said the arc is and WHERE a critic put its alternative, coloured by
how the arbitrator ruled — the justification, on the pixels it was made from.

Geometry (measured from `common/jwst_fetch.render_cutout`, the numbers `golden/views.py`
asserts at import): six 240-px panels a b c / d e f at the `views.PANEL_BOXES` boxes,
row 1 a 10" field (24 px/arcsec), row 2 a 3.5" zoom (68.57 px/arcsec), the ticked galaxy
at each panel's centre (x0+120, y0+120). North is up and East is LEFT (no flips: the
renderer labels NE top-left). Position angles follow the personas' convention — from
North (0) through East (90) — so a PA p sits in image direction (dx, dy) = (−sin p,
−cos p) and a span runs from `pa_deg_from` to `pa_deg_to` in the INCREASING-PA direction
(350 → 10 crosses North). PIL measures arc angles clockwise from +x in image coordinates,
so PA p is PIL angle (270 − p) mod 360 and a PA span from→to is the PIL arc from
angle(to) to angle(from) (`pil_arc_angles`); `pa_to_xy` / `pil_arc_angles` are the two
pure functions the tests pin to the four compass anchors and the wrap span.

Drawing rules:
  * evidence item k: a 2-px cyan arc at r_arcsec over its PA span, in its cited panel, in
    panel (a), and in panel (d) when r ≤ ZOOM_MAX_R (1.7"; a larger radius does not fit the
    3.5" zoom); a panel is used when ANY sampled point of the arc lies inside its box (the
    paste clips the rest — a 6" arc across a corner of the 10" field is drawn where it is
    visible); label "k<k>" at the arc midpoint, offset outward; a zero-length span is a
    small circle marker, a ≥ 360° span a full ring; `counter_image_pos` is a small cyan
    cross;
  * a critic's location box: a dashed annular sector r_from..r_to × pa_from..pa_to in
    panels (a) and (d), each when any point of its outline lies inside the box (an outer
    radius beyond the field is clipped, never dropped), coloured by the arbitrator's ruling
    for that critic (upheld red, partial orange, overruled grey, no ruling / no arbitrator
    yellow), labelled "<Art|Geo|Mor>: <alternative>" at its outer edge; abstaining
    (`no_opinion`) and unnamed critics draw nothing;
  * every overlay is drawn on per-panel layers and pasted inside the panel box (clipped);
    labels go on a layer BELOW the geometry, so a label can never hide an arc or a sector
    edge, and a label is nudged radially (outward, then inward) off any label already
    placed on that panel;
  * a legend strip is appended below the 752x540 kit image: the head (letters, veto,
    p_evidence, S / S_arb — from the `deploy` dict, `aggregate_v2.deploy_letters`, or the
    records — arb letter, scale, layout) wrapped to the canvas width, then one line per
    item (cyan; `*` marks an item the arbitrator kept; `[not drawn]` when no panel could
    show it, `[r > panel X]` when its cited panel's field is too small for that radius)
    and one per critic in its ruling colour (`[not drawn]` likewise).

Zero API, read-only on the run. Nothing here can reach a prompt. `--out-dir` may not be
the images directory or anything under the JWST kit tree (`check_out_dir`).

CLI:
    python lensjudge/golden/annotate.py --preds outputs/<run>.parquet [--votes …] \\
        --images-dir DIR --out-dir NEWDIR [--rule R1|R2] [--thresholds T --model-key K] \\
        [--trace-dir D] [--no-legend] [--limit N]
writes <name>_annot.jpg (quality 90) + <name>_orig.jpg (a byte copy) per item and pins
`annot_index.csv`. Images are looked up as <name>.jpg in --images-dir; a blind
scrambled name scr_NNN maps to NNN.jpg through `regrade_scrambled.scr_to_filename`.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.common import jwst_fetch as jf  # noqa: E402
from lensjudge.golden import _util, aggregate_v2, records as R, views  # noqa: E402

# ------------------------------------------------------------------ geometry (measured)
SLOTS = views.SLOTS                                   # a b c / d e f
PX = views.PX                                         # 240
PANEL_BOXES = dict(views.PANEL_BOXES["color"])        # (left, upper, right, lower) per slot
assert all(views.PANEL_BOXES[lay] == PANEL_BOXES for lay in views.LAYOUTS)   # layout-independent
FOV_ARCSEC = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 3.5, "e": 3.5, "f": 3.5}
assert {k: float(v) for k, v in views.BUILTIN_GLOSS["layouts"]["color"]["fov_arcsec"].items()} == FOV_ARCSEC
ZOOM_MAX_R = 1.7                                      # arcsec: also draw an item in (d) up to here
FIT_MARGIN_PX = 2                                     # an overlay must stay this far inside the box
KIT_SIZE = (jf.COMPOSITE_SIZE[0], jf.FOOTER_Y)        # (752, 540)

# ------------------------------------------------------------------ style
CYAN = (80, 220, 255)
RULING_COLOURS = {"upheld": (255, 80, 80), "partial": (255, 170, 40), "overruled": (170, 170, 170)}
NO_RULING_COLOUR = (250, 230, 80)
MUTED = (150, 150, 160)
LEGEND_BG = (12, 12, 16)
LEGEND_FG = (230, 230, 235)
LINE_W = 2
DASH_PX, GAP_PX = 6, 4
FONT_SIZE = 11
LEGEND_MIN_H = 70
LEGEND_LINE_H = 14
LEGEND_COL2_X = 392
SHORT_LABEL_PX = 60                                   # a sector smaller than this gets "Geo", not "Geo: spiral_arm"
WHAT_MAX = 40
ROLE_ABBREV = {"artifact": "Art", "geometry": "Geo", "morphology": "Mor"}
CRITIC_ROLES = aggregate_v2.CRITIC_ROLES
FONT_CANDIDATES = ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf")
INDEX_COLS = ("name", "layout", "rule", "letter_rank", "letter_final", "veto", "letter_llm", "p_evidence",
              "S", "S_arb", "n_items", "n_arcs_drawn", "n_sectors_drawn", "annot_file", "orig_file")
_SCR = re.compile(r"scr_\d{3}")


# ------------------------------------------------------------------ pure geometry
def panel_scale(slot: str) -> float:
    """Pixels per arcsecond in a panel (24 in row 1, 68.57 in row 2)."""
    return PX / FOV_ARCSEC[slot]


def panel_centre(slot: str) -> tuple[float, float]:
    """The ticked galaxy's pixel in composite coordinates (x0 + 120, y0 + 120)."""
    x0, y0, _, _ = PANEL_BOXES[slot]
    return (x0 + PX / 2.0, y0 + PX / 2.0)


def pa_to_xy(slot: str, r_arcsec: float, pa_deg: float, local: bool = False) -> tuple[float, float]:
    """The pixel at radius r (arcsec) and PA p (deg, N through E) from the panel centre:
    x = cx − r·sin p·s, y = cy − r·cos p·s (North up, East LEFT). `local` gives panel-box
    coordinates (centre (120, 120)) instead of composite coordinates."""
    s = panel_scale(slot)
    cx, cy = (PX / 2.0, PX / 2.0) if local else panel_centre(slot)
    p = math.radians(pa_deg)
    return (cx - r_arcsec * math.sin(p) * s, cy - r_arcsec * math.cos(p) * s)


def pa_to_pil_angle(pa_deg: float) -> float:
    """PIL's angle (clockwise from +x, y down) of the image direction of PA p: (270 − p) mod 360.
    PA 0 → 270 (12 o'clock), 90 → 180 (9 o'clock, East = left), 180 → 90, 270 → 0."""
    return (270.0 - float(pa_deg)) % 360.0


def span_deg(pa_from: float, pa_to: float) -> float:
    """The extent of a PA span in the increasing-PA direction: 0 (a point) .. 360 (a ring).
    350 → 10 is 20°, 10 → 350 is 340°, 0 → 360 is a full ring."""
    d = float(pa_to) - float(pa_from)
    if d >= 360.0:
        return 360.0
    return d % 360.0


def pil_arc_angles(pa_from: float, pa_to: float) -> tuple[float, float]:
    """(start, end) for ImageDraw.arc covering the PA span from→to. Increasing PA is
    counter-clockwise on screen (decreasing PIL angle), and PIL draws clockwise from start
    to end, so the arc runs from angle(pa_to) to angle(pa_from)."""
    return (pa_to_pil_angle(pa_to), pa_to_pil_angle(pa_from))


def mid_pa(pa_from: float, pa_to: float) -> float:
    return (float(pa_from) + span_deg(pa_from, pa_to) / 2.0) % 360.0


def fits(slot: str, r_arcsec: float) -> bool:
    """Whether a circle of radius r (arcsec) about the panel centre stays inside the box."""
    return float(r_arcsec) * panel_scale(slot) <= PX / 2.0 - FIT_MARGIN_PX


def point_inside(xy) -> bool:
    """Whether a panel-local point lies inside the box, FIT_MARGIN_PX from its edges."""
    x, y = xy
    return FIT_MARGIN_PX <= x <= PX - FIT_MARGIN_PX and FIT_MARGIN_PX <= y <= PX - FIT_MARGIN_PX


def any_inside(pts) -> bool:
    return any(point_inside(p) for p in pts)


def r_exceeds_panel(panel: str, r_arcsec: float) -> bool:
    """Whether a radius is larger than half the cited panel's field (10" row: 5"; 3.5" zoom
    row: 1.75") — the model cited a panel that cannot show that radius on-axis."""
    return panel in FOV_ARCSEC and float(r_arcsec) > FOV_ARCSEC[panel] / 2.0


def arc_visible(slot: str, item: Any) -> bool:
    """Whether any sampled point of an evidence item's arc (point marker / ring included)
    lies inside the panel box — the criterion for drawing it there (the paste clips)."""
    r = float(item.r_arcsec)
    span = span_deg(item.pa_deg_from, item.pa_deg_to)
    if span <= 0.0 or r * panel_scale(slot) < 1.0:
        return point_inside(pa_to_xy(slot, r, item.pa_deg_from, local=True))
    if span >= 360.0:
        return any_inside(arc_points(slot, r, 0.0, 360.0))
    return any_inside(arc_points(slot, r, item.pa_deg_from, item.pa_deg_to))


def item_slots(item: Any) -> list[str]:
    """Where an evidence item is drawn: its cited panel (when it is a composite slot), panel
    (a), and panel (d) when r ≤ ZOOM_MAX_R — each only where some of the arc is visible."""
    out: list[str] = []
    panel = str(getattr(item, "panel", ""))
    r = float(item.r_arcsec)
    if panel in SLOTS:
        out.append(panel)
    if "a" not in out:
        out.append("a")
    if r <= ZOOM_MAX_R and "d" not in out:
        out.append("d")
    return [s for s in out if arc_visible(s, item)]


def sector_outline(slot: str, location: Any) -> list:
    """Sampled panel-local points of a location box's outline: outer arc, inner arc and
    (unless a full ring) the two radial edges."""
    r_in, r_out = sorted((float(location.r_arcsec_from), float(location.r_arcsec_to)))
    pa_from, pa_to = float(location.pa_deg_from), float(location.pa_deg_to)
    span = span_deg(pa_from, pa_to)
    if span <= 0.0:
        pa_to, span = pa_from + 4.0, 4.0
    pts = arc_points(slot, r_out, pa_from, pa_to) + arc_points(slot, r_in, pa_from, pa_to)
    if span < 360.0:
        for pa in (pa_from, pa_to):
            pts += [pa_to_xy(slot, r_in + (r_out - r_in) * i / 8.0, pa, local=True) for i in range(9)]
    return pts


def sector_slots(location: Any) -> list[str]:
    """Where a critic's location box is drawn: panels (a) and (d), each when any point of
    the sector's outline lies inside the box (an outer radius beyond the field is clipped)."""
    return [s for s in ("a", "d") if any_inside(sector_outline(s, location))]


def ruling_for(role: str, arbitrator: Any) -> Optional[str]:
    """The arbitrator's ruling for a critic role (upheld / partial / overruled), None without one."""
    if arbitrator is None:
        return None
    for ru in getattr(arbitrator, "rulings", None) or []:
        if str(getattr(ru, "persona", "")) == role:
            return str(ru.ruling)
    return None


def ruling_colour(ruling: Optional[str]) -> tuple[int, int, int]:
    return RULING_COLOURS.get(ruling, NO_RULING_COLOUR) if ruling else NO_RULING_COLOUR


# ------------------------------------------------------------------ PIL helpers
def _pil():
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


_FONT_CACHE: dict = {}


def load_font(size: int = FONT_SIZE):
    """A TrueType face from the system when one exists (Helvetica / Arial / DejaVu), else
    PIL's default face at that size."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    _, _, ImageFont = _pil()
    font = None
    for cand in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(cand, size)
            break
        except (OSError, ValueError):
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:                       # Pillow < 10.1
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _text_size(draw, text: str, font) -> tuple[int, int]:
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:                            # bitmap fonts on old Pillow
        return (int(6 * len(text)), 11)


def _text(draw, xy, text: str, font, fill, anchor: str = "mm", stroke: bool = True) -> None:
    """Anchored text with a 1-px dark stroke; degrades to top-left placement and no stroke
    on a font that supports neither."""
    kw = {"stroke_width": 1, "stroke_fill": (0, 0, 0)} if stroke else {}
    try:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor, **kw)
        return
    except Exception:
        pass
    w, h = _text_size(draw, text, font)
    x, y = xy
    if anchor and anchor[0] == "m":
        x -= w / 2.0
    if anchor and anchor[-1] == "m":
        y -= h / 2.0
    draw.text((x, y), text, font=font, fill=fill)


LABEL_NUDGES_PX = (0, 12, 24, 36, -12, -24, -36, -48)   # along the label's radius from the panel centre
LABEL_SIDE_PX = (14, -14, 28, -28)                    # then across it (three critics on one spot)


def _place_label(xy, w: int, h: int, margin: int = 3, placed: Optional[list] = None,
                 record: bool = True) -> Optional[tuple[float, float]]:
    """A clash-free centre for a w×h label near `xy` inside the panel layer, or None when
    every candidate overlaps a box in `placed`: the clamped origin, then RADIAL nudges —
    outward from the panel centre first, then inward (a vertical nudge would land a label
    from the top of a small overlay on the arc it annotates) — then sideways (perpendicular
    to the radius, for three critics on one corner). With `placed` None the clamped origin
    is returned; `record` appends the chosen box to `placed`."""
    x0, y0 = xy
    cx = cy = PX / 2.0

    def clamp(x, y):
        return (min(max(x, margin + w / 2.0), PX - margin - w / 2.0),
                min(max(y, margin + h / 2.0), PX - margin - h / 2.0))

    if placed is None:
        return clamp(x0, y0)
    dx, dy = x0 - cx, y0 - cy
    norm = math.hypot(dx, dy)
    ux, uy = (dx / norm, dy / norm) if norm > 1e-9 else (0.0, -1.0)

    def clash(xx, yy):
        return any(abs(xx - px) < (w + pw) / 2.0 + 2 and abs(yy - py) < (h + ph) / 2.0 + 1
                   for px, py, pw, ph in placed)

    candidates = [(x0 + ux * d, y0 + uy * d) for d in LABEL_NUDGES_PX]
    candidates += [(x0 - uy * s + ux * d, y0 + ux * s + uy * d) for s in LABEL_SIDE_PX for d in (0, -12, -24)]
    for xc, yc in candidates:
        xx, yy = clamp(xc, yc)
        if not clash(xx, yy):
            if record:
                placed.append((xx, yy, w, h))
            return (xx, yy)
    return None


def _clamp_label(xy, w: int, h: int, margin: int = 3, placed: Optional[list] = None) -> tuple[float, float]:
    """`_place_label`, falling back to the clamped origin (recorded) when nothing is
    clash-free — a label is always drawn."""
    pos = _place_label(xy, w, h, margin, placed)
    if pos is not None:
        return pos
    x, y = (min(max(xy[0], margin + w / 2.0), PX - margin - w / 2.0),
            min(max(xy[1], margin + h / 2.0), PX - margin - h / 2.0))
    if placed is not None:
        placed.append((x, y, w, h))
    return (x, y)


def dashed_polyline(draw, pts: list, fill, width: int = LINE_W, dash: int = DASH_PX, gap: int = GAP_PX) -> None:
    """A polyline drawn as `dash` px on / `gap` px off segments."""
    on, left = True, float(dash)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        t = 0.0
        while seg - t > 1e-9:
            step = min(left, seg - t)
            if on:
                a = (x0 + (x1 - x0) * t / seg, y0 + (y1 - y0) * t / seg)
                b = (x0 + (x1 - x0) * (t + step) / seg, y0 + (y1 - y0) * (t + step) / seg)
                draw.line([a, b], fill=fill, width=width)
            t += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = float(dash if on else gap)


def arc_points(slot: str, r_arcsec: float, pa_from: float, pa_to: float, step_deg: float = 2.0) -> list:
    """Sampled points (panel-local) along the arc at r over the PA span from→to."""
    span = span_deg(pa_from, pa_to)
    n = max(2, int(math.ceil(span / step_deg)) + 1)
    return [pa_to_xy(slot, r_arcsec, float(pa_from) + span * i / (n - 1), local=True) for i in range(n)]


# ------------------------------------------------------------------ overlays (panel-local)
def draw_item(draw, slot: str, item: Any, font, colour=CYAN, placed: Optional[list] = None,
              label_draw=None) -> None:
    """One evidence item on a panel layer: the arc (or point marker / ring) + its "k" label
    (on `label_draw` when given — the label layer that sits below the geometry)."""
    cx, cy = PX / 2.0, PX / 2.0
    r_px = float(item.r_arcsec) * panel_scale(slot)
    span = span_deg(item.pa_deg_from, item.pa_deg_to)
    bbox = [cx - r_px, cy - r_px, cx + r_px, cy + r_px]
    if span <= 0.0 or r_px < 1.0:
        x, y = pa_to_xy(slot, item.r_arcsec, item.pa_deg_from, local=True)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=colour, width=LINE_W)
    elif span >= 360.0:
        draw.ellipse(bbox, outline=colour, width=LINE_W)
    else:
        start, end = pil_arc_angles(item.pa_deg_from, item.pa_deg_to)
        draw.arc(bbox, start, end, fill=colour, width=LINE_W)
    label = f"k{int(item.k)}"
    ld = draw if label_draw is None else label_draw
    w, h = _text_size(ld, label, font)
    pa_mid = 0.0 if span >= 360.0 else mid_pa(item.pa_deg_from, item.pa_deg_to)
    r_lab = r_px + 10.0 if r_px + 10.0 <= PX / 2.0 - 6 else max(r_px - 10.0, 6.0)
    lx, ly = pa_to_xy(slot, r_lab / panel_scale(slot), pa_mid, local=True)
    _text(ld, _clamp_label((lx, ly), w, h, placed=placed), label, font, colour)


def draw_cross(draw, slot: str, r_arcsec: float, pa_deg: float, colour=CYAN, arm: int = 5) -> None:
    x, y = pa_to_xy(slot, r_arcsec, pa_deg, local=True)
    draw.line([x - arm, y, x + arm, y], fill=colour, width=LINE_W)
    draw.line([x, y - arm, x, y + arm], fill=colour, width=LINE_W)


def draw_sector(draw, slot: str, role: str, critic: Any, colour, font, placed: Optional[list] = None,
                label_draw=None) -> None:
    """A critic's location box as a dashed annular sector + "<Role>: <alternative>" label
    (the role abbreviation alone when the sector is under SHORT_LABEL_PX across — the 10"
    panels — the legend carries the alternative). The label goes on `label_draw` when
    given (the layer below the geometry)."""
    loc = critic.location
    r_in, r_out = sorted((float(loc.r_arcsec_from), float(loc.r_arcsec_to)))
    s = panel_scale(slot)
    span = span_deg(loc.pa_deg_from, loc.pa_deg_to)
    pa_from, pa_to = float(loc.pa_deg_from), float(loc.pa_deg_to)
    if span <= 0.0:                               # a degenerate PA range: draw as a thin wedge
        pa_to = pa_from + 4.0
        span = 4.0
    if r_out * s >= 1.0:
        dashed_polyline(draw, arc_points(slot, r_out, pa_from, pa_to), colour)
    if r_in * s >= 2.0:
        dashed_polyline(draw, arc_points(slot, r_in, pa_from, pa_to), colour)
    if span < 360.0:
        for pa in (pa_from, pa_to):
            dashed_polyline(draw, [pa_to_xy(slot, r_in, pa, local=True), pa_to_xy(slot, r_out, pa, local=True)],
                            colour)
    abbrev = ROLE_ABBREV.get(role, role[:3].title())
    label = abbrev if r_out * s < SHORT_LABEL_PX else f"{abbrev}: {critic.alternative}"
    ld = draw if label_draw is None else label_draw
    pa_mid = 0.0 if span >= 360.0 else mid_pa(pa_from, pa_to)
    r_lab = r_out * s + 8.0
    if r_lab > PX / 2.0 - 8:
        r_lab = max(r_out * s - 9.0, 8.0)
    lx, ly = pa_to_xy(slot, r_lab / s, pa_mid, local=True)
    # the full "<Role>: <alternative>" label where it fits without covering another label;
    # when three critics share a corner, the role abbreviation alone (the legend carries the
    # alternative) is tried before anything is allowed to overlap
    pos = None
    for text in ([label, abbrev] if label != abbrev else [abbrev]):
        w, h = _text_size(ld, text, font)
        pos = _place_label((lx, ly), w, h, placed=placed)
        if pos is not None:
            label = text
            break
    if pos is None:
        w, h = _text_size(ld, abbrev, font)
        label, pos = abbrev, _clamp_label((lx, ly), w, h, placed=placed)
    _text(ld, pos, label, font, colour)


# ------------------------------------------------------------------ legend
def _fmt(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(v) else f"{v:.{nd}f}"


def _truncate(s: str, n: int = WHAT_MAX) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


NOT_DRAWN = "[not drawn]"
HEAD_SEP = "   "


def item_marker(item: Any) -> str:
    """"" / "[not drawn]" / "[r > panel X]": whether the item reaches the composite and
    whether its cited panel's field can hold its radius (a zoom panel cited at r > 1.75")."""
    if not item_slots(item):
        return NOT_DRAWN
    panel = str(getattr(item, "panel", ""))
    if r_exceeds_panel(panel, item.r_arcsec):
        return f"[r > panel {panel}]"
    return ""


def legend_lines(records: dict, deploy: Optional[dict] = None, layout: Optional[str] = None) -> dict:
    """{"head": (text, colour), "head_parts": [str], "items": [(text, colour)], "critics":
    [(text, colour)]} — the legend's text. `head` is the head parts joined by HEAD_SEP;
    `wrap_parts` breaks them into canvas-wide rows at draw time. `deploy` is an
    `aggregate_v2.deploy_letters` dict (letter_rank, letter_final, veto, S, S_arb,
    p_evidence, rule); without one the numbers come from the records (p_evidence, score_S /
    score_S_arb, the arbitrator's advisory letter). An item no panel can show carries
    `[not drawn]`, one whose cited panel is too small for its radius `[r > panel X]`; a
    located critic whose sector fits nowhere carries `[not drawn]` too."""
    adv = records.get("advocate")
    arb = records.get("arbitrator")
    critics = {r: records[r] for r in CRITIC_ROLES if r in records}
    parts: list[str] = []
    if deploy:
        rule = deploy.get("rule") or "R1"
        parts.append(f"{rule}: rank {deploy.get('letter_rank') or '-'} -> final {deploy.get('letter_final') or '-'}")
        if deploy.get("veto"):
            parts.append(f"veto {deploy['veto']}")
        p_ev, S, S_arb = deploy.get("p_evidence"), deploy.get("S"), deploy.get("S_arb")
    elif adv is not None:
        p_ev = adv.p_evidence
        S = aggregate_v2.score_S(adv, critics)
        S_arb = aggregate_v2.score_S_arb(adv, critics, arb)
    else:
        p_ev = S = S_arb = None
    if adv is None:
        parts.append("no advocate record (parse failure)")
    parts.append(f"p_ev {_fmt(p_ev, 2)}  S {_fmt(S)}  S_arb {_fmt(S_arb)}")
    if arb is not None:
        parts.append(f"arb letter {arb.letter_llm}" + ("  needs_human" if arb.needs_human else ""))
        parts.append("* = survives")
    elif "arbitrator" in records:
        parts.append("arbitrator: parse failure")
    scale = (arb.scale_class_final if arb is not None and arb.scale_class_final else
             (adv.scale_class if adv is not None else None))
    if scale:
        parts.append(f"scale {scale}")
    if layout:
        parts.append(f"layout {layout}")
    head = (HEAD_SEP.join(parts), LEGEND_FG)

    items: list[tuple[str, tuple]] = []
    if adv is not None:
        surviving = set(int(k) for k in arb.surviving_items) if arb is not None else None
        for it in adv.items:
            mark = "*" if surviving is not None and int(it.k) in surviving else ""
            marker = item_marker(it)
            items.append((f"k{int(it.k)}{mark} {_truncate(it.what)}" + (f" {marker}" if marker else ""), CYAN))
        if not adv.items and adv.nothing_because:
            items.append((f"no items: {_truncate(adv.nothing_because, 60)}", MUTED))
    crit_lines: list[tuple[str, tuple]] = []
    for role in CRITIC_ROLES:
        if role not in records:
            continue
        c, ab = records[role], ROLE_ABBREV[role]
        if c is None:
            crit_lines.append((f"{ab}: parse failure", MUTED))
        elif c.no_opinion:
            crit_lines.append((f"{ab}: no opinion ({c.no_opinion_reason or '-'})", MUTED))
        elif c.alternative is None:
            crit_lines.append((f"{ab}: none named", MUTED))
        else:
            ruling = ruling_for(role, arb)
            marker = f" {NOT_DRAWN}" if c.location is not None and not sector_slots(c.location) else ""
            crit_lines.append((f"{ab}: {c.alternative} - {ruling or 'no ruling'} r={_fmt(c.refutation_strength, 2)}"
                               + marker, ruling_colour(ruling)))
    return {"head": head, "head_parts": parts, "items": items, "critics": crit_lines}


def wrap_parts(draw, parts: list, font, max_w: int, sep: str = HEAD_SEP) -> list[str]:
    """Greedily join `parts` with `sep` into rows no wider than `max_w` px as `draw`
    measures them with `font`. A single part wider than the row is split on ';' (a
    three-role veto) and then on spaces; nothing is ever dropped."""
    def width(s: str) -> int:
        return _text_size(draw, s, font)[0]

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


def legend_height(lines: dict, n_head: int = 1) -> int:
    n = max(1, int(n_head)) + max(len(lines["items"]), len(lines["critics"]))
    return max(LEGEND_MIN_H, 6 + LEGEND_LINE_H * n + 4)


# ------------------------------------------------------------------ the composite
def _as_kit(img):
    """The 752x540 RGB kit view of a composite (footer cropped when present)."""
    if img.size[0] != jf.COMPOSITE_SIZE[0] or img.size[1] not in (jf.COMPOSITE_SIZE[1], jf.FOOTER_Y):
        raise ValueError(f"composite is {img.size}, expected 752x562 or 752x540")
    return views.full_view(img).convert("RGB")


def annotate_composite(img, records: dict, layout: str, deploy: Optional[dict] = None,
                       legend: bool = True):
    """The kit image with the records drawn on it (RGB; 752x540, plus the legend strip
    below when `legend`). `records` is {role: record | None} as `records.records_from_votes`
    yields for one item (absent roles absent, failed ones None); `layout` is the frame /
    manifest layout (color, gray_sw_only, gray_lw_only — validated, geometry identical);
    `deploy` an optional `aggregate_v2.deploy_letters` dict for the legend's first line."""
    Image, ImageDraw, _ = _pil()
    views.layout_kind(layout)                      # refuses an unknown layout
    base = _as_kit(img).copy()
    font = load_font(FONT_SIZE)
    adv = records.get("advocate")
    arb = records.get("arbitrator")
    layers: dict = {}                              # slot -> (geometry layer, label layer)
    placed: dict = {}                              # slot -> label boxes already on that panel

    def layer(slot):
        """(geometry draw, label draw) for a panel: two RGBA layers, the labels pasted
        first so the geometry is never hidden by a label."""
        if slot not in layers:
            G = Image.new("RGBA", (PX, PX), (0, 0, 0, 0))
            T = Image.new("RGBA", (PX, PX), (0, 0, 0, 0))
            layers[slot] = (G, T, ImageDraw.Draw(G), ImageDraw.Draw(T))
            placed[slot] = []
        return layers[slot][2], layers[slot][3]

    if adv is not None:
        for it in adv.items:
            for slot in item_slots(it):
                g, t = layer(slot)
                draw_item(g, slot, it, font, placed=placed[slot], label_draw=t)
        cip = adv.counter_image_pos
        if cip is not None:
            for slot in ("a", "d"):
                if fits(slot, cip.r_arcsec):
                    draw_cross(layer(slot)[0], slot, cip.r_arcsec, cip.pa_deg)
    for role in CRITIC_ROLES:
        c = records.get(role)
        if c is None or c.no_opinion or c.alternative is None or c.location is None:
            continue
        colour = ruling_colour(ruling_for(role, arb))
        for slot in sector_slots(c.location):
            g, t = layer(slot)
            draw_sector(g, slot, role, c, colour, font, placed=placed[slot], label_draw=t)
    for slot, (G, T, _, _) in layers.items():
        x0, y0, _, _ = PANEL_BOXES[slot]
        base.paste(T, (x0, y0), T)               # labels below …
        base.paste(G, (x0, y0), G)               # … the geometry; alpha-pasted: clipped to the panel box

    if not legend:
        return base
    lines = legend_lines(records, deploy, layout)
    scratch = ImageDraw.Draw(base)
    head_rows = wrap_parts(scratch, lines["head_parts"], font, base.width - 16)
    h = legend_height(lines, len(head_rows))
    out = Image.new("RGB", (base.width, base.height + h), LEGEND_BG)
    out.paste(base, (0, 0))
    d = ImageDraw.Draw(out)
    y = base.height + 6
    _, colour = lines["head"]
    for i, text in enumerate(head_rows):
        _text(d, (8, y + LEGEND_LINE_H * i), text, font, colour, anchor="la", stroke=False)
    y += LEGEND_LINE_H * (len(head_rows) - 1)
    for i, (text, colour) in enumerate(lines["items"]):
        _text(d, (8, y + LEGEND_LINE_H * (i + 1)), text, font, colour, anchor="la", stroke=False)
    for i, (text, colour) in enumerate(lines["critics"]):
        _text(d, (LEGEND_COL2_X, y + LEGEND_LINE_H * (i + 1)), text, font, colour, anchor="la", stroke=False)
    return out


def overlay_counts(records: dict) -> dict:
    """{n_items, n_arcs_drawn, n_sectors_drawn}: what `annotate_composite` will draw."""
    adv, arb = records.get("advocate"), records.get("arbitrator")
    n_items = len(adv.items) if adv is not None else 0
    n_arcs = sum(len(item_slots(it)) for it in adv.items) if adv is not None else 0
    n_sec = 0
    for role in CRITIC_ROLES:
        c = records.get(role)
        if c is not None and not c.no_opinion and c.alternative is not None and c.location is not None:
            n_sec += len(sector_slots(c.location))
    return {"n_items": n_items, "n_arcs_drawn": n_arcs, "n_sectors_drawn": n_sec}


# ------------------------------------------------------------------ a run on disk
def kit_filename_for(name: str) -> list[str]:
    """Candidate file names for an item in --images-dir: <name>.jpg, its safe form, and for
    a blind scrambled name scr_NNN the kit's NNN.jpg (`regrade_scrambled.scr_to_filename`)."""
    name = str(name)
    out = [f"{name}.jpg"]
    safe = f"{_util.safe_name(name)}.jpg"
    if safe not in out:
        out.append(safe)
    if _SCR.fullmatch(name):
        from lensjudge.golden import regrade_scrambled
        out.append(regrade_scrambled.scr_to_filename(name))
    return out


def find_image(images_dir, name: str) -> Optional[Path]:
    d = Path(images_dir)
    for fn in kit_filename_for(name):
        p = d / fn
        if p.exists():
            return p
    return None


def deploy_for(recs: dict, thresholds: Optional[dict], rule: str) -> Optional[dict]:
    """`aggregate_v2.deploy_letters` on one item's records, None without thresholds."""
    if thresholds is None:
        return None
    return R.deploy_from_roles(recs, dict(thresholds), rule)   # voids a called-but-failed arbitrator


def _protected_dirs(images_dir) -> list[Path]:
    from lensjudge.golden import regrade_scrambled
    return [Path(images_dir), _util.JWST_REPO, regrade_scrambled.SCRAMBLED_DIR_DEFAULT]


def check_out_dir(out_dir, images_dir) -> Path:
    """The resolved `out_dir`, or SystemExit when it IS or lies under the images directory,
    the JWST repo (`_util.JWST_REPO`) or the scrambled kit — the kit trees are read-only and
    an annotate run drops <name>_annot.jpg / <name>_orig.jpg / annot_index.csv into its
    --out-dir."""
    out = Path(out_dir).resolve()
    for p in _protected_dirs(images_dir):
        p = Path(p).resolve()
        if out == p or p in out.parents:
            raise SystemExit(f"[annotate] REFUSED: --out-dir {out} is {'the' if out == p else 'under the'} "
                             f"read-only directory {p}; write into a NEW directory")
    return out


def annotate_run(preds: pd.DataFrame, records: dict, images_dir, out_dir, rule: str = "R1",
                 thresholds: Optional[dict] = None, legend: bool = True, limit: Optional[int] = None,
                 quality: int = 90) -> pd.DataFrame:
    """Annotate every preds row that has records and an image: <name>_annot.jpg +
    <name>_orig.jpg in `out_dir`; returns the index frame (INDEX_COLS). Thresholds per row
    come from the stored run-tuple columns unless one dict is given for all; a row without
    usable thresholds gets a records-only legend. `out_dir` is checked against the
    images directory and the kit trees first (`check_out_dir`)."""
    Image, _, _ = _pil()
    check_out_dir(out_dir, images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for _, row in preds.iterrows():
        name = str(row["name"])
        recs = records.get(name)
        src = find_image(images_dir, name)
        if recs is None or src is None:
            continue
        if limit is not None and n >= limit:
            break
        thr = thresholds
        if thr is None:
            try:
                thr = R.thresholds_from_row(row)
            except ValueError:
                thr = None
        deploy = deploy_for(recs, thr, rule)
        layout = row.get("layout") if hasattr(row, "get") else None
        layout = "color" if R.is_missing(layout) else str(layout)
        with Image.open(src) as im:
            out = annotate_composite(im, recs, layout, deploy=deploy, legend=legend)
        safe = _util.safe_name(name)
        annot, orig = out_dir / f"{safe}_annot.jpg", out_dir / f"{safe}_orig.jpg"
        out.save(annot, quality=quality)
        shutil.copyfile(src, orig)
        counts = overlay_counts(recs)
        arb = recs.get("arbitrator")
        rows.append({"name": name, "layout": layout, "rule": deploy["rule"] if deploy else None,
                     "letter_rank": deploy["letter_rank"] if deploy else None,
                     "letter_final": deploy["letter_final"] if deploy else None,
                     "veto": deploy["veto"] if deploy else None,
                     "letter_llm": arb.letter_llm if arb is not None else None,
                     "p_evidence": deploy["p_evidence"] if deploy else
                     (recs["advocate"].p_evidence if recs.get("advocate") is not None else None),
                     "S": deploy["S"] if deploy else None, "S_arb": deploy["S_arb"] if deploy else None,
                     **counts, "annot_file": annot.name, "orig_file": orig.name})
        n += 1
    return pd.DataFrame(rows, columns=list(INDEX_COLS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", required=True, help="preds_*.parquet of a finished run")
    ap.add_argument("--votes", default=None, help="its votes parquet (default: the _votes.parquet sibling)")
    ap.add_argument("--images-dir", required=True, help="directory of <name>.jpg kit images (scr_NNN -> NNN.jpg)")
    ap.add_argument("--out-dir", required=True, help="NEW directory for <name>_annot.jpg / <name>_orig.jpg")
    ap.add_argument("--rule", default="R1", choices=list(aggregate_v2.DEPLOY_RULES),
                    help="deployment rule for the legend's letters (default R1)")
    ap.add_argument("--thresholds", default=None,
                    help="thresholds_v2.json to letter with (default: each row's stored tau0/t_A/t_B)")
    ap.add_argument("--model-key", default=None,
                    help="thresholds_v2.json key (default: the run's model through aggregate_v2.MODEL_KEYS)")
    ap.add_argument("--trace-dir", default=None, help="the run's per-role traces: recovers a missing votes raw")
    ap.add_argument("--no-legend", action="store_true", help="omit the legend strip")
    ap.add_argument("--limit", type=int, default=None, help="annotate at most N items")
    args = ap.parse_args(argv)

    preds, records = R.load_run(Path(args.preds), None if args.votes is None else Path(args.votes),
                                trace_dir=args.trace_dir)
    thresholds = None
    if args.thresholds:
        model = R.single(preds, "model")
        key = args.model_key or aggregate_v2.MODEL_KEYS.get(str(model), f"{model}_api")
        thresholds = aggregate_v2.resolve_thresholds(aggregate_v2.load_thresholds(args.thresholds), key)
        print(f"[annotate] thresholds {args.thresholds} key {key}: {thresholds}")
    index = annotate_run(preds, records, args.images_dir, args.out_dir, rule=args.rule, thresholds=thresholds,
                         legend=not args.no_legend, limit=args.limit)
    n_no_rec = sum(1 for n in preds["name"].astype(str) if n not in records)
    n_no_img = sum(1 for n in preds["name"].astype(str) if find_image(args.images_dir, n) is None)
    out = Path(args.out_dir)
    sha = _util.pin(index, out / "annot_index.csv")
    print(f"[annotate] {len(index)}/{len(preds)} items annotated -> {out} (index sha {sha}); "
          f"{n_no_rec} without records, {n_no_img} without an image; "
          f"arcs {int(index['n_arcs_drawn'].sum()) if len(index) else 0}, "
          f"sectors {int(index['n_sectors_drawn'].sum()) if len(index) else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
