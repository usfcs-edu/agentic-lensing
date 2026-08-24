#!/usr/bin/env python3
"""No-network tests for golden/annotate.py (the records drawn onto the composite).

Everything runs on synthetic records and a black 752x540 (or 752x562) image in memory:
  1. the PA → pixel mapping: PA 0 is straight above the panel centre, 90 straight LEFT
     (East), 180 below, 270 right — checked on the pure functions AND on the pixels of a
     single narrow arc (centroid of the cyan pixels in panel (a));
  2. a span 350 → 10 crosses the top, 10 → 350 the bottom; a zero span is a point marker;
  3. the 3.5" panel scale: an item at r 1.0 drawn in (d) lands 68.6 px from (128, 412);
     an item at r ≤ 1.7 is drawn in (a) AND (d); a too-large radius is skipped;
  4. critic sector colours follow the arbitrator's ruling (upheld red, partial orange,
     overruled grey, no arbitrator yellow); a no_opinion critic draws nothing; every
     overlay stays inside its panel box;
  5. the legend extends the canvas (and only then); a 752x562 composite is footer-cropped
     first; gray layouts are accepted, an unknown layout refused; None / absent records;
  6. the CLI on a tmp_path run (preds + votes parquet, <name>.jpg images, scr_NNN → NNN.jpg)
     writes <name>_annot.jpg + <name>_orig.jpg and pins annot_index.csv;
  7. the real scrambled-100 dev run, when present on this machine (skipped otherwise).

Runs under pytest:
    cd reproductions/lensjudge && ~/.venvs/lensjudge/bin/python -m pytest tests/test_golden_annotate.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from lensjudge.golden import annotate as A  # noqa: E402
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import schemas_panel as sp  # noqa: E402

LENSJUDGE = Path(__file__).resolve().parents[1]
DEV_PREDS = LENSJUDGE / "outputs" / "scrambled100" / "preds_scrambled100_a1_sonnet.parquet"
BOX_A, BOX_D = A.PANEL_BOXES["a"], A.PANEL_BOXES["d"]
CA, CD = (128.0, 146.0), (128.0, 412.0)


# ------------------------------------------------------------------ synthetic records
def _item(k=1, panel="a", r=2.0, pa_from=0.0, pa_to=90.0, what="arc"):
    return sp.EvidenceItem(k=k, what=what, panel=panel, r_arcsec=r, pa_deg_from=pa_from, pa_deg_to=pa_to,
                           visible_in_direct=True, criteria=[3])


def _advocate(items, p_ev=0.5, counter=None):
    return sp.AdvocateRecord(
        id="item", persona="advocate",
        criteria=sp.CriteriaV2(source_contrast=6, low_surface_brightness=5, curvature=6, counter_image=3,
                               arc_morphology=6),
        items=items, counter_image_pos=counter, scale_class="galaxy", n_red_neighbours_10as=0,
        bcg_like_halo=False, deflector_is_centre=True, p_evidence=p_ev, notes="synthetic")


def _critic(persona="geometry", alternative="spiral_arm", r_from=0.5, r_to=1.5, pa_from=0.0, pa_to=180.0,
            strength=0.5, no_opinion=False):
    if no_opinion:
        return sp.CriticRecord(id="item", persona=persona, no_opinion=True, no_opinion_reason="outside_competence")
    return sp.CriticRecord(id="item", persona=persona, alternative=alternative, alternative_desc="synthetic",
                           location=sp.LocationBox(r_arcsec_from=r_from, r_arcsec_to=r_to, pa_deg_from=pa_from,
                                                   pa_deg_to=pa_to),
                           accounts_for=[1], refutation_strength=strength)


def _arbitrator(ruling="upheld", persona="geometry", surviving=(), letter="C"):
    rulings = [] if ruling is None else [sp.Ruling(persona=persona, ruling=ruling, covers=[1], why="synthetic")]
    return sp.ArbitratorRecord(id="item", persona="arbitrator", rulings=rulings, surviving_items=list(surviving),
                               letter_llm=letter, scale_class_final="galaxy", needs_human=False, rationale="synthetic")


def _black(h=540):
    return Image.new("RGB", (752, h), (0, 0, 0))


def _mask(img, colour, box=None):
    """Boolean (y, x) mask of pixels of exactly `colour`, optionally within a PIL box."""
    a = np.asarray(img.convert("RGB"))
    m = np.all(a == np.array(colour, dtype=a.dtype), axis=-1)
    if box is not None:
        x0, y0, x1, y1 = box
        sub = np.zeros_like(m)
        sub[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        m = sub
    return m


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "no pixels of that colour"
    return (float(xs.mean()), float(ys.mean()))


def _direction_deg(centre, point):
    """The PA (N through E, East LEFT) of `point` as seen from `centre`, in degrees."""
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    return math.degrees(math.atan2(-dx, -dy)) % 360.0


def _annot(items, critics=None, arbitrator=None, layout="color", legend=False, img=None, counter=None):
    recs = {"advocate": _advocate(items, counter=counter)}
    for c in critics or []:
        recs[c.persona] = c
    if arbitrator is not None:
        recs["arbitrator"] = arbitrator
    return A.annotate_composite(img or _black(), recs, layout, legend=legend)


# ------------------------------------------------------------------ 1. the PA mapping
def test_pa_to_pil_angle_anchors():
    assert A.pa_to_pil_angle(0) == 270.0        # 12 o'clock
    assert A.pa_to_pil_angle(90) == 180.0       # 9 o'clock: East is LEFT
    assert A.pa_to_pil_angle(180) == 90.0       # 6 o'clock
    assert A.pa_to_pil_angle(270) == 0.0        # 3 o'clock
    assert A.pa_to_pil_angle(-90) == 0.0 and A.pa_to_pil_angle(450) == 180.0


def test_pa_to_xy_anchors_and_scale():
    s1, s2 = A.panel_scale("a"), A.panel_scale("d")
    assert s1 == 24.0 and abs(s2 - 68.5714) < 1e-3
    assert A.panel_centre("a") == CA and A.panel_centre("d") == CD and A.panel_centre("f") == (624.0, 412.0)
    x, y = A.pa_to_xy("a", 2.0, 0)
    assert abs(x - 128) < 1e-9 and abs(y - (146 - 48)) < 1e-9        # above
    x, y = A.pa_to_xy("a", 2.0, 90)
    assert abs(x - (128 - 48)) < 1e-9 and abs(y - 146) < 1e-9        # LEFT
    x, y = A.pa_to_xy("a", 2.0, 180)
    assert abs(x - 128) < 1e-9 and abs(y - (146 + 48)) < 1e-9        # below
    x, y = A.pa_to_xy("a", 2.0, 270)
    assert abs(x - (128 + 48)) < 1e-9 and abs(y - 146) < 1e-9        # right
    x, y = A.pa_to_xy("d", 1.0, 90)
    assert abs(x - (128 - 68.5714)) < 1e-3 and abs(y - 412) < 1e-9


def test_span_and_arc_angles():
    assert A.span_deg(350, 10) == 20.0 and A.span_deg(10, 350) == 340.0
    assert A.span_deg(0, 360) == 360.0 and A.span_deg(90, 90) == 0.0 and A.span_deg(0, 90) == 90.0
    assert A.pil_arc_angles(350, 10) == (260.0, 280.0)     # clockwise 260 → 280 passes 270 = the top
    assert A.pil_arc_angles(0, 90) == (180.0, 270.0)       # 9 o'clock → 12 o'clock: the NE quadrant
    assert A.mid_pa(350, 10) == 0.0 and A.mid_pa(0, 90) == 45.0 and A.mid_pa(10, 350) == 180.0


@pytest.mark.parametrize("pa", [0.0, 90.0, 180.0, 270.0])
def test_arc_pixels_sit_at_the_pa(pa):
    """A narrow arc (pa ± 3°) at r 3" in panel (a): all cyan pixels in (a) — arc + label —
    cluster in the PA direction at ~72 px (label a little outward), nothing opposite."""
    out = _annot([_item(panel="a", r=3.0, pa_from=pa - 3, pa_to=pa + 3)])
    assert out.size == (752, 540)
    m = _mask(out, A.CYAN, BOX_A)
    cx, cy = _centroid(m)
    d = _direction_deg(CA, (cx, cy))
    assert min(abs(d - pa), 360 - abs(d - pa)) < 10.0, (pa, d)
    r = math.hypot(cx - CA[0], cy - CA[1])
    assert 66 <= r <= 86, r
    opp = A.pa_to_xy("a", 3.0, pa + 180)
    ys, xs = np.nonzero(m)
    assert np.min(np.hypot(xs - opp[0], ys - opp[1])) > 40


def test_pa0_arc_point_is_straight_above_and_pa90_left():
    out = _annot([_item(panel="a", r=3.0, pa_from=-2, pa_to=2)])
    assert _mask(out, A.CYAN)[146 - 72, 128] or _mask(out, A.CYAN)[146 - 73, 128] or _mask(out, A.CYAN)[146 - 71, 128]
    out = _annot([_item(panel="a", r=3.0, pa_from=88, pa_to=92)])
    col = _mask(out, A.CYAN)[146, 128 - 74:128 - 70]
    assert col.any()


# ------------------------------------------------------------------ 2. wrap spans, points
def test_wrap_span_crosses_the_top_not_the_bottom():
    out = _annot([_item(panel="a", r=3.0, pa_from=350, pa_to=10)])
    m = _mask(out, A.CYAN, BOX_A)
    top = m[146 - 74:146 - 70, 126:131]
    bottom = m[146 + 60:146 + 90, 100:157]
    assert top.any() and not bottom.any()
    # the long way round (10 → 350) crosses the bottom and the sides, not the top
    out = _annot([_item(panel="a", r=3.0, pa_from=10, pa_to=350)])
    m = _mask(out, A.CYAN, BOX_A)
    assert m[146 + 70:146 + 75, 126:131].any()                 # bottom
    assert m[144:149, 128 - 74:128 - 70].any()                 # left (East)
    assert m[144:149, 128 + 70:128 + 75].any()                 # right (West)
    assert not m[146 - 75:146 - 69, 126:131].any()             # top is the gap


def test_zero_span_is_a_point_marker():
    out = _annot([_item(panel="a", r=3.0, pa_from=200, pa_to=200)])
    m = _mask(out, A.CYAN, BOX_A)
    px, py = A.pa_to_xy("a", 3.0, 200)
    ys, xs = np.nonzero(m)
    near = np.hypot(xs - px, ys - py) < 7
    assert near.sum() >= 8                                     # the little circle
    # nothing on a big arc: every cyan pixel is within 22 px of the point (marker + label)
    assert np.hypot(xs - px, ys - py).max() < 22


def test_full_ring_span():
    out = _annot([_item(panel="a", r=3.0, pa_from=0, pa_to=360)])
    m = _mask(out, A.CYAN, BOX_A)
    for pa in (0, 90, 180, 270):
        x, y = A.pa_to_xy("a", 3.0, pa)
        assert m[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].any(), pa


# ------------------------------------------------------------------ 3. the zoom panel
def test_zoom_panel_scale_and_double_draw():
    out = _annot([_item(panel="d", r=1.0, pa_from=87, pa_to=93)])
    md = _mask(out, A.CYAN, BOX_D)
    ys, xs = np.nonzero(md)
    target = (128 - 68.5714, 412)
    assert np.min(np.hypot(xs - target[0], ys - target[1])) < 2.5
    # the same item is also drawn in (a), at 24 px left of (128, 146)
    ma = _mask(out, A.CYAN, BOX_A)
    ys, xs = np.nonzero(ma)
    assert np.min(np.hypot(xs - (128 - 24), ys - 146)) < 2.5
    assert A.item_slots(_item(panel="d", r=1.0)) == ["d", "a"]
    assert A.item_slots(_item(panel="e", r=1.0)) == ["e", "a", "d"]
    assert A.item_slots(_item(panel="b", r=2.0)) == ["b", "a"]          # > ZOOM_MAX_R: no (d)
    # r 2.0" in the 3.5" zoom is 137 px: straight up it leaves the box (nothing to show), but
    # a 0->90 span passes the NE corner at PA 45 (97 px along each axis) — that part is drawn
    assert A.item_slots(_item(panel="e", r=2.0, pa_from=0, pa_to=10)) == ["a"]
    assert A.item_slots(_item(panel="e", r=2.0, pa_from=0, pa_to=90)) == ["e", "a"]
    assert A.item_slots(_item(panel="ctx", r=1.0)) == ["a", "d"]        # ctx is not a slot
    assert A.item_slots(_item(panel="b", r=6.0, pa_from=0, pa_to=10)) == []      # 144 px up: outside every box
    assert A.item_slots(_item(panel="b", r=6.0, pa_from=30, pa_to=60)) == ["b", "a"]   # across the corner: visible
    assert A.arc_visible("a", _item(r=6.0, pa_from=30, pa_to=60)) and not A.arc_visible("a", _item(r=6.0, pa_from=0, pa_to=10))
    assert A.r_exceeds_panel("d", 1.8) and not A.r_exceeds_panel("d", 1.75) and not A.r_exceeds_panel("a", 4.9)
    assert A.r_exceeds_panel("a", 5.1) and not A.r_exceeds_panel("ctx", 99.0)


def test_cited_panel_e_gets_the_arc_and_nothing_leaves_its_box():
    out = _annot([_item(panel="e", r=1.5, pa_from=0, pa_to=180)])
    me = _mask(out, A.CYAN, A.PANEL_BOXES["e"])
    assert me.sum() > 50
    whole = _mask(out, A.CYAN)
    inside = np.zeros_like(whole)
    for slot in ("a", "d", "e"):
        x0, y0, x1, y1 = A.PANEL_BOXES[slot]
        inside[y0:y1, x0:x1] = True
    assert not (whole & ~inside).any()                          # clipped to the panel boxes


def test_too_large_radius_draws_nothing_or_only_the_visible_part():
    # 6" straight up (144 px from the centre of a 240-px box): no point inside any box
    out = _annot([_item(panel="a", r=6.0, pa_from=0, pa_to=10)])
    assert not _mask(out, A.CYAN).any()
    # 6" across the NE corner: the part inside (a) is drawn, clipped to the box, nowhere else
    out = _annot([_item(panel="a", r=6.0, pa_from=30, pa_to=60)])
    m = _mask(out, A.CYAN)
    assert m.sum() > 20
    ys, xs = np.nonzero(m)
    x0, y0, x1, y1 = BOX_A
    assert xs.min() >= x0 and xs.max() < x1 and ys.min() >= y0 and ys.max() < y1
    px, py = A.pa_to_xy("a", 6.0, 45)                          # (26, 44): the arc's mid-point is inside
    assert np.min(np.hypot(xs - px, ys - py)) < 3


def test_counter_image_cross():
    out = _annot([_item(panel="a", r=2.0)], counter=sp.CounterImagePos(r_arcsec=1.0, pa_deg=270))
    m = _mask(out, A.CYAN, BOX_D)
    x, y = A.pa_to_xy("d", 1.0, 270)                            # right of the (d) centre
    assert m[int(y) - 1:int(y) + 2, int(x) - 5:int(x) + 6].sum() >= 8


# ------------------------------------------------------------------ 4. critic sectors
@pytest.mark.parametrize("ruling,colour", [("upheld", A.RULING_COLOURS["upheld"]),
                                           ("partial", A.RULING_COLOURS["partial"]),
                                           ("overruled", A.RULING_COLOURS["overruled"]),
                                           (None, A.NO_RULING_COLOUR)])
def test_sector_colour_by_ruling(ruling, colour):
    arb = None if ruling is None else _arbitrator(ruling)
    out = _annot([_item(panel="a", r=1.0)], critics=[_critic()], arbitrator=arb)
    m = _mask(out, colour, BOX_A)
    assert m.sum() > 30, (ruling, m.sum())
    others = [c for c in list(A.RULING_COLOURS.values()) + [A.NO_RULING_COLOUR] if c != colour]
    for c in others:
        assert not _mask(out, c).any()
    # drawn in (d) too (r_to 1.5 fits the zoom), and the sector sits in the E half (PA 0..180)
    md = _mask(out, colour, BOX_D)
    assert md.sum() > 30
    ys, xs = np.nonzero(md)
    assert xs.max() <= CD[0] + 8


def test_sector_beyond_the_field_is_clipped_not_dropped():
    """A location box whose OUTER radius leaves a panel is still drawn where its outline is
    visible (the paste clips): r 1-3" in the 3.5" zoom shows its inner arc (69 px) and the
    radial edges running out of the box; r 0.5-6" (beyond the 10" field's 4.9") shows in (a);
    only a box with no point inside a panel is skipped there."""
    red = A.RULING_COLOURS["upheld"]
    out = _annot([_item(panel="a", r=1.0)], critics=[_critic(r_from=1.0, r_to=3.0)], arbitrator=_arbitrator("upheld"))
    assert _mask(out, red, BOX_A).sum() > 30
    md = _mask(out, red, BOX_D)
    assert md.sum() > 30                                        # inner arc + radial edges, clipped
    ys, xs = np.nonzero(md)
    x0, y0, x1, y1 = BOX_D
    assert xs.min() >= x0 and xs.max() < x1 and ys.min() >= y0 and ys.max() < y1
    ix, iy = A.pa_to_xy("d", 1.0, 90)                           # the inner arc at PA 90: 69 px left of centre
    assert np.min(np.hypot(xs - ix, ys - iy)) < 3
    assert A.sector_slots(sp.LocationBox(r_arcsec_from=1.0, r_arcsec_to=3.0, pa_deg_from=0, pa_deg_to=90)) == ["a", "d"]
    # the dev-run case: an upheld critic at r 0-6" full ring vanished before; now it is in (a) and (d)
    big = sp.LocationBox(r_arcsec_from=0.0, r_arcsec_to=6.0, pa_deg_from=0, pa_deg_to=360)
    assert A.sector_slots(big) == ["a", "d"]
    out = _annot([_item(panel="a", r=1.0)], critics=[_critic(r_from=0.5, r_to=6.0, pa_from=0, pa_to=360)],
                 arbitrator=_arbitrator("upheld"))
    ma = _mask(out, red, BOX_A)
    assert ma.sum() > 30
    ix, iy = A.pa_to_xy("a", 0.5, 0)                            # the inner ring (12 px) is what fits
    ys, xs = np.nonzero(ma)
    assert np.min(np.hypot(xs - ix, ys - iy)) < 3
    # no point inside the zoom box: r 2.5-3" in a 0->10 wedge (171 px straight up) -> (a) only
    assert A.sector_slots(sp.LocationBox(r_arcsec_from=2.5, r_arcsec_to=3.0, pa_deg_from=0, pa_deg_to=10)) == ["a"]
    # no point inside any box: r 6-7" straight up
    none = sp.LocationBox(r_arcsec_from=6.0, r_arcsec_to=7.0, pa_deg_from=0, pa_deg_to=10)
    assert A.sector_slots(none) == []
    out = _annot([_item(panel="a", r=1.0)], critics=[_critic(r_from=6.0, r_to=7.0, pa_from=0, pa_to=10)],
                 arbitrator=_arbitrator("upheld"))
    assert not _mask(out, red).any()


def test_legend_marks_undrawn_and_oversized_items():
    recs = {"advocate": _advocate([_item(k=1, panel="d", r=3.4, pa_from=0, pa_to=90),      # cited zoom, r > 1.75"
                                   _item(k=2, panel="b", r=6.0, pa_from=0, pa_to=10),      # nowhere visible
                                   _item(k=3, panel="a", r=2.0)]),
            "geometry": _critic(r_from=6.0, r_to=7.0, pa_from=0, pa_to=10),                 # sector fits nowhere
            "morphology": _critic(persona="morphology", alternative="merger"),
            "arbitrator": _arbitrator("upheld")}
    lines = A.legend_lines(recs, None, "color")
    texts = [t for t, _ in lines["items"]]
    assert texts[0].endswith(" [r > panel d]") and texts[1].endswith(" [not drawn]") and not texts[2].endswith("]")
    assert A.item_marker(_item(k=1, panel="d", r=3.4)) == "[r > panel d]"
    assert A.item_marker(_item(k=1, panel="e", r=1.0)) == "" and A.item_marker(_item(panel="b", r=6.0, pa_from=0, pa_to=10)) == A.NOT_DRAWN
    crit = dict(lines["critics"])
    assert "Geo: spiral_arm - upheld r=0.50 [not drawn]" in crit and "Mor: merger - no ruling r=0.50" in crit
    assert A.overlay_counts(recs) == {"n_items": 3, "n_arcs_drawn": 1 + 0 + 1, "n_sectors_drawn": 0 + 2}
    out = A.annotate_composite(_black(), recs, "color", legend=True)
    assert out.size[0] == 752 and out.size[1] > 540


def test_head_wraps_to_the_canvas_width():
    """A three-role veto used to push the head past the 752-px canvas (scale / layout cut
    off); every head row now measures under the width and nothing is dropped."""
    from PIL import ImageDraw
    recs = {"advocate": _advocate([_item()]), "artifact": _critic(persona="artifact", alternative="companion_projection"),
            "geometry": _critic(alternative="companion_projection"),
            "morphology": _critic(persona="morphology", alternative="companion_projection"),
            "arbitrator": _arbitrator("upheld")}
    veto = "artifact:companion_projection;geometry:companion_projection;morphology:companion_projection"
    deploy = {"letter_rank": "A", "letter_final": "D", "veto": veto, "S": 0.25, "S_arb": 0.01,
              "p_evidence": 0.85, "rule": "R1"}
    lines = A.legend_lines(recs, deploy, "gray_lw_only")
    assert lines["head"][0] == A.HEAD_SEP.join(lines["head_parts"]) and f"veto {veto}" in lines["head_parts"]
    font = A.load_font()
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    max_w = 752 - 16
    assert A._text_size(d, lines["head"][0], font)[0] > max_w        # the unwrapped head would overflow
    rows = A.wrap_parts(d, lines["head_parts"], font, max_w)
    assert len(rows) >= 2 and all(A._text_size(d, r, font)[0] <= max_w for r in rows)
    joined = A.HEAD_SEP.join(rows)
    for part in lines["head_parts"]:
        assert part in joined, part                                     # nothing dropped: scale, layout, arb letter
    assert "layout gray_lw_only" in rows[-1] or any("layout gray_lw_only" in r for r in rows)
    n_side = max(len(lines["items"]), len(lines["critics"]))
    assert A.legend_height(lines, len(rows)) == max(A.LEGEND_MIN_H, 10 + A.LEGEND_LINE_H * (len(rows) + n_side))
    out = A.annotate_composite(_black(), recs, "gray_lw_only", deploy=deploy)
    assert out.size[1] >= 540 + A.legend_height(lines, len(rows))
    # a lone part wider than a row is split on ';' and reassembled without loss
    wide = "veto " + ";".join(f"role{i}:some_very_long_alternative_name_{i}" for i in range(12))
    rows = A.wrap_parts(d, [wide], font, 300)
    assert all(A._text_size(d, r, font)[0] <= 300 for r in rows) and len(rows) > 1
    assert "".join(rows).replace(A.HEAD_SEP, "").replace(" ", "") == wide.replace(" ", "")
    assert A.wrap_parts(d, ["a", "b"], font, 10_000) == ["a" + A.HEAD_SEP + "b"]
    assert A.wrap_parts(d, [], font, 100) == []


def test_labels_sit_below_the_geometry_and_nudge_radially():
    """A small critic sector's label lands on a 0.6" ring (14 px): the ring's pixels survive
    because labels are pasted BELOW the geometry; the label-collision nudge moves along the
    label's radius, not down onto the overlay."""
    ring = _item(panel="a", r=0.6, pa_from=0, pa_to=360)
    small = _critic(r_from=0.05, r_to=0.15, pa_from=0, pa_to=360)        # label "Geo" at ~12 px above the centre
    out = _annot([ring], critics=[small], arbitrator=_arbitrator("upheld"))
    m = _mask(out, A.CYAN, BOX_A)
    x, y = A.pa_to_xy("a", 0.6, 0)                                        # the ring's top (128, 131.6)
    assert m[int(round(y)) - 2:int(round(y)) + 3, int(round(x)) - 2:int(round(x)) + 3].sum() >= 3
    assert _mask(out, A.RULING_COLOURS["upheld"], BOX_A).sum() > 5     # the label was drawn
    # radial nudging: a label placed where another sits moves outward along its own radius
    placed = [(120.0, 60.0, 20, 11)]                                      # a label straight above the centre
    x1, y1 = A._clamp_label((120.0, 60.0), 20, 11, placed=placed)
    assert abs(x1 - 120.0) < 1e-9 and y1 < 60.0                           # outward = further up, same x
    placed = [(60.0, 120.0, 20, 11)]                                      # a label straight left (East)
    x2, y2 = A._clamp_label((60.0, 120.0), 20, 11, placed=placed)
    assert abs(y2 - 120.0) < 1e-9 and x2 < 60.0                           # outward = further left
    assert A._clamp_label((300.0, -5.0), 20, 11) == (240 - 3 - 10.0, 3 + 5.5)   # clamped into the layer
    assert len(placed) == 2


def test_check_out_dir_refuses_the_kit_trees(tmp_path):
    imgs = tmp_path / "kit"
    imgs.mkdir()
    for bad in (imgs, imgs / "sub", tmp_path / "kit" / "deeper" / "x"):
        with pytest.raises(SystemExit, match="REFUSED"):
            A.check_out_dir(bad, imgs)
    with pytest.raises(SystemExit, match="REFUSED"):
        A.check_out_dir(A._util.JWST_REPO / "anything", imgs)
    from lensjudge.golden import regrade_scrambled
    with pytest.raises(SystemExit, match="REFUSED"):
        A.check_out_dir(regrade_scrambled.SCRAMBLED_DIR_DEFAULT, imgs)
    assert A.check_out_dir(tmp_path / "annot", imgs) == (tmp_path / "annot").resolve()
    # annotate_run itself refuses before creating anything
    (tmp_path / "run").mkdir()
    p, imgs2 = _write_run(tmp_path / "run")
    preds, records = R.load_run(p)
    with pytest.raises(SystemExit, match="REFUSED"):
        A.annotate_run(preds, records, imgs2, imgs2 / "out")
    assert not (imgs2 / "out").exists() and sorted(f.name for f in imgs2.iterdir()) == ["007.jpg", "item-b.jpg"]


def test_no_opinion_and_unnamed_critics_draw_nothing():
    out = _annot([_item(panel="a", r=1.0)], critics=[_critic(no_opinion=True)], arbitrator=_arbitrator("upheld"))
    for c in list(A.RULING_COLOURS.values()) + [A.NO_RULING_COLOUR]:
        assert not _mask(out, c).any()
    unnamed = sp.CriticRecord(id="item", persona="morphology", no_opinion=False, alternative=None, notes="nothing fits")
    out = _annot([_item(panel="a", r=1.0)], critics=[unnamed], arbitrator=None)
    assert not _mask(out, A.NO_RULING_COLOUR).any()


def test_ruling_lookup():
    arb = _arbitrator("partial", persona="morphology")
    assert A.ruling_for("morphology", arb) == "partial"
    assert A.ruling_for("geometry", arb) is None and A.ruling_for("geometry", None) is None
    assert A.ruling_colour(None) == A.NO_RULING_COLOUR and A.ruling_colour("upheld") == (255, 80, 80)


# ------------------------------------------------------------------ 5. canvas, layouts, None
def test_legend_extends_canvas_and_footer_is_cropped():
    recs = {"advocate": _advocate([_item(), _item(k=2, panel="d", r=1.0)]), "geometry": _critic(),
            "arbitrator": _arbitrator("upheld", surviving=[2])}
    out = A.annotate_composite(_black(), recs, "color", legend=True)
    assert out.size[0] == 752 and out.size[1] >= 540 + A.LEGEND_MIN_H
    assert out.getpixel((3, 541)) == A.LEGEND_BG
    assert _mask(out, A.CYAN)[540:, :].any()                    # item lines in the legend
    assert _mask(out, A.RULING_COLOURS["upheld"])[540:, :].any()  # the critic line in its colour
    assert A.annotate_composite(_black(), recs, "color", legend=False).size == (752, 540)
    tall = A.annotate_composite(_black(562), recs, "color", legend=False)
    assert tall.size == (752, 540)
    with pytest.raises(ValueError):
        A.annotate_composite(Image.new("RGB", (700, 540)), recs, "color")
    lines = A.legend_lines(recs, None, "color")
    assert lines["items"][1][0].startswith("k2* ") and lines["items"][0][0].startswith("k1 ")
    assert lines["critics"] == [("Geo: spiral_arm - upheld r=0.50", A.RULING_COLOURS["upheld"])]
    assert "arb letter C" in lines["head"][0] and "p_ev 0.50" in lines["head"][0]


def test_legend_from_deploy_dict():
    recs = {"advocate": _advocate([_item()]), "geometry": _critic(), "arbitrator": _arbitrator("upheld")}
    deploy = {"letter_rank": "B", "letter_final": "D", "veto": "geometry:spiral_arm", "S": 0.25, "S_arb": 0.25,
              "p_evidence": 0.5, "rule": "R1"}
    head = A.legend_lines(recs, deploy, "color")["head"][0]
    assert head.startswith("R1: rank B -> final D") and "veto geometry:spiral_arm" in head and "S_arb 0.250" in head
    out = A.annotate_composite(_black(), recs, "color", deploy=deploy)
    assert out.size[1] > 540


def test_gray_layouts_and_unknown_layout():
    items = [_item(panel="c", r=2.0)]
    for layout in ("gray_sw_only", "gray_lw_only", "gray", "color"):
        out = _annot(items, layout=layout, legend=True)
        assert out.size[0] == 752 and _mask(out, A.CYAN, A.PANEL_BOXES["c"]).any()
    with pytest.raises(ValueError):
        _annot(items, layout="sepia")


def test_none_and_absent_records():
    for recs in ({}, {"advocate": None}, {"advocate": None, "geometry": None, "arbitrator": None}):
        out = A.annotate_composite(_black(), recs, "color", legend=True)
        assert out.size[0] == 752 and out.size[1] >= 540 + A.LEGEND_MIN_H
        assert not _mask(out, A.CYAN)[:540, :].any()
        assert A.overlay_counts(recs) == {"n_items": 0, "n_arcs_drawn": 0, "n_sectors_drawn": 0}
    # an advocate with a failed critic and no arbitrator: arcs yes, sectors none, legend notes the failure
    recs = {"advocate": _advocate([_item()]), "geometry": None}
    out = A.annotate_composite(_black(), recs, "color")
    assert _mask(out, A.CYAN, BOX_A).any()
    assert ("Geo: parse failure", A.MUTED) in A.legend_lines(recs)["critics"]


def test_overlay_counts():
    recs = {"advocate": _advocate([_item(panel="e", r=1.0), _item(k=2, panel="b", r=3.0)]),
            "geometry": _critic(), "artifact": _critic(persona="artifact", no_opinion=True)}
    assert A.overlay_counts(recs) == {"n_items": 2, "n_arcs_drawn": 3 + 2, "n_sectors_drawn": 2}


# ------------------------------------------------------------------ 6. the CLI
def _vote(name, role, rec):
    return {"name": name, "unit_id": name, "role": role, "k": 1, "parse_ok": True,
            "raw": "```json\n" + rec.model_dump_json() + "\n```", "cost_usd": 0.01, "system_sha16": "0" * 16}


def _write_run(tmp_path):
    names = ["scr_007", "item-b", "no_image"]
    votes, preds = [], []
    for i, name in enumerate(names):
        adv = _advocate([_item(panel="e", r=1.0, pa_from=60, pa_to=120), _item(k=2, panel="a", r=2.5)], p_ev=0.6)
        geo = _critic()
        arb = _arbitrator("partial", surviving=[2], letter="B")
        votes += [_vote(name, "advocate", adv), _vote(name, "geometry", geo), _vote(name, "arbitrator", arb)]
        preds.append({"name": name, "layout": "color" if i else "gray_lw_only", "parse_ok": True, "k": 1,
                      "arm": "a1", "model": "sonnet", "tau0": 0.15, "t_A": 0.192, "t_B": 0.1318,
                      "letter_source": "sonnet_api_calibrated", "cost_usd": 0.03})
    p = tmp_path / "preds_run.parquet"
    pd.DataFrame(preds).to_parquet(p, index=False)
    pd.DataFrame(votes).to_parquet(tmp_path / "preds_run_votes.parquet", index=False)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _black().save(imgs / "007.jpg", quality=95)                 # scr_007 -> 007.jpg
    _black(562).save(imgs / "item-b.jpg", quality=95)           # <name>.jpg, full composite
    return p, imgs


def test_kit_filename_mapping():
    assert A.kit_filename_for("scr_007")[-1] == "007.jpg" and A.kit_filename_for("scr_007")[0] == "scr_007.jpg"
    assert A.kit_filename_for("abc")[0] == "abc.jpg" and A.kit_filename_for("a b")[1] == "a_b.jpg"


def test_cli_end_to_end(tmp_path):
    p, imgs = _write_run(tmp_path)
    out = tmp_path / "annot"
    assert A.main(["--preds", str(p), "--images-dir", str(imgs), "--out-dir", str(out)]) == 0
    files = sorted(f.name for f in out.iterdir())
    assert files == ["annot_index.csv", "annot_index.csv.sha", "item-b_annot.jpg", "item-b_orig.jpg",
                     "scr_007_annot.jpg", "scr_007_orig.jpg"]
    idx = pd.read_csv(out / "annot_index.csv")
    assert list(idx.columns) == list(A.INDEX_COLS) and list(idx["name"]) == ["scr_007", "item-b"]
    assert list(idx["rule"]) == ["R1", "R1"] and list(idx["letter_rank"]) == ["A", "A"]   # p_ev 0.6 > t_A
    assert list(idx["letter_final"]) == ["A", "A"] and list(idx["letter_llm"]) == ["B", "B"]
    assert list(idx["n_items"]) == [2, 2] and list(idx["n_arcs_drawn"]) == [3 + 1, 3 + 1] and list(idx["n_sectors_drawn"]) == [2, 2]
    assert list(idx["layout"]) == ["gray_lw_only", "color"]
    with Image.open(out / "scr_007_annot.jpg") as im:
        assert im.size[0] == 752 and im.size[1] > 540
    with Image.open(out / "item-b_orig.jpg") as im:
        assert im.size == (752, 562)                            # a byte copy of the source
    assert (out / "scr_007_orig.jpg").read_bytes() == (imgs / "007.jpg").read_bytes()
    # R2 + a thresholds file + --limit
    thr = tmp_path / "thr.json"
    thr.write_text('{"provisional": {"tau0": 0.15, "t_A": 0.8, "t_B": 0.5}, "sonnet_api": {"t_A": 0.2, "t_B": 0.1}}')
    out2 = tmp_path / "annot2"
    assert A.main(["--preds", str(p), "--images-dir", str(imgs), "--out-dir", str(out2), "--rule", "R2",
                   "--thresholds", str(thr), "--limit", "1", "--no-legend"]) == 0
    idx2 = pd.read_csv(out2 / "annot_index.csv")
    assert len(idx2) == 1 and idx2["rule"][0] == "R2"
    with Image.open(out2 / "scr_007_annot.jpg") as im:
        assert im.size == (752, 540)


def test_annotate_run_without_thresholds(tmp_path):
    p, imgs = _write_run(tmp_path)
    preds, records = R.load_run(p)
    preds = preds.drop(columns=["t_A", "t_B"])
    idx = A.annotate_run(preds, records, imgs, tmp_path / "o")
    assert len(idx) == 2 and idx["rule"].isna().all() and list(idx["letter_llm"]) == ["B", "B"]
    assert (idx["p_evidence"] == 0.6).all()


# ------------------------------------------------------------------ 7. the real dev run
@pytest.mark.skipif(not DEV_PREDS.exists(), reason="scrambled-100 dev run not on this machine")
def test_dev_run_records_annotate(tmp_path):
    preds, records = R.load_run(DEV_PREDS)
    name = next(n for n in preds["name"].astype(str) if records.get(n, {}).get("advocate") is not None)
    row = preds[preds["name"] == name].iloc[0]
    recs = records[name]
    deploy = A.deploy_for(recs, R.thresholds_from_row(row), "R1")
    out = A.annotate_composite(_black(), recs, str(row["layout"]), deploy=deploy)
    assert out.size[0] == 752 and out.size[1] > 540
    assert A.overlay_counts(recs)["n_items"] == len(recs["advocate"].items)
