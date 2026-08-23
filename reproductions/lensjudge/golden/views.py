"""golden/views.py — per-role views of the v1 composite (PIL crops) and the 20" context pair.

The evidence-first scheme (plan PART 2, des2_scheme §A4) gives each panel role a DIFFERENT
view of the SAME composite so the three critics are decorrelated by what they see, not only
by what they are told: the advocate, the artifact critic and the arbitrator get the whole
footer-cropped composite; the geometry and morphology critics get 2x-enlarged crops of
individual panels and never a deflector-subtracted panel (58% of the incumbent's fail notes
attacked the circular radial-profile residual, panel (f)). The crops are cut from the kit
JPEG itself — no re-fetch, no re-render — so a view is a deterministic function of the
served bytes.

Panel geometry is MEASURED from the vendored renderer (common/jwst_fetch.render_cutout,
px=240, gap=8, title strip th=18, footer strip 22): column origins x = 8 / 256 / 504, row
origins y = 26 / 292 (gap + title strip; row 2 adds px + th + gap), every panel 240 px
square, footer below y = 540. PANEL_BOXES hard-codes those numbers, an import-time
assertion ties them to the renderer's constants, and `outline_ok` checks a real composite
against them (the renderer draws a (70,70,80) 1-px outline around every panel).

Which panels a role sees, and the model-facing words that describe them, live in ONE place:
prompts/personas/jwst_v1/panel_gloss.json (WP-1; embedded verbatim in the Nate drop-in).
It is layout-conditional (critique M2): the colour layout has row 1 = normal | deep | colour
and row 2 = deep | colour | subtracted, but a GRAY layout (one channel missing or below the
finite gate) puts a 10" radial-profile SUBTRACTION in slot (c) and a normal-stretch zoom in
slot (e), so the colour-layout map "(c),(d),(e) for morphology" would hand a subtraction
panel to a critic that must never see one. It is also RENDER-conditional: the A3 / gated-R
arm serves the "jwst_v2r" composite (golden/render_v2.py: slot (f) becomes a signed-chi
SW | LW montage of an ELLIPTICAL model residual), and the description that ships with that
image is load-bearing (R10 — at DESI the model read red/blue as "artifact" until told not
to), so `renders["v2r"]` maps every composite view set to a `_v2r` twin whose (f) sentence is
the chi description and which carries `render_desc: true`: `view_text(..., render="v2r",
render_desc=<golden/render_v2_desc.md>)` appends the full description to the VIEW paragraph,
and refuses to describe the v2r image without it. Crop roles (geometry, morphology) never
see slot (f), so their view sets are render-independent. The circular-subtraction caveat
(butterfly / bowtie lobes, concentric rings, an off-centre dipole) lives in the v1 composite
texts here, NOT in the persona .md files or the note, so a role is never told two different
things about the same panel. `load_gloss` refuses any gloss — file or built-in — whose
geometry/morphology view set contains a slot the same file lists as subtracted, whose
`renders` map points at a non-composite or description-less view set, or whose boxes
disagree with the measured ones. BUILTIN_GLOSS is a same-schema copy used only when the
file is absent (dev machines, tests), so the code never has to know which source it read.

ctx20_image renders the optional 20" context pair (deep | colour, one tile when a band is
absent) from golden/stamps/<id>/<id>_{SW,LW}_20as.fits with the composite's own stretch
parameters (deep: asinh soft=0.7 cap=30; colour: 1.1-px smoothing + asinh soft=1.2 cap=120
per band) — the geometry critic's wide view for the group/cluster question.

  python lensjudge/golden/views.py --stamp-dir golden/stamps/<id> --layout color --out /tmp/v
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from lensjudge.common import jwst_fetch as jf  # noqa: E402
from lensjudge.golden import _util  # noqa: E402

# ------------------------------------------------------------------ measured geometry
# render_cutout(px=240): gap, th (title strip), foot_h are its locals; W/H/FOOTER_Y follow.
PX, GAP, TH, FOOT_H = 240, 8, 18, 22
COL_X = tuple(GAP + c * (PX + GAP) for c in range(3))           # (8, 256, 504)
ROW_Y = tuple(GAP + r * (PX + TH + GAP) + TH for r in range(2))  # (26, 292)
SLOTS = ("a", "b", "c", "d", "e", "f")                          # row-major: a b c / d e f
_BOXES = {s: (COL_X[i % 3], ROW_Y[i // 3], COL_X[i % 3] + PX, ROW_Y[i // 3] + PX)
          for i, s in enumerate(SLOTS)}                          # PIL (left, upper, right, lower)
_TITLED = {s: (x0, y0 - TH, x1, y1) for s, (x0, y0, x1, y1) in _BOXES.items()}   # + title strip
LAYOUTS = ("color", "gray_sw_only", "gray_lw_only")
# the geometry is layout-independent (same canvas, same six slots); only the CONTENT of
# slots (c) and (e) changes — keyed by layout anyway so a caller never has to know that
PANEL_BOXES = {layout: dict(_BOXES) for layout in LAYOUTS}
TITLED_BOXES = {layout: dict(_TITLED) for layout in LAYOUTS}
OUTLINE_RGB = (70, 70, 80)        # render_cutout's panel outline
BACKGROUND_RGB = (12, 12, 16)     # render_cutout's canvas

assert COL_X == (8, 256, 504) and ROW_Y == (26, 292)
assert 3 * PX + 4 * GAP == jf.COMPOSITE_SIZE[0] and 2 * (PX + TH) + 3 * GAP + FOOT_H == jf.COMPOSITE_SIZE[1]
assert ROW_Y[1] + PX + GAP == jf.FOOTER_Y == 540
assert PANEL_BOXES["color"]["f"] == (504, 292, 744, 532) and TITLED_BOXES["color"]["a"] == (8, 8, 248, 266)

FULL_ROLES = ("advocate", "artifact", "arbitrator")   # whole footer-cropped composite
CROP_ROLES = ("geometry", "morphology")               # per-panel crops, never a subtraction
ROLES = FULL_ROLES + CROP_ROLES
CTX_PX = 320                       # one 20" tile = the 640-px stamp downsampled exactly 2x

# ------------------------------------------------------------------ gloss
GLOSS_PATH = _util.LENSJUDGE / "prompts" / "personas" / "jwst_v1" / "panel_gloss.json"
RENDERS = ("v1", "v2r")            # v1 = the run's composite; v2r = golden/render_v2.py (slot f)
RENDER_DESC_SEP = "\n\n"           # VIEW paragraph + blank line + the render description
# The circular-subtraction caveat, stated ONCE (here, with the panel it describes) for every
# composite role — the persona .md files and the note defer to the VIEW description so that a
# v2r role is never told (f) is something it is not. Wording kept from the v1 note.
_CIRCULAR_CAVEAT = ("The subtraction is a one-dimensional azimuthally-averaged profile about the stamp centre: "
                    "no ellipticity, bar, disc or off-centre position is modelled, so on ANY elliptical, barred "
                    "or inclined galaxy it leaves a four-lobed butterfly / bowtie and concentric positive/negative "
                    "rings, and on an off-centre or companion-blended deflector a dipole. Those patterns are "
                    "properties of the subtraction, not of the sky - evidence neither for nor against a lens; "
                    "never count one as an item or name one as an alternative. A lensed arc in {slots} is an "
                    "OFFSET, tangential feature at roughly constant radius that is also traceable in (d) or (e).")
_CIRCULAR_CAVEAT_F = _CIRCULAR_CAVEAT.format(slots="(f)")
_CIRCULAR_CAVEAT_CF = _CIRCULAR_CAVEAT.format(slots="(c) or (f)")
_CIRCULAR_CAVEAT_C = _CIRCULAR_CAVEAT.format(slots="(c)")
# Same schema as the file (composite boxes, layout_aliases, layouts, view_sets, roles,
# headers); used only when the file is absent. Item-agnostic by construction: it describes
# the render, never the object. Keep the texts byte-equal to the file when editing either.
BUILTIN_GLOSS = {
    "version": "jwst_v1",
    "composite": {"width": 752, "height": 562, "footer_y": 540, "panel_px": 240, "gap": 8,
                  "title_h": 18, "x0": list(COL_X), "y0": list(ROW_Y),
                  "boxes": {s: list(b) for s, b in _BOXES.items()},
                  "boxes_titled": {s: list(b) for s, b in _TITLED.items()}},
    "layout_aliases": {"color": "color", "gray_sw_only": "gray", "gray_lw_only": "gray"},
    "layouts": {
        "color": {
            "panels": {
                "a": "normal stretch, 10\" field",
                "b": "deep stretch, 10\" field (faint low-surface-brightness features; cores saturate)",
                "c": "two-band colour, 10\" field (red = long-wavelength channel, blue = short-wavelength channel, green = their mean)",
                "d": "deep stretch, 3.5\" zoom on the catalogued galaxy",
                "e": "two-band colour, 3.5\" zoom",
                "f": "deflector-subtracted residual, 3.5\" zoom (a CIRCULAR radial-profile model of the central galaxy removed)"},
            "fov_arcsec": {"a": 10, "b": 10, "c": 10, "d": 3.5, "e": 3.5, "f": 3.5},
            "direct": ["a", "b", "c", "d", "e"], "subtracted": ["f"], "has_colour": True},
        "gray": {
            "panels": {
                "a": "normal stretch, 10\" field",
                "b": "deep stretch, 10\" field (faint low-surface-brightness features; cores saturate)",
                "c": "deflector-subtracted residual, 10\" field (a CIRCULAR radial-profile model of the central galaxy removed)",
                "d": "deep stretch, 3.5\" zoom on the catalogued galaxy",
                "e": "normal stretch, 3.5\" zoom",
                "f": "deflector-subtracted residual, 3.5\" zoom (the same circular radial-profile subtraction)"},
            "fov_arcsec": {"a": 10, "b": 10, "c": 10, "d": 3.5, "e": 3.5, "f": 3.5},
            "direct": ["a", "b", "d", "e"], "subtracted": ["c", "f"], "has_colour": False},
    },
    "view_sets": {
        "composite_color": {
            "panels": "composite", "crop": "footer", "upscale": 1, "ctx20": False,
            "text": ("VIEW: one JWST NIRCam composite, six panels in two rows, north up and east left in every "
                     "panel. Row 1, 10\" field: (a) normal stretch; (b) deep stretch (faint low-surface-brightness "
                     "features, cores saturate); (c) two-band colour (red = long-wavelength channel, blue = "
                     "short-wavelength channel, green = their mean). Row 2, 3.5\" zoom on the catalogued galaxy: "
                     "(d) deep stretch; (e) two-band colour; (f) deflector-subtracted residual, a CIRCULAR "
                     "radial-profile model of the central galaxy removed. Direct (un-subtracted) panels: a, b, c, "
                     "d, e. Subtracted panels: f. " + _CIRCULAR_CAVEAT_F + " Four yellow ticks mark the catalogued "
                     "galaxy (the putative deflector) in every panel; a 1\" white scale bar sits in the first panel "
                     "of each row; row-1 panels carry NE/NW/SE/SW corner labels and panel (a) a green N/E compass. "
                     "Colour is two-band only (one short- and one long-wavelength filter). No tools, scores or "
                     "photometry are available; the image is the complete evidence set."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, as a deep-stretch "
                           "tile and a two-band colour tile, same orientation, catalogued galaxy at the centre - "
                           "for group/cluster-scale assessment (neighbours, envelope, second red members).")},
        "composite_gray": {
            "panels": "composite", "crop": "footer", "upscale": 1, "ctx20": False,
            "text": ("VIEW: one JWST NIRCam composite from a SINGLE band - six grayscale panels in two rows and no "
                     "colour information anywhere; north up and east left in every panel. Row 1, 10\" field: (a) "
                     "normal stretch; (b) deep stretch (faint low-surface-brightness features, cores saturate); "
                     "(c) deflector-subtracted residual at 10\", a CIRCULAR radial-profile model of the central "
                     "galaxy removed. Row 2, 3.5\" zoom on the catalogued galaxy: (d) deep stretch; (e) normal "
                     "stretch; (f) deflector-subtracted residual at 3.5\", the same circular subtraction. Direct "
                     "(un-subtracted) panels: a, b, d, e. Subtracted panels: c, f. " + _CIRCULAR_CAVEAT_CF + " Four "
                     "yellow ticks mark the catalogued galaxy (the putative deflector) in every panel; a 1\" white "
                     "scale bar sits in the first panel of each row; row-1 panels carry NE/NW/SE/SW corner labels "
                     "and panel (a) a green N/E compass. No tools, scores or photometry are available; the image "
                     "is the complete evidence set."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, a single-band "
                           "deep-stretch tile, same orientation, catalogued galaxy at the centre - for "
                           "group/cluster-scale assessment (neighbours, envelope, second red members).")},
        # the "jwst_v2r" render (golden/render_v2.py): slot (f) is a signed-chi SW | LW montage of an
        # ELLIPTICAL-model residual; `render_desc: True` makes view_text() append the full
        # golden/render_v2_desc.md (the load-bearing description, R10) and refuse to run without it
        "composite_color_v2r": {
            "panels": "composite", "crop": "footer", "upscale": 1, "ctx20": False, "render_desc": True,
            "text": ("VIEW: one JWST NIRCam composite, six panels in two rows, north up and east left in every "
                     "panel. Row 1, 10\" field: (a) normal stretch; (b) deep stretch (faint low-surface-brightness "
                     "features, cores saturate); (c) two-band colour (red = long-wavelength channel, blue = "
                     "short-wavelength channel, green = their mean). Row 2, 3.5\" zoom on the catalogued galaxy: "
                     "(d) deep stretch; (e) two-band colour; (f) a SIGNED-CHI residual after removing a smooth "
                     "ELLIPTICAL model of the central galaxy, shown as two tiles SW | LW on a red/blue scale - NOT "
                     "a circular subtraction; its description follows this paragraph and governs how (f) is read. "
                     "Direct (un-subtracted) panels: a, b, c, d, e. Subtracted panels: f (elliptical-model chi "
                     "residual). Four yellow ticks mark the catalogued galaxy (the putative deflector) in every "
                     "panel; a 1\" white scale bar sits in the first panel of each row; row-1 panels carry "
                     "NE/NW/SE/SW corner labels and panel (a) a green N/E compass. Colour is two-band only (one "
                     "short- and one long-wavelength filter). No tools, scores or photometry are available; the "
                     "image is the complete evidence set."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, as a deep-stretch "
                           "tile and a two-band colour tile, same orientation, catalogued galaxy at the centre - "
                           "for group/cluster-scale assessment (neighbours, envelope, second red members).")},
        "composite_gray_v2r": {
            "panels": "composite", "crop": "footer", "upscale": 1, "ctx20": False, "render_desc": True,
            "text": ("VIEW: one JWST NIRCam composite from a SINGLE band - six grayscale panels in two rows and no "
                     "colour information anywhere; north up and east left in every panel. Row 1, 10\" field: (a) "
                     "normal stretch; (b) deep stretch (faint low-surface-brightness features, cores saturate); "
                     "(c) deflector-subtracted residual at 10\", a CIRCULAR radial-profile model of the central "
                     "galaxy removed. Row 2, 3.5\" zoom on the catalogued galaxy: (d) deep stretch; (e) normal "
                     "stretch; (f) a SIGNED-CHI residual after removing a smooth ELLIPTICAL model of the central "
                     "galaxy, a single tile on a red/blue scale - NOT the circular subtraction of (c); its "
                     "description follows this paragraph and governs how (f) is read. Direct (un-subtracted) "
                     "panels: a, b, d, e. Subtracted panels: c, f (c: circular radial-profile residual; f: "
                     "elliptical-model chi residual). " + _CIRCULAR_CAVEAT_C + " Four yellow ticks mark the "
                     "catalogued galaxy (the putative deflector) in every panel; a 1\" white scale bar sits in the "
                     "first panel of each row; row-1 panels carry NE/NW/SE/SW corner labels and panel (a) a green "
                     "N/E compass. No tools, scores or photometry are available; the image is the complete "
                     "evidence set."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, a single-band "
                           "deep-stretch tile, same orientation, catalogued galaxy at the centre - for "
                           "group/cluster-scale assessment (neighbours, envelope, second red members).")},
        "geometry_color": {
            "panels": ["b", "d", "e"], "crop": "boxes_titled", "upscale": 2, "ctx20": True,
            "text": ("VIEW: three direct (un-subtracted) panels cropped from one JWST NIRCam composite and enlarged "
                     "2x, each under its own title strip; north up, east left. (b) deep stretch, 10\" field, "
                     "NE/NW/SE/SW corner labels; (d) deep stretch, 3.5\" zoom on the catalogued galaxy, with a 1\" "
                     "white scale bar; (e) two-band colour, 3.5\" zoom (red = long-wavelength channel, blue = "
                     "short-wavelength channel, green = their mean). No deflector-subtracted panel is included. "
                     "Four yellow ticks mark the catalogued galaxy (the putative deflector) in every panel; the "
                     "field of view stated for each panel sets the scale where no bar is drawn. Colour is two-band "
                     "only. No tools, scores or photometry are available."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, as a deep-stretch "
                           "tile and a two-band colour tile, same orientation, catalogued galaxy at the centre - "
                           "for group/cluster-scale assessment (neighbours, envelope, second red members).")},
        "geometry_gray": {
            "panels": ["b", "d", "e"], "crop": "boxes_titled", "upscale": 2, "ctx20": True,
            "text": ("VIEW: three direct (un-subtracted) grayscale panels cropped from one SINGLE-BAND JWST NIRCam "
                     "composite and enlarged 2x, each under its own title strip; north up, east left; no colour "
                     "information exists for this object. (b) deep stretch, 10\" field, NE/NW/SE/SW corner "
                     "labels; (d) deep stretch, 3.5\" zoom on the catalogued galaxy, with a 1\" white scale bar; "
                     "(e) normal stretch, 3.5\" zoom. No deflector-subtracted panel is included. Four yellow ticks "
                     "mark the catalogued galaxy (the putative deflector) in every panel; the field of view stated "
                     "for each panel sets the scale where no bar is drawn. No tools, scores or photometry are "
                     "available."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, a single-band "
                           "deep-stretch tile, same orientation, catalogued galaxy at the centre - for "
                           "group/cluster-scale assessment (neighbours, envelope, second red members).")},
        "morphology_color": {
            "panels": ["c", "d", "e"], "crop": "boxes_titled", "upscale": 2, "ctx20": False,
            "text": ("VIEW: three direct (un-subtracted) panels cropped from one JWST NIRCam composite and enlarged "
                     "2x, each under its own title strip; north up, east left. (c) two-band colour, 10\" field "
                     "(red = long-wavelength channel, blue = short-wavelength channel, green = their mean), "
                     "NE/NW/SE/SW corner labels; (d) deep stretch, 3.5\" zoom on the catalogued galaxy, with a 1\" "
                     "white scale bar; (e) two-band colour, 3.5\" zoom. No deflector-subtracted panel is included. "
                     "Four yellow ticks mark the catalogued galaxy (the putative deflector) in every panel; the "
                     "field of view stated for each panel sets the scale where no bar is drawn. Colour is two-band "
                     "only. No tools, scores or photometry are available."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, as a deep-stretch "
                           "tile and a two-band colour tile, same orientation, catalogued galaxy at the centre - "
                           "for group/cluster-scale assessment (neighbours, envelope, second red members).")},
        "morphology_gray": {
            "panels": ["a", "d", "e"], "crop": "boxes_titled", "upscale": 2, "ctx20": False,
            "text": ("VIEW: three direct (un-subtracted) grayscale panels cropped from one SINGLE-BAND JWST NIRCam "
                     "composite and enlarged 2x, each under its own title strip; north up, east left; no colour "
                     "information exists for this object. (a) normal stretch, 10\" field, with a 1\" white scale "
                     "bar, NE/NW/SE/SW corner labels and a green N/E compass; (d) deep stretch, 3.5\" zoom on the "
                     "catalogued galaxy, with a 1\" white scale bar; (e) normal stretch, 3.5\" zoom. No "
                     "deflector-subtracted panel is included. Four yellow ticks mark the catalogued galaxy (the "
                     "putative deflector) in every panel. No tools, scores or photometry are available."),
            "ctx20_text": ("Also supplied: a wide context view of the same field, 20\" across, a single-band "
                           "deep-stretch tile, same orientation, catalogued galaxy at the centre - for "
                           "group/cluster-scale assessment (neighbours, envelope, second red members).")},
    },
    "roles": {"advocate": {"color": "composite_color", "gray": "composite_gray"},
              "artifact": {"color": "composite_color", "gray": "composite_gray"},
              "geometry": {"color": "geometry_color", "gray": "geometry_gray"},
              "morphology": {"color": "morphology_color", "gray": "morphology_gray"},
              "arbitrator": {"color": "composite_color", "gray": "composite_gray"}},
    # render -> {view set -> replacement view set}; a view set absent from the map is render-
    # independent (the crop roles never see slot (f))
    "renders": {"v2r": {"composite_color": "composite_color_v2r", "composite_gray": "composite_gray_v2r"}},
    "headers": {
        "items": ("EVIDENCE ITEMS located by the scorer (k, what, panel, r_arcsec from the ticked galaxy, "
                  "pa_deg_from, pa_deg_to with North 0 and East 90, visible_in_direct, criteria):"),
        "items_none": "EVIDENCE ITEMS: none were located.",
        "scale_class": "Scale class reported by the scorer:",
        "arbitrator_texts": "REPORTS (the scorer's record, then each critic's record, verbatim JSON):"},
}
_GLOSS_CACHE: dict = {}


def _validate_gloss(g: dict, where: str) -> None:
    """Structure + the two invariants: no crop role sees a subtracted slot in any layout kind,
    and the boxes are the measured ones. Raises ValueError (never silently degrades)."""
    try:
        kinds = set(g["layout_aliases"].values())
        for k in kinds:
            lay = g["layouts"][k]
            assert set(lay["panels"]) == set(SLOTS) and set(lay["subtracted"]) <= set(SLOTS)
        for role in ROLES:
            for k in kinds:
                vs = g["view_sets"][g["roles"][role][k]]
                assert isinstance(vs["text"], str) and vs["text"].strip()
                if vs["panels"] == "composite":
                    assert vs["crop"] == "footer"
                else:
                    assert vs["crop"] in ("boxes", "boxes_titled") and set(vs["panels"]) <= set(SLOTS)
                    bad = set(vs["panels"]) & set(g["layouts"][k]["subtracted"])
                    if bad:
                        raise ValueError(f"{where}: {role}/{k} view set shows subtracted slot(s) {sorted(bad)}")
                if vs.get("ctx20"):
                    assert isinstance(vs.get("ctx20_text"), str) and vs["ctx20_text"].strip()
        # renders: every replacement view set exists, is a composite view and carries the
        # render description flag (the image must never ship without its description)
        for render, sub in (g.get("renders") or {}).items():
            assert render in RENDERS and render != "v1", render
            for src, dst in sub.items():
                assert src in g["view_sets"] and g["view_sets"][src]["panels"] == "composite", src
                vs = g["view_sets"][dst]
                assert vs["panels"] == "composite" and vs["crop"] == "footer", dst
                assert isinstance(vs["text"], str) and vs["text"].strip()
                if not vs.get("render_desc"):
                    raise ValueError(f"{where}: render {render!r} view set {dst!r} lacks render_desc: true")
        for s in SLOTS:
            if tuple(g["composite"]["boxes"][s]) != _BOXES[s]:
                raise ValueError(f"{where}: box {s} {g['composite']['boxes'][s]} != measured {_BOXES[s]}")
            if tuple(g["composite"]["boxes_titled"][s]) != _TITLED[s]:
                raise ValueError(f"{where}: titled box {s} != measured {_TITLED[s]}")
        for h in ("items", "items_none", "arbitrator_texts"):
            assert isinstance(g["headers"][h], str)
    except (KeyError, AssertionError, TypeError) as e:
        raise ValueError(f"{where}: malformed panel gloss ({type(e).__name__}: {e})") from e


_validate_gloss(BUILTIN_GLOSS, "BUILTIN_GLOSS")


def load_gloss(path: Path = GLOSS_PATH, required: bool = False) -> dict:
    """The panel gloss: the JSON file when it exists (validated; a bad file RAISES — it is
    the one place that decides what a critic sees), else BUILTIN_GLOSS. `required=True`
    also refuses an absent file (production runs)."""
    key = str(path)
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"{p}: panel gloss required (prompts/personas/jwst_v1 incomplete)")
        return json.loads(json.dumps(BUILTIN_GLOSS))   # a private copy; never cached
    if key not in _GLOSS_CACHE:
        g = json.loads(p.read_text())
        _validate_gloss(g, str(p))
        _GLOSS_CACHE[key] = g
    return _GLOSS_CACHE[key]


def gloss_sha16(gloss: Optional[dict] = None) -> str:
    """sha16 of the canonical JSON of the gloss in use (a run-tuple ingredient)."""
    return _util.sha_json(gloss or load_gloss())


def layout_kind(layout, gloss: Optional[dict] = None) -> str:
    """'color' or 'gray' for a frame/manifest layout value (color, gray_sw_only, gray_lw_only)."""
    g = gloss or load_gloss()
    s = str(layout or "").strip().lower()
    if s in g["layout_aliases"]:
        return g["layout_aliases"][s]
    if s in ("colour", "color"):
        return "color"
    if s.startswith("gray") or s.startswith("grey"):
        return "gray"
    raise ValueError(f"unknown layout {layout!r} (expected one of {LAYOUTS})")


def view_set_name(role: str, layout, gloss: Optional[dict] = None, render: str = "v1") -> str:
    """The view-set key for role x layout x render: the role's layout entry, replaced through
    `renders[render]` when that render substitutes it (composite roles under v2r)."""
    g = gloss or load_gloss()
    if role not in g["roles"]:
        raise ValueError(f"unknown role {role!r} (expected one of {ROLES})")
    if render not in RENDERS:
        raise ValueError(f"unknown render {render!r} (expected one of {RENDERS})")
    name = g["roles"][role][layout_kind(layout, g)]
    if render != "v1":
        name = (g.get("renders") or {}).get(render, {}).get(name, name)
    return name


def view_set(role: str, layout, gloss: Optional[dict] = None, render: str = "v1") -> dict:
    """The view-set entry (panels, crop, upscale, ctx20, text[, ctx20_text, render_desc]) for
    role x layout x render."""
    g = gloss or load_gloss()
    return g["view_sets"][view_set_name(role, layout, g, render)]


def role_slots(role: str, layout, gloss: Optional[dict] = None) -> Optional[tuple]:
    """Slots this role sees in this layout; None means the whole composite."""
    vs = view_set(role, layout, gloss)
    return None if vs["panels"] == "composite" else tuple(vs["panels"])


def view_text(role: str, layout, gloss: Optional[dict] = None, with_ctx20: bool = False,
              render: str = "v1", render_desc: Optional[str] = None) -> str:
    """The model-facing VIEW paragraph for role x layout x render (+ the ctx20 sentence when
    the context pair is actually supplied). A view set flagged `render_desc` (the v2r
    composite) gets `render_desc` — golden/render_v2_desc.md — appended after a blank line
    and REFUSES to be described without it: the description ships with the image or the
    image does not ship (R10)."""
    vs = view_set(role, layout, gloss, render)
    t = vs["text"]
    if with_ctx20 and vs.get("ctx20") and vs.get("ctx20_text"):
        t = t + " " + vs["ctx20_text"]
    if vs.get("render_desc"):
        if not (isinstance(render_desc, str) and render_desc.strip()):
            raise ValueError(f"view set for {role}/{layout}/{render} needs the render description "
                             f"(golden/render_v2_desc.md) — the image never ships without it")
        t = t + RENDER_DESC_SEP + render_desc.strip()
    return t


def panel_gloss(layout, slot: str, gloss: Optional[dict] = None) -> str:
    """'panel (b): deep stretch, 10" field (...)' — the per-image label for a crop."""
    g = gloss or load_gloss()
    return f"panel ({slot}): {g['layouts'][layout_kind(layout, g)]['panels'][slot]}"


def gloss_strings(gloss: Optional[dict] = None, render_descs: Optional[dict] = None) -> list[str]:
    """Every model-facing string the gloss can emit (audit_traces.known_template_shas
    registers their sha16s so a long VIEW block is verifiable from its sha). `render_descs`
    = {render: description text}: the `render_desc` view sets are also emitted with each
    description appended, exactly as view_text() sends them."""
    g = gloss or load_gloss()
    descs = [d for d in (render_descs or {}).values() if isinstance(d, str) and d.strip()]
    out = []
    for vs in g["view_sets"].values():
        texts = [vs["text"]]
        if vs.get("ctx20_text"):
            texts.append(vs["text"] + " " + vs["ctx20_text"])
            out.append(vs["ctx20_text"])
        if vs.get("render_desc"):
            texts = [x + RENDER_DESC_SEP + d.strip() for x in texts for d in descs] + texts
        out += texts
    for k in g["layouts"]:
        out += [f"panel ({s}): {g['layouts'][k]['panels'][s]}" for s in SLOTS]
    out += [str(v) for v in g["headers"].values()]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ------------------------------------------------------------------ crops
def _pil():
    from PIL import Image
    return Image


def full_view(composite_img):
    """The footer-cropped composite (752x540); an already-cropped image is returned as is."""
    if composite_img.height > jf.FOOTER_Y:
        return jf.crop_footer(composite_img)
    return composite_img


def crop_panel(composite_img, slot: str, layout="color", scale: int = 2, crop: str = "boxes"):
    """One panel cut at its measured box ('boxes') or box + title strip ('boxes_titled'),
    enlarged `scale`x (LANCZOS). The image may be the full 752x562 composite or the
    footer-cropped 752x540 kit JPEG — the panel coordinates are the same."""
    if composite_img.width != jf.COMPOSITE_SIZE[0] or composite_img.height not in (jf.COMPOSITE_SIZE[1], jf.FOOTER_Y):
        raise ValueError(f"composite is {composite_img.size}, expected 752x562 or 752x540")
    table = TITLED_BOXES if crop == "boxes_titled" else PANEL_BOXES
    box = table[layout if layout in table else LAYOUTS[0]][slot]
    tile = composite_img.convert("RGB").crop(box)
    if scale and scale != 1:
        tile = tile.resize((tile.width * scale, tile.height * scale), _pil().LANCZOS)
    return tile


def role_views(composite_img, layout, role: str, ctx20=None, gloss: Optional[dict] = None,
               render: str = "v1", render_desc: Optional[str] = None) -> list:
    """[(label, PIL.Image), ...] for `role` in `layout` under `render`: the full footer-cropped
    composite (label = the VIEW paragraph, render description included for v2r) for advocate
    / artifact / arbitrator; per-panel crops (label = 'panel (x): ...') for geometry and
    morphology, with the 20" context pair appended when the view set takes it and `ctx20`
    is given. The first view is the role's primary image; the rest go to
    grader_jwst.grade_candidate(extra_views=...)."""
    g = gloss or load_gloss()
    vs = view_set(role, layout, g, render)
    if vs["panels"] == "composite":
        return [(view_text(role, layout, g, render=render, render_desc=render_desc), full_view(composite_img))]
    views = [(panel_gloss(layout, s, g),
              crop_panel(composite_img, s, layout, scale=int(vs.get("upscale", 2)), crop=vs["crop"]))
             for s in vs["panels"]]
    if vs.get("ctx20") and ctx20 is not None:
        views.append((vs["ctx20_text"], ctx20))
    return views


def image_block(img, fmt: str = "PNG") -> dict:
    """One Anthropic-style base64 image block for a PIL image (PNG: lossless crops)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    media = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


# ------------------------------------------------------------------ box verification
def outline_ok(composite_img, layout="color", tol: int = 12) -> dict:
    """slot -> True when the 1-px ring at the box border is the renderer's outline colour
    and the pixels just outside it are canvas background (a measurement of a real or
    synthetic composite against PANEL_BOXES)."""
    a = np.asarray(composite_img.convert("RGB")).astype(int)
    out = {}
    for slot, (x0, y0, x1, y1) in PANEL_BOXES[layout if layout in PANEL_BOXES else LAYOUTS[0]].items():
        ring = np.concatenate([a[y0, x0:x1], a[y1 - 1, x0:x1], a[y0:y1, x0], a[y0:y1, x1 - 1]])
        # the renderer draws the quadrant labels / compass / scale bar INSIDE the box, so
        # the ring itself is clean; use the median against the outline colour
        ring_ok = np.all(np.abs(np.median(ring, axis=0) - OUTLINE_RGB) <= tol)
        outside = np.concatenate([a[y0 - 1, x0:x1], a[y0:y1, x0 - 1]])   # above + left
        out_ok = np.all(np.abs(np.median(outside, axis=0) - BACKGROUND_RGB) <= tol)
        out[slot] = bool(ring_ok and out_ok)
    return out


# ------------------------------------------------------------------ 20" context pair
def _colour_rgb(lw, sw):
    """render_cutout's `color` closure (vendored, not importable): per-band noise-normalised
    asinh after a matched 1.1-px smoothing; red = LW, blue = SW, green = their mean."""
    from scipy.ndimage import gaussian_filter
    sm = lambda z: gaussian_filter(np.nan_to_num(z, nan=0.0), 1.1)  # noqa: E731
    rr = jf.asinh_stretch(sm(lw), soft=1.2, cap=120.0)
    bb = jf.asinh_stretch(sm(sw), soft=1.2, cap=120.0)
    return rr, (rr + bb) / 2.0, bb


def _finite_ok(header) -> bool:
    try:
        return float(header.get("FINITE", 1.0)) >= jf.MIN_FINITE
    except (TypeError, ValueError):
        return True


def _decorate(tile, fov_arcsec: float, title: str, bar_arcsec: float = 5.0):
    """Title strip above, yellow centre ticks and a scale bar inside — the composite's own
    conventions at this field of view (render_cutout L384-417), so a critic reads the
    context tile the way it reads the panels."""
    from PIL import Image, ImageDraw
    px = tile.width
    canvas = Image.new("RGB", (px, px + TH), BACKGROUND_RGB)
    canvas.paste(tile, (0, TH))
    d = ImageDraw.Draw(canvas)
    f9 = jf._font(11)
    d.text((1, 3), title, font=f9, fill=(205, 205, 215))
    d.rectangle([0, TH, px - 1, TH + px - 1], outline=OUTLINE_RGB)
    cxp, cyp = px / 2.0, TH + px / 2.0
    off, ln = px * 1.2 / fov_arcsec, px * 0.6 / fov_arcsec
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        d.line([cxp + dx * off, cyp + dy * off, cxp + dx * (off + ln), cyp + dy * (off + ln)],
               fill=(255, 215, 90), width=1)
    bar = px * bar_arcsec / fov_arcsec
    d.line([6, TH + px - 8, 6 + bar, TH + px - 8], fill=(255, 255, 255), width=2)
    d.text((8 + bar, TH + px - 17), f'{bar_arcsec:g}"', font=f9, fill=(255, 255, 255))
    return canvas


def find_20as(stamp_dir: Path) -> dict:
    """{'SW': path|None, 'LW': path|None} for the 20" stamps in a golden/stamps/<id>/ dir."""
    d = Path(stamp_dir)
    out = {}
    for ch in jf.CHANNELS:
        hits = sorted(d.glob(f"*_{ch}_20as.fits"))
        out[ch] = hits[0] if hits else None
    return out


def ctx20_image(stamp_dir, px: int = CTX_PX):
    """The 20" context pair (deep | colour) as one PIL image, or None when no 20" stamp
    exists (or none passes the finite gate). Deep uses the composite's row-1 deep stretch;
    colour needs both bands (a gray layout gets the deep tile alone). Tiles are px square
    (default 320 = the 640-px stamp at exactly 2x down), gap 8, same canvas colour."""
    paths = find_20as(stamp_dir)
    arrs = {}
    for ch, p in paths.items():
        if p is None:
            continue
        arr, hdr = jf.read_stamp_fits(p)
        if _finite_ok(hdr):
            arrs[ch] = arr
    if not arrs:
        return None
    sw, lw = arrs.get("SW"), arrs.get("LW")
    base = sw if sw is not None else lw
    fov = 2 * jf.CUT_ARCSEC
    tiles = [_decorate(jf._gray_panel(jf.asinh_stretch(base, soft=0.7, cap=30.0), px), fov, f'deep {fov:g}"')]
    if sw is not None and lw is not None:
        tiles.append(_decorate(jf._rgb_panel(*_colour_rgb(lw, sw), px), fov, f'colour {fov:g}"'))
    Image = _pil()
    W = len(tiles) * px + (len(tiles) + 1) * GAP
    H = px + TH + 2 * GAP
    canvas = Image.new("RGB", (W, H), BACKGROUND_RGB)
    for i, t in enumerate(tiles):
        canvas.paste(t, (GAP + i * (px + GAP), GAP))
    return canvas


def main(argv=None) -> int:
    """Render every role's views for one stamp dir (eyeballing the fan-out): writes
    <out>/<role>_<k>.png and prints the labels."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stamp-dir", required=True, help="golden/stamps/<id>/ (has <id>_v1.jpg)")
    ap.add_argument("--layout", default="color", choices=LAYOUTS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    Image = _pil()
    sd = Path(args.stamp_dir)
    jpgs = sorted(sd.glob("*_v1.jpg"))
    if not jpgs:
        raise SystemExit(f"no *_v1.jpg under {sd}")
    comp = Image.open(jpgs[0]).convert("RGB")
    print("gloss:", "file" if GLOSS_PATH.exists() else "built-in", gloss_sha16())
    print("outline_ok:", outline_ok(comp, args.layout))
    ctx = ctx20_image(sd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        print(f"{role}: {view_text(role, args.layout, with_ctx20=ctx is not None)[:90]}...")
        for k, (label, img) in enumerate(role_views(comp, args.layout, role, ctx20=ctx), 1):
            p = out / f"{role}_{k}.png"
            img.save(p)
            print(f"    view {k}: {img.size} {p.name}  [{label[:70]}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
