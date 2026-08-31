"""Canonical deterministic renderer: ``LensMarkFile`` + original image -> ``<id>.annot.png``.

Rules are CONTRACT.md's; constants come from ``file.style_defaults`` (== lensmark/schema/style_defaults.json,
``fraction_of_min_dim`` units, so ``value * min(W, H)`` is pixels) and ``palette.json``. Everything is drawn
on a 3x supersampled RGBA overlay that is LANCZOS-downsampled (premultiplied, so edges do not darken) and
alpha-composited on the original; the PNG is written with ``optimize=False`` and no metadata.

Layers, bottom to top: labels (arrow labels, text notes, θ_E labels) -> geometry (mask circles ->
Einstein rings -> arrows) -> legend plate. A label can therefore never hide a mark; the legend is on top.
Items with status ``rejected`` / ``invalid`` are never drawn; ``proposed`` items are drawn unless
``include_proposed=False``.

Label placement (arrow labels and the θ_E label): the label box is placed so that its *near edge* is
``offset`` px beyond the anchor point along the outward direction (the box centre is offset + the box's
half-extent along that direction), then clamped inside the image with a 4-px (x scale) margin;
``label_offset`` is added after placement and the box is clamped again. ``label_anchor: auto`` = the tail
side; if the clamped box would cover the pointed-at feature (head or apex) the box is moved beside the
arrow line (perpendicular, the side farther from the head); only if that still covers the feature does it
fall back to the head side. The legend's ``auto`` corner is the one whose plate overlaps the fewest items
and labels; ties -> fewest item anchors in that quadrant, then top_left/top_right/bottom_left/bottom_right.
"""
from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

from .. import config
from ..model import Arrow, EinsteinRing, ItemBase, LensMarkFile, MaskCircle, RenderInfo, TextNote, now_iso
from ..store import Campaign, atomic_write_bytes, sha256_file
from . import primitives as P

SUPERSAMPLE = 3
LABEL_MARGIN_PX = 4.0                          # at scale 1
DRAWN_STATUSES = ("accepted", "edited")
LABEL_FONT_DEFAULT = "DejaVuSans-Bold.ttf"
TEXT_FONT = "DejaVuSans.ttf"
THETA_DIR = (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)))   # screen -45 deg = below-right

Point = P.Point
Box = P.Box


# ----------------------------------------------------------------------------- context / layout
@dataclass
class _Ctx:
    file: LensMarkFile
    W: int                       # output size (scale applied)
    H: int
    scale: float
    ss: int = SUPERSAMPLE

    @property
    def Wss(self) -> float:
        return float(self.W * self.ss)

    @property
    def Hss(self) -> float:
        return float(self.H * self.ss)

    def px(self, uv) -> Point:
        return (float(uv[0]) * self.W * self.ss, float(uv[1]) * self.H * self.ss)

    def mm(self, frac: float) -> float:
        """style value (fraction of min(W, H)) -> overlay px."""
        return float(frac) * min(self.W, self.H) * self.ss

    def arcsec(self, a: float) -> float:
        """arcsec -> overlay px (radius); exact for the output width, not the nominal scale."""
        return float(a) / self.file.image.pixel_scale_arcsec * (self.W / self.file.image.width) * self.ss

    def style(self, key: str, item: Optional[ItemBase] = None) -> dict[str, Any]:
        st = dict(config.STYLE_DEFAULTS.get(key, {}))
        st.update(self.file.style_defaults.get(key, {}) or {})
        if item is not None and item.style:
            own = item.style.get(key)
            st.update(own if isinstance(own, dict) else
                      {k: v for k, v in item.style.items() if not isinstance(v, dict)})
        return st

    def color(self, name: str, alpha: int = 255) -> P.RGBA:
        r, g, b = config.palette_rgb(name)
        return (r, g, b, alpha)

    @property
    def margin(self) -> float:
        return LABEL_MARGIN_PX * self.scale * self.ss


@dataclass
class Placement:
    kind: str                                 # label | theta | text
    item_id: Optional[str]
    center: Point                             # overlay px
    w: float
    h: float
    text: str
    font_name: str
    size: float
    fill: P.RGBA
    halo_fill: Optional[P.RGBA] = None
    halo_w: float = 0.0
    parts: Optional[list[P.RichPart]] = None
    anchor_used: Optional[str] = None

    @property
    def box(self) -> Box:
        return P.box_of(self.center, self.w, self.h)


@dataclass
class LegendLayout:
    corner: str
    box: Box
    pad: float
    row_h: float
    size: float
    font_name: str
    bg: P.RGBA
    rows: list[tuple[list[str], P.RGBA]] = field(default_factory=list)   # (lines, colour)


@dataclass
class Layout:
    placements: list[Placement] = field(default_factory=list)
    legend: Optional[LegendLayout] = None


def visible_items(file: LensMarkFile, include_proposed: bool = True) -> list[ItemBase]:
    ok = set(DRAWN_STATUSES) | ({"proposed"} if include_proposed else set())
    return [it for it in file.items if it.status in ok]


def _ring_center(ctx: _Ctx, ring: EinsteinRing) -> Point:
    if ring.center_ref:
        ref = ctx.file.item(ring.center_ref)
        if isinstance(ref, Arrow):
            return ctx.px(ref.head)
    return ctx.px(ring.center)


def _place(ctx: _Ctx, base: Point, dirv: Point, offset: float, w: float, h: float) -> Point:
    dist = offset + P.rect_support(dirv[0], dirv[1], w, h)
    return (base[0] + dirv[0] * dist, base[1] + dirv[1] * dist)


def _covers(box: Box, pt: Point) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _auto_label(ctx: _Ctx, it: Arrow, cands: dict[str, Point], d_out: Point, offset: float, w: float, h: float
                ) -> tuple[Point, str]:
    """``label_anchor: auto``: tail side, clamped; moved beside the arrow line when the clamped box would cover
    the pointed-at feature; head side as the last resort. Returns (centre, anchor_used)."""
    W, H, m = ctx.Wss, ctx.Hss, ctx.margin
    head = ctx.px(it.head)
    st = ctx.style("arrow", it)
    apex = (head[0] + d_out[0] * ctx.mm(st["tip_gap"]), head[1] + d_out[1] * ctx.mm(st["tip_gap"]))
    feature = (head, apex)

    def clear(c: Point) -> bool:
        return not any(_covers(P.box_of(c, w, h), q) for q in feature)

    c = P.clamp_center(cands["tail"], w, h, W, H, m)
    if clear(c):
        return c, "tail"
    nx, ny = -d_out[1], d_out[0]
    shift = P.rect_support(nx, ny, w, h) + offset
    alts = [P.clamp_center((c[0] + s * nx * shift, c[1] + s * ny * shift), w, h, W, H, m) for s in (1.0, -1.0)]
    alts.sort(key=lambda q: -math.hypot(q[0] - head[0], q[1] - head[1]))
    for q in alts:
        if clear(q):
            return q, "tail_side"
    return P.clamp_center(cands["head"], w, h, W, H, m), "head"


def _layout(ctx: _Ctx, items: list[ItemBase]) -> Layout:
    lay = Layout()
    halo_default = ctx.style("label").get("halo", "#000000CC")
    for it in items:
        if isinstance(it, Arrow) and it.label:
            st = ctx.style("label", it)
            font_name = st.get("font", LABEL_FONT_DEFAULT)
            size = ctx.mm(st["size"])
            w, h = P.text_size(it.label, P.load_font(font_name, size))
            tail, head = ctx.px(it.tail), ctx.px(it.head)
            d_out = P.unit(tail[0] - head[0], tail[1] - head[1])       # head -> tail
            offset = ctx.mm(st["offset"])
            cands = {"tail": _place(ctx, tail, d_out, offset, w, h),
                     "head": _place(ctx, head, (-d_out[0], -d_out[1]), offset, w, h)}
            anchor = it.label_anchor
            if anchor == "auto":
                c, anchor = _auto_label(ctx, it, cands, d_out, offset, w, h)
            else:
                c = P.clamp_center(cands[anchor], w, h, ctx.Wss, ctx.Hss, ctx.margin)
            if it.label_offset:
                c = P.clamp_center((c[0] + it.label_offset[0] * ctx.Wss, c[1] + it.label_offset[1] * ctx.Hss),
                                   w, h, ctx.Wss, ctx.Hss, ctx.margin)
            lay.placements.append(Placement("label", it.id, c, w, h, it.label, font_name, size, ctx.color(it.color),
                                            P.hex_to_rgba(st.get("halo", halo_default)),
                                            float(st.get("halo_px", 2)) * ctx.scale * ctx.ss, anchor_used=anchor))
        elif isinstance(it, TextNote):
            st = ctx.style("text", it)
            size = ctx.mm(st["size"])
            w, h = P.text_size(it.text, P.load_font(TEXT_FONT, size))
            c = P.clamp_center(ctx.px(it.pos), w, h, ctx.Wss, ctx.Hss, ctx.margin)
            lay.placements.append(Placement("text", it.id, c, w, h, it.text, TEXT_FONT, size, ctx.color(it.color),
                                            P.hex_to_rgba(halo_default), float(st.get("halo_px", 2)) * ctx.scale * ctx.ss))
    for it in items:
        if isinstance(it, EinsteinRing):
            st = ctx.style("theta_label", it)
            size = ctx.mm(st["size"])
            text = it.label if it.label else f"θ_E ≈ {it.theta_e_arcsec:.2g}″"
            parts = P.theta_parts(text)
            w, h = P.rich_text_size(parts, TEXT_FONT, size)
            if it.label_pos:
                c = ctx.px(it.label_pos)
            else:
                c = _place(ctx, _ring_center(ctx, it), THETA_DIR, ctx.arcsec(it.theta_e_arcsec) + ctx.mm(st["offset"]), w, h)
            c = P.clamp_center(c, w, h, ctx.Wss, ctx.Hss, ctx.margin)
            lay.placements.append(Placement("theta", it.id, c, w, h, text, TEXT_FONT, size, ctx.color("white"),
                                            P.hex_to_rgba(halo_default), float(ctx.style("label").get("halo_px", 2)) * ctx.scale * ctx.ss,
                                            parts=parts))
    lay.legend = _layout_legend(ctx, items, lay.placements)
    return lay


def _legend_items(ctx: _Ctx, items: list[ItemBase]) -> list[ItemBase]:
    rows = [it for it in items if it.show_in_legend and it.label]
    order = ctx.file.legend.order
    if order:
        rank = {iid: k for k, iid in enumerate(order)}
        rows.sort(key=lambda it: (rank.get(it.id, len(order)), items.index(it)))
    return rows


def _item_extents(ctx: _Ctx, items: list[ItemBase]) -> list[Box]:
    """Rough bounding boxes (overlay px) of every drawn mark - what the legend plate should not cover."""
    out: list[Box] = []
    for it in items:
        if isinstance(it, Arrow):
            st = ctx.style("arrow", it)
            (x0, y0), (x1, y1) = ctx.px(it.tail), ctx.px(it.head)
            pad = ctx.mm(st["head_w"])
            out.append((min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad))
        elif isinstance(it, MaskCircle):
            cx, cy = ctx.px(it.center)
            r = ctx.arcsec(it.radius_arcsec)
            out.append((cx - r, cy - r, cx + r, cy + r))
        elif isinstance(it, EinsteinRing):
            cx, cy = _ring_center(ctx, it)
            r = ctx.arcsec(it.theta_e_arcsec)
            out.append((cx - r, cy - r, cx + r, cy + r))
    return out


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _layout_legend(ctx: _Ctx, items: list[ItemBase], placements: list[Placement]) -> Optional[LegendLayout]:
    if not ctx.file.legend.show:
        return None
    rows = _legend_items(ctx, items)
    if not rows:
        return None
    st = ctx.style("legend")
    size = ctx.mm(st["size"])
    font = P.load_font(TEXT_FONT, size)
    pad = ctx.mm(st["pad"])
    row_h = float(st.get("line_h", 1.3)) * size
    glyph = st.get("glyph", "→")
    max_text_w = ctx.Wss - 6.0 * pad
    lines_rows: list[tuple[list[str], P.RGBA]] = []
    for it in rows:
        lines = P.wrap_parts([f"{glyph} {it.label}"], font, max_text_w)
        lines_rows.append((lines, ctx.color(it.color)))
    n_lines = sum(len(ls) for ls, _ in lines_rows)
    plate_w = max(P.text_size(l, font)[0] for ls, _ in lines_rows for l in ls) + 2.0 * pad
    plate_h = n_lines * row_h + 2.0 * pad
    position = ctx.file.legend.position
    if position == "auto":
        anchors: list[Point] = []
        for it in items:
            if isinstance(it, Arrow):
                anchors += [tuple(it.tail), tuple(it.head)]
            elif isinstance(it, (MaskCircle, EinsteinRing)):
                anchors.append(tuple(it.center))
        quadrant = {c: 0 for c in P.CORNERS}
        for u, v in anchors:
            quadrant[("top" if v < 0.5 else "bottom") + "_" + ("left" if u < 0.5 else "right")] += 1
        extents = _item_extents(ctx, items) + [p.box for p in placements]

        def score(corner: str) -> tuple[int, int, int]:
            plate = P.legend_plate_box(corner, plate_w, plate_h, ctx.Wss, ctx.Hss, 2.0 * pad)
            return (sum(_overlaps(plate, b) for b in extents), quadrant[corner], P.CORNERS.index(corner))

        position = min(P.CORNERS, key=score)
    box = P.legend_plate_box(position, plate_w, plate_h, ctx.Wss, ctx.Hss, 2.0 * pad)
    return LegendLayout(position, box, pad, row_h, size, TEXT_FONT, P.hex_to_rgba(st.get("bg", "#000000B0")), lines_rows)


def label_boxes(file: LensMarkFile, *, scale: float = 1.0, include_proposed: bool = True) -> list[dict[str, Any]]:
    """Where the renderer puts every label / note / θ_E label and the legend plate, in output pixels:
    ``[{kind, id, box: (x0, y0, x1, y1), anchor}]`` (``kind`` label | text | theta | legend)."""
    ctx = _Ctx(file, max(1, round(file.image.width * scale)), max(1, round(file.image.height * scale)), scale)
    lay = _layout(ctx, visible_items(file, include_proposed))
    out = [{"kind": p.kind, "id": p.item_id, "box": tuple(v / ctx.ss for v in p.box), "anchor": p.anchor_used,
            "text": p.text} for p in lay.placements]
    if lay.legend is not None:
        out.append({"kind": "legend", "id": None, "box": tuple(v / ctx.ss for v in lay.legend.box),
                    "anchor": lay.legend.corner, "text": "\n".join(l for ls, _ in lay.legend.rows for l in ls)})
    return out


# ----------------------------------------------------------------------------- drawing
def _draw_mask(ctx: _Ctx, d: ImageDraw.ImageDraw, it: MaskCircle) -> None:
    cx, cy = ctx.px(it.center)
    r = ctx.arcsec(it.radius_arcsec)
    fill = ctx.color(it.color)
    if it.kind == "star":
        st = ctx.style("mask_star", it)
        dot_r = ctx.mm(st["dot_r"])
        P.dotted_circle(d, cx, cy, r, fill, dot_r, (2.0 + float(st["gap_mult"])) * dot_r)
    else:
        st = ctx.style("mask_galaxy", it)
        dash = ctx.mm(st["dash_len"]) * (0.5 if it.kind == "artifact" else 1.0)
        P.dashed_circle(d, cx, cy, r, fill, ctx.mm(st["line_w"]), dash, ctx.mm(st["gap_len"]))


def _draw_ring(ctx: _Ctx, d: ImageDraw.ImageDraw, it: EinsteinRing) -> None:
    cx, cy = _ring_center(ctx, it)
    st = ctx.style("einstein_ring", it)
    dot_r = ctx.mm(st["dot_r"])
    P.dotted_circle(d, cx, cy, ctx.arcsec(it.theta_e_arcsec), ctx.color(it.color), dot_r,
                    (2.0 + float(st["gap_mult"])) * dot_r)


def _draw_arrow(ctx: _Ctx, d: ImageDraw.ImageDraw, it: Arrow) -> None:
    st = ctx.style("arrow", it)
    P.draw_arrow(d, ctx.px(it.tail), ctx.px(it.head), ctx.color(it.color), ctx.mm(st["line_w"]),
                 ctx.mm(st["tip_gap"]), ctx.mm(st["head_len"]), ctx.mm(st["head_w"]))


def _draw_placement(d: ImageDraw.ImageDraw, p: Placement) -> None:
    if p.parts is not None:
        P.rich_halo_text(d, p.center, p.parts, p.font_name, p.size, p.fill, p.halo_fill, p.halo_w)
    else:
        P.halo_text(d, p.center, p.text, P.load_font(p.font_name, p.size), p.fill, p.halo_fill, p.halo_w, anchor="mm")


def _draw_legend(d: ImageDraw.ImageDraw, lg: LegendLayout) -> None:
    d.rectangle([lg.box[0], lg.box[1], lg.box[2], lg.box[3]], fill=lg.bg)
    font = P.load_font(lg.font_name, lg.size)
    y = lg.box[1] + lg.pad + lg.row_h / 2.0
    for lines, fill in lg.rows:
        for line in lines:
            d.text((lg.box[0] + lg.pad, y), line, font=font, fill=fill, anchor="lm")
            y += lg.row_h


def render_image(file: LensMarkFile, base: Image.Image, *, scale: float = 1.0,
                 include_proposed: bool = True) -> Image.Image:
    """The annotated image (RGB) for ``file`` over ``base`` (the original at ``image.width x height``).
    ``scale`` multiplies the output size (export only; the canonical render is scale 1)."""
    W0, H0 = file.image.width, file.image.height
    if base.size != (W0, H0):
        raise ValueError(f"base image is {base.size[0]}x{base.size[1]}, file says {W0}x{H0}")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    W, H = max(1, round(W0 * scale)), max(1, round(H0 * scale))
    ctx = _Ctx(file, W, H, scale)
    items = visible_items(file, include_proposed)
    lay = _layout(ctx, items)
    size = (W * ctx.ss, H * ctx.ss)
    labels = Image.new("RGBA", size, (0, 0, 0, 0))
    geo = Image.new("RGBA", size, (0, 0, 0, 0))
    top = Image.new("RGBA", size, (0, 0, 0, 0))
    dl, dg, dt = (ImageDraw.Draw(im, "RGBA") for im in (labels, geo, top))
    order = {"text": 0, "label": 1, "theta": 2}
    for p in sorted(lay.placements, key=lambda p: order[p.kind]):
        _draw_placement(dl, p)
    for it in items:
        if isinstance(it, MaskCircle):
            _draw_mask(ctx, dg, it)
    for it in items:
        if isinstance(it, EinsteinRing):
            _draw_ring(ctx, dg, it)
    for it in items:
        if isinstance(it, Arrow):
            _draw_arrow(ctx, dg, it)
    if lay.legend is not None:
        _draw_legend(dt, lay.legend)
    overlay = Image.alpha_composite(Image.alpha_composite(labels, geo), top)
    small = overlay.convert("RGBa").resize((W, H), Image.LANCZOS).convert("RGBA")
    bg = base.convert("RGBA")
    if bg.size != (W, H):
        bg = bg.resize((W, H), Image.LANCZOS)
    out = Image.alpha_composite(bg, small).convert("RGB")
    return Image.frombytes("RGB", out.size, out.tobytes())      # a fresh image: no inherited info/metadata


def png_bytes(im: Image.Image) -> bytes:
    """PNG with ``optimize=False`` and no ancillary metadata chunks (byte-stable for the goldens)."""
    clean = Image.frombytes(im.mode, im.size, im.tobytes())
    buf = io.BytesIO()
    clean.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------- campaign I/O
def load_base(campaign: Campaign, file: LensMarkFile) -> Image.Image:
    """The original image, verified against ``file.image`` (sha256 + size) - a mismatch is a ValueError."""
    path = campaign.image_path(file.id)
    sha = sha256_file(path)
    if sha != file.image.sha256:
        raise ValueError(f"{file.id}: original {path.name} sha256 {sha[:12]}… != file.image.sha256 {file.image.sha256[:12]}…")
    with Image.open(path) as im:
        base = im.convert("RGB")
    if base.size != (file.image.width, file.image.height):
        raise ValueError(f"{file.id}: original is {base.size[0]}x{base.size[1]}, file says "
                         f"{file.image.width}x{file.image.height}")
    return base


def render_png_bytes(campaign: Campaign, image_id: str, *, include_proposed: bool = True) -> bytes:
    """Render in memory (nothing is written); an image without a JSON renders as a bare copy."""
    file = campaign.load_or_new(image_id)
    return png_bytes(render_image(file, load_base(campaign, file), include_proposed=include_proposed))


def render_to_file(campaign: Campaign, image_id: str, *, scale: float = 1.0, out: Path | None = None,
                   save_json: bool = True) -> Path:
    """Write ``<id>.annot.png`` (atomically) next to the original or under ``out``. For the canonical case
    (scale 1, no ``out``, ``save_json``) the JSON's ``render`` block is pinned to ``content_sha256()`` and
    saved with ``touch_modified=False`` (only when the JSON already exists - rendering never creates one)."""
    file = campaign.load_or_new(image_id)
    base = load_base(campaign, file)
    data = png_bytes(render_image(file, base, scale=scale))
    if out is None:
        dest = campaign.annot_path(image_id)
    else:
        dest = Path(out).expanduser() / campaign.annot_path(image_id).name
        dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(dest, data)
    if scale == 1 and out is None and save_json and campaign.exists(image_id):
        file.render = RenderInfo(renderer=config.RENDERER_VERSION, output=dest.name,
                                 of_json_sha256=file.content_sha256(), rendered_at=now_iso())
        campaign.save(image_id, file, actor="render", source="cli", touch_modified=False)
    return dest


def is_stale(campaign: Campaign, image_id: str) -> bool:
    """True when ``<id>.annot.png`` is missing or was rendered from a different JSON content."""
    return campaign.annot_stale(image_id)


def cli_render(dir: str, *, image_id: str | None = None, check: bool = False, scale: float = 1.0,
               out: str | None = None) -> int:
    """``lensmark render DIR [--id ID] [--check] [--scale N] [--out DIR]``. Without ``--id`` every image that
    has a JSON is rendered; ``--check`` writes nothing and exits 1 listing the stale ids."""
    campaign = Campaign(dir)
    ids = [image_id] if image_id else [i for i in campaign.list_ids() if campaign.exists(i)]
    if check:
        stale = [i for i in ids if is_stale(campaign, i)]
        for i in ids:
            print(f"{i:<20} {'STALE' if i in stale else 'ok'}")
        if stale:
            print(f"{len(stale)} stale: {' '.join(stale)}")
            return 1
        print(f"{len(ids)} rendered PNG(s) current")
        return 0
    rc = 0
    for i in ids:
        try:
            dest = render_to_file(campaign, i, scale=scale, out=Path(out) if out else None)
        except (ValueError, FileNotFoundError) as e:
            print(f"{i:<20} ERROR {e}", file=sys.stderr)
            rc = 1
            continue
        note = "" if campaign.exists(i) else "  (no JSON - bare copy, nothing pinned)"
        print(f"{i:<20} -> {dest.name}{note}")
    return rc
