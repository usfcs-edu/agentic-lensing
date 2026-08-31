/**
 * Drawing maths shared by the SVG overlay, the tools and the tests - the browser-side statement of the
 * geometry rules in CONTRACT.md ("Items" table). The PIL renderer implements the same rules from the
 * same style constants, so the preview and the canonical .annot.png agree.
 *
 * All functions here work in IMAGE PIXELS (the overlay's viewBox) and take `m = min(W, H)`: every
 * style value is a fraction of `m` (style_defaults.unit == "fraction_of_min_dim").
 */
import type { Arrow, ColorName, Item, LegendPosition, LensMarkFile, MaskKind, PaletteDoc, StyleDefaults, UV } from "./types";
import { CORNERS } from "./types";

export type XY = [number, number];
export type Corner = Exclude<LegendPosition, "auto">;

export const LABEL_MARGIN_PX = 4;

export function minDim(W: number, H: number): number {
  return Math.min(W, H);
}

// ----------------------------------------------------------------------------- arrows
export interface ArrowGeom {
  /** shaft start (the tail) */
  x1: number; y1: number;
  /** shaft end: the base of the head (the head triangle covers the rest up to the apex) */
  x2: number; y2: number;
  /** apex of the head = `tip_gap*m` short of the pointed-at feature */
  apex: XY;
  /** filled triangle: apex, base-left, base-right */
  head: [XY, XY, XY];
  lineW: number;
  /** unit vector tail -> head (NaN-free; degenerate arrows point right) */
  dir: XY;
  length: number;
}

/**
 * Arrow from `tail` towards `head` (image px). The shaft stops `tip_gap*m` short of `head` (the
 * feature stays visible, 21_annotate.py:54); the filled triangular head has length `head_len*m`,
 * base width `head_w*m` and its apex at that gap point. For very short arrows the head is
 * shortened so it never overshoots the tail.
 */
export function arrowGeometry(tail: XY, head: XY, style: StyleDefaults, m: number): ArrowGeom {
  const a = style.arrow;
  const dx = head[0] - tail[0], dy = head[1] - tail[1];
  const len = Math.hypot(dx, dy);
  const dir: XY = len > 1e-9 ? [dx / len, dy / len] : [1, 0];
  const gap = a.tip_gap * m;
  const headLen = Math.min(a.head_len * m, Math.max(0, len - gap) * 0.9);
  const halfW = (a.head_w * m) / 2;
  const apex: XY = [head[0] - dir[0] * gap, head[1] - dir[1] * gap];
  const base: XY = [apex[0] - dir[0] * headLen, apex[1] - dir[1] * headLen];
  const nx = -dir[1], ny = dir[0];          // unit normal
  return {
    x1: tail[0], y1: tail[1], x2: base[0], y2: base[1], apex,
    head: [apex, [base[0] + nx * halfW, base[1] + ny * halfW], [base[0] - nx * halfW, base[1] - ny * halfW]],
    lineW: a.line_w * m, dir, length: len,
  };
}

// ----------------------------------------------------------------------------- dashes / dots
export interface StrokePattern { dasharray: string; strokeWidth: number; linecap: "butt" | "round" }

/**
 * Mask circle stroke. galaxy: `dash_len*m` on / `gap_len*m` off; artifact: half the dash length;
 * star: dots of radius `dot_r*m` whose centres are `(2 + gap_mult)*dot_r*m` apart (edge gap =
 * gap_mult * dot_r, deck PROMPT 6 "gap = 3x dot radius"). An SVG circle's stroke starts at the
 * rightmost point (screen angle 0) and walks clockwise on screen, which is exactly the contract's
 * dash phase, so a plain stroke-dasharray reproduces the PIL pattern.
 */
export function maskPattern(kind: MaskKind, style: StyleDefaults, m: number): StrokePattern {
  if (kind === "star") {
    const s = style.mask_star;
    const dotR = s.dot_r * m;
    return { dasharray: `0 ${fmt(((2 + s.gap_mult) * dotR))}`, strokeWidth: 2 * dotR, linecap: "round" };
  }
  const g = style.mask_galaxy;
  const dash = g.dash_len * m * (kind === "artifact" ? 0.5 : 1);
  return { dasharray: `${fmt(dash)} ${fmt(g.gap_len * m)}`, strokeWidth: g.line_w * m, linecap: "butt" };
}

/** Einstein ring: fine dots (`einstein_ring.dot_r*m`) with the same `(2 + gap_mult)` spacing rule. */
export function ringPattern(style: StyleDefaults, m: number): StrokePattern {
  const r = style.einstein_ring;
  const dotR = r.dot_r * m;
  return { dasharray: `0 ${fmt((2 + r.gap_mult) * dotR)}`, strokeWidth: 2 * dotR, linecap: "round" };
}

function fmt(x: number): string {
  return String(Math.round(x * 1000) / 1000);
}

// ----------------------------------------------------------------------------- text metrics
/** Approximate advance widths (em) for DejaVu Sans Bold - enough to clamp labels and size the legend. */
const NARROW = new Set("ijl|!.,:;'’ ");
const MEDIUM = new Set("ftrI()[]-\"");
const WIDE = new Set("mwMW@→≈");
export function estimateTextWidth(text: string, fontPx: number): number {
  let em = 0;
  for (const ch of text) {
    if (NARROW.has(ch)) em += ch === " " ? 0.35 : 0.33;
    else if (MEDIUM.has(ch)) em += 0.45;
    else if (WIDE.has(ch)) em += 1.0;
    else if (ch >= "A" && ch <= "Z") em += 0.78;
    else if (ch >= "0" && ch <= "9") em += 0.7;
    else em += 0.66;
  }
  return em * fontPx;
}
export type Measure = (text: string, fontPx: number) => number;

// ----------------------------------------------------------------------------- arrow labels
export interface LabelPlacement { x: number; y: number; side: "tail" | "head"; w: number; h: number; fontPx: number }

/**
 * Arrow label centre. `tail` (default for `auto`): `label.offset*m` beyond the tail along the
 * head->tail direction; `head`: the same beyond the head; `auto` switches to the head side only
 * when the tail-side box would leave the image. The box is then clamped inside the image
 * (margin 4 px) and `label_offset` (fractions of W, H) is added last.
 */
export function labelPlacement(arrow: Arrow, style: StyleDefaults, W: number, H: number,
                               measure: Measure = estimateTextWidth): LabelPlacement | null {
  const text = (arrow.label || "").trim();
  if (!text) return null;
  const m = minDim(W, H);
  const fontPx = style.label.size * m;
  const w = measure(text, fontPx), h = fontPx * 1.15;
  const tail: XY = [arrow.tail[0] * W, arrow.tail[1] * H];
  const head: XY = [arrow.head[0] * W, arrow.head[1] * H];
  const g = arrowGeometry(tail, head, style, m);
  const off = style.label.offset * m;
  const tailSide: XY = [tail[0] - g.dir[0] * off, tail[1] - g.dir[1] * off];
  const headSide: XY = [head[0] + g.dir[0] * off, head[1] + g.dir[1] * off];
  const anchor = arrow.label_anchor || "auto";
  let side: "tail" | "head" = anchor === "head" ? "head" : "tail";
  let p = side === "head" ? headSide : tailSide;
  if (anchor === "auto" && boxLeavesImage(p, w, h, W, H)) { side = "head"; p = headSide; }
  let [x, y] = clampBox(p, w, h, W, H);
  if (arrow.label_offset) { x += arrow.label_offset[0] * W; y += arrow.label_offset[1] * H; }
  return { x, y, side, w, h, fontPx };
}

export function boxLeavesImage(c: XY, w: number, h: number, W: number, H: number, margin = LABEL_MARGIN_PX): boolean {
  return c[0] - w / 2 < margin || c[0] + w / 2 > W - margin || c[1] - h / 2 < margin || c[1] + h / 2 > H - margin;
}

/** Clamp a w x h box centred at `c` inside the image (golden/annotate.py:339 `_clamp_label`, whole image). */
export function clampBox(c: XY, w: number, h: number, W: number, H: number, margin = LABEL_MARGIN_PX): XY {
  const x = Math.min(Math.max(c[0], margin + w / 2), W - margin - w / 2);
  const y = Math.min(Math.max(c[1], margin + h / 2), H - margin - h / 2);
  return [x, y];
}

// ----------------------------------------------------------------------------- theta_E label
/** Python's `f"{x:.2g}"` for the values theta_E takes (0.01 .. 99). */
export function fmtG2(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const s = x.toPrecision(2);
  if (s.includes("e")) {
    const [mant, e] = s.split("e");
    return stripZeros(mant) + "e" + e[0] + e.slice(1).padStart(2, "0");
  }
  return stripZeros(s);
}
function stripZeros(s: string): string {
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}

export function thetaLabelText(theta_e_arcsec: number, label?: string | null): string {
  return label && label.trim() ? label : `θ_E ≈ ${fmtG2(theta_e_arcsec)}″`;
}

/**
 * Centre of the theta label: at `label_pos` when set, else below-right of the ring (screen angle
 * -45 deg) at distance `r + theta_label.offset*m` from the centre; clamped inside the image.
 */
export function thetaLabelPlacement(center: XY, rPx: number, style: StyleDefaults, W: number, H: number,
                                    text: string, labelPos?: UV | null, measure: Measure = estimateTextWidth): LabelPlacement {
  const m = minDim(W, H);
  const fontPx = style.theta_label.size * m;
  const w = measure(text, fontPx), h = fontPx * 1.15;
  let p: XY;
  if (labelPos) p = [labelPos[0] * W, labelPos[1] * H];
  else {
    const d = rPx + style.theta_label.offset * m;
    const k = Math.SQRT1_2;                           // cos(45) = sin(45)
    p = [center[0] + d * k, center[1] + d * k];       // +y is down on screen -> "below-right"
  }
  const [x, y] = clampBox(p, w, h, W, H);
  return { x, y, side: "tail", w, h, fontPx };
}

// ----------------------------------------------------------------------------- legend
export interface LegendRow { id: string; text: string; color: ColorName; status: string }
export interface LegendLayout {
  corner: Corner; x: number; y: number; w: number; h: number;
  fontPx: number; lineH: number; pad: number; rows: LegendRow[];
}

export const DRAWN_STATUSES = new Set(["accepted", "edited", "proposed"]);

/** Rows: items with `show_in_legend` and a label that are drawn, in `legend.order` then file order. */
export function legendRows(file: LensMarkFile): LegendRow[] {
  const glyph = file.style_defaults.legend.glyph || "→";
  const byId = new Map(file.items.map((it) => [it.id, it]));
  const ordered: Item[] = [];
  for (const id of file.legend.order || []) { const it = byId.get(id); if (it) ordered.push(it); }
  for (const it of file.items) if (!ordered.includes(it)) ordered.push(it);
  return ordered
    .filter((it) => it.show_in_legend && (it.label || "").trim() && DRAWN_STATUSES.has(it.status))
    .map((it) => ({ id: it.id, text: `${glyph} ${(it.label || "").trim()}`, color: it.color, status: it.status }));
}

/** Anchor points (u, v) used for collision-free legend placement: arrow tail/head, circle centres. */
export function itemAnchors(it: Item): UV[] {
  switch (it.type) {
    case "arrow": return [it.tail, it.head];
    case "mask_circle": return [it.center];
    case "einstein_ring": return [it.center];
    case "text": return [it.pos];
  }
}

export function quadrant(u: number, v: number): Corner {
  return v < 0.5 ? (u < 0.5 ? "top_left" : "top_right") : (u < 0.5 ? "bottom_left" : "bottom_right");
}

/** `auto` -> the corner whose quadrant holds the fewest anchor points; ties -> first in CORNERS order. */
export function legendCorner(file: LensMarkFile): Corner {
  if (file.legend.position !== "auto") return file.legend.position;
  const counts: Record<Corner, number> = { top_left: 0, top_right: 0, bottom_left: 0, bottom_right: 0 };
  for (const it of file.items) {
    if (!DRAWN_STATUSES.has(it.status) || it.type === "text") continue;
    for (const [u, v] of itemAnchors(it)) counts[quadrant(u, v)] += 1;
  }
  let best: Corner = "top_left";
  for (const c of CORNERS) if (counts[c] < counts[best]) best = c;
  return best;
}

export function legendLayout(file: LensMarkFile, measure: Measure = estimateTextWidth): LegendLayout | null {
  const rows = legendRows(file);
  if (!file.legend.show || rows.length === 0) return null;
  const W = file.image.width, H = file.image.height, m = minDim(W, H);
  const lg = file.style_defaults.legend;
  const fontPx = lg.size * m, lineH = lg.line_h * fontPx, pad = lg.pad * m;
  const w = Math.max(...rows.map((r) => measure(r.text, fontPx))) + 2 * pad;
  const h = rows.length * lineH + 2 * pad;
  const inset = 2 * pad;
  const corner = legendCorner(file);
  const x = corner.endsWith("left") ? inset : W - inset - w;
  const y = corner.startsWith("top") ? inset : H - inset - h;
  return { corner, x, y, w, h, fontPx, lineH, pad, rows };
}

export function nextCorner(pos: LegendPosition): LegendPosition {
  const order: LegendPosition[] = ["auto", ...CORNERS];
  return order[(order.indexOf(pos) + 1) % order.length];
}

// ----------------------------------------------------------------------------- ids / colours
const ID_PREFIX: Record<Item["type"], string> = { arrow: "ann-arrow-", mask_circle: "ann-mask-", einstein_ring: "ann-ring-", text: "ann-text-" };

/** Same rule as model.py `LensMarkFile.next_id`: first free `ann-<kind>-NNN` from 001. */
export function nextId(items: Item[], type: Item["type"]): string {
  const ids = new Set(items.map((i) => i.id));
  const prefix = ID_PREFIX[type];
  let n = 1;
  while (ids.has(`${prefix}${String(n).padStart(3, "0")}`)) n += 1;
  return `${prefix}${String(n).padStart(3, "0")}`;
}

export function isDeflectorLabel(label?: string | null): boolean {
  return !!label && label.toLowerCase().includes("deflector");
}

/**
 * Colour for a new arrow: the deflector colour when the label says "deflector", else the least-used
 * non-deflector colour in `palette.arrow_order` (first unused wins).
 */
export function nextArrowColor(items: Item[], palette: PaletteDoc, label?: string | null): ColorName {
  if (isDeflectorLabel(label)) return palette.deflector;
  const used = new Map<string, number>();
  for (const it of items) if (it.type === "arrow" && it.status !== "rejected") used.set(it.color, (used.get(it.color) || 0) + 1);
  let best: ColorName | null = null, bestN = Infinity;
  for (const c of palette.arrow_order) {
    if (c === palette.deflector) continue;
    const n = used.get(c) || 0;
    if (n < bestN) { best = c; bestN = n; }
  }
  return best || palette.arrow_order[0];
}

// ----------------------------------------------------------------------------- edits
/** The geometry fields that `edit_of` snapshots before a proposed item is dragged. */
export function geometryOf(it: Item): Record<string, unknown> {
  switch (it.type) {
    case "arrow": return { tail: [...it.tail], head: [...it.head] };
    case "mask_circle": return { center: [...it.center], radius_arcsec: it.radius_arcsec };
    case "einstein_ring": return { center: [...it.center], theta_e_arcsec: it.theta_e_arcsec };
    case "text": return { pos: [...it.pos] };
  }
}

/** Primary anchor (the point a reviewer judges): arrow head, circle centre, text position. */
export function primaryAnchor(it: Item | Record<string, unknown>): UV | null {
  const g = it as Record<string, unknown>;
  const p = (g.head ?? g.center ?? g.pos) as UV | undefined;
  return Array.isArray(p) && p.length === 2 ? [p[0], p[1]] : null;
}

/** Angular distance (arcsec) between an item's current primary anchor and its `edit_of` snapshot. */
export function deltaArcsec(it: Item, W: number, H: number, cutout: number): number | null {
  if (!it.edit_of) return null;
  const a = primaryAnchor(it), b = primaryAnchor(it.edit_of);
  if (!a || !b) return null;
  const ps = cutout / W;
  return Math.hypot((a[0] - b[0]) * W * ps, (a[1] - b[1]) * H * ps);
}
