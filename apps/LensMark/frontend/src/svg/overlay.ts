/**
 * State -> SVG. The overlay's viewBox is in image pixels ("0 0 W H"), so every coordinate here is
 * an image pixel and the CSS zoom/pan transform on #stage is irrelevant to the geometry.
 *
 * DOM (CONTRACT.md "Browser DOM"):
 *   <g id="items">  committed items   <g id="ghost"> proposed items (dashed outline, 50% opacity)
 *   <g id="legend"> plate             <g id="handles"> selection handles
 *   + <g id="draft"> (the shape being drawn) and <g id="voice-ghost"> (unapplied voice ops)
 * Draw order inside a group: mask circles -> einstein ring -> arrows -> text -> (legend last).
 */
import { arcsec_to_px } from "../coords";
import { clear, svgEl } from "../dom";
import * as G from "../geometry";
import { getItem, state, subscribe } from "../state";
import type { Arrow, EinsteinRing, Item, LensMarkFile, MaskCircle, MaskKind, PatchOp, StyleDefaults, TextNote, UV } from "../types";

type XY = [number, number];
export type DraftShape =
  | { type: "arrow"; tail: XY; head: XY; color: string }
  | { type: "circle"; center: XY; r: number; kind: MaskKind | "ring" }
  | null;

export const FONT = "'DejaVu Sans', Verdana, 'Bitstream Vera Sans', sans-serif";
const ORDER: Record<Item["type"], number> = { mask_circle: 0, einstein_ring: 1, arrow: 2, text: 3 };
const HIT_W = 14;

let svg: SVGSVGElement | null = null;
let draft: DraftShape = null;

export function mountOverlay(root: SVGSVGElement): void {
  svg = root;
  subscribe((ev) => {
    if (["file", "geometry", "selection", "view", "voice", "boot"].includes(ev.kind)) render();
  });
  render();
}

export function setDraft(d: DraftShape): void {
  draft = d;
  render();
}

function colorHex(name: string): string {
  return state.style?.palette.colors[name] || "#FFFFFF";
}

// ----------------------------------------------------------------------------- render
export function render(): void {
  if (!svg) return;
  const file = state.file;
  clear(svg);
  if (!file) { svg.setAttribute("viewBox", "0 0 1 1"); return; }
  const W = file.image.width, H = file.image.height;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", String(W));
  svg.setAttribute("height", String(H));

  const items = svgEl("g", { id: "items" });
  const ghost = svgEl("g", { id: "ghost" });
  const legend = svgEl("g", { id: "legend" });
  const voice = svgEl("g", { id: "voice-ghost" });
  const draftG = svgEl("g", { id: "draft" });
  const handles = svgEl("g", { id: "handles" });

  const sorted = [...file.items].sort((a, b) => ORDER[a.type] - ORDER[b.type]);
  for (const it of sorted) {
    const hidden = it.status === "rejected" || it.status === "invalid";
    if (hidden && !state.showRejected) continue;
    const isGhost = it.status === "proposed";
    const g = drawItem(it, file, isGhost);
    if (hidden) g.classList.add("hidden-status");
    if (it.id === state.selectedId) g.classList.add("selected");
    (isGhost ? ghost : items).appendChild(g);
  }
  drawLegend(legend, file);
  if (state.patch) drawVoiceOps(voice, state.patch.ops, file);
  drawDraft(draftG, file);
  const sel = getItem(state.selectedId);
  if (sel && (state.showRejected || (sel.status !== "rejected" && sel.status !== "invalid"))) drawHandles(handles, sel, file);

  svg.append(items, ghost, legend, voice, draftG, handles);
}

// ----------------------------------------------------------------------------- items
export function drawItem(it: Item, file: LensMarkFile, ghost: boolean): SVGGElement {
  const cls = it.type === "mask_circle" ? `annot mask ${it.kind}` : it.type === "einstein_ring" ? "annot ring" : `annot ${it.type}`;
  const g = svgEl("g", {
    class: cls + (ghost ? " ghost" : ""), "data-id": it.id, "data-type": it.type, "data-status": it.status,
    "data-kind": it.type === "mask_circle" ? it.kind : null, opacity: ghost ? 0.5 : null,
  });
  switch (it.type) {
    case "arrow": drawArrow(g, it, file, ghost); break;
    case "mask_circle": drawMask(g, it, file); break;
    case "einstein_ring": drawRing(g, it, file); break;
    case "text": drawText(g, it, file); break;
  }
  return g;
}

function px(p: UV, file: LensMarkFile): XY {
  return [p[0] * file.image.width, p[1] * file.image.height];
}

function drawArrow(g: SVGGElement, it: Arrow, file: LensMarkFile, ghost: boolean): void {
  const W = file.image.width, H = file.image.height, m = G.minDim(W, H);
  const style = file.style_defaults;
  const color = colorHex(it.color);
  const a = G.arrowGeometry(px(it.tail, file), px(it.head, file), style, m);
  // invisible wide hit area so a 2-px shaft is clickable
  g.appendChild(svgEl("line", { class: "hit", x1: a.x1, y1: a.y1, x2: it.head[0] * W, y2: it.head[1] * H, stroke: "transparent", "stroke-width": HIT_W, "pointer-events": "stroke" }));
  g.appendChild(svgEl("line", {
    class: "shaft", x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2, stroke: color, "stroke-width": a.lineW,
    "stroke-linecap": "butt", "stroke-dasharray": ghost ? `${a.lineW * 2.5} ${a.lineW * 2}` : null,
  }));
  g.appendChild(svgEl("polygon", {
    class: "head", points: a.head.map((p) => `${p[0]},${p[1]}`).join(" "), fill: color,
    "fill-opacity": ghost ? 0.35 : null, stroke: ghost ? color : null, "stroke-width": ghost ? a.lineW * 0.6 : null,
    "stroke-dasharray": ghost ? `${a.lineW} ${a.lineW}` : null,
  }));
  const lp = G.labelPlacement(it, style, W, H);
  if (lp) g.appendChild(haloText("label", lp.x, lp.y, it.label || "", lp.fontPx, color, style));
}

function drawMask(g: SVGGElement, it: MaskCircle, file: LensMarkFile): void {
  const [cx, cy] = px(it.center, file);
  const r = arcsec_to_px(it.radius_arcsec, file.image.width, file.image.cutout_arcsec);
  const pat = G.maskPattern(it.kind, file.style_defaults, G.minDim(file.image.width, file.image.height));
  g.appendChild(svgEl("circle", { class: "hit", cx, cy, r, fill: "none", stroke: "transparent", "stroke-width": HIT_W, "pointer-events": "stroke" }));
  g.appendChild(svgEl("circle", {
    class: "mask-circle", cx, cy, r, fill: "none", stroke: colorHex(it.color), "stroke-width": pat.strokeWidth,
    "stroke-dasharray": pat.dasharray, "stroke-linecap": pat.linecap,
  }));
}

/** Ring centre: the head of the `center_ref` arrow when that arrow exists, else `center`. */
export function ringCenter(it: EinsteinRing, file: LensMarkFile): UV {
  if (it.center_ref) {
    const ref = file.items.find((x) => x.id === it.center_ref);
    if (ref && ref.type === "arrow") return ref.head;
  }
  return it.center;
}

function drawRing(g: SVGGElement, it: EinsteinRing, file: LensMarkFile): void {
  const W = file.image.width, H = file.image.height, m = G.minDim(W, H);
  const style = file.style_defaults;
  const [cx, cy] = px(ringCenter(it, file), file);
  const r = arcsec_to_px(it.theta_e_arcsec, W, file.image.cutout_arcsec);
  const pat = G.ringPattern(style, m);
  g.appendChild(svgEl("circle", { class: "hit", cx, cy, r, fill: "none", stroke: "transparent", "stroke-width": HIT_W, "pointer-events": "stroke" }));
  g.appendChild(svgEl("circle", {
    class: "ring-circle", cx, cy, r, fill: "none", stroke: colorHex(it.color), "stroke-width": pat.strokeWidth,
    "stroke-dasharray": pat.dasharray, "stroke-linecap": pat.linecap,
  }));
  const text = G.thetaLabelText(it.theta_e_arcsec, it.label);
  const lp = G.thetaLabelPlacement([cx, cy], r, style, W, H, text, it.label_pos);
  const t = haloText("theta-label", lp.x, lp.y, text, lp.fontPx, "#FFFFFF", style);
  t.setAttribute("font-style", "italic");
  g.appendChild(t);
}

function drawText(g: SVGGElement, it: TextNote, file: LensMarkFile): void {
  const m = G.minDim(file.image.width, file.image.height);
  const [x, y] = px(it.pos, file);
  const style = file.style_defaults;
  g.appendChild(haloText("note", x, y, it.text, style.text.size * m, colorHex(it.color), style));
}

/** Bold text centred at (x, y) with the dark halo (`label.halo`, `halo_px`) drawn under the glyphs. */
function haloText(cls: string, x: number, y: number, text: string, fontPx: number, fill: string, style: StyleDefaults): SVGTextElement {
  const haloPx = style.label.halo_px ?? 2;
  return svgEl("text", {
    class: cls, x, y, text, fill, "font-size": fontPx, "font-family": FONT, "font-weight": "bold",
    "text-anchor": "middle", "dominant-baseline": "central", "paint-order": "stroke",
    stroke: style.label.halo || "#000000CC", "stroke-width": 2 * haloPx, "stroke-linejoin": "round",
  });
}

// ----------------------------------------------------------------------------- legend
function drawLegend(g: SVGGElement, file: LensMarkFile): void {
  const lay = G.legendLayout(file);
  if (!lay) return;
  const lg = file.style_defaults.legend;
  g.setAttribute("data-corner", lay.corner);
  g.appendChild(svgEl("rect", { class: "legend-plate", x: lay.x, y: lay.y, width: lay.w, height: lay.h, fill: lg.bg, stroke: "#FFFFFF40", "stroke-width": 0.5, rx: 2 }));
  lay.rows.forEach((row, i) => {
    g.appendChild(svgEl("text", {
      class: "legend-row", "data-id": row.id, x: lay.x + lay.pad, y: lay.y + lay.pad + (i + 0.5) * lay.lineH, text: row.text,
      fill: colorHex(row.color), "font-size": lay.fontPx, "font-family": FONT, "font-weight": "bold",
      "dominant-baseline": "central", opacity: row.status === "proposed" ? 0.6 : null,
    }));
  });
}

// ----------------------------------------------------------------------------- handles
function drawHandles(g: SVGGElement, it: Item, file: LensMarkFile): void {
  const W = file.image.width, H = file.image.height, m = G.minDim(W, H);
  const r = Math.max(4, 0.014 * m);
  const add = (name: string, p: XY) => g.appendChild(svgEl("circle", { class: "handle", "data-handle": name, cx: p[0], cy: p[1], r }));
  switch (it.type) {
    case "arrow": add("tail", px(it.tail, file)); add("head", px(it.head, file)); break;
    case "mask_circle": {
      const c = px(it.center, file);
      add("center", c);
      add("radius", [c[0] + arcsec_to_px(it.radius_arcsec, W, file.image.cutout_arcsec), c[1]]);
      break;
    }
    case "einstein_ring": {
      const c = px(ringCenter(it, file), file);
      add("center", c);
      add("radius", [c[0] + arcsec_to_px(it.theta_e_arcsec, W, file.image.cutout_arcsec), c[1]]);
      break;
    }
    case "text": add("pos", px(it.pos, file)); break;
  }
}

// ----------------------------------------------------------------------------- draft + voice ghost
function drawDraft(g: SVGGElement, file: LensMarkFile): void {
  if (!draft) return;
  const m = G.minDim(file.image.width, file.image.height);
  const style = file.style_defaults;
  if (draft.type === "arrow") {
    const a = G.arrowGeometry(draft.tail, draft.head, style, m);
    g.appendChild(svgEl("line", { x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2, stroke: draft.color, "stroke-width": a.lineW }));
    g.appendChild(svgEl("polygon", { points: a.head.map((p) => `${p[0]},${p[1]}`).join(" "), fill: draft.color }));
  } else {
    const pat = draft.kind === "ring" ? G.ringPattern(style, m) : G.maskPattern(draft.kind, style, m);
    const color = colorHex(draft.kind === "ring" ? "ring_white" : "mask_red");
    g.appendChild(svgEl("circle", {
      cx: draft.center[0], cy: draft.center[1], r: Math.max(draft.r, 0.5), fill: "none", stroke: color,
      "stroke-width": pat.strokeWidth, "stroke-dasharray": pat.dasharray, "stroke-linecap": pat.linecap,
    }));
    g.appendChild(svgEl("circle", { cx: draft.center[0], cy: draft.center[1], r: 1.5, fill: color }));
  }
}

/** A voice `add`/`update` op as a pseudo-item so the reviewer sees it before applying. */
export function pseudoItem(op: PatchOp, i: number): Item | null {
  const base = op.op === "add" ? op.item : op.op === "update" ? { ...(getItem(op.id || "") || {}), ...(op.set || {}) } : null;
  if (!base || typeof base !== "object") return null;
  const d = base as Record<string, unknown>;
  const type = String(d.type || "");
  const uv = (k: string): UV | null => Array.isArray(d[k]) && (d[k] as unknown[]).length === 2 ? [Number((d[k] as unknown[])[0]), Number((d[k] as unknown[])[1])] : null;
  const common = { id: op.id || `voice-${i}`, label: (d.label as string) || null, show_in_legend: false, created_by: { kind: "voice" as const }, created_at: "", status: "proposed" as const };
  if (type === "arrow" && uv("head")) {
    const head = uv("head")!;
    const tail = uv("tail") || [Math.min(0.98, head[0] + 0.1), Math.min(0.98, head[1] + 0.1)];
    return { ...common, type: "arrow", tail, head, color: (d.color as Arrow["color"]) || "cyan", label_anchor: "auto", show_in_legend: true };
  }
  if (type === "mask_circle" && uv("center")) {
    return { ...common, type: "mask_circle", center: uv("center")!, radius_arcsec: Number(d.radius_arcsec) || 0.5, kind: (["galaxy", "star", "artifact"].includes(String(d.kind)) ? d.kind : "galaxy") as MaskKind, color: "mask_red" };
  }
  if (type === "einstein_ring" && uv("center")) {
    return { ...common, type: "einstein_ring", center: uv("center")!, theta_e_arcsec: Number(d.theta_e_arcsec) || 1.5, color: "ring_white" };
  }
  if (type === "text" && uv("pos")) {
    return { ...common, type: "text", pos: uv("pos")!, text: String(d.text || d.label || ""), color: (d.color as TextNote["color"]) || "white" };
  }
  return null;
}

function drawVoiceOps(g: SVGGElement, ops: PatchOp[], file: LensMarkFile): void {
  ops.forEach((op, i) => {
    const it = pseudoItem(op, i);
    if (!it) return;
    const node = drawItem(it, file, true);
    node.classList.add("voice");
    node.setAttribute("data-op", String(i));
    g.appendChild(node);
  });
}
