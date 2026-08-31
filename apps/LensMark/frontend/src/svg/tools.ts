/**
 * Pointer handling on the stage: view transform (wheel zoom around the cursor, space / middle-button
 * pan, fit, 2x toggle), the drawing tools (arrow, galaxy/star mask, ring, text) and selection with
 * drag handles. Mouse -> image px goes through `overlay.getScreenCTM().inverse()`, so it is correct
 * at any zoom / DPR.
 */
import * as A from "../actions";
import { px_to_arcsec } from "../coords";
import { qs } from "../dom";
import { getItem, select, setStatus, setTab, setTool, state, subscribe } from "../state";
import type { Item, MaskKind, UV } from "../types";
import { ringCenter, setDraft } from "./overlay";

type XY = [number, number];

export interface View { scale: number; tx: number; ty: number; fit: number }
export const view: View = { scale: 1, tx: 0, ty: 0, fit: 1 };

let svg: SVGSVGElement, stage: HTMLElement, wrap: HTMLElement, prompt: HTMLElement, promptInput: HTMLInputElement;
let spaceDown = false;
let sizedFor = "";

type Drag =
  | { kind: "pan"; x: number; y: number; tx: number; ty: number }
  | { kind: "draft-arrow"; tail: XY; color: string }
  | { kind: "draft-circle"; center: XY; mask: MaskKind }
  | { kind: "handle"; id: string; handle: string; start: XY; orig: Item }
  | { kind: "move"; id: string; start: XY; orig: Item; moved: boolean }
  | null;
let drag: Drag = null;

export function installTools(): void {
  svg = qs<SVGSVGElement>("#overlay");
  stage = qs("#stage");
  wrap = qs("#stage-wrap");
  prompt = qs("#text-prompt");
  promptInput = qs<HTMLInputElement>("#text-prompt input");

  wrap.addEventListener("pointerdown", onDown);
  wrap.addEventListener("pointermove", onMove);
  wrap.addEventListener("pointerup", onUp);
  wrap.addEventListener("pointercancel", onUp);
  wrap.addEventListener("wheel", onWheel, { passive: false });
  wrap.addEventListener("contextmenu", (e) => e.preventDefault());
  wrap.addEventListener("dblclick", (e) => { if (state.tool === "select") { fitToView(); e.preventDefault(); } });
  promptInput.addEventListener("keydown", onPromptKey);
  window.addEventListener("resize", () => fitToView());

  subscribe((ev) => {
    if (ev.kind === "file" || ev.kind === "view") sizeStage();
    if (ev.kind === "tool") wrap.setAttribute("data-tool", state.tool);
  });
  wrap.setAttribute("data-tool", state.tool);
}

export function setSpace(down: boolean): void {
  spaceDown = down;
  wrap.classList.toggle("space", down);
}

// ----------------------------------------------------------------------------- view transform
function sizeStage(): void {
  const f = state.file;
  if (!f) return;
  const key = `${state.id}:${f.image.width}x${f.image.height}`;
  stage.style.width = `${f.image.width}px`;
  stage.style.height = `${f.image.height}px`;
  if (key !== sizedFor) { sizedFor = key; fitToView(); }
}

export function applyTransform(): void {
  stage.style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`;
  wrap.setAttribute("data-zoom", view.scale.toFixed(2));
}

/** Fit the image into the viewport with a small margin and centre it. */
export function fitToView(): void {
  const f = state.file;
  if (!f) return;
  const pad = 16;
  const r = wrap.getBoundingClientRect();
  const s = Math.max(0.05, Math.min((r.width - 2 * pad) / f.image.width, (r.height - 2 * pad) / f.image.height));
  view.fit = s;
  view.scale = s;
  view.tx = (r.width - f.image.width * s) / 2;
  view.ty = (r.height - f.image.height * s) / 2;
  applyTransform();
}

/** Zoom by factor k keeping the image point under (clientX, clientY) fixed (default: viewport centre). */
export function zoomAt(k: number, clientX?: number, clientY?: number): void {
  const r = wrap.getBoundingClientRect();
  const cx = (clientX ?? r.left + r.width / 2) - r.left;
  const cy = (clientY ?? r.top + r.height / 2) - r.top;
  const s2 = Math.min(40, Math.max(0.05, view.scale * k));
  const kk = s2 / view.scale;
  // screen = t + s*p  =>  p fixed under the cursor: t' = c - kk*(c - t)
  view.tx = cx - kk * (cx - view.tx);
  view.ty = cy - kk * (cy - view.ty);
  view.scale = s2;
  applyTransform();
}

/** `Z`: toggle between fit and 2x fit. */
export function zoomToggle(): void {
  if (Math.abs(view.scale - view.fit) < 1e-6) zoomAt(2);
  else fitToView();
}

export function zoomTo(scale: number): void {
  zoomAt(scale / view.scale);
}

function onWheel(e: WheelEvent): void {
  e.preventDefault();
  const k = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
  zoomAt(k, e.clientX, e.clientY);
}

// ----------------------------------------------------------------------------- coordinates
export function clientToImage(clientX: number, clientY: number): XY {
  const ctm = svg.getScreenCTM();
  if (!ctm) return [0, 0];
  const p = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
  return [p.x, p.y];
}

function toUV(p: XY): UV {
  const f = state.file!;
  return A.clampUV([p[0] / f.image.width, p[1] / f.image.height]);
}

// ----------------------------------------------------------------------------- pointer handlers
function onDown(e: PointerEvent): void {
  if ((e.target as Element).closest?.("#text-prompt")) return;      // typing in the text-note prompt
  if (!state.file || state.rendered) {
    if (e.button === 1 || spaceDown) startPan(e);
    return;
  }
  if (e.button === 1 || (e.button === 0 && spaceDown)) { startPan(e); return; }
  if (e.button !== 0) return;
  hidePrompt();
  const p = clientToImage(e.clientX, e.clientY);
  const target = e.target as Element;
  wrap.setPointerCapture(e.pointerId);

  switch (state.tool) {
    case "select": {
      const handle = target.closest?.(".handle") as SVGElement | null;
      const sel = getItem(state.selectedId);
      if (handle && sel) {
        A.beginGeometryEdit(sel.id);
        drag = { kind: "handle", id: sel.id, handle: handle.getAttribute("data-handle") || "", start: p, orig: JSON.parse(JSON.stringify(sel)) };
        return;
      }
      const g = target.closest?.("g.annot:not(.voice)") as SVGGElement | null;
      const id = g?.getAttribute("data-id");
      if (id && getItem(id)) {
        select(id);
        drag = { kind: "move", id, start: p, orig: JSON.parse(JSON.stringify(getItem(id))), moved: false };
        return;
      }
      select(null);
      startPan(e);
      return;
    }
    case "arrow": {
      const pal = state.style?.palette;
      const color = pal ? pal.colors[nextColor()] : "#00E5FF";
      drag = { kind: "draft-arrow", tail: p, color };
      setDraft({ type: "arrow", tail: p, head: p, color });
      return;
    }
    case "galaxy":
    case "star": {
      const mask: MaskKind = state.tool === "galaxy" ? "galaxy" : "star";
      drag = { kind: "draft-circle", center: p, mask };
      setDraft({ type: "circle", center: p, r: 0, kind: mask });
      return;
    }
    case "ring": {
      const id = A.createRing(toUV(p));
      setStatus(`ring ${id} (θ_E ${getItem(id)?.type === "einstein_ring" ? (getItem(id) as { theta_e_arcsec: number }).theta_e_arcsec : ""}″)`);
      setTool("select");
      return;
    }
    case "text": {
      showPrompt(e.clientX, e.clientY, toUV(p));
      return;
    }
  }
}

function nextColor(): string {
  const pal = state.style?.palette;
  if (!pal || !state.file) return "cyan";
  const used = new Set(state.file.items.filter((i) => i.type === "arrow" && i.status !== "rejected").map((i) => i.color));
  return pal.arrow_order.find((c) => c !== pal.deflector && !used.has(c)) || pal.arrow_order[1] || "cyan";
}

function startPan(e: PointerEvent): void {
  wrap.setPointerCapture(e.pointerId);
  drag = { kind: "pan", x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
  wrap.classList.add("panning");
}

function onMove(e: PointerEvent): void {
  if (!drag) return;
  if (drag.kind === "pan") {
    view.tx = drag.tx + (e.clientX - drag.x);
    view.ty = drag.ty + (e.clientY - drag.y);
    applyTransform();
    return;
  }
  const p = clientToImage(e.clientX, e.clientY);
  const f = state.file!;
  switch (drag.kind) {
    case "draft-arrow":
      setDraft({ type: "arrow", tail: drag.tail, head: p, color: drag.color });
      return;
    case "draft-circle":
      setDraft({ type: "circle", center: drag.center, r: Math.hypot(p[0] - drag.center[0], p[1] - drag.center[1]), kind: drag.mask });
      return;
    case "handle": {
      const d = drag;
      A.updateGeometry(d.id, (it) => applyHandle(it, d.handle, p));
      return;
    }
    case "move": {
      const d = drag;
      const du = (p[0] - d.start[0]) / f.image.width, dv = (p[1] - d.start[1]) / f.image.height;
      if (!d.moved) {
        if (Math.hypot(p[0] - d.start[0], p[1] - d.start[1]) < 3) return;   // click, not a drag
        d.moved = true;
        A.beginGeometryEdit(d.id);
      }
      const orig = d.orig;
      A.updateGeometry(d.id, (it) => translate(it, orig, du, dv));
      return;
    }
  }
}

function onUp(e: PointerEvent): void {
  if (!drag) return;
  const d = drag;
  drag = null;
  wrap.classList.remove("panning");
  try { wrap.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
  if (d.kind === "pan") return;
  const p = clientToImage(e.clientX, e.clientY);
  switch (d.kind) {
    case "draft-arrow": {
      setDraft(null);
      const id = A.createArrow(toUV(d.tail), toUV(p));
      setTool("select");
      if (id) { setStatus(`arrow ${id}: type its label`); focusLabel(); }
      return;
    }
    case "draft-circle": {
      setDraft(null);
      const rPx = Math.hypot(p[0] - d.center[0], p[1] - d.center[1]);
      const id = A.createMask(toUV(d.center), rPx, d.mask);
      const it = getItem(id) as { radius_arcsec: number };
      setStatus(`${d.mask} mask ${id}: r = ${it.radius_arcsec.toFixed(2)}″`);
      setTool("select");
      return;
    }
    case "handle":
      A.endGeometryEdit(d.id);
      return;
    case "move":
      if (d.moved) A.endGeometryEdit(d.id);
      return;
  }
}

function focusLabel(): void {
  setTab("items");
  window.setTimeout(() => {
    const inp = document.querySelector<HTMLInputElement>('[data-testid="label-input"]');
    inp?.focus();
    inp?.select();
  }, 0);
}

// ----------------------------------------------------------------------------- geometry edits
function applyHandle(it: Item, handle: string, p: XY): void {
  const f = state.file!;
  const uv = toUV(p);
  switch (it.type) {
    case "arrow":
      if (handle === "tail") it.tail = uv;
      else if (handle === "head") it.head = uv;
      return;
    case "mask_circle":
      if (handle === "center") it.center = uv;
      else if (handle === "radius") {
        const c: XY = [it.center[0] * f.image.width, it.center[1] * f.image.height];
        it.radius_arcsec = Math.max(0.05, +px_to_arcsec(Math.hypot(p[0] - c[0], p[1] - c[1]), f.image.width, f.image.cutout_arcsec).toFixed(3));
      }
      return;
    case "einstein_ring":
      if (handle === "center") { it.center = uv; it.center_ref = null; }
      else if (handle === "radius") {
        const cc = ringCenter(it, f);
        const c: XY = [cc[0] * f.image.width, cc[1] * f.image.height];
        it.theta_e_arcsec = Math.max(0.05, +px_to_arcsec(Math.hypot(p[0] - c[0], p[1] - c[1]), f.image.width, f.image.cutout_arcsec).toFixed(3));
      }
      return;
    case "text":
      if (handle === "pos") it.pos = uv;
      return;
  }
}

function translate(it: Item, orig: Item, du: number, dv: number): void {
  const mv = (p: UV): UV => A.clampUV([p[0] + du, p[1] + dv]);
  switch (it.type) {
    case "arrow": { const o = orig as typeof it; it.tail = mv(o.tail); it.head = mv(o.head); return; }
    case "mask_circle": { const o = orig as typeof it; it.center = mv(o.center); return; }
    case "einstein_ring": { const o = orig as typeof it; it.center = mv(o.center); it.center_ref = null; return; }
    case "text": { const o = orig as typeof it; it.pos = mv(o.pos); return; }
  }
}

// ----------------------------------------------------------------------------- text prompt
let promptUV: UV | null = null;

function showPrompt(clientX: number, clientY: number, uv: UV): void {
  const r = wrap.getBoundingClientRect();
  promptUV = uv;
  prompt.style.left = `${Math.min(clientX - r.left, r.width - 260)}px`;
  prompt.style.top = `${Math.min(clientY - r.top + 8, r.height - 40)}px`;
  prompt.hidden = false;
  promptInput.value = "";
  promptInput.focus();
}

export function hidePrompt(): void {
  prompt.hidden = true;
  promptUV = null;
}

function onPromptKey(e: KeyboardEvent): void {
  e.stopPropagation();
  if (e.key === "Enter") {
    const uv = promptUV;
    const text = promptInput.value;
    hidePrompt();
    if (uv) { const id = A.createText(uv, text); if (id) setStatus(`text ${id}`); }
    setTool("select");
  } else if (e.key === "Escape") {
    hidePrompt();
    setTool("select");
  }
}
