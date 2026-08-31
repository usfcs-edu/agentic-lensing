/**
 * The verbs of the app (load / save / add / delete / review / navigate). UI modules, the pointer
 * tools and the keyboard map all call these; `window.__lensmark` exposes a subset for tests.
 */
import { api, ApiError } from "./api";
import { dist_uv, px_to_arcsec } from "./coords";
import * as G from "./geometry";
import {
  clearDraft, emit, getItem, mutate, pushUndo, readDraft, select, setFile, setStatus, state, latestRun, markSaved,
} from "./state";
import { toast } from "./ui/toast";
import type { Arrow, EinsteinRing, Item, LegendPosition, MaskCircle, MaskKind, Status, TextNote, UV, Verdict } from "./types";

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export const clampUV = (p: UV): UV => [clamp01(p[0]), clamp01(p[1])];

function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

// ----------------------------------------------------------------------------- load / save
let loadSeq = 0;

export async function load(id: string): Promise<void> {
  const seq = ++loadSeq;
  const sameImage = state.id === id;
  state.id = id;
  if (!sameImage) {                                    // a reload after propose/critique keeps the log + pending ops
    state.propose = null;
    state.patch = null;
    state.file = null;
    state.fileExists = false;
    emit("file", "load");
  }
  state.draft = null;
  select(null);
  if (location.hash.slice(1) !== id) history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  emit("view");
  try {
    const { file, exists } = await api.ann(id);
    if (seq !== loadSeq) return;                       // a newer load won
    const draft = readDraft(id);
    if (draft && (!exists || draft.saved_at > file.modified) && JSON.stringify(draft.file) !== JSON.stringify(file)) state.draft = draft;
    setFile(file, { exists });
    state.reviewStartedAt = Date.now();
    state.renderTs = Date.now();
    document.title = `LensMark – ${id}`;
    setStatus(exists ? `loaded ${id} (${file.items.length} items)` : `${id}: new file (not saved yet)`);
    emit("draft");
    emit("view");
  } catch (e) {
    setStatus(`load ${id} failed: ${describe(e)}`, "error");
  }
}

export function restoreDraft(): void {
  if (!state.draft) return;
  setFile(state.draft.file, { exists: state.fileExists, dirty: true });
  state.draft = null;
  setStatus("draft restored (unsaved)");
  emit("draft");
}

export function discardDraft(): void {
  if (state.id) clearDraft(state.id);
  state.draft = null;
  emit("draft");
}

export async function save(): Promise<boolean> {
  if (!state.id || !state.file) return false;
  const id = state.id;
  try {
    const resp = await api.putAnn(id, state.file, state.reviewer || "ui");
    markSaved(resp);
    setStatus(`saved ${id} ${resp.lint?.length ? `(${resp.lint.length} lint)` : ""}`.trim(), "ok");
    toast(`saved ${id}`, "ok");
    emit("file", "save");
    void refreshImages();
    return true;
  } catch (e) {
    const msg = describe(e);
    setStatus(`save failed: ${msg}`, "error");
    toast(`save failed: ${msg}`, "error", 8000);
    return false;
  }
}

export async function refreshImages(): Promise<void> {
  try {
    state.images = await api.images();
    emit("images");
  } catch (e) {
    setStatus(`image list: ${describe(e)}`, "error");
  }
}

export function describe(e: unknown): string {
  if (e instanceof ApiError) return e.describe();
  return (e as Error)?.message || String(e);
}

export function navigate(delta: number): void {
  if (!state.images.length) return;
  const i = state.images.findIndex((im) => im.id === state.id);
  const j = (i < 0 ? 0 : i + delta + state.images.length) % state.images.length;
  location.hash = encodeURIComponent(state.images[j].id);
}

// ----------------------------------------------------------------------------- items
/** Fill in id / created_by / created_at / status / colour and append. Returns the new id. */
export function addItem(json: Partial<Item> & { type: Item["type"] }): string {
  if (!state.file) throw new Error("no file loaded");
  const file = state.file;
  const pal = state.style?.palette;
  const id = json.id && !getItem(json.id) ? json.id : G.nextId(file.items, json.type);
  const base = {
    id, label: json.label ?? null, show_in_legend: json.show_in_legend ?? (json.type === "arrow"),
    created_by: json.created_by ?? { kind: "human" as const, reviewer: state.reviewer || null },
    created_at: json.created_at ?? nowIso(), status: (json.status ?? "accepted") as Status,
  };
  let item: Item;
  switch (json.type) {
    case "arrow": {
      const a = json as Partial<Arrow>;
      if (!a.tail || !a.head) throw new Error("arrow needs tail and head");
      const color = a.color ?? (pal ? G.nextArrowColor(file.items, pal, a.label) : "cyan");
      item = { ...base, type: "arrow", tail: clampUV(a.tail), head: clampUV(a.head), color, label_anchor: a.label_anchor ?? "auto", ...(a.label_offset ? { label_offset: a.label_offset } : {}) };
      break;
    }
    case "mask_circle": {
      const c = json as Partial<MaskCircle>;
      if (!c.center) throw new Error("mask needs center");
      item = { ...base, type: "mask_circle", center: clampUV(c.center), radius_arcsec: c.radius_arcsec || defaultMaskRadius(), kind: c.kind ?? "galaxy", color: "mask_red", show_in_legend: json.show_in_legend ?? false };
      break;
    }
    case "einstein_ring": {
      const r = json as Partial<EinsteinRing>;
      if (!r.center) throw new Error("ring needs center");
      item = { ...base, type: "einstein_ring", center: clampUV(r.center), theta_e_arcsec: r.theta_e_arcsec || thetaDefault(), color: "ring_white", show_in_legend: json.show_in_legend ?? false, ...(r.center_ref ? { center_ref: r.center_ref } : {}), ...(r.label_pos ? { label_pos: r.label_pos } : {}) };
      break;
    }
    case "text": {
      const t = json as Partial<TextNote>;
      if (!t.pos) throw new Error("text needs pos");
      item = { ...base, type: "text", pos: clampUV(t.pos), text: t.text ?? "", color: t.color ?? "white", show_in_legend: json.show_in_legend ?? false };
      break;
    }
    default:
      throw new Error(`unknown item type ${(json as { type: string }).type}`);
  }
  if (json.notes) item.notes = json.notes;
  if (json.review) item.review = json.review;
  mutate((f) => { f.items.push(item); }, { source: "add" });
  select(id);
  return id;
}

export function defaultMaskRadius(): number {
  return +(0.04 * (state.file?.image.cutout_arcsec || 16)).toFixed(2);
}

/** theta_E for a new ring: the System theta_E input, else 1.5". */
export function thetaDefault(): number {
  const v = state.file?.system.theta_e.value_arcsec;
  return v && v > 0 ? v : 1.5;
}

export function createArrow(tail: UV, head: UV): string | null {
  if (Math.hypot(tail[0] - head[0], tail[1] - head[1]) < 0.01) { setStatus("arrow too short (drag from tail to head)"); return null; }
  return addItem({ type: "arrow", tail, head, label: "" });
}

export function createMask(center: UV, radiusPx: number, kind: MaskKind): string {
  const file = state.file!;
  const r = radiusPx < 2 ? defaultMaskRadius() : +px_to_arcsec(radiusPx, file.image.width, file.image.cutout_arcsec).toFixed(3);
  return addItem({ type: "mask_circle", center, radius_arcsec: Math.max(0.05, r), kind });
}

/** Ring at the click; snaps to (and tracks) a deflector arrow head within ~2 % of the image. */
export function createRing(center: UV): string {
  const file = state.file!;
  const defl = file.items.find((it): it is Arrow => it.type === "arrow" && G.isDeflectorLabel(it.label) && it.status !== "rejected" && dist_uv(it.head, center) < 0.02);
  return addItem({ type: "einstein_ring", center: defl ? defl.head : center, theta_e_arcsec: thetaDefault(), ...(defl ? { center_ref: defl.id } : {}) });
}

export function createText(pos: UV, text: string): string | null {
  const t = text.trim();
  if (!t) return null;
  return addItem({ type: "text", pos, text: t });
}

export function deleteItem(id: string): void {
  if (!getItem(id)) return;
  mutate((f) => {
    f.items = f.items.filter((it) => it.id !== id);
    for (const it of f.items) if (it.type === "einstein_ring" && it.center_ref === id) it.center_ref = null;
    if (f.legend.order) f.legend.order = f.legend.order.filter((x) => x !== id);
  }, { source: "delete" });
  if (state.selectedId === id) select(null);
  setStatus(`deleted ${id}`);
}

export function deleteSelected(): void {
  if (state.selectedId) deleteItem(state.selectedId);
}

/** `coalesce`: a key such as "label:ann-arrow-001" merges rapid successive edits into one undo step. */
export function updateItem(id: string, fn: (it: Item) => void, source = "items-editor", coalesce?: string): void {
  const it = getItem(id);
  if (!it) return;
  mutate(() => fn(it), { source, coalesce });
}

// ----------------------------------------------------------------------------- geometry edits (drags)
/** Call once when a drag starts: one undo step for the whole drag; a proposed item becomes `edited`. */
export function beginGeometryEdit(id: string): void {
  const it = getItem(id);
  if (!it) return;
  pushUndo();
  if (it.status === "proposed" || (it.status === "accepted" && it.created_by.kind === "claude" && !it.edit_of)) {
    it.edit_of = G.geometryOf(it);
    it.status = "edited";
  }
}

export function updateGeometry(id: string, fn: (it: Item) => void): void {
  const it = getItem(id);
  if (!it) return;
  mutate(() => fn(it), { undo: false, kind: "geometry", source: "drag" });
}

export function endGeometryEdit(id: string): void {
  const it = getItem(id);
  if (it && it.edit_of && it.status === "edited") {
    const d = G.deltaArcsec(it, state.file!.image.width, state.file!.image.height, state.file!.image.cutout_arcsec);
    if (d != null) it.review = { ...(it.review || { verdict: "wrong_position" }), delta_arcsec: +d.toFixed(3) };
  }
  emit("file", "drag-end");
}

// ----------------------------------------------------------------------------- review verbs
export function applyVerdict(id: string, verdict: Verdict, comment?: string): void {
  updateItem(id, (it) => {
    it.review = { ...(it.review || {}), verdict, reviewer: state.reviewer || null, reviewed_at: nowIso(), comment: comment ?? it.review?.comment ?? "" };
    if (verdict === "correct" && it.status === "proposed") it.status = "accepted";
    if ((verdict === "spurious" || verdict === "redundant") && it.status === "proposed") it.status = "rejected";
  }, "review");
}

/** Is this item part of the critique of the latest run (Claude/voice-made, or human-added since the run)? */
export function inReviewSet(it: Item): boolean {
  if (it.created_by.kind === "claude" || it.created_by.kind === "voice") return true;
  const run = latestRun();
  return !!run && !!run.started_at && it.created_at >= run.started_at;
}

export function acceptItem(id: string): void {
  updateItem(id, (it) => {
    if (it.status === "proposed" || it.status === "rejected") it.status = it.edit_of ? "edited" : "accepted";
    if (!it.review && inReviewSet(it)) it.review = { verdict: it.created_by.kind === "human" ? "missed_by_model" : "correct", reviewer: state.reviewer || null, reviewed_at: nowIso(), comment: "" };
  }, "review");
}

export function rejectItem(id: string): void {
  updateItem(id, (it) => {
    it.status = "rejected";
    if (inReviewSet(it) && (!it.review || it.review.verdict === "correct")) it.review = { ...(it.review || {}), verdict: "spurious", reviewer: state.reviewer || null, reviewed_at: nowIso(), comment: it.review?.comment || "" };
  }, "review");
}

export function acceptAll(): void {
  mutate((f) => { for (const it of f.items) if (it.status === "proposed") { it.status = "accepted"; it.review = it.review || { verdict: "correct", reviewer: state.reviewer || null, reviewed_at: nowIso(), comment: "" }; } }, { source: "review" });
}

export function rejectAll(): void {
  mutate((f) => { for (const it of f.items) if (it.status === "proposed") { it.status = "rejected"; it.review = it.review || { verdict: "spurious", reviewer: state.reviewer || null, reviewed_at: nowIso(), comment: "" }; } }, { source: "review" });
}

export function proposedItems(): Item[] {
  return state.file?.items.filter((it) => it.status === "proposed") || [];
}

/** Items a critique covers: everything Claude proposed (any status now) + human items added since the run. */
export function reviewSet(): Item[] {
  return state.file ? state.file.items.filter(inReviewSet) : [];
}

// ----------------------------------------------------------------------------- misc
export function cycleLegend(): void {
  mutate((f) => { f.legend.position = G.nextCorner(f.legend.position); }, { source: "legend" });
  setStatus(`legend: ${state.file!.legend.position}`);
}

export function setLegendPosition(pos: LegendPosition): void {
  mutate((f) => { f.legend.position = pos; }, { source: "system" });
}

export function setRendered(on: boolean): void {
  state.rendered = on;
  if (on) state.renderTs = Date.now();
  emit("view");
}
