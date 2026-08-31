/**
 * Application state: the current LensMarkFile (the JSON IS the state), dirty flag, selection, tool,
 * undo stack of snapshots (max 50), a localStorage draft per image, and a tiny pub/sub so panels
 * re-render on the changes they care about.
 */
import type {
  CampaignConfig, HealthResponse, ImageSummary, Item, LensMarkFile, ModelsResponse, PaletteDoc, Patch,
  ProposalRun, PutResponse, StyleResponse, ToolName,
} from "./types";

export type ChangeKind =
  | "boot" | "images" | "file" | "geometry" | "selection" | "tool" | "dirty" | "view" | "status" | "propose" | "voice" | "draft";
export interface ChangeEvent { kind: ChangeKind; source?: string }
export type Listener = (ev: ChangeEvent) => void;

export interface ProposeState {
  runId: string;
  running: boolean;
  log: string[];
  cost?: number | null;
  startedAt: number;
  error?: string | null;
}

export interface Draft { saved_at: string; file: LensMarkFile }

export const UNDO_MAX = 50;
const LS_REVIEWER = "lensmark:reviewer";
const LS_TAB = "lensmark:tab";

export interface AppState {
  style: StyleResponse | null;
  models: ModelsResponse | null;
  config: CampaignConfig | null;
  health: HealthResponse | null;
  images: ImageSummary[];
  id: string | null;
  file: LensMarkFile | null;
  fileExists: boolean;
  dirty: boolean;
  selectedId: string | null;
  tool: ToolName;
  reviewer: string;
  showRejected: boolean;
  rendered: boolean;
  lint: string[];
  status: string;
  statusKind: "info" | "error" | "ok";
  /** when the current file (or its latest proposal) became reviewable; lead_time_s counts from here */
  reviewStartedAt: number;
  propose: ProposeState | null;
  patch: Patch | null;
  draft: Draft | null;
  tab: string;
  /** monotonically increasing; bumps when the rendered PNG may have changed */
  renderTs: number;
}

export const state: AppState = {
  style: null, models: null, config: null, health: null, images: [],
  id: null, file: null, fileExists: false, dirty: false, selectedId: null, tool: "select",
  reviewer: "", showRejected: false, rendered: false, lint: [], status: "", statusKind: "info",
  reviewStartedAt: Date.now(), propose: null, patch: null, draft: null, tab: lsGet(LS_TAB) || "items", renderTs: Date.now(),
};

// ----------------------------------------------------------------------------- pub/sub
const listeners = new Set<Listener>();
export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
export function emit(kind: ChangeKind, source?: string): void {
  const ev: ChangeEvent = { kind, source };
  for (const fn of listeners) {
    try { fn(ev); } catch (e) { console.error("listener failed", kind, e); }
  }
}

export function setStatus(msg: string, kind: AppState["statusKind"] = "info"): void {
  state.status = msg;
  state.statusKind = kind;
  if (kind === "error") console.error("[lensmark]", msg);
  emit("status");
}

// ----------------------------------------------------------------------------- localStorage helpers
export function lsGet(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
export function lsSet(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* private mode / quota: drafts are best-effort */ }
}
export function lsDel(key: string): void {
  try { localStorage.removeItem(key); } catch { /* ignore */ }
}

export function campaignKey(): string {
  return state.health?.campaign_dir || state.config?.campaign || "campaign";
}
export function draftKey(id: string): string {
  return `lensmark:${campaignKey()}:${id}`;
}
export function readDraft(id: string): Draft | null {
  const raw = lsGet(draftKey(id));
  if (!raw) return null;
  try {
    const d = JSON.parse(raw) as Draft;
    return d && d.file && d.saved_at ? d : null;
  } catch { return null; }
}
export function saveDraft(): void {
  if (!state.id || !state.file) return;
  lsSet(draftKey(state.id), JSON.stringify({ saved_at: new Date().toISOString(), file: state.file }));
}
export function clearDraft(id: string): void {
  lsDel(draftKey(id));
}

export function setReviewer(name: string): void {
  state.reviewer = name.trim();
  lsSet(LS_REVIEWER, state.reviewer);
  emit("view");
}
export function initReviewer(defaultName: string): void {
  state.reviewer = lsGet(LS_REVIEWER) || defaultName || "reviewer";
}
export function setTab(tab: string): void {
  state.tab = tab;
  lsSet(LS_TAB, tab);
  emit("view");
}

// ----------------------------------------------------------------------------- file / undo
let undoStack: string[] = [];

export function snapshot(): string {
  return JSON.stringify(state.file);
}

/** Push the current file onto the undo stack (call BEFORE mutating; drags call it once at drag start). */
export function pushUndo(): void {
  if (!state.file) return;
  undoStack.push(snapshot());
  if (undoStack.length > UNDO_MAX) undoStack = undoStack.slice(-UNDO_MAX);
}

export function undo(): boolean {
  const prev = undoStack.pop();
  if (!prev) { setStatus("nothing to undo"); return false; }
  state.file = JSON.parse(prev) as LensMarkFile;
  if (state.selectedId && !getItem(state.selectedId)) state.selectedId = null;
  state.dirty = true;
  saveDraft();
  refreshLint();
  emit("file", "undo");
  emit("dirty");
  return true;
}

export function undoDepth(): number {
  return undoStack.length;
}

export function setFile(file: LensMarkFile, opts: { exists?: boolean; dirty?: boolean; keepSelection?: boolean } = {}): void {
  state.file = file;
  state.fileExists = opts.exists ?? true;
  state.dirty = opts.dirty ?? false;
  undoStack = [];
  if (!opts.keepSelection || (state.selectedId && !getItem(state.selectedId))) state.selectedId = null;
  refreshLint();
  emit("file", "load");
  emit("dirty");
  emit("selection");
}

export interface MutateOptions { source?: string; kind?: "file" | "geometry"; undo?: boolean; coalesce?: string }
const COALESCE_MS = 1200;
let lastCoalesce: { key: string; t: number } | null = null;

/** Apply an edit to the file: undo snapshot (default on), dirty, draft autosave, notify.
 * Edits sharing a `coalesce` key within 1.2 s (typing in one field) form a single undo step. */
export function mutate(fn: (file: LensMarkFile) => void, opts: MutateOptions = {}): void {
  if (!state.file) return;
  const now = Date.now();
  const merge = !!opts.coalesce && lastCoalesce?.key === opts.coalesce && now - lastCoalesce.t < COALESCE_MS;
  lastCoalesce = opts.coalesce ? { key: opts.coalesce, t: now } : null;
  if (opts.undo !== false && !merge) pushUndo();
  fn(state.file);
  state.dirty = true;
  saveDraft();
  refreshLint();
  emit(opts.kind || "file", opts.source);
  emit("dirty");
}

export function markSaved(resp: PutResponse): void {
  if (!state.file) return;
  state.file.modified = resp.modified;
  if (resp.render) state.file.render = resp.render;
  state.dirty = false;
  state.fileExists = true;
  state.lint = resp.lint || [];
  state.renderTs = Date.now();
  if (state.id) clearDraft(state.id);
  emit("dirty");
  emit("status");
}

export function getItem(id: string | null | undefined): Item | undefined {
  if (!id || !state.file) return undefined;
  return state.file.items.find((it) => it.id === id);
}

export function select(id: string | null): void {
  if (state.selectedId === id) return;
  state.selectedId = id;
  emit("selection");
}

export function setTool(tool: ToolName): void {
  state.tool = tool;
  emit("tool");
}

export function latestRun(): ProposalRun | null {
  const runs = state.file?.provenance.proposal_runs || [];
  return runs.length ? runs[runs.length - 1] : null;
}

export function palette(): PaletteDoc | null {
  return state.style?.palette || null;
}

// ----------------------------------------------------------------------------- lint (mirror of model.py LensMarkFile.lint)
const COLOR_WORD_RE = /\b(magenta|cyan|green|yellow|white|orange|gr[ae]y)\b/gi;

export function computeLint(file: LensMarkFile, pal: PaletteDoc | null): string[] {
  const warns: string[] = [];
  const ids = new Set(file.items.map((i) => i.id));
  const colors = new Set(file.items.filter((i) => ["accepted", "edited", "proposed"].includes(i.status)).map((i) => i.color));
  const words = new Set<string>();
  for (const mth of file.system.description.matchAll(COLOR_WORD_RE)) {
    const w = mth[1].toLowerCase();
    words.add(w === "grey" ? "gray" : w);
  }
  for (const w of words) if (!colors.has(w as never)) warns.push(`description mentions '${w}' but no item has that colour`);
  for (const ref of file.system.description_refs || []) if (!ids.has(ref)) warns.push(`description_refs: unknown item '${ref}'`);
  for (const it of file.items) {
    if (it.type === "einstein_ring" && it.center_ref && !ids.has(it.center_ref)) warns.push(`${it.id}: center_ref '${it.center_ref}' does not exist`);
    const reserved = pal?.reserved?.[it.color];
    if (reserved && reserved !== it.type) warns.push(`${it.id}: colour '${it.color}' is reserved for ${reserved}`);
  }
  return warns;
}

export function refreshLint(): void {
  state.lint = state.file ? computeLint(state.file, palette()) : [];
}
