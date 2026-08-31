/**
 * Typed fetch wrappers for every route in API.md. Every call checks `res.ok` and turns the server's
 * `{error, detail}` body into an ApiError so the UI can surface it in the status line / toast.
 */
import type {
  CampaignConfig, Critique, ExportResponse, HealthResponse, ImageSummary, LensMarkFile, ModelsResponse, Patch,
  PatchOp, ProposalRun, ProposeStartResponse, PutResponse, SseEvent, StyleResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
  /** One line for the status bar: the message plus a compact rendering of `detail` (422 bodies). */
  describe(): string {
    if (this.detail == null || this.detail === "") return `${this.status}: ${this.message}`;
    const d = typeof this.detail === "string" ? this.detail : JSON.stringify(this.detail);
    return `${this.status}: ${this.message} — ${d.length > 400 ? d.slice(0, 400) + "…" : d}`;
  }
}

interface Reply<T> { data: T; headers: Headers }

async function request<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {}): Promise<Reply<T>> {
  const init: RequestInit = { method, headers: { Accept: "application/json", ...headers }, cache: "no-store" };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    throw new ApiError(0, `network error: ${(e as Error).message}`);
  }
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try { data = JSON.parse(text); } catch { data = text; }
  }
  if (!res.ok) {
    const obj = (data && typeof data === "object" ? data : {}) as Record<string, unknown>;
    const msg = typeof obj.error === "string" ? obj.error : (typeof obj.detail === "string" ? obj.detail : res.statusText || "request failed");
    throw new ApiError(res.status, msg, obj.detail ?? (typeof data === "string" ? data : undefined));
  }
  return { data: data as T, headers: res.headers };
}

const enc = encodeURIComponent;

export const urls = {
  original: (id: string) => `/api/images/${enc(id)}/original`,
  annot: (id: string, ts: number = Date.now()) => `/api/images/${enc(id)}/annot?t=${ts}`,
  thumb: (id: string, px = 96) => `/api/images/${enc(id)}/thumb?px=${px}`,
};

export const api = {
  health: () => request<HealthResponse>("GET", "/api/health").then((r) => r.data),
  models: () => request<ModelsResponse>("GET", "/api/models").then((r) => r.data),
  style: () => request<StyleResponse>("GET", "/api/style").then((r) => r.data),
  config: () => request<CampaignConfig>("GET", "/api/config").then((r) => r.data),
  images: () => request<ImageSummary[]>("GET", "/api/images").then((r) => r.data),

  /** The file; `exists` is false when the server returned a fresh unsaved one (X-LensMark-Exists: 0). */
  ann: async (id: string): Promise<{ file: LensMarkFile; exists: boolean }> => {
    const r = await request<LensMarkFile>("GET", `/api/ann/${enc(id)}`);
    return { file: r.data, exists: r.headers.get("X-LensMark-Exists") !== "0" };
  },
  putAnn: (id: string, file: LensMarkFile, actor: string) =>
    request<PutResponse>("PUT", `/api/ann/${enc(id)}`, file, { "X-LensMark-Actor": actor }).then((r) => r.data),
  log: (id: string) => request<Record<string, unknown>[]>("GET", `/api/ann/${enc(id)}/log`).then((r) => r.data),
  render: (id: string) => request<{ output: string; sha256: string; stale: boolean }>("POST", `/api/render/${enc(id)}`, {}).then((r) => r.data),

  propose: (id: string, body: { model?: string; effort?: string | null; budget?: number; fewshot?: string }) =>
    request<ProposeStartResponse>("POST", `/api/propose/${enc(id)}`, body).then((r) => r.data),
  cancelPropose: (id: string, runId: string) =>
    request<{ ok: boolean }>("POST", `/api/propose/${enc(id)}/${enc(runId)}/cancel`, {}).then((r) => r.data),
  proposals: (id: string) => request<ProposalRun[]>("GET", `/api/proposals/${enc(id)}`).then((r) => r.data),

  critique: (id: string, doc: Critique) => request<{ file: string }>("POST", `/api/critique/${enc(id)}`, doc).then((r) => r.data),
  exportFmt: (fmt: "coco" | "ds9" | "masks" | "fewshot", body: { ids?: string[]; k?: number; require_flag?: boolean }) =>
    request<ExportResponse>("POST", `/api/export/${fmt}`, body).then((r) => r.data),

  patch: (id: string, body: { transcript: string; model?: string; effort?: string }) =>
    request<Patch>("POST", `/api/patch/${enc(id)}`, body).then((r) => r.data),
  applyPatch: (id: string, body: { ops: PatchOp[]; transcript?: string }) =>
    request<LensMarkFile>("POST", `/api/patch/${enc(id)}/apply`, body).then((r) => r.data),
};

/**
 * Subscribe to the proposal SSE stream. Keepalive comments are swallowed by EventSource; the
 * stream ends after `done` / `error`, at which point we close (a closed server stream would
 * otherwise make EventSource reconnect forever). Returns a closer.
 */
export function openProposeEvents(id: string, runId: string, onEvent: (ev: SseEvent) => void,
                                  onError: (msg: string) => void): () => void {
  const es = new EventSource(`/api/propose/${enc(id)}/${enc(runId)}/events`);
  let finished = false;
  es.onmessage = (e: MessageEvent) => {
    let ev: SseEvent;
    try { ev = JSON.parse(e.data); } catch { onError(`bad SSE payload: ${String(e.data).slice(0, 120)}`); return; }
    onEvent(ev);
    if (ev.phase === "done" || ev.phase === "error") { finished = true; es.close(); }
  };
  es.onerror = () => {
    if (finished) return;
    // EventSource retries on its own; a CLOSED state means the server dropped us for good.
    if (es.readyState === EventSource.CLOSED) { onError("event stream closed before done"); }
  };
  return () => { finished = true; es.close(); };
}
