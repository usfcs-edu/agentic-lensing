/**
 * Voice tab. Text is the contract: a transcript (typed, or dictated via the Web Speech API when the
 * browser has it) -> POST /api/patch/{id} -> ops shown as ghost items + a list with per-op apply /
 * reject and apply-all -> POST /api/patch/{id}/apply -> the server returns the saved file.
 */
import * as A from "../actions";
import { api } from "../api";
import { clear, el } from "../dom";
import { emit, setFile, setStatus, state, subscribe } from "../state";
import type { PatchOp } from "../types";
import { toast } from "./toast";

// Minimal typing for the (prefixed) Web Speech API; lib.dom does not declare it everywhere.
interface SpeechRecognitionLike {
  lang: string; interimResults: boolean; continuous: boolean;
  onresult: ((ev: { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null;
  onend: (() => void) | null; onerror: ((ev: { error?: string }) => void) | null;
  start(): void; stop(): void;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function speechCtor(): SpeechRecognitionCtor | null {
  const w = window as unknown as { SpeechRecognition?: SpeechRecognitionCtor; webkitSpeechRecognition?: SpeechRecognitionCtor };
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

export function mountVoice(root: HTMLElement): void {
  root.appendChild(el("h3", { text: "Voice / natural-language patch" }));
  const text = el("input", { type: "text", "data-testid": "voice-text", placeholder: "e.g. put a dashed circle around the galaxy at upper left" });
  const mic = el("button", { "data-testid": "voice-mic", text: "🎤", title: "dictate (Web Speech API)" });
  const send = el("button", { class: "primary", "data-testid": "voice-send", text: "Send" });
  const hint = el("div", { class: "hint", "data-testid": "voice-hint" });
  root.append(el("div", { class: "voice-bar" }, mic, text, send), hint);
  const clarification = el("div", { class: "notes", "data-testid": "voice-clarification" });
  const ops = el("div", { class: "ops", "data-testid": "voice-ops" });
  const applyAll = el("button", { class: "primary", "data-testid": "apply-all", text: "Apply all", disabled: true });
  const discard = el("button", { "data-testid": "discard-ops", text: "Discard", disabled: true });
  root.append(clarification, ops, el("div", { class: "actions" }, applyAll, discard));

  // ---- speech
  const Ctor = speechCtor();
  let rec: SpeechRecognitionLike | null = null;
  let listening = false;
  if (!Ctor) {
    mic.disabled = true;
    mic.title = "Web Speech API not available in this browser — type the instruction instead (Chrome/Safari have it)";
    hint.textContent = "No speech recognition in this browser; type the instruction.";
  } else {
    hint.textContent = "Click the mic and speak, or type; Enter sends.";
  }
  mic.addEventListener("click", () => {
    if (!Ctor) return;
    if (listening && rec) { rec.stop(); return; }
    rec = new Ctor();
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.continuous = false;
    rec.onresult = (ev) => {
      let s = "";
      for (let i = 0; i < ev.results.length; i++) s += ev.results[i][0].transcript;
      text.value = s.trim();
    };
    rec.onend = () => { listening = false; mic.classList.remove("listening"); };
    rec.onerror = (ev) => { listening = false; mic.classList.remove("listening"); toast(`speech: ${ev.error || "error"}`, "error"); };
    try { rec.start(); listening = true; mic.classList.add("listening"); } catch (e) { toast(`speech: ${(e as Error).message}`, "error"); }
  });

  // ---- send transcript -> ops
  async function sendTranscript(): Promise<void> {
    const id = state.id;
    const transcript = text.value.trim();
    if (!id || !state.file || !transcript) return;
    if (state.dirty) { const ok = await A.save(); if (!ok) return; }
    send.disabled = true;
    setStatus("asking Claude for a patch…");
    try {
      const patch = await api.patch(id, { transcript });
      state.patch = patch;
      emit("voice");
      setStatus(`${patch.ops.length} op(s) proposed — apply or reject each`, "ok");
    } catch (e) {
      const msg = A.describe(e);
      setStatus(`patch failed: ${msg}`, "error");
      toast(`patch failed: ${msg}`, "error", 8000);
    } finally {
      send.disabled = false;
    }
  }
  send.addEventListener("click", () => void sendTranscript());
  text.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); void sendTranscript(); } });

  async function apply(keep: PatchOp[]): Promise<void> {
    const id = state.id;
    if (!id || !state.patch) return;
    if (!keep.length) { state.patch = null; emit("voice"); return; }
    try {
      const file = await api.applyPatch(id, { ops: keep, transcript: state.patch.transcript || text.value });
      const remaining = state.patch.ops.filter((op) => !keep.includes(op));
      state.patch = remaining.length ? { ...state.patch, ops: remaining } : null;
      setFile(file, { exists: true, dirty: false, keepSelection: true });
      state.renderTs = Date.now();
      setStatus(`applied ${keep.length} op(s) (saved)`, "ok");
      toast(`applied ${keep.length} op(s)`, "ok");
      emit("voice");
      void A.refreshImages();
    } catch (e) {
      const msg = A.describe(e);
      setStatus(`apply failed: ${msg}`, "error");
      toast(`apply failed: ${msg}`, "error", 8000);
    }
  }
  applyAll.addEventListener("click", () => void apply(state.patch?.ops || []));
  discard.addEventListener("click", () => { state.patch = null; emit("voice"); });

  function describeOp(op: PatchOp): string {
    const it = (op.item || op.set || {}) as Record<string, unknown>;
    const what = op.op === "add" ? `add ${it.type || "?"}${it.label ? ` "${it.label}"` : ""}${it.kind ? ` (${it.kind})` : ""}`
      : op.op === "update" ? `update ${op.id}: ${Object.keys(op.set || {}).join(", ")}`
      : `delete ${op.id}`;
    return what + (op.confidence != null ? ` · conf ${op.confidence.toFixed(2)}` : "");
  }

  function update(): void {
    clear(ops);
    const p = state.patch;
    clarification.textContent = p?.clarification ? `clarification: ${p.clarification}` : "";
    applyAll.disabled = discard.disabled = !p || !p.ops.length;
    send.disabled = !state.file;
    if (!p) return;
    if (!p.ops.length) { ops.appendChild(el("div", { class: "empty", text: "no ops" })); return; }
    p.ops.forEach((op, i) => {
      const a = el("button", { class: "small", "data-testid": "apply-op", text: "✓ apply" });
      const r = el("button", { class: "small", "data-testid": "reject-op", text: "✗" });
      a.addEventListener("click", () => void apply([op]));
      r.addEventListener("click", () => { if (!state.patch) return; state.patch = { ...state.patch, ops: state.patch.ops.filter((o) => o !== op) }; if (!state.patch.ops.length) state.patch = null; emit("voice"); });
      ops.appendChild(el("div", { class: "op", "data-op": String(i), "data-testid": "voice-op" },
        el("span", { class: "label", text: describeOp(op) }), a, r,
        op.rationale ? el("div", { class: "notes", text: op.rationale }) : null));
    });
  }
  subscribe((ev) => { if (["voice", "file", "boot", "view"].includes(ev.kind)) update(); });
  update();
}
