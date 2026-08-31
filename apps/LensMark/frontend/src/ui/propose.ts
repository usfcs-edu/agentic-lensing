/** Propose tab: model x effort x budget -> POST /api/propose -> SSE log -> reload file (ghost items). */
import * as A from "../actions";
import { api, openProposeEvents } from "../api";
import { el, fillSelect } from "../dom";
import { emit, setStatus, setTab, state, subscribe } from "../state";
import type { SseEvent } from "../types";
import { toast } from "./toast";

export function mountPropose(root: HTMLElement): void {
  root.appendChild(el("h3", { text: "Propose (Claude)" }));
  const form = el("div", { class: "form" });
  root.appendChild(form);
  const row = (label: string, ...ctl: (HTMLElement | string)[]) => form.appendChild(el("label", { class: "row" }, el("span", { class: "k", text: label }), ...ctl));

  const model = el("select", { "data-testid": "model-select" });
  const effort = el("select", { "data-testid": "effort-select" });
  const budget = el("input", { type: "number", step: "0.05", min: "0.01", value: "0.50", "data-testid": "budget-input", title: "max USD for this call" });
  const fewshot = el("input", { type: "text", "data-testid": "fewshot-input", placeholder: "exports/fewshot (optional)" });
  const propose = el("button", { class: "primary", "data-testid": "propose", text: "Propose" });
  const cancel = el("button", { "data-testid": "propose-cancel", text: "Cancel", disabled: true });
  const log = el("pre", { class: "log", "data-testid": "propose-log", "aria-live": "polite" });
  const cost = el("span", { class: "cost", "data-testid": "propose-cost" });

  row("model", model);
  row("effort", effort);
  row("budget $", budget);
  row("few-shot", fewshot);
  form.appendChild(el("div", { class: "actions" }, propose, cancel, cost));
  root.appendChild(log);

  let closeEvents: (() => void) | null = null;

  function populate(): void {
    const ms = state.models;
    if (!ms) return;
    const want = state.config?.default_model || ms.default.model;
    const defModel = ms.models.find((m) => m.alias === want || m.id === want)?.alias || ms.models[0]?.alias || "";
    const defEffort = state.config?.default_effort || ms.default.effort;
    fillSelect(model, ms.models.map((m) => ({ value: m.alias, label: `${m.label} (${m.id})` })), model.value || defModel);
    fillSelect(effort, ms.efforts.map((e) => ({ value: e })), effort.value || defEffort);
    syncEffort();
  }
  function syncEffort(): void {
    const info = state.models?.models.find((m) => m.alias === model.value || m.id === model.value);
    effort.disabled = !!info && !info.supports_effort;
    effort.title = effort.disabled ? `${info?.label} does not take an effort level` : "";
  }
  model.addEventListener("change", syncEffort);

  function append(line: string): void {
    if (!state.propose) return;
    state.propose.log.push(line);
    log.textContent = state.propose.log.join("\n");
    log.scrollTop = log.scrollHeight;
  }

  propose.addEventListener("click", async () => {
    const id = state.id;
    if (!id || !state.file) return;
    if (state.dirty) {
      const ok = await A.save();
      if (!ok) { toast("save before proposing", "error"); return; }
    }
    const body = { model: model.value, effort: effort.disabled ? null : effort.value, budget: Number(budget.value) || undefined, ...(fewshot.value.trim() ? { fewshot: fewshot.value.trim() } : {}) };
    try {
      const { run_id } = await api.propose(id, body);
      state.propose = { runId: run_id, running: true, log: [`run ${run_id} · ${body.model}${body.effort ? ` · ${body.effort}` : ""} · budget $${body.budget ?? "default"}`], startedAt: Date.now() };
      emit("propose");
      setStatus(`proposing ${id} (${run_id})…`);
      closeEvents = openProposeEvents(id, run_id, (ev) => onEvent(id, run_id, ev), (msg) => { append(`[stream] ${msg}`); finish(id, false, msg); });
    } catch (e) {
      const msg = A.describe(e);
      setStatus(`propose failed: ${msg}`, "error");
      toast(`propose failed: ${msg}`, "error", 8000);
    }
  });

  cancel.addEventListener("click", async () => {
    const p = state.propose;
    if (!p || !state.id) return;
    try { await api.cancelPropose(state.id, p.runId); append("[cancel] requested"); }
    catch (e) { append(`[cancel] ${A.describe(e)}`); }
  });

  function onEvent(id: string, runId: string, ev: SseEvent): void {
    const p = state.propose;
    if (!p || p.runId !== runId) return;
    const parts = [`[${ev.phase}]`];
    if (ev.detail) parts.push(ev.detail);
    if (ev.text) parts.push(ev.text.trim());
    if (ev.n_items != null) parts.push(`n_items=${ev.n_items}`);
    if (ev.cost_usd != null) { p.cost = ev.cost_usd; parts.push(`$${ev.cost_usd.toFixed(4)}`); }
    append(parts.join(" "));
    if (ev.phase === "done") {
      const run = ev.run;
      const c = run?.cost_usd ?? ev.cost_usd ?? p.cost;
      append(`done · ${run?.n_items_proposed ?? ev.n_items ?? "?"} items · ${run?.n_invalid ?? 0} invalid · ${run?.n_repaired ?? 0} repaired · cost $${c != null ? c.toFixed(4) : "?"} · ${((Date.now() - p.startedAt) / 1000).toFixed(0)} s`);
      if (c != null) p.cost = c;
      finish(id, true);
    } else if (ev.phase === "error") {
      finish(id, false, ev.detail || "error");
    } else {
      emit("propose");
    }
  }

  async function finish(id: string, ok: boolean, err?: string): Promise<void> {
    const p = state.propose;
    if (closeEvents) { closeEvents(); closeEvents = null; }
    if (p) { p.running = false; p.error = ok ? null : err || "error"; }
    emit("propose");
    if (!ok) { setStatus(`propose ${id}: ${err}`, "error"); toast(`propose failed: ${err}`, "error", 8000); }
    // reload the file: the proposed items are now merged server-side with status "proposed"
    if (state.id === id) {
      await A.load(id);
      state.reviewStartedAt = Date.now();
      void A.refreshImages();
      if (ok) {
        const n = state.file?.items.filter((it) => it.status === "proposed").length || 0;
        setStatus(`proposal ready: ${n} proposed items — review them (keys 1–7, Enter, X)`, "ok");
        toast(`${n} proposed items`, "ok");
        if (n) setTab("review");
      }
    }
  }

  function update(): void {
    const p = state.propose;
    const running = !!p?.running;
    propose.disabled = !state.file || running;
    cancel.disabled = !running;
    model.disabled = running;
    effort.disabled = running || effort.disabled;
    if (!running) syncEffort();
    log.textContent = p ? p.log.join("\n") : "";
    cost.textContent = p?.cost != null ? `$${p.cost.toFixed(4)}` : "";
    root.classList.toggle("running", running);
  }
  subscribe((ev) => {
    if (ev.kind === "boot") populate();
    if (["boot", "propose", "file", "view"].includes(ev.kind)) update();
  });
  populate();
  update();
}
