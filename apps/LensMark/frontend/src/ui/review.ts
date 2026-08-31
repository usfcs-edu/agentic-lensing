/**
 * Review tab (critique). Per proposed item: accept / reject / verdict / comment; accept-all / reject-all;
 * panel scores; submit = PUT the file (statuses + reviews applied) then POST the Critique document.
 */
import * as A from "../actions";
import { api } from "../api";
import { clear, el, fillSelect } from "../dom";
import { deltaArcsec } from "../geometry";
import { getItem, latestRun, mutate, select, setStatus, state, subscribe } from "../state";
import type { Critique, CritiqueItem, CritiquePanel, Item, ThetaEVerdict, Verdict } from "../types";
import { THETA_E_VERDICTS, VERDICTS } from "../types";
import { toast } from "./toast";

function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function mountReview(root: HTMLElement): void {
  root.setAttribute("data-testid", "critique-panel");
  root.appendChild(el("h3", { text: "Review proposal" }));
  const runInfo = el("div", { class: "run-info", "data-testid": "review-run" });
  const bulk = el("div", { class: "actions" });
  const acceptAll = el("button", { "data-testid": "accept-all", text: "Accept all" });
  const rejectAll = el("button", { "data-testid": "reject-all", text: "Reject all" });
  acceptAll.addEventListener("click", () => A.acceptAll());
  rejectAll.addEventListener("click", () => A.rejectAll());
  bulk.append(acceptAll, rejectAll, el("span", { class: "hint", text: "keys: 1–7 verdict · Enter accept · X reject · drag = edit" }));
  const list = el("div", { class: "review-list" });
  root.append(runInfo, bulk, list);

  // ---- panel scores
  root.appendChild(el("h4", { text: "Panel" }));
  const form = el("div", { class: "form" });
  root.appendChild(form);
  const row = (label: string, ...ctl: (HTMLElement | string)[]) => form.appendChild(el("label", { class: "row" }, el("span", { class: "k", text: label }), ...ctl));
  const score = (name: string) => {
    const s = el("select", { "data-testid": `score-${name}` });
    fillSelect(s, [{ value: "", label: "—" }, ...["1", "2", "3", "4", "5"].map((v) => ({ value: v }))], "");
    return s;
  };
  const completeness = score("completeness"), geometric = score("geometric_accuracy"), labels = score("label_quality"), description = score("description_quality");
  row("completeness", completeness);
  row("geometry", geometric);
  row("labels", labels);
  row("description", description);
  const thetaVerdict = el("select", { "data-testid": "theta-e-verdict" });
  fillSelect(thetaVerdict, [{ value: "", label: "—" }, ...THETA_E_VERDICTS.map((v) => ({ value: v }))], "");
  const thetaHuman = el("input", { type: "number", step: "0.01", min: "0", "data-testid": "theta-e-human", placeholder: "human θ_E ″" });
  row("θ_E", thetaVerdict, thetaHuman);
  const fewshot = el("input", { type: "checkbox", "data-testid": "would-use-as-fewshot" });
  row("few-shot worthy", fewshot);
  const freeText = el("textarea", { rows: 3, "data-testid": "free-text", placeholder: "free text" });
  form.appendChild(el("label", { class: "row col" }, el("span", { class: "k", text: "notes" }), freeText));
  const submit = el("button", { class: "primary", "data-testid": "submit-critique", text: "Submit critique" });
  const result = el("div", { class: "result", "data-testid": "critique-result" });
  root.append(el("div", { class: "actions" }, submit), result);

  function renderList(): void {
    clear(list);
    const items = A.reviewSet();
    if (!items.length) { list.appendChild(el("div", { class: "empty", text: "no proposed items — run Propose first" })); return; }
    for (const it of items) list.appendChild(reviewRow(it));
  }

  function reviewRow(it: Item): HTMLElement {
    const color = state.style?.palette.colors[it.color] || "#fff";
    const title = it.type === "arrow" ? it.label || "(no label)" : it.type === "mask_circle" ? `${it.kind} mask ${it.radius_arcsec.toFixed(2)}″` : it.type === "einstein_ring" ? `ring θ_E ${it.theta_e_arcsec}″` : it.text;
    const row = el("div", { class: `review-row status-${it.status}` + (it.id === state.selectedId ? " active" : ""), "data-id": it.id, "data-status": it.status, "data-testid": "review-row" });
    const head = el("div", { class: "rr-head" },
      el("span", { class: "swatch", style: `background:${color}` }),
      el("span", { class: "label", text: `${title}` }),
      el("code", { text: it.id }),
      el("span", { class: `status-badge s-${it.status}`, text: it.status }),
      el("span", { class: "who", text: it.created_by.kind === "human" ? "human-added" : "" }));
    head.addEventListener("click", () => select(it.id));
    const accept = el("button", { class: "small", "data-testid": "accept-item", text: "✓ accept" });
    const reject = el("button", { class: "small", "data-testid": "reject-item", text: "✗ reject" });
    accept.addEventListener("click", () => A.acceptItem(it.id));
    reject.addEventListener("click", () => A.rejectItem(it.id));
    const verdict = el("select", { "data-testid": "verdict-select" });
    fillSelect(verdict, [{ value: "", label: "verdict…" }, ...VERDICTS.map((v, i) => ({ value: v, label: i < 7 ? `${i + 1} ${v}` : v }))], it.review?.verdict || "");
    verdict.addEventListener("change", () => { if (verdict.value) A.applyVerdict(it.id, verdict.value as Verdict); });
    const severity = el("select", { "data-testid": "severity-select" });
    fillSelect(severity, [{ value: "", label: "severity" }, { value: "minor" }, { value: "major" }], it.review?.severity || "");
    severity.addEventListener("change", () => A.updateItem(it.id, (x) => { if (x.review) x.review.severity = (severity.value || null) as "minor" | "major" | null; }, "review"));
    const comment = el("input", { type: "text", "data-testid": "review-comment", placeholder: "comment", value: it.review?.comment || "" });
    comment.addEventListener("change", () => A.updateItem(it.id, (x) => { x.review = { ...(x.review || { verdict: defaultVerdict(x) }), comment: comment.value }; }, "review"));
    const extra = [];
    if (it.notes) extra.push(el("div", { class: "notes", text: `rationale: ${it.notes}` }));
    if (it.edit_of) extra.push(el("div", { class: "notes", text: `edited: Δ ${deltaArcsec(it, state.file!.image.width, state.file!.image.height, state.file!.image.cutout_arcsec)?.toFixed(2) ?? "?"}″ from the proposal` }));
    if (it.invalid_reason) extra.push(el("div", { class: "notes warn", text: `invalid: ${it.invalid_reason}` }));
    row.append(head, el("div", { class: "rr-controls" }, accept, reject, verdict, severity), comment, ...extra);
    return row;
  }

  function defaultVerdict(it: Item): Verdict {
    if (it.created_by.kind === "human") return "missed_by_model";
    if (it.status === "rejected") return "spurious";
    if (it.status === "edited") return "wrong_position";
    return "correct";
  }

  submit.addEventListener("click", async () => {
    const f = state.file;
    const id = state.id;
    if (!f || !id) return;
    const run = latestRun();
    if (!run) { toast("no proposal run to critique", "error"); return; }
    const items = A.reviewSet();
    const W = f.image.width, H = f.image.height, cut = f.image.cutout_arcsec;
    // 1. apply defaults into the file: every reviewed item gets a review record; still-proposed items stay proposed
    const critItems: CritiqueItem[] = [];
    const counts: Record<string, number> = { proposed: 0, accepted: 0, edited: 0, rejected: 0, invalid: 0, added_by_human: 0 };
    mutate(() => {                                       // one undo step for the whole submit
      for (const it of items) {
        const live = getItem(it.id)!;
        const verdict = live.review?.verdict || defaultVerdict(live);
        const delta = live.edit_of ? deltaArcsec(live, W, H, cut) : null;
        live.review = { ...(live.review || {}), verdict, reviewer: state.reviewer || null, reviewed_at: nowIso(), comment: live.review?.comment || "", ...(delta != null ? { delta_arcsec: +delta.toFixed(3) } : {}) };
        counts[live.status] = (counts[live.status] || 0) + 1;
        if (live.created_by.kind === "human") counts.added_by_human += 1;
        critItems.push({ item_id: live.id, verdict, severity: live.review.severity ?? null, comment: live.review.comment || "", delta_arcsec: live.review.delta_arcsec ?? null });
      }
    }, { source: "review" });
    const panel: CritiquePanel = {
      completeness: completeness.value ? Number(completeness.value) : null,
      geometric_accuracy: geometric.value ? Number(geometric.value) : null,
      label_quality: labels.value ? Number(labels.value) : null,
      description_quality: description.value ? Number(description.value) : null,
      theta_e_verdict: (thetaVerdict.value || null) as ThetaEVerdict | null,
      theta_e_human_arcsec: thetaHuman.value ? Number(thetaHuman.value) : null,
      free_text: freeText.value,
      would_use_as_fewshot: fewshot.checked,
    };
    const doc: Critique = {
      schema_version: "lensmark-critique/1.0", image_id: id, run_id: run.run_id, model: run.model, effort: run.effort ?? null,
      reviewer: state.reviewer || "reviewer", reviewed_at: nowIso(), lead_time_s: +((Date.now() - state.reviewStartedAt) / 1000).toFixed(1),
      items: critItems, panel, counts: {}, // backend fills authoritative counts (critique.run_counts)
    };
    submit.disabled = true;
    try {
      const ok = await A.save();                       // 2. PUT the file with statuses/reviews applied
      if (!ok) return;
      const res = await api.critique(id, doc);         // 3. POST the critique document
      result.textContent = `critique written: ${res.file}`;
      setStatus(`critique saved (${critItems.length} items, lead ${doc.lead_time_s}s)`, "ok");
      toast("critique submitted", "ok");
      await A.load(id);
      void A.refreshImages();
    } catch (e) {
      const msg = A.describe(e);
      result.textContent = `critique failed: ${msg}`;
      setStatus(`critique failed: ${msg}`, "error");
      toast(`critique failed: ${msg}`, "error", 8000);
    } finally {
      submit.disabled = false;
    }
  });

  function update(): void {
    const run = latestRun();
    runInfo.textContent = run
      ? `run ${run.run_id} · ${run.model}${run.effort ? ` · ${run.effort}` : ""} · ${run.n_items_proposed ?? 0} proposed${run.cost_usd != null ? ` · $${run.cost_usd.toFixed(4)}` : ""} · reviewer ${state.reviewer || "?"}`
      : "no proposal run for this image yet";
    const nProposed = A.proposedItems().length;
    acceptAll.disabled = rejectAll.disabled = nProposed === 0;
    submit.disabled = !run || !state.file;
    renderList();
    const f = state.file;
    if (f && document.activeElement !== thetaHuman && !thetaHuman.value && f.system.theta_e.value_arcsec) thetaHuman.placeholder = `human θ_E ″ (system: ${f.system.theta_e.value_arcsec})`;
  }
  subscribe((ev) => { if (["file", "selection", "boot", "view", "propose"].includes(ev.kind)) update(); });
  update();
}
