/** System tab: description, theta_E, grade, verdict, tags, rank, legend settings, image facts. */
import * as A from "../actions";
import { clear, el, fillSelect } from "../dom";
import { mutate, state, subscribe } from "../state";
import { LEGEND_POSITIONS, SYSTEM_VERDICTS, type Grade, type LegendPosition, type SystemVerdict } from "../types";

export function mountSystem(root: HTMLElement): void {
  root.appendChild(el("h3", { text: "System" }));
  const form = el("div", { class: "form" });
  root.appendChild(form);
  const row = (label: string, ...ctl: (HTMLElement | string)[]) => form.appendChild(el("label", { class: "row" }, el("span", { class: "k", text: label }), ...ctl));

  const desc = el("textarea", { "data-testid": "description", rows: 6, placeholder: "Describe the system; refer to arrows by colour (\"the cyan arrow marks a tight arc…\")" });
  desc.addEventListener("input", () => mutate((f) => { f.system.description = desc.value; }, { source: "system", coalesce: "description" }));
  form.appendChild(el("label", { class: "row col" }, el("span", { class: "k", text: "description" }), desc));

  const theta = el("input", { type: "number", step: "0.01", min: "0", "data-testid": "theta-e-input", placeholder: "arcsec" });
  theta.addEventListener("change", () => mutate((f) => { const v = Number(theta.value); f.system.theta_e.value_arcsec = theta.value === "" || !Number.isFinite(v) ? null : v; if (v > 0 && !f.system.theta_e.method) f.system.theta_e.method = "human"; }, { source: "system" }));
  const method = el("input", { type: "text", "data-testid": "theta-method", placeholder: "method", size: 9 });
  method.addEventListener("change", () => mutate((f) => { f.system.theta_e.method = method.value || null; }, { source: "system" }));
  const alt = el("input", { type: "number", step: "0.01", min: "0", "data-testid": "theta-alt", placeholder: "alt", size: 5 });
  alt.addEventListener("change", () => mutate((f) => { const v = Number(alt.value); f.system.theta_e.alt_arcsec = alt.value === "" || !Number.isFinite(v) ? null : v; }, { source: "system" }));
  row("θ_E ″", theta, method, alt);

  const grade = el("select", { "data-testid": "grade-select" });
  fillSelect(grade, [{ value: "", label: "—" }, ...["A", "B", "C", "D"].map((g) => ({ value: g }))], "");
  grade.addEventListener("change", () => mutate((f) => { f.system.grade = (grade.value || null) as Grade | null; }, { source: "system" }));
  const score = el("select", { "data-testid": "score-select", title: "golden-campaign 1-4 score" });
  fillSelect(score, [{ value: "", label: "1–4: —" }, ...["1", "2", "3", "4"].map((g) => ({ value: g }))], "");
  score.addEventListener("change", () => mutate((f) => { f.system.score_1_4 = score.value ? Number(score.value) : null; }, { source: "system" }));
  const conf = el("select", { "data-testid": "confidence-select", title: "confidence L/M/H" });
  fillSelect(conf, [{ value: "", label: "L/M/H: —" }, ...["L", "M", "H"].map((g) => ({ value: g }))], "");
  conf.addEventListener("change", () => mutate((f) => { f.system.confidence_lmh = (conf.value || null) as "L" | "M" | "H" | null; }, { source: "system" }));
  row("grade", grade, score, conf);

  const verdict = el("select", { "data-testid": "system-verdict" });
  fillSelect(verdict, [{ value: "", label: "—" }, ...SYSTEM_VERDICTS.map((v) => ({ value: v }))], "");
  verdict.addEventListener("change", () => mutate((f) => { f.system.verdict = (verdict.value || null) as SystemVerdict | null; }, { source: "system" }));
  row("verdict", verdict);

  const tags = el("input", { type: "text", "data-testid": "tags-input", placeholder: "comma-separated" });
  tags.addEventListener("change", () => mutate((f) => { f.system.tags = tags.value.split(",").map((t) => t.trim()).filter(Boolean); }, { source: "system" }));
  row("tags", tags);

  const objectId = el("input", { type: "text", "data-testid": "object-id", placeholder: "object id" });
  objectId.addEventListener("change", () => mutate((f) => { f.system.object_id = objectId.value || null; }, { source: "system" }));
  const rank = el("input", { type: "text", "data-testid": "rank", readonly: true, size: 6 });
  row("object / rank", objectId, rank);

  const legendShow = el("input", { type: "checkbox", "data-testid": "legend-show" });
  legendShow.addEventListener("change", () => mutate((f) => { f.legend.show = legendShow.checked; }, { source: "system" }));
  const legendPos = el("select", { "data-testid": "legend-position" });
  fillSelect(legendPos, LEGEND_POSITIONS.map((v) => ({ value: v })), "auto");
  legendPos.addEventListener("change", () => A.setLegendPosition(legendPos.value as LegendPosition));
  row("legend", legendShow, legendPos, el("span", { class: "hint", text: "(L cycles)" }));

  const facts = el("dl", { class: "facts", "data-testid": "image-facts" });
  root.append(el("h4", { text: "Image" }), facts);
  const runs = el("div", { class: "runs" });
  root.append(el("h4", { text: "Proposal runs" }), runs);

  function update(source?: string): void {
    const f = state.file;
    const dis = !f;
    for (const c of [desc, theta, method, alt, grade, score, conf, verdict, tags, objectId, legendShow, legendPos]) (c as HTMLInputElement).disabled = dis;
    if (!f) { clear(facts); clear(runs); return; }
    if (source === "system") return;                       // our own edit: inputs already hold the value
    const setIf = (inp: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement, v: string) => { if (document.activeElement !== inp && inp.value !== v) inp.value = v; };
    setIf(desc, f.system.description || "");
    setIf(theta, f.system.theta_e.value_arcsec != null ? String(f.system.theta_e.value_arcsec) : "");
    setIf(method, f.system.theta_e.method || "");
    setIf(alt, f.system.theta_e.alt_arcsec != null ? String(f.system.theta_e.alt_arcsec) : "");
    setIf(grade, f.system.grade || "");
    setIf(score, f.system.score_1_4 != null ? String(f.system.score_1_4) : "");
    setIf(conf, f.system.confidence_lmh || "");
    setIf(verdict, f.system.verdict || "");
    setIf(tags, (f.system.tags || []).join(", "));
    setIf(objectId, f.system.object_id || "");
    rank.value = f.system.rank != null ? String(f.system.rank) : "";
    legendShow.checked = f.legend.show;
    setIf(legendPos, f.legend.position);

    clear(facts);
    const im = f.image;
    const dd = (k: string, v: string | HTMLElement) => facts.append(el("dt", { text: k }), el("dd", {}, v));
    dd("file", im.file);
    dd("size", `${im.width} × ${im.height} px`);
    dd("cutout", el("span", {}, `${im.cutout_arcsec}″ `, el("span", { class: `badge ${im.scale_source === "assumed" ? "warn" : ""}`, "data-testid": "scale-source", text: im.scale_source || "config" })));
    dd("pixel scale", `${im.pixel_scale_arcsec.toFixed(5)} ″/px`);
    dd("orientation", `${im.north_up ? "N up" : "N down"}, ${im.east_left ? "E left" : "E right"}, origin ${im.array_origin}`);
    if (im.survey) dd("survey", im.survey);
    if (im.wcs) dd("wcs", `${im.wcs.ra_deg.toFixed(5)}, ${im.wcs.dec_deg.toFixed(5)}`);
    dd("sha256", im.sha256.slice(0, 12) + "…");
    if (f.render) dd("render", `${f.render.output} (${f.render.rendered_at || "?"})`);
    dd("modified", f.modified);

    clear(runs);
    if (!f.provenance.proposal_runs.length) runs.appendChild(el("div", { class: "empty", text: "none yet" }));
    for (const r of f.provenance.proposal_runs) {
      runs.appendChild(el("div", { class: "run", "data-run": r.run_id },
        el("code", { text: r.run_id }), ` ${r.model}${r.effort ? ` · ${r.effort}` : ""} · ${r.n_items_proposed ?? 0} items` +
        `${r.n_invalid ? ` · ${r.n_invalid} invalid` : ""}${r.n_repaired ? ` · ${r.n_repaired} repaired` : ""}` +
        `${r.cost_usd != null ? ` · $${r.cost_usd.toFixed(4)}` : ""}${r.error ? ` · ERROR ${r.error}` : ""}`));
      if (r.proposed_system && typeof r.proposed_system === "object") {
        const ps = r.proposed_system as Record<string, unknown>;
        const desc2 = typeof ps.description === "string" ? ps.description : "";
        if (desc2) {
          const use = el("button", { class: "small", text: "use description" });
          use.addEventListener("click", () => { mutate((ff) => { ff.system.description = desc2; }, { source: "system-run" }); });
          runs.appendChild(el("div", { class: "proposed-system" }, el("span", { text: `model says: ${ps.verdict || ""} — ${desc2}` }), use));
        }
      }
    }
  }
  subscribe((ev) => { if (["file", "boot", "view"].includes(ev.kind)) update(ev.source); });
  update();
}
