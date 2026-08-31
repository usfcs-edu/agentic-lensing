/** Export tab: COCO / DS9 / masks / few-shot -> POST /api/export/{fmt}; lists the files written. */
import * as A from "../actions";
import { api } from "../api";
import { clear, el } from "../dom";
import { setStatus, state, subscribe } from "../state";
import { toast } from "./toast";

type Fmt = "coco" | "ds9" | "masks" | "fewshot";

export function mountExport(root: HTMLElement): void {
  root.appendChild(el("h3", { text: "Export" }));
  const onlyCurrent = el("input", { type: "checkbox", "data-testid": "export-current-only" });
  const k = el("input", { type: "number", min: "1", value: "6", "data-testid": "export-k", size: 3, title: "few-shot: number of examples" });
  const requireFlag = el("input", { type: "checkbox", "data-testid": "export-require-flag", title: "few-shot: only panels flagged would_use_as_fewshot" });
  const form = el("div", { class: "form" },
    el("label", { class: "row" }, el("span", { class: "k", text: "scope" }), onlyCurrent, "current image only"),
    el("label", { class: "row" }, el("span", { class: "k", text: "few-shot k" }), k, requireFlag, "require flag"));
  const buttons = el("div", { class: "actions wrap" });
  const files = el("ul", { class: "files", "data-testid": "export-files" });
  root.append(form, buttons, el("h4", { text: "Files" }), files);

  const labels: Record<Fmt, string> = { coco: "COCO", ds9: "DS9 .reg", masks: "Mask PNGs", fewshot: "Few-shot bundle" };
  const btns: HTMLButtonElement[] = [];
  for (const fmt of ["coco", "ds9", "masks", "fewshot"] as Fmt[]) {
    const b = el("button", { "data-testid": `export-${fmt}`, text: labels[fmt] });
    b.addEventListener("click", () => void run(fmt));
    btns.push(b);
    buttons.appendChild(b);
  }

  async function run(fmt: Fmt): Promise<void> {
    const body: { ids?: string[]; k?: number; require_flag?: boolean } = {};
    if (onlyCurrent.checked && state.id) body.ids = [state.id];
    if (fmt === "fewshot") { body.k = Number(k.value) || 6; body.require_flag = requireFlag.checked; }
    for (const b of btns) b.disabled = true;
    setStatus(`exporting ${fmt}…`);
    try {
      const res = await api.exportFmt(fmt, body);
      clear(files);
      if (!res.files?.length) files.appendChild(el("li", { class: "empty", text: "(no files)" }));
      for (const f of res.files || []) files.appendChild(el("li", { text: f }));
      setStatus(`${fmt}: ${res.files?.length ?? 0} file(s)`, "ok");
      toast(`${labels[fmt]}: ${res.files?.length ?? 0} file(s)`, "ok");
    } catch (e) {
      const msg = A.describe(e);
      setStatus(`export ${fmt} failed: ${msg}`, "error");
      toast(`export failed: ${msg}`, "error", 8000);
    } finally {
      for (const b of btns) b.disabled = false;
    }
  }
  subscribe((ev) => { if (ev.kind === "boot" || ev.kind === "view") for (const b of btns) b.disabled = !state.images.length; });
}
