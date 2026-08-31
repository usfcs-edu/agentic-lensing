/** Centre column chrome: tool buttons, zoom, Rendered toggle, save + dirty, status line + lint. */
import * as A from "../actions";
import { urls } from "../api";
import { clear, el } from "../dom";
import { emit, setTool, state, subscribe, undo, undoDepth } from "../state";
import { fitToView, zoomAt, zoomTo } from "../svg/tools";
import { TOOLS, type ToolName } from "../types";

const TOOL_LABELS: Record<ToolName, [string, string]> = {
  select: ["Select", "V"], arrow: ["Arrow", "A"], galaxy: ["Galaxy mask", "G"], star: ["Star mask", "S"], ring: ["Ring", "R"], text: ["Text", "T"],
};

export function mountToolbar(root: HTMLElement, statusRoot: HTMLElement): void {
  const toolBtns = new Map<ToolName, HTMLButtonElement>();
  const tools = el("div", { class: "group tools" });
  for (const t of TOOLS) {
    const [label, key] = TOOL_LABELS[t];
    const b = el("button", { class: "tool", "data-testid": `tool-${t}`, "data-tool": t, title: `${label} (${key})` }, el("span", { class: "key", text: key }), label);
    b.addEventListener("click", () => setTool(t));
    toolBtns.set(t, b);
    tools.appendChild(b);
  }

  const zoomOut = el("button", { title: "zoom out", text: "−" });
  const zoomIn = el("button", { title: "zoom in", text: "+" });
  const zoomFit = el("button", { title: "fit (double-click the stage)", text: "fit" });
  const zoom1 = el("button", { title: "100 %", text: "1:1" });
  zoomOut.addEventListener("click", () => zoomAt(1 / 1.25));
  zoomIn.addEventListener("click", () => zoomAt(1.25));
  zoomFit.addEventListener("click", () => fitToView());
  zoom1.addEventListener("click", () => zoomTo(1));

  const renderToggle = el("input", { type: "checkbox", "data-testid": "render-toggle", id: "render-toggle" });
  renderToggle.addEventListener("change", () => A.setRendered(renderToggle.checked));
  const showRejected = el("input", { type: "checkbox", id: "show-rejected", "data-testid": "show-rejected" });
  showRejected.addEventListener("change", () => { state.showRejected = showRejected.checked; emit("view"); });

  const undoBtn = el("button", { title: "undo (Cmd/Ctrl+Z)", text: "undo" });
  undoBtn.addEventListener("click", () => undo());
  const saveBtn = el("button", { class: "primary", "data-testid": "save", title: "save (Cmd/Ctrl+S)", text: "Save" });
  saveBtn.addEventListener("click", () => void A.save());
  const dirty = el("span", { class: "dirty", "data-testid": "dirty", "data-dirty": "0", text: "saved" });
  const title = el("span", { class: "image-title", id: "image-title" });

  root.append(
    tools,
    el("div", { class: "group" }, zoomOut, zoomIn, zoomFit, zoom1),
    el("div", { class: "group" },
      el("label", { class: "check", title: "show the server-rendered .annot.png instead of the live preview" }, renderToggle, "Rendered"),
      el("label", { class: "check", title: "draw rejected / invalid items too" }, showRejected, "rejected")),
    el("div", { class: "spacer" }, title),
    el("div", { class: "group" }, undoBtn, saveBtn, dirty),
  );

  const status = el("span", { class: "status", id: "status" });
  const lint = el("span", { class: "lint", "data-testid": "lint", id: "lint" });
  statusRoot.append(status, lint);

  const base = document.getElementById("base") as HTMLImageElement;
  const overlay = document.getElementById("overlay") as unknown as SVGSVGElement;
  let baseSrcFor = "";

  function update(): void {
    for (const [t, b] of toolBtns) b.classList.toggle("active", state.tool === t);
    dirty.textContent = state.dirty ? "● unsaved" : state.fileExists ? "saved" : "new";
    dirty.setAttribute("data-dirty", state.dirty ? "1" : "0");
    dirty.classList.toggle("on", state.dirty);
    undoBtn.disabled = undoDepth() === 0;
    saveBtn.disabled = !state.file;
    renderToggle.checked = state.rendered;
    renderToggle.disabled = !state.file;
    showRejected.checked = state.showRejected;
    title.textContent = state.id ? `${state.id}${state.file ? ` · ${state.file.image.width}×${state.file.image.height} · ${state.file.image.cutout_arcsec}″` : ""}` : "";
    status.textContent = state.status;
    status.className = `status ${state.statusKind}`;
    clear(lint);
    for (const w of state.lint) lint.appendChild(el("span", { class: "lint-item", text: w }));
    // stage image: original vs rendered
    if (state.id && state.file) {
      const want = state.rendered ? urls.annot(state.id, state.renderTs) : urls.original(state.id);
      if (want !== baseSrcFor) { baseSrcFor = want; base.src = want; }
      base.width = state.file.image.width;
      base.height = state.file.image.height;
      overlay.style.display = state.rendered ? "none" : "";
      base.classList.toggle("rendered", state.rendered);
    } else if (!state.id) {
      base.removeAttribute("src");
      baseSrcFor = "";
    }
  }
  subscribe(() => update());
  update();
}
