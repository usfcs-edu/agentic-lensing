/** Items tab: list rows (swatch, type icon, label, status) + the editor for the selected item. */
import * as A from "../actions";
import { clear, el, fillSelect, option } from "../dom";
import { isDeflectorLabel } from "../geometry";
import { getItem, select, state, subscribe } from "../state";
import type { ColorName, Item, MaskKind } from "../types";

const TYPE_ICON: Record<Item["type"], string> = { arrow: "→", mask_circle: "◌", einstein_ring: "◯", text: "T" };

export function mountItems(root: HTMLElement): void {
  const list = el("div", { class: "item-list", id: "item-list" });
  const editor = el("div", { class: "item-editor", id: "item-editor" });
  root.append(el("h3", { text: "Items" }), list, editor);
  let editorFor: string | null = null;

  subscribe((ev) => {
    if (ev.kind === "file" || ev.kind === "selection" || ev.kind === "boot" || ev.kind === "view") {
      renderList(list);
      const sel = getItem(state.selectedId);
      // rebuild the editor on selection change / external file change; own edits only patch the list
      if (!sel) { clear(editor); editorFor = null; }
      else if (sel.id !== editorFor || (ev.kind === "file" && ev.source !== "items-editor")) { renderEditor(editor, sel); editorFor = sel.id; }
    }
    if (ev.kind === "geometry") {
      const sel = getItem(state.selectedId);
      if (sel && sel.id === editorFor) patchGeometryFields(editor, sel);
    }
  });
}

function colorHex(c: string): string {
  return state.style?.palette.colors[c] || "#fff";
}

function itemTitle(it: Item): string {
  switch (it.type) {
    case "arrow": return it.label || "(no label)";
    case "mask_circle": return `${it.kind} mask r=${it.radius_arcsec.toFixed(2)}″${it.label ? ` · ${it.label}` : ""}`;
    case "einstein_ring": return `θ_E ${it.theta_e_arcsec}″${it.label ? ` · ${it.label}` : ""}`;
    case "text": return it.text;
  }
}

function renderList(list: HTMLElement): void {
  clear(list);
  const f = state.file;
  if (!f) { list.appendChild(el("div", { class: "empty", text: "no image loaded" })); return; }
  if (!f.items.length) { list.appendChild(el("div", { class: "empty", text: "no items — press A / G / S / R / T and draw on the image" })); return; }
  for (const it of f.items) {
    const row = el("div", {
      class: `item-row status-${it.status}` + (it.id === state.selectedId ? " active" : ""), "data-testid": "item-row",
      "data-id": it.id, "data-status": it.status, "data-type": it.type, role: "button", tabindex: 0,
    },
      el("span", { class: "swatch", style: `background:${colorHex(it.color)}` }),
      el("span", { class: "type-icon", title: it.type, text: TYPE_ICON[it.type] }),
      el("span", { class: "label", text: itemTitle(it) }),
      el("span", { class: `status-badge s-${it.status}`, text: it.status + (it.review ? ` · ${it.review.verdict}` : "") }),
      el("span", { class: "who", text: it.created_by.kind === "claude" ? "claude" : it.created_by.kind === "voice" ? "voice" : "" }),
    );
    row.addEventListener("click", () => select(it.id));
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") select(it.id); });
    list.appendChild(row);
  }
}

function num(value: number, step: string, testid: string, onChange: (v: number) => void): HTMLInputElement {
  const inp = el("input", { type: "number", step, min: "0.01", value: String(value), "data-testid": testid });
  inp.addEventListener("change", () => { const v = Number(inp.value); if (Number.isFinite(v) && v > 0) onChange(v); });
  return inp;
}

function renderEditor(editor: HTMLElement, it: Item): void {
  clear(editor);
  const pal = state.style?.palette;
  const id = it.id;
  const head = el("div", { class: "editor-head" },
    el("span", { class: "swatch", style: `background:${colorHex(it.color)}` }),
    el("code", { text: id }),
    el("span", { class: `status-badge s-${it.status}`, text: it.status }),
    el("span", { class: "who", text: describeCreator(it) }));
  editor.appendChild(head);
  const form = el("div", { class: "form" });
  editor.appendChild(form);
  const row = (label: string, ...ctl: (HTMLElement | string)[]) => form.appendChild(el("label", { class: "row" }, el("span", { class: "k", text: label }), ...ctl));

  // label (arrow / mask / ring) or text body
  if (it.type === "text") {
    const inp = el("input", { type: "text", value: it.text, "data-testid": "note-input", placeholder: "text" });
    inp.addEventListener("input", () => A.updateItem(id, (x) => { if (x.type === "text") x.text = inp.value; }, "items-editor", `text:${id}`));
    row("text", inp);
  } else {
    const inp = el("input", { type: "text", value: it.label || "", "data-testid": "label-input", placeholder: it.type === "arrow" ? "e.g. tight arc, deflector, counter-image" : "optional label" });
    inp.addEventListener("input", () => A.updateItem(id, (x) => {
      x.label = inp.value;
      // green is reserved for the deflector: promote an auto-coloured arrow when the label says so
      if (x.type === "arrow" && pal && isDeflectorLabel(inp.value) && x.color !== pal.deflector) {
        const otherGreen = state.file!.items.some((o) => o.id !== x.id && o.type === "arrow" && o.color === pal.deflector && o.status !== "rejected");
        if (!otherGreen) { x.color = pal.deflector; colorSel.value = pal.deflector; }
      }
    }, "items-editor", `label:${id}`));
    row("label", inp);
  }

  // colour
  const colorSel = el("select", { "data-testid": "color-select" });
  const locked: ColorName | null = it.type === "mask_circle" ? "mask_red" : it.type === "einstein_ring" ? "ring_white" : null;
  const choices = locked ? [locked] : (pal ? Object.keys(pal.colors).filter((c) => !pal.reserved[c]) : ["white"]);
  fillSelect(colorSel, choices.map((c) => ({ value: c })), it.color);
  colorSel.disabled = !!locked;
  colorSel.addEventListener("change", () => A.updateItem(id, (x) => { x.color = colorSel.value as ColorName; }));
  row("colour", colorSel, el("span", { class: "swatch", style: `background:${colorHex(it.color)}` }));

  // legend
  const legend = el("input", { type: "checkbox", "data-testid": "legend-checkbox" });
  legend.checked = it.show_in_legend;
  legend.addEventListener("change", () => A.updateItem(id, (x) => { x.show_in_legend = legend.checked; }));
  row("in legend", legend);

  // type-specific geometry
  switch (it.type) {
    case "arrow": {
      const anchor = el("select", { "data-testid": "anchor-select" });
      fillSelect(anchor, ["auto", "tail", "head"].map((v) => ({ value: v })), it.label_anchor);
      anchor.addEventListener("change", () => A.updateItem(id, (x) => { if (x.type === "arrow") x.label_anchor = anchor.value as "auto" | "tail" | "head"; }));
      row("label side", anchor);
      row("tail", el("span", { class: "geo", "data-geo": "tail", text: uvText(it.tail) }));
      row("head", el("span", { class: "geo", "data-geo": "head", text: uvText(it.head) }));
      break;
    }
    case "mask_circle": {
      const kind = el("select", { "data-testid": "kind-select" });
      fillSelect(kind, ["galaxy", "star", "artifact"].map((v) => ({ value: v })), it.kind);
      kind.addEventListener("change", () => A.updateItem(id, (x) => { if (x.type === "mask_circle") x.kind = kind.value as MaskKind; }));
      row("kind", kind);
      row("radius ″", num(it.radius_arcsec, "0.01", "radius-input", (v) => A.updateItem(id, (x) => { if (x.type === "mask_circle") x.radius_arcsec = v; })));
      row("centre", el("span", { class: "geo", "data-geo": "center", text: uvText(it.center) }));
      break;
    }
    case "einstein_ring": {
      row("θ_E ″", num(it.theta_e_arcsec, "0.01", "ring-theta-input", (v) => A.updateItem(id, (x) => { if (x.type === "einstein_ring") x.theta_e_arcsec = v; })));
      const refSel = el("select", { "data-testid": "center-ref-select" });
      const arrows = state.file!.items.filter((x) => x.type === "arrow");
      refSel.appendChild(option("", "(fixed centre)", !it.center_ref));
      for (const a of arrows) refSel.appendChild(option(a.id, `${a.id} ${a.label || ""}`, it.center_ref === a.id));
      refSel.addEventListener("change", () => A.updateItem(id, (x) => { if (x.type === "einstein_ring") x.center_ref = refSel.value || null; }));
      row("track arrow", refSel);
      row("centre", el("span", { class: "geo", "data-geo": "center", text: uvText(it.center) }));
      break;
    }
    case "text":
      row("pos", el("span", { class: "geo", "data-geo": "pos", text: uvText(it.pos) }));
      break;
  }

  // notes / review summary
  if (it.notes) form.appendChild(el("div", { class: "notes", text: `notes: ${it.notes}` }));
  if (it.invalid_reason) form.appendChild(el("div", { class: "notes warn", text: `invalid: ${it.invalid_reason}` }));
  if (it.review) form.appendChild(el("div", { class: "notes", text: `review: ${it.review.verdict}${it.review.comment ? ` — ${it.review.comment}` : ""}` }));

  const del = el("button", { class: "danger", "data-testid": "delete-item", text: "Delete item (⌫)" });
  del.addEventListener("click", () => A.deleteItem(id));
  const actions = el("div", { class: "actions" }, del);
  if (it.status === "proposed") {
    const acc = el("button", { text: "Accept (Enter)" }); acc.addEventListener("click", () => A.acceptItem(id));
    const rej = el("button", { text: "Reject (X)" }); rej.addEventListener("click", () => A.rejectItem(id));
    actions.prepend(acc, rej);
  }
  editor.appendChild(actions);
}

function describeCreator(it: Item): string {
  const c = it.created_by;
  if (c.kind === "claude") return `claude ${c.model || ""} ${c.effort || ""}`.trim();
  if (c.kind === "human") return c.reviewer ? `by ${c.reviewer}` : "human";
  return c.kind;
}

function uvText(p: [number, number]): string {
  return `(${p[0].toFixed(3)}, ${p[1].toFixed(3)})`;
}

/** During a drag only the geometry read-outs change; keep the inputs (and focus) intact. */
function patchGeometryFields(editor: HTMLElement, it: Item): void {
  const set = (k: string, v: string) => { const e = editor.querySelector<HTMLElement>(`[data-geo="${k}"]`); if (e) e.textContent = v; };
  switch (it.type) {
    case "arrow": set("tail", uvText(it.tail)); set("head", uvText(it.head)); break;
    case "mask_circle": set("center", uvText(it.center)); { const r = editor.querySelector<HTMLInputElement>('[data-testid="radius-input"]'); if (r && document.activeElement !== r) r.value = String(it.radius_arcsec); } break;
    case "einstein_ring": set("center", uvText(it.center)); { const r = editor.querySelector<HTMLInputElement>('[data-testid="ring-theta-input"]'); if (r && document.activeElement !== r) r.value = String(it.theta_e_arcsec); } break;
    case "text": set("pos", uvText(it.pos)); break;
  }
}
