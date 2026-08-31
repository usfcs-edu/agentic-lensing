/** Tiny DOM helpers (no framework). */
export const SVG_NS = "http://www.w3.org/2000/svg";

type Attrs = Record<string, string | number | boolean | null | undefined>;
type Child = Node | string | null | undefined | false;

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  setAttrs(e, attrs);
  append(e, children);
  return e;
}

export function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: Child[]): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  setAttrs(e, attrs);
  append(e, children);
  return e;
}

export function setAttrs(e: Element, attrs: Attrs): void {
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.setAttribute("class", String(v));
    else if (k === "text") e.textContent = String(v);
    else if (v === true) e.setAttribute(k, "");
    else e.setAttribute(k, String(v));
  }
}

export function append(parent: Node, children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    parent.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
}

export function clear(e: Element): void {
  while (e.firstChild) e.removeChild(e.firstChild);
}

export function qs<T extends Element = HTMLElement>(sel: string, root: ParentNode = document): T {
  const e = root.querySelector<T>(sel);
  if (!e) throw new Error(`missing element ${sel}`);
  return e;
}

export function option(value: string, label: string = value, selected = false): HTMLOptionElement {
  const o = el("option", { value, text: label });
  o.selected = selected;
  return o;
}

export function fillSelect(sel: HTMLSelectElement, values: { value: string; label?: string }[], current?: string | null): void {
  clear(sel);
  for (const v of values) sel.appendChild(option(v.value, v.label ?? v.value, v.value === (current ?? "")));
  if (current != null) sel.value = current;
}

export function isEditable(t: EventTarget | null): boolean {
  const e = t as HTMLElement | null;
  if (!e || !e.tagName) return false;
  const tag = e.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.isContentEditable;
}

export function fmtNum(x: number | null | undefined, digits = 2): string {
  return x == null || !Number.isFinite(x) ? "—" : x.toFixed(digits);
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return iso;
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
