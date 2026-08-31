/** Transient notifications (bottom-right). Errors from every fetch end up here and in the status line. */
import { el } from "../dom";

export type ToastKind = "info" | "ok" | "error";

export function toast(msg: string, kind: ToastKind = "info", ms = 3500): void {
  const host = document.getElementById("toasts");
  if (!host) return;
  const node = el("div", { class: `toast ${kind}`, role: "status", text: msg });
  host.appendChild(node);
  window.setTimeout(() => { node.classList.add("fade"); window.setTimeout(() => node.remove(), 400); }, ms);
}
