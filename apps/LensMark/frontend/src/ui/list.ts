/** Left column: one row per image (thumbnail, id, rank, badges). Click -> #id. */
import { urls } from "../api";
import { clear, el } from "../dom";
import { state, subscribe } from "../state";
import type { ImageSummary } from "../types";

export function mountList(root: HTMLElement): void {
  // full rebuild only when the list itself changes (thumbnails would otherwise refetch on every edit)
  subscribe((ev) => {
    if (ev.kind === "images" || ev.kind === "boot") render(root);
    else if (ev.kind === "view" || ev.kind === "dirty") patch(root);
  });
  render(root);
}

/** Cheap update: active row highlight + the unsaved badge on the current image. */
function patch(root: HTMLElement): void {
  for (const row of root.querySelectorAll<HTMLElement>(".image-row")) {
    const active = row.dataset.id === state.id;
    row.classList.toggle("active", active);
    const badges = row.querySelector(".badges");
    const existing = badges?.querySelector(".badge.dirty");
    if (active && state.dirty && !existing) badges?.appendChild(el("span", { class: "badge dirty", text: "unsaved" }));
    else if ((!active || !state.dirty) && existing) existing.remove();
  }
}

function rowStatus(im: ImageSummary): string {
  const proposed = im.by_status?.proposed || 0;
  if (proposed > 0) return "proposed";
  if (!im.has_json) return "new";
  return im.n_items > 0 ? "annotated" : "empty";
}

function render(root: HTMLElement): void {
  clear(root);
  if (!state.images.length) { root.appendChild(el("div", { class: "empty", text: "no images" })); return; }
  for (const im of state.images) {
    const proposed = im.by_status?.proposed || 0;
    const badges = el("div", { class: "badges" });
    badges.appendChild(el("span", { class: "badge", title: "items", text: String(im.n_items) }));
    if (im.annot_stale && im.has_json) badges.appendChild(el("span", { class: "badge warn", text: "stale" }));
    if (proposed) badges.appendChild(el("span", { class: "badge ghost", text: `proposed: ${proposed}` }));
    if (im.grade) badges.appendChild(el("span", { class: `badge grade grade-${im.grade}`, text: im.grade }));
    if (im.id === state.id && state.dirty) badges.appendChild(el("span", { class: "badge dirty", text: "unsaved" }));
    const row = el("div", {
      class: "image-row" + (im.id === state.id ? " active" : ""), "data-testid": "image-row", "data-id": im.id,
      "data-status": rowStatus(im), role: "button", tabindex: 0, title: `${im.file} ${im.width}x${im.height}`,
    },
      el("img", { class: "thumb", src: urls.thumb(im.id, 96), alt: im.id, loading: "lazy", width: 48, height: 48 }),
      el("div", { class: "meta" },
        el("div", { class: "id", text: im.id }),
        el("div", { class: "sub", text: [im.rank != null ? `rank ${im.rank}` : "", im.theta_e_arcsec != null ? `θ_E ${im.theta_e_arcsec}″` : ""].filter(Boolean).join(" · ") }),
        badges),
    );
    row.addEventListener("click", () => { location.hash = encodeURIComponent(im.id); });
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") location.hash = encodeURIComponent(im.id); });
    root.appendChild(row);
  }
}
