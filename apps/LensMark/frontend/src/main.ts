/**
 * Boot: fetch /api/style, /api/models, /api/config, /api/health, /api/images; mount the panels; route by
 * location.hash (#<image id>); expose window.__lensmark for tests.
 */
import "./style.css";
import * as A from "./actions";
import { api } from "./api";
import { el, qs } from "./dom";
import { installKeys } from "./keys";
import { emit, initReviewer, select, setReviewer, setStatus, setTab, setTool, state, subscribe } from "./state";
import { mountOverlay } from "./svg/overlay";
import { installTools } from "./svg/tools";
import { mountExport } from "./ui/export";
import { mountItems } from "./ui/items";
import { mountList } from "./ui/list";
import { mountPropose } from "./ui/propose";
import { mountReview } from "./ui/review";
import { mountSystem } from "./ui/system";
import { toast } from "./ui/toast";
import { mountToolbar } from "./ui/toolbar";
import { mountVoice } from "./ui/voice";
import type { Item, ToolName } from "./types";

const TABS: [string, string][] = [["items", "Items"], ["system", "System"], ["propose", "Propose"], ["review", "Review"], ["voice", "Voice"], ["export", "Export"]];

declare global {
  interface Window {
    __lensmark: {
      state: typeof state;
      addItem: (json: Partial<Item> & { type: Item["type"] }) => string;
      selectItem: (id: string | null) => void;
      save: () => Promise<boolean>;
      load: (id: string) => Promise<void>;
      setTool: (name: ToolName) => void;
    };
  }
}

function mountTabs(): void {
  const nav = qs("#tabs");
  const buttons = new Map<string, HTMLButtonElement>();
  for (const [key, label] of TABS) {
    const b = el("button", { class: "tab", "data-tab": key, "data-testid": `tab-${key}`, text: label });
    b.addEventListener("click", () => setTab(key));
    buttons.set(key, b);
    nav.appendChild(b);
  }
  const panels = Array.from(document.querySelectorAll<HTMLElement>("#right section[data-panel]"));
  function update(): void {
    const tab = buttons.has(state.tab) ? state.tab : "items";
    for (const [k, b] of buttons) b.classList.toggle("active", k === tab);
    for (const p of panels) p.hidden = p.dataset.panel !== tab;
  }
  subscribe((ev) => { if (ev.kind === "view" || ev.kind === "boot") update(); });
  update();
}

function mountDraftBar(): void {
  const bar = qs("#draft-bar");
  const restore = el("button", { class: "primary", "data-testid": "restore-draft", text: "Restore draft" });
  const discard = el("button", { "data-testid": "discard-draft", text: "Discard" });
  const msg = el("span", {});
  restore.addEventListener("click", () => A.restoreDraft());
  discard.addEventListener("click", () => A.discardDraft());
  bar.append(msg, restore, discard);
  subscribe((ev) => {
    if (ev.kind !== "draft" && ev.kind !== "file") return;
    bar.hidden = !state.draft;
    if (state.draft) msg.textContent = `Unsaved draft from ${new Date(state.draft.saved_at).toLocaleString()} (${state.draft.file.items.length} items) is newer than the saved file.`;
  });
}

function mountHeader(): void {
  const reviewer = qs<HTMLInputElement>("#reviewer");
  reviewer.addEventListener("change", () => { setReviewer(reviewer.value); toast(`reviewer: ${state.reviewer}`); });
  subscribe((ev) => {
    if (ev.kind === "boot" || ev.kind === "view") {
      if (document.activeElement !== reviewer) reviewer.value = state.reviewer;
      qs("#campaign-name").textContent = state.config?.campaign || state.health?.campaign_dir?.split("/").pop() || "";
      const h = state.health;
      qs("#health").textContent = h ? `v${h.version} · engine ${h.engine}${h.claude_version ? ` · claude ${h.claude_version}` : ""} · ${h.n_images} images` : "";
      qs("#health").title = h?.claude_bin || "";
    }
  });
}

function route(): void {
  const id = decodeURIComponent(location.hash.slice(1));
  if (!id) {
    if (state.images.length) location.hash = encodeURIComponent(state.images[0].id);
    return;
  }
  if (id !== state.id) void A.load(id);
}

async function boot(): Promise<void> {
  mountTabs();
  mountHeader();
  mountDraftBar();
  mountList(qs("#image-list"));
  mountToolbar(qs("#toolbar"), qs("#statusbar"));
  mountOverlay(qs<SVGSVGElement>("#overlay"));
  mountItems(qs("#panel-items"));
  mountSystem(qs("#panel-system"));
  mountPropose(qs("#panel-propose"));
  mountReview(qs("#panel-review"));
  mountVoice(qs("#panel-voice"));
  mountExport(qs("#panel-export"));
  installTools();
  installKeys();

  window.__lensmark = {
    state,
    addItem: (json) => A.addItem(json),
    selectItem: (id) => select(id),
    save: () => A.save(),
    load: (id) => A.load(id),
    setTool: (name) => setTool(name),
  };

  const [style, models, config, health, images] = await Promise.allSettled([api.style(), api.models(), api.config(), api.health(), api.images()]);
  const errs: string[] = [];
  if (style.status === "fulfilled") state.style = style.value; else errs.push(`style: ${A.describe(style.reason)}`);
  if (models.status === "fulfilled") state.models = models.value; else errs.push(`models: ${A.describe(models.reason)}`);
  if (config.status === "fulfilled") state.config = config.value; else errs.push(`config: ${A.describe(config.reason)}`);
  if (health.status === "fulfilled") state.health = health.value;      // optional
  if (images.status === "fulfilled") state.images = images.value; else errs.push(`images: ${A.describe(images.reason)}`);
  initReviewer(state.config?.reviewer || "");
  emit("boot");
  emit("images");
  if (errs.length) {
    setStatus(`boot: ${errs.join(" · ")}`, "error");
    toast(`server error — ${errs[0]}`, "error", 10000);
  } else {
    setStatus(`${state.images.length} images · reviewer ${state.reviewer}`);
  }
  window.addEventListener("hashchange", route);
  route();
}

void boot();
