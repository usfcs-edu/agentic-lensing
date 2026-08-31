/**
 * Keyboard map (CONTRACT.md "Keyboard"). Single-letter keys are ignored while typing in a field;
 * Cmd/Ctrl+S and Cmd/Ctrl+Z work everywhere (undo only outside fields, where the browser's own
 * text undo applies).
 */
import * as A from "./actions";
import { isEditable } from "./dom";
import { getItem, select, setTool, state, undo } from "./state";
import { setSpace, zoomToggle } from "./svg/tools";
import { KEY_VERDICTS } from "./types";

export function installKeys(): void {
  document.addEventListener("keydown", onKey);
  document.addEventListener("keyup", (e) => { if (e.key === " ") setSpace(false); });
  window.addEventListener("blur", () => setSpace(false));
}

function onKey(e: KeyboardEvent): void {
  const mod = e.metaKey || e.ctrlKey;
  const inField = isEditable(e.target);
  const key = e.key;

  if (mod && key.toLowerCase() === "s") { e.preventDefault(); void A.save(); return; }
  if (mod && key.toLowerCase() === "z" && !e.shiftKey && !inField) { e.preventDefault(); undo(); return; }
  if (mod) return;

  if (inField) {
    if (key === "Escape") (e.target as HTMLElement).blur();
    return;
  }
  // Enter / Space on a focused button is the button's own activation, not a review verdict
  if ((e.target as HTMLElement)?.tagName === "BUTTON" && (key === "Enter" || key === " ")) return;
  if (key === " ") { setSpace(true); e.preventDefault(); return; }

  const sel = getItem(state.selectedId);
  switch (key) {
    case "a": case "A": setTool("arrow"); break;
    case "g": case "G": setTool("galaxy"); break;
    case "s": case "S": setTool("star"); break;
    case "r": case "R": setTool("ring"); break;
    case "t": case "T": setTool("text"); break;
    case "v": case "V": setTool("select"); break;
    case "Escape": setTool("select"); select(null); break;
    case "Delete": case "Backspace": A.deleteSelected(); break;
    case "[": A.navigate(-1); break;
    case "]": A.navigate(+1); break;
    case "z": case "Z": zoomToggle(); break;
    case "l": case "L": if (state.file) A.cycleLegend(); break;
    case "Enter": if (sel) A.acceptItem(sel.id); break;
    case "x": case "X": if (sel) A.rejectItem(sel.id); break;
    default: {
      const n = Number(key);
      if (n >= 1 && n <= 7 && sel && (sel.status === "proposed" || sel.review || sel.created_by.kind === "claude")) {
        A.applyVerdict(sel.id, KEY_VERDICTS[n - 1]);
      } else return;
    }
  }
  e.preventDefault();
}
