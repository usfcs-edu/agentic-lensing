"""golden/grader_jwst.py — DIRECT grader for the golden JWST set: one inline composite, one call.

The golden campaign shows Xiaosheng exactly one image per item: the run's 6-panel NIRCam
composite, footer-cropped, served from the blind kit. The model arms must see the SAME
bytes, no more and no less, so this grader feeds grader_direct.grade_candidate(content=...)
the kit JPEG as a single base64 image block (the tier-2 pattern of
finetune/distill_euclid.euclid_content) behind an item-agnostic panel gloss. Nothing
identifying reaches the model: no name, no coordinates (the JWST ids encode them, and the
L_known stratum is in the literature by name), no pipeline grade, no CNN score.

System prompt = prompts/rubric_imaging_v2.md + prompts/jwst_note.md (the note voids the
rubric's tool/photometry/CNN sentences and states the JWST resolution regime); any other
`system_prompt` handed in (run_batch --rubric, the E3 arm) gets the same note appended, so
the two entry points never disagree about what the model read. E2 passes `fewshot_blocks`
from golden/fewshot.py; they go in front of the candidate blocks.

Every call writes a `golden_content_audit` event to its trace BEFORE the request: sha16 +
first 200 chars of every text block, sha16 of every image payload (== the kit render_sha
for kit JPEGs), the system-prompt sha16 and the exemplar unit ids. golden/audit_traces.py
replays those events against the banned lexicon and the split key, so a leak is caught
from the trace rather than trusted from the code.

Drop-in with run_batch (`--mode jwst`): same GradeResult, same grade_candidate keywords.
On the Anthropic backend the single call runs at the API default temperature (grader_direct
passes none) — the k=3 replicate parquets of run_golden_eval.py are real samples.

Seams for the evidence-first panel (plan PART 2; golden/panel.py is the caller): `note=`
(default = this module's JWST_NOTE, so every existing caller is unchanged; the panel passes
prompts/jwst_note_v2.md and the incumbent re-implementation passes "" because its wrapper is
self-contained — the v1 note carries "bluer" and "default C or D", which must never be
auto-appended to a persona), `schema=` (the pydantic record the reply validates into,
forwarded to grader_direct; a reply that fails it is a parse failure, never an ImageGrade
coercion), `extra_views=` (list of (label, PIL.Image | path) appended as [label text, image]
block pairs AFTER the candidate blocks — the geometry critic's extra panels and the 20"
context pair), and `audit_full_text=` (record every text block in full in the audit event
rather than its 200-char head, for the panel's item-specific blocks — the located-items JSON
— so audit_traces can still lexicon-check all of it). The audit event records
`n_extra_views`; audit_traces' image rule is n_images == 1 + n_exemplars + n_extra_views.
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.common import hooks  # noqa: E402
from lensjudge.golden import _util  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.imaging import grader_direct  # noqa: E402
from lensjudge.imaging.grader_lean import GradeResult  # noqa: E402

PROMPTS = _util.LENSJUDGE / "prompts"
RUBRIC_V2_PATH = PROMPTS / "rubric_imaging_v2.md"
JWST_NOTE_PATH = PROMPTS / "jwst_note.md"
JWST_NOTE = JWST_NOTE_PATH.read_text()
RUBRIC_V2 = RUBRIC_V2_PATH.read_text()
DIRECT_SYS_JWST = RUBRIC_V2 + JWST_NOTE
KEYS_DIR = _util.HERE / "keys"
KITS_DIR = _util.HERE / "kits"

# The only candidate-side text the model ever sees. Item-agnostic by construction: it
# describes the render, not the object.
PANEL_GLOSS = (
    "Grade this JWST NIRCam strong-lens candidate from the single composite below. "
    "Six panels, two rows, north up and east left. Row 1 (10\" field): normal stretch, "
    "deep stretch, two-band colour (red = long-wavelength channel, blue = short-wavelength "
    "channel). Row 2 (3.5\" zoom on the catalogued galaxy): deep stretch, two-band colour, "
    "deflector-subtracted residual. A 1\" scale bar sits in each row; four yellow ticks "
    "point at the catalogued galaxy (the putative deflector). No tools, scores or "
    "photometry are available; the composite is the complete evidence set.\n\n"
    "Respond with ONLY the JSON object for the required schema.")


def _sha16_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def with_note(rubric_text: str, note: str = JWST_NOTE) -> str:
    """Any rubric + the JWST note (E3 rubric arms build their system prompt through this).
    `note` may be another note text (jwst_note_v2.md for the panel) or "" (nothing appended);
    a prompt that already ends with the note is returned unchanged."""
    if not note or rubric_text.endswith(note):
        return rubric_text
    return rubric_text + note


# ------------------------------------------------------------------ image resolution
_KEY_CACHE: Optional[pd.DataFrame] = None


def _keys(keys_dir: Path = KEYS_DIR) -> pd.DataFrame:
    """All tracked kit keys concatenated (pass-1 rows resolve the canonical composite)."""
    global _KEY_CACHE
    if _KEY_CACHE is None:
        parts = []
        for p in sorted(Path(keys_dir).glob("*_key.csv")):
            parts.append(_util.read_pinned(p) if p.with_suffix(".csv.sha").exists() else pd.read_csv(p))
        _KEY_CACHE = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _KEY_CACHE


def resolve_image_path(cand: dict) -> Optional[Path]:
    """cand['image_path'] if present, else the pass-1 kit JPEG for cand['name'] (candidate_id)
    or cand['unit_id'] via golden/keys/*_key.csv -> kits/<kit_id>/items/NNN.jpg."""
    p = cand.get("image_path")
    if isinstance(p, str) and p.strip():
        return Path(p)
    key = _keys()
    if key.empty:
        return None
    sel = key[pd.to_numeric(key["pass"], errors="coerce").fillna(1) == 1]
    if cand.get("unit_id"):
        sel = sel[sel["unit_id"].astype(str) == str(cand["unit_id"])]
    else:
        sel = sel[sel["candidate_id"].astype(str) == str(cand.get("name"))]
    if sel.empty:
        return None
    r = sel.iloc[0]
    return KITS_DIR / str(r["kit_id"]) / "items" / f"{str(r['item_id']).zfill(3)}.jpg"


def jwst_content(cand: dict, gloss: str = PANEL_GLOSS) -> Optional[list]:
    """[text: panel gloss + JSON instruction] + [one base64 JPEG block]. None if no composite.
    `gloss` is the item-agnostic text in front of the image (a panel role passes its own
    VIEW paragraph + the located-items JSON); the JPEG bytes are always the served kit bytes."""
    path = resolve_image_path(cand)
    if path is None or not Path(path).exists():
        return None
    data = Path(path).read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"{path}: not a JPEG")
    want = cand.get("render_sha")
    if isinstance(want, str) and want.strip() and want.strip() != _sha16_bytes(data):
        raise ValueError(f"{path}: sha {_sha16_bytes(data)} != manifest render_sha {want}")
    return [{"type": "text", "text": gloss},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(data).decode()}}]


def view_blocks(extra_views) -> list:
    """[(label, PIL.Image | path), ...] -> [text label, image] block pairs. A PIL image is
    encoded as PNG (lossless crops); a .jpg/.jpeg path is sent byte-for-byte as JPEG, any
    other path as PNG. Labels are the item-agnostic gloss strings of golden/views.py."""
    blocks: list = []
    for label, view in (extra_views or []):
        if isinstance(view, (str, Path)):
            raw = Path(view).read_bytes()
            media = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
            block = {"type": "image", "source": {"type": "base64", "media_type": media,
                                                 "data": base64.b64encode(raw).decode()}}
        else:
            from lensjudge.golden import views as _views
            block = _views.image_block(view, fmt="PNG")
        blocks.append({"type": "text", "text": str(label)})
        blocks.append(block)
    return blocks


# ------------------------------------------------------------------ audit event
AUDIT_HEAD = 200   # chars of each text block the audit event keeps (audit_traces.HEAD)


def content_audit(content: list, system_prompt: str, n_exemplars: int,
                  exemplar_unit_ids=None, n_extra_views: int = 0, note: str = JWST_NOTE,
                  full_text: bool = False) -> dict:
    """The `golden_content_audit` record: everything audit_traces.py needs, no pixels, no
    full text (sha16 + a 200-char head per text block; sha16 per image payload) — unless
    `full_text`, when every text block is kept whole (the panel's item-specific blocks,
    which the audit must be able to read). Image order is exemplars, then the ONE candidate
    image, then `n_extra_views` extra views (golden/panel.py), so the candidate is image
    number n_exemplars, not the last one."""
    texts, images = [], []
    for b in content:
        if b.get("type") == "text":
            t = b.get("text", "")
            head = t if full_text else t[:AUDIT_HEAD]
            texts.append({"sha16": _util.sha_text(t), "head": head, "n_chars": len(t)})
        elif b.get("type") == "image":
            src = b.get("source", {})
            raw = base64.b64decode(src.get("data", "")) if src.get("type") == "base64" else b""
            images.append({"sha16": _sha16_bytes(raw), "media_type": src.get("media_type"),
                           "n_bytes": len(raw)})
    n_ex, n_xv = int(n_exemplars), int(n_extra_views)
    return {"text_blocks": texts,
            "image_shas": [i["sha16"] for i in images],
            "images": images,
            # exemplars precede the candidate; extra views follow it
            "exemplar_image_shas": [i["sha16"] for i in images[:n_ex]],
            "candidate_image_sha": images[n_ex]["sha16"] if len(images) > n_ex else None,
            "extra_view_shas": [i["sha16"] for i in images[n_ex + 1:]],
            "n_images": len(images), "n_exemplars": n_ex, "n_extra_views": n_xv,
            "exemplar_unit_ids": list(exemplar_unit_ids or []),
            "system_sha16": _util.sha_text(system_prompt),
            # the note actually appended to the system prompt (None when note="")
            "jwst_note_sha16": _util.sha_text(note) if note else None,
            "full_text": bool(full_text)}


# ------------------------------------------------------------------ grader
async def grade_candidate(cand: dict, *, model: Optional[str] = None,
                          tools=None,                       # ignored (no loop), run_batch passes it
                          system_prompt: Optional[str] = None,
                          views=None,                       # ignored: the composite is the view
                          content: Optional[list] = None,
                          trace_path: Optional[str] = None,
                          fewshot_blocks: Optional[list] = None,
                          exemplar_unit_ids=None,
                          note: str = JWST_NOTE,
                          schema=ImageGrade,
                          extra_views: Optional[list] = None,
                          audit_full_text: bool = False) -> GradeResult:
    """Drop-in for run_batch: builds fewshot_blocks + jwst_content(cand), writes the audit
    event, delegates to grader_direct.grade_candidate(content=..., system_prompt=...,
    schema=...). `note` is appended to the system prompt iff non-empty and the prompt does
    not already end with it; `extra_views` ([(label, PIL.Image | path)]) become [label,
    image] block pairs after the candidate blocks (also after a caller-supplied `content`)."""
    if os.environ.get("LENSJUDGE_FEWSHOT_MANIFEST"):
        raise RuntimeError("LENSJUDGE_FEWSHOT_MANIFEST is set: unset it for golden JWST runs "
                           "(that path fetches DESI cutouts for survey_key=jwst)")
    # a caller-supplied rubric (run_batch --rubric, E3) gets the JWST note appended exactly
    # as run_golden_eval does, so the system prompt never silently differs from the one that
    # was registered; a prompt that already ends with the note is left alone; note="" (the
    # incumbent wrapper) appends nothing; the default with no prompt is DIRECT_SYS_JWST
    sysp = with_note(system_prompt or RUBRIC_V2, note or "")
    few = list(fewshot_blocks or [])
    if content is None:
        cand_blocks = jwst_content(cand)
        if cand_blocks is None:
            return GradeResult(None, "", error="no composite", parse_ok=False,
                               meta={"name": cand.get("name"), "mode": "jwst"})
        content = few + cand_blocks
    xv = view_blocks(extra_views)
    if xv:
        content = list(content) + xv
    n_ex = sum(1 for b in few if b.get("type") == "image")
    n_xv = sum(1 for b in xv if b.get("type") == "image")
    audit = content_audit(content, sysp, n_ex, exemplar_unit_ids, n_extra_views=n_xv,
                          note=note or "", full_text=audit_full_text)
    if trace_path:
        hooks.Trace(trace_path).write("golden_content_audit", name=cand.get("name"),
                                      unit_id=cand.get("unit_id"),
                                      schema=getattr(schema, "__name__", str(schema)), **audit)
    res = await grader_direct.grade_candidate(cand, model=model, system_prompt=sysp,
                                              content=content, trace_path=trace_path,
                                              schema=schema)
    res.meta["mode"] = "jwst"
    res.meta["render_sha"] = audit["candidate_image_sha"]
    res.meta["n_exemplars"] = n_ex
    res.meta["n_extra_views"] = n_xv
    res.meta["exemplar_unit_ids"] = "|".join(str(u) for u in (exemplar_unit_ids or []))
    res.meta["system_sha16"] = audit["system_sha16"]
    res.meta["schema"] = getattr(schema, "__name__", str(schema))
    return res
