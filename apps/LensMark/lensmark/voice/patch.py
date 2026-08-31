"""Natural-language / voice patch: transcript -> Claude (patch schema, effort low) -> id-addressed ops;
``apply_patch`` applies approved ops to the file with full re-validation and logs them ``source:"voice"``."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from .. import config, imaging
from ..claude.engine_base import EngineRequest, EngineResult
from ..model import (CreatedBy, EinsteinRing, LensMarkFile, Patch, PatchOp, Review, SystemBlock, now_iso,
                     parse_item)
from ..store import Campaign
from ..validate import ValidationResult, item_from_proposal

PROMPT_FILE = config.PROMPT_DIR / "patch_v1.md"
PATCH_SCHEMA_FILE = config.SCHEMA_DIR / "lensmark-patch-1.0.schema.json"
GEOMETRY_FIELDS = ("tail", "head", "center", "pos", "radius_arcsec", "theta_e_arcsec")
_ITEM_STATE_FIELDS = ("id", "type", "status", "label", "color", "tail", "head", "center", "radius_arcsec", "kind",
                      "theta_e_arcsec", "center_ref", "text", "pos")


def patch_schema() -> dict[str, Any]:
    with open(PATCH_SCHEMA_FILE, encoding="utf-8") as f:
        schema = json.load(f)
    # claude --json-schema rejects a top-level "$schema" key (same fix as claude.propose.engine_schema)
    schema.pop("$schema", None)
    return schema


def system_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def file_state(file: LensMarkFile) -> dict[str, Any]:
    """Compact view of the file for the model: image facts, system block, items with ids."""
    items = []
    for it in file.items:
        d = it.model_dump(mode="json", exclude_none=True)
        row = {k: d[k] for k in _ITEM_STATE_FIELDS if k in d}
        row["created_by"] = it.created_by.kind
        items.append(row)
    im = file.image
    return {"image": {"id": file.id, "width": im.width, "height": im.height, "cutout_arcsec": im.cutout_arcsec,
                      "pixel_scale_arcsec": round(im.pixel_scale_arcsec, 6), "north_up": im.north_up,
                      "east_left": im.east_left},
            "system": file.system.model_dump(mode="json", exclude_none=True),
            "items": items}


def build_request(campaign: Campaign, file: LensMarkFile, transcript: str, *, model: Optional[str] = None,
                  effort: Optional[str] = "low") -> EngineRequest:
    im = imaging.upsample(imaging.load_rgb(campaign.image_path(file.id)))
    model_id = config.resolve_model(model or str(campaign.config.get("default_model") or config.DEFAULT_MODEL))
    content = [
        {"type": "text", "text": "Current LensMark file (ids are the op targets):\n"
                                 + json.dumps(file_state(file), ensure_ascii=False, separators=(",", ":"))},
        {"type": "text", "text": "The cutout, then the same cutout with a labelled u/v grid (lines every 0.1):"},
        imaging.image_block(im),
        imaging.image_block(imaging.grid_overlay(im)),
        {"type": "text", "text": "Instruction (transcript):\n" + transcript.strip()},
    ]
    return EngineRequest(system=system_prompt(), content=content, schema=patch_schema(), model=model_id,
                         effort=effort if config.model_supports_effort(model_id) else None,
                         max_budget_usd=config.max_budget_usd(), max_turns=config.MAX_TURNS, cwd=campaign.root,
                         fixture_key=file.id, purpose="patch")


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Fallback when the engine gave no structured output: the outermost {...} of the final text."""
    if not text:
        return None
    try:
        from ..claude.parse import extract_json_block  # type: ignore[import-not-found]
        obj = extract_json_block(text)
        return obj if isinstance(obj, dict) else None
    except ImportError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_patch(result: EngineResult, transcript: str) -> Patch:
    """Lenient: keep every well-formed op, drop the rest (mentioned in ``clarification``); never raise."""
    obj = result.structured if isinstance(result.structured, dict) else _extract_json(result.text)
    if result.error and obj is None:
        return Patch(transcript=transcript, ops=[], clarification=f"model returned no valid patch: {result.error}")
    if obj is None:
        return Patch(transcript=transcript, ops=[], clarification="model returned no valid patch: no JSON in the reply")
    ops: list[PatchOp] = []
    dropped: list[str] = []
    raw_ops = obj.get("ops")
    for i, raw in enumerate(raw_ops if isinstance(raw_ops, list) else []):
        try:
            ops.append(PatchOp.model_validate(raw))
        except ValueError as e:
            dropped.append(f"op {i}: {str(e).splitlines()[0]}")
    clar = obj.get("clarification") if isinstance(obj.get("clarification"), str) else None
    if dropped:
        clar = ((clar + " ") if clar else "") + "model returned no valid patch for: " + "; ".join(dropped)
    if not ops and clar is None:
        clar = "model returned no valid patch: empty ops"
    return Patch(transcript=str(obj.get("transcript") or transcript), ops=ops, clarification=clar)


async def make_patch(campaign: Campaign, image_id: str, transcript: str, *, engine=None, model: Optional[str] = None,
                     effort: Optional[str] = "low") -> Patch:
    """Ask Claude for the ops implementing ``transcript`` on ``image_id`` (nothing is applied)."""
    if engine is None:
        from ..claude.engine import get_engine  # lazy: the SDK engine is heavy and optional in tests
        engine = get_engine(None)
    file = campaign.load_or_new(image_id)
    req = build_request(campaign, file, transcript, model=model, effort=effort)
    result = await engine.run(req)
    return parse_patch(result, transcript)


# ----------------------------------------------------------------------------- apply
def _as_op(op: PatchOp | dict[str, Any]) -> PatchOp:
    return op if isinstance(op, PatchOp) else PatchOp.model_validate(op)


def _apply_add(file: LensMarkFile, op: PatchOp, used: set[str], res: ValidationResult) -> None:
    if not op.item:
        raise ValueError("add op without an item")
    item = item_from_proposal(op.item, file, CreatedBy(kind="voice"), used_ids=used, res=res, status="accepted")
    if item is None:
        raise ValueError(f"add op: unknown item type {op.item.get('type')!r}")
    if item.status == "invalid":
        raise ValueError(f"add op: item geometry invalid ({item.invalid_reason})")
    if op.rationale and not item.notes:
        item.notes = op.rationale[:300]
    file.items.append(item)


def _apply_system(file: LensMarkFile, changes: dict[str, Any]) -> None:
    data = file.system.model_dump(mode="json", exclude_none=True)
    for k, v in changes.items():
        if k == "theta_e" and isinstance(v, dict):
            cur = dict(data.get("theta_e") or {})
            cur.update({kk: vv for kk, vv in v.items()})
            data["theta_e"] = cur
        else:
            data[k] = v
    file.system = SystemBlock.model_validate(data)


def _apply_update(file: LensMarkFile, op: PatchOp) -> None:
    changes = {k: v for k, v in (op.set or {}).items() if v is not None or k in ("label", "review", "notes")}
    if not changes:
        raise ValueError(f"update op on {op.id!r} sets nothing")
    if op.id == "$system":
        _apply_system(file, changes)
        return
    it = file.item(op.id or "")
    if it is None:
        raise ValueError(f"update op: unknown item id {op.id!r}")
    d = it.model_dump(mode="json", exclude_none=True)
    geometry_changed = any(k in GEOMETRY_FIELDS and d.get(k) != v for k, v in changes.items())
    if (geometry_changed and it.created_by.kind == "claude" and it.status in ("proposed", "accepted")
            and "status" not in changes and not d.get("edit_of")):
        d["edit_of"] = {k: d[k] for k in GEOMETRY_FIELDS if k in d}
        d["status"] = "edited"
    if "review" in changes and isinstance(changes["review"], dict):
        rv = dict(changes["review"])
        rv.setdefault("reviewer", "voice")
        rv.setdefault("reviewed_at", now_iso())
        changes["review"] = rv
    d.update(changes)
    new = parse_item(d)                       # strict model -> ValueError (pydantic) on bad values
    file.items[file.items.index(it)] = new


def _apply_delete(file: LensMarkFile, op: PatchOp, actor: str) -> None:
    it = file.item(op.id or "")
    if it is None:
        raise ValueError(f"delete op: unknown item id {op.id!r}")
    review = (op.set or {}).get("review")
    if isinstance(review, dict) and review.get("verdict"):
        rv = dict(review)
        rv.setdefault("reviewer", actor)
        rv.setdefault("reviewed_at", now_iso())
        it.status = "rejected"
        it.review = Review.model_validate(rv)
    else:
        file.items.remove(it)


def apply_patch(campaign: Campaign, image_id: str, ops: list[dict[str, Any] | PatchOp], transcript: str = "",
                actor: str = "voice") -> LensMarkFile:
    """Apply approved ops, re-validate the whole file, save (diff-logged) and log the patch itself."""
    file = campaign.load_or_new(image_id)
    parsed = [_as_op(o) for o in ops]
    used: set[str] = set()
    res = ValidationResult()
    for op in parsed:
        if op.op == "add":
            _apply_add(file, op, used, res)
        elif op.op == "update":
            _apply_update(file, op)
        elif op.op == "delete":
            _apply_delete(file, op, actor)
    file = LensMarkFile.model_validate(file.to_dict())      # bad values -> ValueError before anything is written
    campaign.save(image_id, file, actor=actor, source="voice")
    campaign.append_log(image_id, actor=actor, source="voice", op="patch", item_id="$patch", transcript=transcript,
                        ops=[o.model_dump(mode="json", exclude_none=True) for o in parsed],
                        repairs=res.repairs or None)
    return file


def cli_patch(dir: str, *, image_id: str, transcript: str, apply: bool = False, model: Optional[str] = None,
              engine: Optional[str] = None) -> int:
    campaign = Campaign(dir)
    from ..claude.engine import get_engine
    eng = get_engine(engine)
    patch = asyncio.run(make_patch(campaign, image_id, transcript, engine=eng, model=model))
    print(json.dumps(patch.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False))
    if not patch.ops:
        print(f"no ops: {patch.clarification}")
        return 2
    if apply:
        f = apply_patch(campaign, image_id, [o.model_dump() for o in patch.ops], transcript=transcript)
        print(f"applied {len(patch.ops)} op(s) to {image_id}: {len(f.items)} items now")
    return 0
