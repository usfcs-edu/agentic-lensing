"""Voice / natural-language patching: ``/api/patch/{id}`` (dry run), ``.../apply`` and ``/api/stt``."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from . import campaign_of, engine_or_none, lazy_module, require_image

router = APIRouter(tags=["voice"])


@router.post("/api/patch/{image_id}")
async def post_patch(image_id: str, request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """``{transcript, model?, effort?}`` -> the ``Patch`` Claude proposes (ops are NOT applied)."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    transcript = str(body.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript is required")
    mod = lazy_module("lensmark.voice.patch", "make_patch")
    engine = engine_or_none(request.app.state.engine_name)
    patch = await mod.make_patch(campaign, image_id, transcript, engine=engine,
                                 model=body.get("model"), effort=body.get("effort") or "low")
    return patch.model_dump(mode="json")


@router.post("/api/patch/{image_id}/apply")
def post_patch_apply(image_id: str, request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """``{ops, transcript?}`` -> apply + save + log ``source:"voice"`` -> the updated ``LensMarkFile``."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    ops = body.get("ops")
    if not isinstance(ops, list):
        raise HTTPException(status_code=400, detail="ops must be a list of {op, id?, item?, set?}")
    mod = lazy_module("lensmark.voice.patch", "apply_patch")
    f = mod.apply_patch(campaign, image_id, ops, transcript=str(body.get("transcript") or ""), actor="voice")
    return f.to_dict()


@router.post("/api/stt")
async def post_stt(audio: UploadFile = File(...)) -> dict[str, Any]:
    """multipart ``audio`` -> ``{transcript, backend}``; 501 while no STT backend is configured."""
    mod = lazy_module("lensmark.voice.stt", "transcribe")
    data = await audio.read()
    mime = audio.content_type or "application/octet-stream"
    try:
        transcript, backend = await run_in_threadpool(mod.transcribe, data, mime)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=f"no STT backend configured: {e}") from e
    return {"transcript": transcript, "backend": backend}
