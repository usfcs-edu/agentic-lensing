"""``/api/ann/{id}`` - the LensMarkFile round trip, ``/api/ann/{id}/log`` and ``/api/render/{id}``."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from ..model import LensMarkFile
from ..store import sha256_file
from . import campaign_of, lazy_module, optional_module, require_image

router = APIRouter(tags=["annotations"])


@router.get("/api/ann/{image_id}")
def get_ann(image_id: str, request: Request) -> JSONResponse:
    """The saved file, or a fresh unsaved one (``X-LensMark-Exists: 0``) built from image + config."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    exists = campaign.exists(image_id)
    f = campaign.load_or_new(image_id)
    return JSONResponse(f.to_dict(), headers={"X-LensMark-Exists": "1" if exists else "0"})


@router.put("/api/ann/{image_id}")
def put_ann(image_id: str, request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Validate (``extra=forbid`` -> 422), check the id (400), atomic save + log diff, re-render."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    file = LensMarkFile.model_validate(body)          # pydantic ValidationError -> 422 (app handler)
    if file.id != image_id:
        raise HTTPException(status_code=400, detail=f"file id {file.id!r} does not match path id {image_id!r}")
    actor = request.headers.get("x-lensmark-actor") or "ui"
    campaign.save(image_id, file, actor=actor, source="ui")
    out: dict[str, Any] = {"ok": True, "modified": file.modified, "render": None, "lint": file.lint()}
    draw = optional_module("lensmark.render.draw", "render_to_file")
    if draw is not None:
        try:
            draw.render_to_file(campaign, image_id)
            saved = campaign.load(image_id)
            if saved is not None and saved.render is not None:
                out["render"] = saved.render.model_dump(mode="json", exclude_none=True)
        except Exception as e:  # noqa: BLE001 - a render failure must not lose the save
            out["render_error"] = f"{type(e).__name__}: {e}"
    return out


@router.get("/api/ann/{image_id}/log")
def get_log(image_id: str, request: Request) -> list[dict[str, Any]]:
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    return campaign.read_log(image_id)


@router.post("/api/render/{image_id}")
def post_render(image_id: str, request: Request, body: Optional[dict[str, Any]] = Body(None)) -> dict[str, Any]:
    """Regenerate ``<id>.annot.png`` from the JSON -> ``{output, sha256 (of the PNG), of_json_sha256, stale: false}``."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    draw = lazy_module("lensmark.render.draw", "render_to_file")
    body = body or {}
    out_path = draw.render_to_file(campaign, image_id, scale=float(body.get("scale", 1.0)))
    saved = campaign.load(image_id)
    info = saved.render.model_dump(mode="json", exclude_none=True) if saved is not None and saved.render else {}
    return {"output": info.get("output", out_path.name), "sha256": sha256_file(out_path),
            "of_json_sha256": info.get("of_json_sha256"), "stale": campaign.annot_stale(image_id, saved)}
