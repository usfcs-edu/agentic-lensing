"""``/api/critique/{id}`` (submit a Critique document) and ``/api/eval`` (aggregate rows)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder

from ..model import Critique
from . import campaign_of, clean_json, lazy_module, rel_path, require_image

router = APIRouter(tags=["critique"])


@router.post("/api/critique/{image_id}")
def post_critique(image_id: str, request: Request, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Validate a ``Critique`` (422), check ``image_id`` (400), then ``critique.submit_critique`` -> ``{file}``."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    critique = Critique.model_validate(body)
    if critique.image_id != image_id:
        raise HTTPException(status_code=400, detail=f"critique image_id {critique.image_id!r} != path id {image_id!r}")
    mod = lazy_module("lensmark.critique", "submit_critique")
    path = mod.submit_critique(campaign, critique)
    return {"file": rel_path(campaign, path)}


@router.get("/api/eval")
def get_eval(request: Request, by: str = Query("model,effort")) -> dict[str, Any]:
    mod = lazy_module("lensmark.evaluate", "eval_rows")
    rows = mod.eval_rows(campaign_of(request), by=by)
    return {"rows": clean_json(jsonable_encoder(rows)), "by": by}
