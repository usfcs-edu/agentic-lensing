"""``POST /api/export/{coco|ds9|masks|fewshot}`` -> ``{files: [...]}`` under ``exports/``."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Request

from . import campaign_of, lazy_module, rel_path

router = APIRouter(tags=["export"])
FORMATS = ("coco", "ds9", "masks", "fewshot")


@router.post("/api/export/{fmt}")
def post_export(fmt: str, request: Request, body: Optional[dict[str, Any]] = Body(None)) -> dict[str, Any]:
    if fmt not in FORMATS:
        raise HTTPException(status_code=404, detail=f"unknown export format {fmt!r}; one of {list(FORMATS)}")
    body = body or {}
    ids = body.get("ids")
    if ids is not None and not (isinstance(ids, list) and all(isinstance(i, str) for i in ids)):
        raise HTTPException(status_code=400, detail="ids must be a list of image ids")
    campaign = campaign_of(request)
    mod = lazy_module("lensmark.exports", "run_export")
    files = mod.run_export(campaign, fmt, out=body.get("out"), k=int(body.get("k", 6)),
                           require_flag=bool(body.get("require_flag", False)), ids=ids)
    return {"format": fmt, "files": [rel_path(campaign, p) for p in files]}
