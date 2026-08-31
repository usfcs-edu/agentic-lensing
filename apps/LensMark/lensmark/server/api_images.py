"""``/api/images`` - listing, original bytes (ETag = sha256), rendered overlay, thumbnails."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response

from .. import imaging
from ..store import sha256_file
from . import campaign_of, optional_module, require_image

router = APIRouter(tags=["images"])
MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@router.get("/api/images")
def list_images(request: Request) -> list[dict]:
    return campaign_of(request).manifest()


@router.get("/api/images/{image_id}/original")
def get_original(image_id: str, request: Request) -> Response:
    campaign = campaign_of(request)
    path = require_image(campaign, image_id)
    saved = campaign.load(image_id)
    sha = saved.image.sha256 if saved is not None else sha256_file(path)
    inm = request.headers.get("if-none-match", "")
    if inm and sha in {t.strip().strip('"').removeprefix("W/").strip('"') for t in inm.split(",")}:
        return Response(status_code=304, headers={"ETag": sha})
    return Response(path.read_bytes(), media_type=MEDIA.get(path.suffix.lower(), "application/octet-stream"),
                    headers={"ETag": sha})


@router.get("/api/images/{image_id}/annot")
def get_annot(image_id: str, request: Request) -> Response:
    """The canonical rendered PNG; re-rendered in memory when the file is stale or missing."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    png = campaign.annot_path(image_id)
    stale = campaign.annot_stale(image_id)
    if png.exists() and not stale:
        return Response(png.read_bytes(), media_type="image/png", headers={"X-LensMark-Stale": "0"})
    draw = optional_module("lensmark.render.draw", "render_png_bytes")
    if draw is None:
        if png.exists():
            return Response(png.read_bytes(), media_type="image/png", headers={"X-LensMark-Stale": "1"})
        raise HTTPException(status_code=404, detail="no annotated PNG yet and the render module is not available")
    data = draw.render_png_bytes(campaign, image_id, include_proposed=True)
    return Response(data, media_type="image/png", headers={"X-LensMark-Stale": "0", "X-LensMark-Rendered": "on-demand"})


@router.get("/api/images/{image_id}/thumb")
def get_thumb(image_id: str, request: Request, px: int = Query(160, ge=16, le=1024)) -> Response:
    path = require_image(campaign_of(request), image_id)
    return Response(imaging.thumbnail(path, px), media_type="image/jpeg")
