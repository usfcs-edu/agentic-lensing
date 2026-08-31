"""``/api/propose`` (202 + SSE + cancel) and ``/api/proposals`` (immutable run files)."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.encoders import jsonable_encoder

from .. import config
from . import campaign_of, clean_json, lazy_module, optional_module, require_image
from .sse import sse_response

router = APIRouter(tags=["propose"])


def _state_or_404(request: Request, image_id: str, run_id: str):
    st = request.app.state.runs.get(run_id)
    if st is None or st.image_id != image_id:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r} for image {image_id!r}")
    return st


@router.post("/api/propose/{image_id}", status_code=202)
async def post_propose(image_id: str, request: Request, body: Optional[dict[str, Any]] = Body(None)) -> dict[str, str]:
    """Start a background proposal run -> ``202 {run_id}``. Defaults come from the campaign config."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    propose = lazy_module("lensmark.claude.propose", "ProposeRequest", "run_propose")
    body = body or {}
    cfg = campaign.config
    engine_name = body.get("engine") or request.app.state.engine_name
    kwargs: dict[str, Any] = {
        "model": body.get("model") or cfg.get("default_model") or config.DEFAULT_MODEL,
        "effort": body.get("effort") or cfg.get("default_effort") or config.DEFAULT_EFFORT,
        "budget": float(body["budget"]) if body.get("budget") is not None else config.max_budget_usd(),
        "engine": engine_name,
    }
    for key in ("fewshot", "include_grid"):
        if body.get(key) is not None:
            kwargs[key] = body[key]
    req = propose.ProposeRequest(**kwargs)                 # pydantic ValidationError -> 422
    run_id = request.app.state.runs.start(campaign, image_id, req, engine_name)
    return {"run_id": run_id}


@router.get("/api/propose/{image_id}/{run_id}")
async def get_propose_status(image_id: str, run_id: str, request: Request) -> dict[str, Any]:
    """Poll-able snapshot of a run (not in API.md; complements the SSE stream)."""
    return _state_or_404(request, image_id, run_id).snapshot()


@router.get("/api/propose/{image_id}/{run_id}/events")
async def get_propose_events(image_id: str, run_id: str, request: Request):
    st = _state_or_404(request, image_id, run_id)
    return sse_response(request.app.state.runs.events(st.run_id))


@router.post("/api/propose/{image_id}/{run_id}/cancel")
async def post_propose_cancel(image_id: str, run_id: str, request: Request) -> dict[str, Any]:
    st = _state_or_404(request, image_id, run_id)
    cancelled = request.app.state.runs.cancel(st.run_id)
    return {"ok": True, "cancelled": cancelled, "phase": st.phase}


@router.get("/api/proposals/{image_id}")
def list_proposals(image_id: str, request: Request) -> list[dict[str, Any]]:
    """Runs from ``provenance.proposal_runs`` + ``proposals/`` files (falls back to the file's
    provenance block while the propose module is missing)."""
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    propose = optional_module("lensmark.claude.propose", "list_runs")
    if propose is not None:
        rows = propose.list_runs(campaign, image_id)
    else:
        f = campaign.load(image_id)
        rows = list(f.provenance.proposal_runs) if f is not None else []
    return clean_json(jsonable_encoder(rows))


@router.get("/api/proposals/{image_id}/{run_id}")
def get_proposal(image_id: str, run_id: str, request: Request) -> dict[str, Any]:
    campaign = campaign_of(request)
    require_image(campaign, image_id)
    propose = lazy_module("lensmark.claude.propose", "load_run")
    try:
        doc = propose.load_run(campaign, image_id, run_id)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=404, detail=f"unknown proposal run {run_id!r}: {e}") from e
    return clean_json(jsonable_encoder(doc))
