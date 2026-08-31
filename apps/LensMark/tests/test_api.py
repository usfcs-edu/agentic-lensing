"""HTTP contract tests (API.md) against ``create_app`` on a throw-away copy of examples/nine.

The propose / render / critique / export / patch flows that depend on sibling modules are guarded with
``pytest.importorskip`` so this file passes before and after those modules land. The proposal
plumbing itself (202 -> SSE -> cancel) is exercised deterministically with a fake ``lensmark.claude.propose``
injected into ``sys.modules``.
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lensmark import __version__
from lensmark.model import Arrow, CreatedBy, ProposalRun
from lensmark.server.app import create_app, serve
from lensmark.server.sse import format_event, parse_events, sse_response

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
SCHEMAS = ("lensmark", "proposal", "critique", "patch")


@pytest.fixture
def client(nine: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENSMARK_ENGINE", "fixture")
    app = create_app(nine, engine="fixture")
    with TestClient(app) as c:      # the context keeps ONE event loop alive across requests (background runs)
        c.campaign_dir = nine       # type: ignore[attr-defined]
        yield c


def _arrow_body(client: TestClient, image_id: str = "deck-01", **kw) -> dict:
    """The GET body of a fresh file with one accepted arrow added."""
    body = client.get(f"/api/ann/{image_id}").json()
    item = {"id": "ann-arrow-001", "type": "arrow", "tail": [0.41, 0.70], "head": [0.446, 0.586],
            "color": "cyan", "label": "tight arc"}
    item.update(kw)
    body["items"].append(item)
    return body


def _stream_phases(client: TestClient, image_id: str, run_id: str) -> list[dict]:
    with client.stream("GET", f"/api/propose/{image_id}/{run_id}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("x-accel-buffering") == "no"
        text = "".join(r.iter_text())
    assert "event: message\ndata: " in text
    return parse_events(text)


def _fake_propose(monkeypatch: pytest.MonkeyPatch, *, delay: float = 0.0, fail: bool = False,
                  accept_run_id: bool = True) -> types.ModuleType:
    """A stand-in ``lensmark.claude.propose`` with the contract signatures; writes one proposed arrow."""
    from pydantic import BaseModel

    class ProposeRequest(BaseModel):
        model: str = "opus"
        effort: str | None = "xhigh"
        budget: float = 0.5
        fewshot: str | None = None
        engine: str = "fixture"
        include_grid: bool = True

    async def _body(campaign, image_id, req, engine, on_event, run_id):
        emit = on_event or (lambda e: None)
        emit({"phase": "started", "detail": f"{req.model}/{req.effort} budget {req.budget}"})
        emit({"phase": "thinking", "detail": "…", "text": "looking at the cutout"})
        if delay:
            await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("boom")
        f = campaign.load_or_new(image_id)
        f.items.append(Arrow(id=f.next_id("arrow"), tail=[0.7, 0.7], head=[0.55, 0.55], label="arc", status="proposed",
                             created_by=CreatedBy(kind="claude", model=req.model, effort=req.effort, run_id=run_id)))
        run = ProposalRun(run_id=run_id, model=req.model, effort=req.effort,
                          engine=getattr(engine, "name", None) or "fake", n_items_proposed=1, cost_usd=0.01)
        f.provenance.proposal_runs.append(run)
        campaign.save(image_id, f, actor="claude", source="claude")
        emit({"phase": "validated", "detail": "1 item", "n_items": 1})
        emit({"phase": "done", "detail": "ok", "cost_usd": 0.01})
        return run

    if accept_run_id:
        async def run_propose(campaign, image_id, req, *, engine=None, on_event=None, run_id=None):
            return await _body(campaign, image_id, req, engine, on_event, run_id or "run-fake")
    else:
        async def run_propose(campaign, image_id, req, *, engine=None, on_event=None):  # type: ignore[misc]
            return await _body(campaign, image_id, req, engine, on_event, "run-minted-by-propose")

    def list_runs(campaign, image_id):
        f = campaign.load(image_id)
        return [r.model_dump(mode="json", exclude_none=True) for r in f.provenance.proposal_runs] if f else []

    def load_run(campaign, image_id, run_id):
        raise FileNotFoundError(f"no proposal file for {run_id}")

    mod = types.ModuleType("lensmark.claude.propose")
    mod.ProposeRequest, mod.run_propose, mod.list_runs, mod.load_run = ProposeRequest, run_propose, list_runs, load_run
    monkeypatch.setitem(sys.modules, "lensmark.claude.propose", mod)
    return mod


# ----------------------------------------------------------------------------- meta
def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"version", "campaign_dir", "engine", "claude_bin", "claude_version", "n_images"}
    assert d["version"] == __version__ and d["engine"] == "fixture" and d["n_images"] == 9
    assert d["campaign_dir"] == str(client.campaign_dir.resolve())


def test_models(client: TestClient):
    d = client.get("/api/models").json()
    assert [m["alias"] for m in d["models"]] == ["fable", "opus", "sonnet", "haiku"]
    assert all({"alias", "id", "label", "supports_effort", "price_in", "price_out"} <= set(m) for m in d["models"])
    assert d["efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert d["default"] == {"model": "opus", "effort": "xhigh"}


def test_style(client: TestClient):
    d = client.get("/api/style").json()
    assert d["palette"]["colors"]["magenta"] == "#FF00FF" and d["palette"]["deflector"] == "green"
    assert set(d["palette"]) >= {"colors", "arrow_order", "deflector", "reserved"}
    assert d["style_defaults"]["unit"] == "fraction_of_min_dim" and "arrow" in d["style_defaults"]


@pytest.mark.parametrize("name", SCHEMAS)
def test_schema(client: TestClient, name: str):
    r = client.get(f"/api/schema/{name}")
    assert r.status_code == 200
    d = r.json()
    assert d.get("type") == "object" and "properties" in d
    if name == "lensmark":
        assert "items" in d["properties"] and d.get("additionalProperties") is False


def test_schema_unknown(client: TestClient):
    r = client.get("/api/schema/nope")
    assert r.status_code == 404 and "error" in r.json()


def test_config(client: TestClient):
    d = client.get("/api/config").json()
    assert d["cutout_arcsec"] == 16.0 and d["cutout_arcsec_source"] == "assumed"
    assert d["overrides"]["deck-01"]["rank"] == 91 and d["default_model"] == "opus"


def test_root_is_html(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text.lower()


def test_no_store_everywhere(client: TestClient):
    for url in ("/", "/api/health", "/api/images", "/api/images/deck-01/original", "/api/images/deck-01/thumb",
                "/api/ann/deck-01", "/api/schema/nope", "/api/images/nope/original"):
        assert client.get(url).headers.get("cache-control") == "no-store", url


def test_error_shape(client: TestClient):
    r = client.get("/api/images/nope/original")
    assert r.status_code == 404
    assert set(r.json()) == {"error", "detail"}
    r = client.get("/api/ann/bad%20id")
    assert r.status_code in (400, 404) and "error" in r.json()


# ----------------------------------------------------------------------------- images
def test_images_list(client: TestClient):
    rows = client.get("/api/images").json()
    assert [r["id"] for r in rows] == [f"deck-{i:02d}" for i in range(1, 10)]
    row = rows[0]
    assert {"id", "file", "width", "height", "cutout_arcsec", "scale_source", "has_json", "has_annot", "annot_stale",
            "n_items", "by_status", "grade", "verdict", "theta_e_arcsec", "rank", "modified", "n_proposals"} <= set(row)
    assert row["width"] == row["height"] and row["rank"] == 91 and row["theta_e_arcsec"] == 1.5
    assert row["has_json"] is False and row["annot_stale"] is True


def test_original_bytes_and_etag(client: TestClient):
    r = client.get("/api/images/deck-01/original")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content.startswith(PNG_MAGIC)
    etag = r.headers["etag"].strip('"')
    assert etag == hashlib.sha256(r.content).hexdigest() == hashlib.sha256((client.campaign_dir / "deck-01.png").read_bytes()).hexdigest()
    assert client.get("/api/images/deck-01/original", headers={"If-None-Match": f'"{etag}"'}).status_code == 304


def test_thumb_is_jpeg(client: TestClient):
    r = client.get("/api/images/deck-03/thumb", params={"px": 64})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"
    from io import BytesIO
    from PIL import Image
    assert max(Image.open(BytesIO(r.content)).size) == 64


def test_annot_without_render_module_and_png(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "lensmark.render.draw", None)   # simulate the module being absent
    r = client.get("/api/images/deck-01/annot")
    assert r.status_code == 404 and "error" in r.json()
    # a stale PNG on disk is still served (flagged) when nothing can re-render it
    (client.campaign_dir / "deck-01.annot.png").write_bytes(PNG_MAGIC + b"stub")
    r = client.get("/api/images/deck-01/annot")
    assert r.status_code == 200 and r.content.startswith(PNG_MAGIC) and r.headers["x-lensmark-stale"] == "1"


# ----------------------------------------------------------------------------- annotations
def test_ann_round_trip(client: TestClient):
    r = client.get("/api/ann/deck-01")
    assert r.status_code == 200 and r.headers["x-lensmark-exists"] == "0"
    fresh = r.json()
    assert fresh["id"] == "deck-01" and fresh["items"] == [] and fresh["system"]["rank"] == 91
    assert fresh["image"]["cutout_arcsec"] == 16.0 and len(fresh["image"]["sha256"]) == 64
    assert not (client.campaign_dir / "deck-01.lensmark.json").exists()

    body = _arrow_body(client)
    body["system"]["description"] = "the cyan arrow marks a tight arc; the magenta one is missing"
    r = client.put("/api/ann/deck-01", json=body, headers={"X-LensMark-Actor": "xhuang"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True and d["modified"] and isinstance(d["lint"], list)
    assert any("magenta" in w for w in d["lint"])          # lint runs (colour word with no item)
    assert "render" in d

    r = client.get("/api/ann/deck-01")
    assert r.headers["x-lensmark-exists"] == "1"
    saved = r.json()
    assert saved["items"][0]["id"] == "ann-arrow-001" and saved["items"][0]["label"] == "tight arc"
    assert saved["items"][0]["status"] == "accepted" and saved["modified"] == d["modified"]

    log = client.get("/api/ann/deck-01/log").json()
    ops = [(e["op"], e["item_id"]) for e in log]
    assert ("create", "$file") in ops and ("add", "ann-arrow-001") in ops
    assert all(e["actor"] == "xhuang" and e["source"] == "ui" for e in log)
    assert client.get("/api/images").json()[0]["has_json"] is True


def test_put_extra_key_422(client: TestClient):
    body = _arrow_body(client, "deck-02")
    body["surprise"] = 1
    r = client.put("/api/ann/deck-02", json=body)
    assert r.status_code == 422
    d = r.json()
    assert "error" in d and isinstance(d["detail"], list) and any("surprise" in str(e.get("loc")) for e in d["detail"])
    body = _arrow_body(client, "deck-02", nonsense=True)   # extra key inside an item
    assert client.put("/api/ann/deck-02", json=body).status_code == 422
    assert client.put("/api/ann/deck-02", json=[1, 2]).status_code == 422
    assert not (client.campaign_dir / "deck-02.lensmark.json").exists()


def test_put_id_mismatch_400(client: TestClient):
    body = _arrow_body(client, "deck-03")
    r = client.put("/api/ann/deck-04", json=body)
    assert r.status_code == 400 and "error" in r.json()
    assert not (client.campaign_dir / "deck-04.lensmark.json").exists()


def test_put_unknown_image_404(client: TestClient):
    body = _arrow_body(client, "deck-03")
    body["id"] = "nope"
    assert client.put("/api/ann/nope", json=body).status_code == 404


# ----------------------------------------------------------------------------- propose plumbing (fake module)
def test_propose_flow_with_fake_module(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_propose(monkeypatch)
    r = client.post("/api/propose/deck-01", json={"model": "sonnet", "effort": "low", "budget": 0.1})
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    assert run_id.startswith("run-")

    events = _stream_phases(client, "deck-01", run_id)
    phases = [e["phase"] for e in events]
    assert phases[0] == "queued" and phases[-1] == "done"
    assert "started" in phases and "validated" in phases
    assert all("detail" in e for e in events)
    done = events[-1]
    assert done["run"]["run_id"] == run_id and done["run"]["model"] == "sonnet" and done["cost_usd"] == 0.01
    assert done["n_items"] == 1

    # replay: a second consumer gets the full history again and terminates
    assert [e["phase"] for e in _stream_phases(client, "deck-01", run_id)] == phases

    saved = client.get("/api/ann/deck-01").json()
    assert [it["status"] for it in saved["items"]] == ["proposed"]
    assert saved["items"][0]["created_by"] == {"kind": "claude", "model": "sonnet", "effort": "low", "run_id": run_id}
    assert client.get("/api/images").json()[0]["by_status"] == {"proposed": 1}

    runs = client.get("/api/proposals/deck-01").json()
    assert [x["run_id"] for x in runs] == [run_id] and runs[0]["n_items_proposed"] == 1
    assert client.get("/api/proposals/deck-01/nope").status_code == 404

    snap = client.get(f"/api/propose/deck-01/{run_id}").json()
    assert snap["finished"] is True and snap["phase"] == "done" and snap["propose_run_id"] == run_id
    # cancelling a finished run is a no-op
    r = client.post(f"/api/propose/deck-01/{run_id}/cancel")
    assert r.status_code == 200 and r.json() == {"ok": True, "cancelled": False, "phase": "done"}


def test_propose_defaults_from_config(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_propose(monkeypatch, accept_run_id=False)   # run_propose without a run_id kwarg
    run_id = client.post("/api/propose/deck-05").json()["run_id"]
    events = _stream_phases(client, "deck-05", run_id)
    assert events[-1]["phase"] == "done"
    assert events[1]["detail"].startswith("opus/xhigh")            # campaign default model/effort
    assert events[-1]["run"]["run_id"] == "run-minted-by-propose"
    # both ids resolve in the registry
    assert client.get("/api/propose/deck-05/run-minted-by-propose").json()["run_id"] == run_id
    assert client.get("/api/propose/deck-05/run-minted-by-propose/events").status_code == 200


def test_propose_cancel(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_propose(monkeypatch, delay=30.0)
    run_id = client.post("/api/propose/deck-02", json={}).json()["run_id"]
    r = client.post(f"/api/propose/deck-02/{run_id}/cancel")
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["cancelled"] is True
    events = _stream_phases(client, "deck-02", run_id)
    assert events[-1]["phase"] == "error" and "cancelled" in events[-1]["detail"]
    assert client.get("/api/ann/deck-02").headers["x-lensmark-exists"] == "0"   # nothing was written
    assert client.post("/api/propose/deck-02/run-unknown/cancel").status_code == 404
    assert client.get("/api/propose/deck-01/run-unknown/events").status_code == 404
    assert client.get(f"/api/propose/deck-01/{run_id}/events").status_code == 404   # wrong image


def test_propose_failure_becomes_error_event(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_propose(monkeypatch, fail=True)
    run_id = client.post("/api/propose/deck-03").json()["run_id"]
    events = _stream_phases(client, "deck-03", run_id)
    assert events[-1]["phase"] == "error" and "boom" in events[-1]["detail"]


def test_propose_bad_request_422(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    _fake_propose(monkeypatch)
    r = client.post("/api/propose/deck-01", json={"budget": "lots"})
    assert r.status_code in (400, 422) and "error" in r.json()
    assert client.post("/api/propose/nope", json={}).status_code == 404


def test_propose_501_when_module_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "lensmark.claude.propose", None)
    r = client.post("/api/propose/deck-01", json={})
    assert r.status_code == 501
    assert r.json()["error"] == "module not available: lensmark.claude.propose"


# ----------------------------------------------------------------------------- SSE framing
def test_sse_format_and_keepalive():
    assert format_event({"phase": "done", "detail": "θ_E"}) == 'event: message\ndata: {"phase": "done", "detail": "θ_E"}\n\n'.encode()

    async def slow():
        yield {"phase": "queued", "detail": ""}
        await asyncio.sleep(0.35)
        yield {"phase": "done", "detail": ""}

    app = FastAPI()

    @app.get("/s")
    async def s():
        return sse_response(slow(), keepalive_s=0.1)

    with TestClient(app) as c, c.stream("GET", "/s") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert text.count(": keepalive\n\n") >= 2
    assert [e["phase"] for e in parse_events(text)] == ["queued", "done"]


# ----------------------------------------------------------------------------- serve()
def test_serve_sets_engine_and_runs_uvicorn(nine: Path, monkeypatch: pytest.MonkeyPatch):
    calls: dict = {}
    monkeypatch.delenv("LENSMARK_ENGINE", raising=False)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.update(app=app, **kw))
    assert serve(nine, port=8999, bind="127.0.0.1", engine="fixture", open_browser=False) == 0
    assert calls["host"] == "127.0.0.1" and calls["port"] == 8999 and calls["log_level"] == "info"
    assert calls["app"].state.engine_name == "fixture" and calls["app"].state.campaign.root == nine.resolve()
    import os
    assert os.environ["LENSMARK_ENGINE"] == "fixture"
    assert serve(nine / "missing", open_browser=False) == 2


def test_create_app_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "nope")


# ----------------------------------------------------------------------------- sibling modules (skip until they land)
def test_render_flow(client: TestClient):
    pytest.importorskip("lensmark.render.draw")
    r = client.put("/api/ann/deck-06", json=_arrow_body(client, "deck-06"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["render"] is not None and d["render"]["output"] == "deck-06.annot.png" and "render_error" not in d
    assert (client.campaign_dir / "deck-06.annot.png").exists()
    r = client.post("/api/render/deck-06", json={})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["output"] == "deck-06.annot.png" and len(d["sha256"]) == 64 and d["stale"] is False
    assert d["sha256"] == hashlib.sha256((client.campaign_dir / "deck-06.annot.png").read_bytes()).hexdigest()
    r = client.get("/api/images/deck-06/annot")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content.startswith(PNG_MAGIC)
    row = next(x for x in client.get("/api/images").json() if x["id"] == "deck-06")
    assert row["has_annot"] is True and row["annot_stale"] is False
    # an unsaved image renders on demand too (no PNG on disk)
    r = client.get("/api/images/deck-07/annot")
    assert r.status_code == 200 and r.content.startswith(PNG_MAGIC)


def test_propose_flow_real_module(client: TestClient):
    pytest.importorskip("lensmark.claude.propose")
    pytest.importorskip("lensmark.claude.engine")
    if not list((FIXTURES / "proposals").glob("deck-01*")):
        pytest.skip("no fixture proposal for deck-01 yet")
    r = client.post("/api/propose/deck-01", json={"model": "opus", "effort": "xhigh", "engine": "fixture"})
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    events = _stream_phases(client, "deck-01", run_id)
    phases = [e["phase"] for e in events]
    assert phases[0] == "queued" and phases[-1] == "done", phases
    assert events[-1].get("run", {}).get("model")
    saved = client.get("/api/ann/deck-01").json()
    assert any(it["status"] == "proposed" for it in saved["items"])
    assert saved["provenance"]["proposal_runs"]
    runs = client.get("/api/proposals/deck-01").json()
    assert len(runs) >= 1 and all("run_id" in x for x in runs)
    pr = runs[-1]["run_id"]
    r = client.get(f"/api/proposals/deck-01/{pr}")
    assert r.status_code == 200 and isinstance(r.json(), dict)
    assert client.post(f"/api/propose/deck-01/{run_id}/cancel").json()["ok"] is True


def test_critique_and_eval_flow(client: TestClient):
    pytest.importorskip("lensmark.critique")
    pytest.importorskip("lensmark.evaluate")
    body = _arrow_body(client, "deck-08", status="proposed",
                       created_by={"kind": "claude", "model": "claude-opus-5", "effort": "xhigh", "run_id": "run-test"})
    body["provenance"]["proposal_runs"] = [{"run_id": "run-test", "model": "claude-opus-5", "effort": "xhigh",
                                            "engine": "fixture", "n_items_proposed": 1}]
    assert client.put("/api/ann/deck-08", json=body).status_code == 200
    crit = {"image_id": "deck-08", "run_id": "run-test", "model": "claude-opus-5", "effort": "xhigh", "reviewer": "xhuang",
            "items": [{"item_id": "ann-arrow-001", "verdict": "correct"}],
            "panel": {"completeness": 4, "would_use_as_fewshot": True}, "lead_time_s": 12.5}
    r = client.post("/api/critique/deck-08", json=crit)
    assert r.status_code == 200, r.text
    assert r.json()["file"].startswith("critiques/")
    assert (client.campaign_dir / r.json()["file"]).exists()
    saved = client.get("/api/ann/deck-08").json()
    assert saved["items"][0].get("review", {}).get("verdict") == "correct"
    assert saved["provenance"]["critiques"]
    r = client.get("/api/eval", params={"by": "model,effort"})
    assert r.status_code == 200 and isinstance(r.json()["rows"], list)
    # validation: bad verdict -> 422, wrong id -> 400
    bad = dict(crit, items=[{"item_id": "ann-arrow-001", "verdict": "meh"}])
    assert client.post("/api/critique/deck-08", json=bad).status_code == 422
    assert client.post("/api/critique/deck-09", json=crit).status_code == 400


def test_export_flow(client: TestClient):
    pytest.importorskip("lensmark.exports")
    assert client.put("/api/ann/deck-09", json=_arrow_body(client, "deck-09")).status_code == 200
    for fmt in ("coco", "ds9"):
        r = client.post(f"/api/export/{fmt}", json={"ids": ["deck-09"]})
        assert r.status_code == 200, (fmt, r.text)
        files = r.json()["files"]
        assert isinstance(files, list) and all((client.campaign_dir / f).exists() for f in files)
    assert client.post("/api/export/nope", json={}).status_code == 404
    assert client.post("/api/export/coco", json={"ids": "deck-09"}).status_code == 400


def test_patch_apply_flow(client: TestClient):
    pytest.importorskip("lensmark.voice.patch")
    ops = [{"op": "add", "item": {"type": "mask_circle", "center": [0.2, 0.2], "radius_arcsec": 1.0, "kind": "galaxy"},
            "confidence": 0.9, "rationale": "upper-left galaxy"}]
    r = client.post("/api/patch/deck-04/apply", json={"ops": ops, "transcript": "circle the galaxy at upper left"})
    assert r.status_code == 200, r.text
    d = r.json()
    masks = [it for it in d["items"] if it["type"] == "mask_circle"]
    assert masks and masks[0]["kind"] == "galaxy" and masks[0]["center"][0] < 0.5
    log = client.get("/api/ann/deck-04/log").json()
    assert any(e.get("source") == "voice" for e in log)
    assert client.post("/api/patch/deck-04/apply", json={"ops": "nope"}).status_code == 400


def test_patch_dry_run_flow(client: TestClient):
    pytest.importorskip("lensmark.voice.patch")
    pytest.importorskip("lensmark.claude.engine")
    if not any((FIXTURES / "patches").iterdir()):
        pytest.skip("no fixture patch yet")
    assert client.post("/api/patch/deck-04", json={"transcript": ""}).status_code == 400
    r = client.post("/api/patch/deck-04", json={"transcript": "put a dashed circle around the galaxy at upper left"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "ops" in d and isinstance(d["ops"], list) and d.get("schema_version", "").startswith("lensmark-patch")
    assert client.get("/api/ann/deck-04").headers["x-lensmark-exists"] == "0"   # dry run: nothing applied


def test_stt(client: TestClient):
    r = client.post("/api/stt", files={"audio": ("clip.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")})
    assert r.status_code in (200, 501), r.text
    if r.status_code == 501:
        assert "error" in r.json()
    else:
        assert {"transcript", "backend"} <= set(r.json())
