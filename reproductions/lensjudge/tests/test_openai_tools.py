#!/usr/bin/env python3
"""No-API unit tests for the v4 Phase-3 open-weight agentic path (tools/openai_tools + grader_lean).

Pure-logic, mocked transport/render — NO network, NO GPU, NO API spend. Run under the lensjudge venv:
    PYTHONPATH=reproductions python reproductions/lensjudge/tests/test_openai_tools.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lensjudge.tools import openai_tools  # noqa: E402
from lensjudge.common import fetch, render, llm_client  # noqa: E402
from lensjudge.imaging import grader_lean  # noqa: E402


# ---------------------------------------------------------------- tool schemas
def test_tool_schemas():
    s = openai_tools.tool_schemas(["fetch_cutout", "get_photometry", "bogus"])
    names = [t["function"]["name"] for t in s]
    assert names == ["fetch_cutout", "get_photometry"]      # bogus skipped
    assert all(t["type"] == "function" for t in s)
    assert "name" in s[0]["function"]["parameters"]["properties"]


# ---------------------------------------------------------------- executor
def test_execute_fetch_cutout(monkeypatch=None):
    o, orv, ob, ord_ = (fetch.get_cube, render.render_views, render.png_b64, render.residual_view_desc)
    fetch.get_cube = lambda **k: object()
    render.render_views = lambda cube, views: {v: object() for v in views}
    render.png_b64 = lambda img: "B64"
    render.residual_view_desc = lambda: "resid-desc"
    try:
        text, images = asyncio.run(openai_tools.execute_tool(
            "fetch_cutout", {"name": "DESI-1", "survey": "storfer", "views": ["full", "residual"]}))
        assert "[full]" in text and "[residual]" in text and "resid-desc" in text
        assert len(images) == 2
        assert images[0]["type"] == "image" and images[0]["source"]["data"] == "B64"
    finally:
        fetch.get_cube, render.render_views, render.png_b64, render.residual_view_desc = o, orv, ob, ord_


def test_execute_fetch_cutout_no_cube():
    o = fetch.get_cube
    fetch.get_cube = lambda **k: None
    try:
        text, images = asyncio.run(openai_tools.execute_tool("fetch_cutout", {"name": "X"}))
        assert text.startswith("ERROR") and images == []
    finally:
        fetch.get_cube = o


def test_execute_photometry():
    o, oap = fetch.get_cube, openai_tools._aperture_colors
    fetch.get_cube = lambda **k: object()
    openai_tools._aperture_colors = lambda cube: {"core": {"g_minus_r": 0.9}, "annulus_bluer_than_core": True}
    try:
        text, images = asyncio.run(openai_tools.execute_tool("get_photometry", {"name": "X"}))
        assert images == [] and '"annulus_bluer_than_core": true' in text
    finally:
        fetch.get_cube, openai_tools._aperture_colors = o, oap


def test_execute_unknown():
    text, images = asyncio.run(openai_tools.execute_tool("nope", {}))
    assert text.startswith("ERROR") and images == []


# --------------------------------------------------- fake OpenAI client (tool loop)
class _U:
    prompt_tokens = 10
    completion_tokens = 5


class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _TC:
    def __init__(self, tid, name, args):
        self.id, self.function = tid, _Fn(name, args)

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Resp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg, "finish_reason": "stop"})()]
        self.usage = _U()


class _Comp:
    def __init__(self, responses):
        self._r = list(responses); self.calls = []

    async def create(self, **kw):
        self.calls.append(kw); return self._r.pop(0)


class _Fake:
    def __init__(self, responses):
        self.chat = type("X", (), {"completions": _Comp(responses)})()


_GRADE = ('{"grade":"A","criteria":{"blue_source":7,"low_surface_brightness":6,"curvature":7,'
          '"counter_images":4,"arc_morphology":7},"p_lens":0.78,"confidence":0.6,'
          '"contaminant":null,"escalate_to_human":false,"rationale":"blue arc around red LRG"}')


def test_grade_openai_agentic_end_to_end():
    """grader_lean routes to the OpenAI tool loop, executes a tool call, parses the final grade."""
    saved = {k: os.environ.get(k) for k in ("LENSJUDGE_BACKEND", "LENSJUDGE_BASE_URL", "LENSJUDGE_MODEL_GRADER")}
    og = llm_client._get_client
    oexec = openai_tools.execute_tool
    os.environ["LENSJUDGE_BACKEND"] = "openai"
    os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
    os.environ["LENSJUDGE_MODEL_GRADER"] = "lensjudge-qwen8b-distilled"
    # turn 1: model calls fetch_cutout; turn 2: model returns the final ImageGrade JSON
    tc = _TC("c1", "fetch_cutout", '{"name":"DESI-1","survey":"storfer"}')
    llm_client._get_client = lambda: _Fake([_Resp(_Msg("", [tc])), _Resp(_Msg(_GRADE))])

    async def _fake_exec(name, args):
        return ("cutout: 3 views", [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "PX"}}])
    openai_tools.execute_tool = _fake_exec
    try:
        res = asyncio.run(grader_lean.grade_candidate(
            {"name": "DESI-1", "survey_key": "storfer", "ra": 1.0, "dec": 2.0}))
        assert res.parse_ok and res.grade.grade == "A" and abs(res.grade.p_lens - 0.78) < 1e-6
        assert res.meta["backend"] == "openai" and res.meta["mode"] == "lean_openai"
        assert res.meta["tool_calls"] == 1 and res.num_turns == 2
    finally:
        llm_client._get_client = og
        openai_tools.execute_tool = oexec
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
