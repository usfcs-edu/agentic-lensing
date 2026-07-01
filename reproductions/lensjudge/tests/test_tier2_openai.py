#!/usr/bin/env python3
"""No-API unit tests for the v4 tier-2 open-weight port (HSC/Euclid).

Covers: the HSC/Euclid OpenAI tool schemas + executor branches (tools/openai_tools), the
open-weight backend branch in eval/run_hsc + eval/run_euclid, and the credential-free warm-cache
path (common/hsc_fetch + common/highres) that lets the offline GPU host grade staged HSC cutouts.

Pure-logic, mocked transport/render/loaders — NO network, NO GPU, NO API spend, NO HSC creds.
    PYTHONPATH=reproductions python reproductions/lensjudge/tests/test_tier2_openai.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lensjudge.tools import openai_tools  # noqa: E402
from lensjudge.common import render, llm_client, hsc_fetch, highres  # noqa: E402
import lensjudge.common.hsc as hsc_mod  # noqa: E402
import lensjudge.common.euclid as euclid_mod  # noqa: E402
from lensjudge.eval import run_hsc, run_euclid  # noqa: E402


# ---------------------------------------------------------------- tool schemas
def test_tier2_tool_schemas():
    s = openai_tools.tool_schemas(["fetch_hsc_cutout", "fetch_euclid_cutout"])
    names = [t["function"]["name"] for t in s]
    assert names == ["fetch_hsc_cutout", "fetch_euclid_cutout"]
    hsc = next(t for t in s if t["function"]["name"] == "fetch_hsc_cutout")
    assert set(hsc["function"]["parameters"]["required"]) == {"ra", "dec"}
    euc = next(t for t in s if t["function"]["name"] == "fetch_euclid_cutout")
    assert euc["function"]["parameters"]["required"] == ["id_str"]


# ---------------------------------------------------------------- HSC executor
def test_execute_hsc_cutout():
    o_load, o_render, o_b64 = hsc_mod.load_hsc, hsc_mod.render_hsc_views, render.png_b64
    hsc_mod.load_hsc = lambda ra, dec: {"i": object(), "r": object(), "g": object()}
    hsc_mod.render_hsc_views = lambda bands, views: {v: object() for v in views}
    render.png_b64 = lambda img: "B64"
    try:
        text, images = asyncio.run(openai_tools.execute_tool(
            "fetch_hsc_cutout", {"ra": 0.0788, "dec": 0.2716}))
        assert "HSC-SSP PDR3" in text and "[lum_sub]" in text
        assert len(images) == 4  # default full, lum, zoom, lum_sub
        assert images[0]["type"] == "image" and images[0]["source"]["data"] == "B64"
    finally:
        hsc_mod.load_hsc, hsc_mod.render_hsc_views, render.png_b64 = o_load, o_render, o_b64


def test_execute_hsc_no_coverage():
    o = hsc_mod.load_hsc
    hsc_mod.load_hsc = lambda ra, dec: None
    try:
        text, images = asyncio.run(openai_tools.execute_tool(
            "fetch_hsc_cutout", {"ra": 1.0, "dec": 2.0}))
        assert text.startswith("ERROR") and images == []
    finally:
        hsc_mod.load_hsc = o


# ------------------------------------------------------------- Euclid executor
def test_execute_euclid_cutout():
    o_load, o_render, o_b64 = euclid_mod.load_euclid, euclid_mod.render_euclid_views, render.png_b64
    euclid_mod.load_euclid = lambda idd: {"VIS_FLUX": object(), "NIR_H_FLUX": object()}
    euclid_mod.render_euclid_views = lambda bands, views: {v: object() for v in views}
    render.png_b64 = lambda img: "B64"
    try:
        text, images = asyncio.run(openai_tools.execute_tool(
            "fetch_euclid_cutout", {"id_str": "102_NEG509"}))
        assert "Euclid Q1 candidate 102_NEG509" in text and "[vis_sub]" in text
        assert len(images) == 4  # default full, vis, zoom, vis_sub
        assert images[0]["source"]["data"] == "B64"
    finally:
        euclid_mod.load_euclid, euclid_mod.render_euclid_views, render.png_b64 = o_load, o_render, o_b64


def test_execute_euclid_no_data():
    o = euclid_mod.load_euclid
    euclid_mod.load_euclid = lambda idd: None
    try:
        text, images = asyncio.run(openai_tools.execute_tool(
            "fetch_euclid_cutout", {"id_str": "missing"}))
        assert text.startswith("ERROR") and images == []
    finally:
        euclid_mod.load_euclid = o


def test_load_euclid_corrupt_fits_is_none():
    """A corrupt/empty on-disk FITS must degrade to None, not raise (data staging is imperfect)."""
    o_root = euclid_mod.EUCLID_ROOT
    tmp = Path(tempfile.mkdtemp())
    idd = "999_CORRUPT"
    d = tmp / "lens" / idd
    d.mkdir(parents=True)
    (d / f"{idd}.fits").write_bytes(b"not a fits file at all")
    euclid_mod.EUCLID_ROOT = tmp
    try:
        assert euclid_mod.load_euclid(idd) is None          # no raise
        # and the executor turns it into a recoverable ERROR observation
        text, images = asyncio.run(openai_tools.execute_tool("fetch_euclid_cutout", {"id_str": idd}))
        assert text.startswith("ERROR") and images == []
    finally:
        euclid_mod.EUCLID_ROOT = o_root


def test_execute_euclid_missing_id_str():
    """A tool call that omits id_str must return a recoverable ERROR, not raise (small VLMs drop args)."""
    for args in ({}, {"id_str": ""}, {"views": ["full"]}):
        text, images = asyncio.run(openai_tools.execute_tool("fetch_euclid_cutout", args))
        assert text.startswith("ERROR") and "id_str" in text and images == []


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


_GRADE = ('{"grade":"A","criteria":{"blue_source":8,"low_surface_brightness":7,"curvature":8,'
          '"counter_images":5,"arc_morphology":8},"p_lens":0.86,"confidence":0.7,'
          '"contaminant":null,"escalate_to_human":false,"rationale":"resolved tangential arc"}')


def _with_openai_backend(fn):
    """Run fn() with LENSJUDGE_BACKEND=openai + endpoint env set, restoring afterward."""
    keys = ("LENSJUDGE_BACKEND", "LENSJUDGE_BASE_URL", "LENSJUDGE_MODEL_GRADER")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["LENSJUDGE_BACKEND"] = "openai"
    os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
    os.environ["LENSJUDGE_MODEL_GRADER"] = "lensjudge-qwen8b"
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _fake_exec_factory():
    async def _fake_exec(name, args):
        return ("hi-res: 4 views", [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "PX"}}])
    return _fake_exec


def test_grade_hsc_openai_end_to_end():
    og, oexec = llm_client._get_client, openai_tools.execute_tool
    tc = _TC("c1", "fetch_hsc_cutout", '{"ra":0.0788,"dec":0.2716}')
    llm_client._get_client = lambda: _Fake([_Resp(_Msg("", [tc])), _Resp(_Msg(_GRADE))])
    openai_tools.execute_tool = _fake_exec_factory()
    try:
        res = _with_openai_backend(lambda: asyncio.run(
            run_hsc.grade_hsc({"ra": 0.0788, "dec": 0.2716, "name": "DESI-X"})))
        assert res["agent_grade"] == "A" and abs(res["p_lens"] - 0.86) < 1e-6
        assert res["backend"] == "openai" and res["tool_calls"] == 1
        assert res["name"] == "DESI-X"  # obj fields preserved
    finally:
        llm_client._get_client, openai_tools.execute_tool = og, oexec


def test_run_tool_loop_survives_tool_crash():
    """A crashing executor must become a recoverable ERROR observation, not abort the loop."""
    og = llm_client._get_client
    tc = _TC("c1", "fetch_euclid_cutout", '{"id_str":"x"}')
    llm_client._get_client = lambda: _Fake([_Resp(_Msg("", [tc])), _Resp(_Msg(_GRADE))])

    async def _boom(name, args):
        raise RuntimeError("kaboom")
    try:
        res = asyncio.run(llm_client.run_tool_loop(
            system="s", user_content=[{"type": "text", "text": "go"}],
            tools=openai_tools.tool_schemas(["fetch_euclid_cutout"]),
            execute_tool=_boom, model="m", max_turns=4))
        assert res.text == _GRADE and res.num_turns == 2 and res.tool_calls == 1
    finally:
        llm_client._get_client = og


def test_grade_euclid_openai_end_to_end():
    og, oexec = llm_client._get_client, openai_tools.execute_tool
    tc = _TC("c1", "fetch_euclid_cutout", '{"id_str":"102_NEG509"}')
    llm_client._get_client = lambda: _Fake([_Resp(_Msg("", [tc])), _Resp(_Msg(_GRADE))])
    openai_tools.execute_tool = _fake_exec_factory()
    try:
        res = _with_openai_backend(lambda: asyncio.run(
            run_euclid.grade_euclid({"id_str": "102_NEG509", "grade": "A"})))
        assert res["agent_grade"] == "A" and abs(res["p_lens"] - 0.86) < 1e-6
        assert res["backend"] == "openai" and res["tool_calls"] == 1
        assert res["id_str"] == "102_NEG509"
    finally:
        llm_client._get_client, openai_tools.execute_tool = og, oexec


# --------------------------------------------------- offline warm-cache (no creds/network)
def _write_fake_hsc(cache_root: Path, ra: float, dec: float, bands=("i", "r")):
    d = cache_root / "pdr3_wide" / hsc_fetch._key(ra, dec)
    d.mkdir(parents=True, exist_ok=True)
    for b in bands:
        fits.PrimaryHDU(np.ones((8, 8), np.float32)).writeto(d / f"{b}.fits", overwrite=True)


def _no_hsc_creds(fn):
    saved = {k: os.environ.pop(k, None) for k in ("HSC_USER", "HSC_PASSWORD")}
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_hsc_warm_cache_serves_without_creds():
    """A staged cache must serve credential-free (the decoupled offline grade host)."""
    o_cache = hsc_fetch.HSC_CACHE
    tmp = Path(tempfile.mkdtemp())
    hsc_fetch.HSC_CACHE = tmp
    ra, dec = 10.5, -3.25
    _write_fake_hsc(tmp, ra, dec)
    try:
        def _check():
            assert hsc_fetch.have_credentials() is False
            assert hsc_fetch.cached(ra, dec) is True
            bands = hsc_fetch.fetch_hsc_cutout(ra, dec)          # no auth, no network
            assert bands is not None and "i" in bands and bands["i"].shape == (8, 8)
        _no_hsc_creds(_check)
    finally:
        hsc_fetch.HSC_CACHE = o_cache


def test_hsc_no_cache_no_creds_is_none():
    """No creds AND no cache -> None (the existing safe no-op is preserved)."""
    o_cache = hsc_fetch.HSC_CACHE
    hsc_fetch.HSC_CACHE = Path(tempfile.mkdtemp())    # empty
    try:
        def _check():
            assert hsc_fetch.cached(1.0, 2.0) is False
            assert hsc_fetch.fetch_hsc_cutout(1.0, 2.0) is None
        _no_hsc_creds(_check)
    finally:
        hsc_fetch.HSC_CACHE = o_cache


def test_highres_resolves_hsc_from_warm_cache():
    """highres._resolve_hsc must recognize warm-cache coverage without credentials."""
    o_cache = hsc_fetch.HSC_CACHE
    tmp = Path(tempfile.mkdtemp())
    hsc_fetch.HSC_CACHE = tmp
    ra, dec = 42.0, 12.0
    _write_fake_hsc(tmp, ra, dec)
    try:
        def _check():
            hit = highres._resolve_hsc("DESI-Y", ra, dec)
            assert hit is not None and hit["survey"] == "hsc"
            assert abs(hit["ra"] - ra) < 1e-9 and abs(hit["dec"] - dec) < 1e-9
            # and no coverage where nothing is staged
            assert highres._resolve_hsc("DESI-Z", 99.0, 1.0) is None
        _no_hsc_creds(_check)
    finally:
        hsc_fetch.HSC_CACHE = o_cache


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
