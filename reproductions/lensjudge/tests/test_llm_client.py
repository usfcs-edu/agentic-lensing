#!/usr/bin/env python3
"""No-API unit tests for the lensjudge-v4 open-weight backend (common/llm_client.py).

Pure-logic + mocked-transport only — NO network, NO API spend, NO `openai` install required
(the client is mocked). Runs under the lensjudge venv via pytest, or directly:
    PYTHONPATH=reproductions python reproductions/lensjudge/tests/test_llm_client.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lensjudge.common import llm_client  # noqa: E402


# --------------------------------------------------------------- env helpers
_ENV_KEYS = ("LENSJUDGE_BACKEND", "LENSJUDGE_BASE_URL", "LENSJUDGE_API_KEY",
             "LENSJUDGE_TEMPERATURE", "LENSJUDGE_JSON_MODE", "LENSJUDGE_GUIDED_JSON",
             "LENSJUDGE_STRUCTURED", "LENSJUDGE_LOGPROBS", "LENSJUDGE_STRICT_TOOLS",
             "LENSJUDGE_NOTHINK", "LENSJUDGE_MAX_TOKENS")


def _save_env():
    llm_client._STRUCTURED_RESOLVED = None   # reset the auto-probe cache between tests
    return {k: os.environ.pop(k, None) for k in _ENV_KEYS}


def _restore(saved):
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# --------------------------------------------------------- fake OpenAI client
class _Usage:
    def __init__(self, pin=11, pout=7):
        self.prompt_tokens, self.completion_tokens = pin, pout


class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, tid, name, arguments):
        self.id, self.function = tid, _Fn(name, arguments)

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _LPTop:
    def __init__(self, token, logprob):
        self.token, self.logprob = token, logprob


class _LPTok:
    def __init__(self, token, logprob=-0.05, top=None):
        self.token, self.logprob, self.top_logprobs = token, logprob, top or []


class _LP:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, msg, finish_reason="stop", logprobs=None):
        self.message, self.finish_reason, self.logprobs = msg, finish_reason, logprobs


class _Resp:
    def __init__(self, msg, usage=None, finish_reason="stop", logprobs=None):
        self.choices = [_Choice(msg, finish_reason, logprobs)]
        self.usage = usage or _Usage()


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.chat = type("C", (), {"completions": _FakeCompletions(responses)})()


def _patch_client(responses):
    """Override llm_client._get_client to return a fake; returns the fake for assertions."""
    fake = _FakeClient(responses)
    llm_client._get_client = lambda: fake  # type: ignore[assignment]
    return fake


# ------------------------------------------------------------ backend switch
def test_get_backend_default_and_switch():
    saved = _save_env()
    try:
        assert llm_client.get_backend() == "anthropic"   # default, no env
        assert llm_client.is_open() is False
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        assert llm_client.get_backend() == "openai"
        assert llm_client.is_open() is True
        os.environ["LENSJUDGE_BACKEND"] = "OpenAI"        # case-insensitive
        assert llm_client.get_backend() == "openai"
        os.environ["LENSJUDGE_BACKEND"] = "bogus"
        try:
            llm_client.get_backend()
            raise AssertionError("expected ValueError for bogus backend")
        except ValueError:
            pass
    finally:
        _restore(saved)


def test_base_url_required():
    saved = _save_env()
    # ensure a clean client cache + no base url -> clear error
    llm_client._clients.clear()
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        try:
            llm_client._get_client()
            raise AssertionError("expected RuntimeError without LENSJUDGE_BASE_URL")
        except RuntimeError as e:
            assert "LENSJUDGE_BASE_URL" in str(e)
    finally:
        _restore(saved)


# ----------------------------------------------------- content-block convert
def test_anthropic_content_to_openai():
    blocks = [
        {"type": "text", "text": "grade this"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": "AAAB"}},
        {"type": "image", "source": {"type": "url", "url": "http://x/y.png"}},
    ]
    out = llm_client.anthropic_content_to_openai(blocks)
    assert out[0] == {"type": "text", "text": "grade this"}
    assert out[1]["type"] == "image_url"
    assert out[1]["image_url"]["url"] == "data:image/png;base64,AAAB"
    assert out[2]["image_url"]["url"] == "http://x/y.png"


# ----------------------------------------------------------- chat_with_images
def test_chat_with_images_request_shape():
    saved = _save_env()
    orig = llm_client._get_client
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        fake = _patch_client([_Resp(_Msg(content='{"grade":"A"}'), _Usage(20, 5))])
        content = [{"type": "text", "text": "hi"},
                   {"type": "image", "source": {"type": "base64",
                                                "media_type": "image/png", "data": "ZZ"}}]
        res = asyncio.run(llm_client.chat_with_images(
            system="RUBRIC", content=content, model="Qwen/Qwen3-VL-8B-Instruct",
            max_tokens=2048))
        assert res.text == '{"grade":"A"}'
        assert res.input_tokens == 20 and res.output_tokens == 5
        assert res.cost_usd == 0.0          # local inference, no price entry
        assert res.model == "Qwen/Qwen3-VL-8B-Instruct"
        req = fake.chat.completions.calls[0]
        assert req["model"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert req["messages"][0] == {"role": "system", "content": "RUBRIC"}
        user = req["messages"][1]
        assert user["role"] == "user"
        assert user["content"][1]["type"] == "image_url"
        assert req["temperature"] == 0.0    # deterministic grading default
    finally:
        llm_client._get_client = orig
        _restore(saved)


def test_chat_with_images_guided_json_optin():
    saved = _save_env()
    orig = llm_client._get_client
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        os.environ["LENSJUDGE_GUIDED_JSON"] = "1"
        os.environ["LENSJUDGE_JSON_MODE"] = "1"
        fake = _patch_client([_Resp(_Msg(content="{}"))])
        schema = {"type": "object", "properties": {"grade": {"type": "string"}}}
        asyncio.run(llm_client.chat_with_images(
            system="S", content=[{"type": "text", "text": "x"}],
            model="m", json_schema=schema))
        req = fake.chat.completions.calls[0]
        assert req["response_format"] == {"type": "json_object"}
        assert req["extra_body"] == {"guided_json": schema}
    finally:
        llm_client._get_client = orig
        _restore(saved)


def test_chat_with_images_json_kwargs_off_by_default():
    saved = _save_env()
    orig = llm_client._get_client
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        fake = _patch_client([_Resp(_Msg(content="{}"))])
        asyncio.run(llm_client.chat_with_images(
            system="S", content=[{"type": "text", "text": "x"}], model="m",
            json_schema={"type": "object"}))
        req = fake.chat.completions.calls[0]
        assert "response_format" not in req and "extra_body" not in req
    finally:
        llm_client._get_client = orig
        _restore(saved)


# --------------------------------------------------------------- tool loop
def test_run_tool_loop_executes_and_injects_images():
    saved = _save_env()
    orig = llm_client._get_client
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        # turn 1: model asks for fetch_cutout; turn 2: model returns final JSON
        tcall = _ToolCall("call_1", "fetch_cutout", '{"name":"DESI-1"}')
        fake = _patch_client([
            _Resp(_Msg(content="", tool_calls=[tcall])),
            _Resp(_Msg(content='{"grade":"B","p_lens":0.4}')),
        ])

        executed = []

        async def _exec(name, args):
            executed.append((name, args))
            img = [{"type": "image", "source": {"type": "base64",
                                                "media_type": "image/png", "data": "PX"}}]
            return ("cutout fetched: 3 views", img)

        tools = [{"type": "function", "function": {"name": "fetch_cutout",
                  "parameters": {"type": "object",
                                 "properties": {"name": {"type": "string"}}}}}]
        res = asyncio.run(llm_client.run_tool_loop(
            system="RUBRIC", user_content=[{"type": "text", "text": "grade DESI-1"}],
            tools=tools, execute_tool=_exec, model="m", max_turns=6))
        assert res.text == '{"grade":"B","p_lens":0.4}'
        assert res.tool_calls == 1 and res.num_turns == 2
        assert executed == [("fetch_cutout", {"name": "DESI-1"})]
        # second request must contain the tool result + the injected image user turn
        req2 = fake.chat.completions.calls[1]
        roles = [m["role"] for m in req2["messages"]]
        assert "tool" in roles
        tool_msg = next(m for m in req2["messages"] if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "call_1"
        injected = [m for m in req2["messages"]
                    if m["role"] == "user" and isinstance(m["content"], list)
                    and any(p.get("type") == "image_url" for p in m["content"])]
        assert len(injected) == 1
    finally:
        llm_client._get_client = orig
        _restore(saved)


# --------------------------------------------- v5 A0: structured-outputs shim
def test_structured_mode_shim():
    """LENSJUDGE_STRUCTURED=new -> structured_outputs key; =legacy -> guided_json (vLLM 0.24 shim)."""
    import math  # noqa: F401  (parity with client module imports)
    saved = _save_env()
    orig = llm_client._get_client
    schema = {"type": "object", "properties": {"grade": {"type": "string"}}}
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        os.environ["LENSJUDGE_GUIDED_JSON"] = "1"
        os.environ["LENSJUDGE_LOGPROBS"] = "0"
        os.environ["LENSJUDGE_STRUCTURED"] = "new"
        fake = _patch_client([_Resp(_Msg(content="{}")), _Resp(_Msg(content="{}"))])
        asyncio.run(llm_client.chat_with_images(
            system="S", content=[{"type": "text", "text": "x"}], model="m", json_schema=schema))
        assert fake.chat.completions.calls[0]["extra_body"] == {
            "structured_outputs": {"json": schema}}
        os.environ["LENSJUDGE_STRUCTURED"] = "legacy"
        asyncio.run(llm_client.chat_with_images(
            system="S", content=[{"type": "text", "text": "x"}], model="m", json_schema=schema))
        assert fake.chat.completions.calls[1]["extra_body"] == {"guided_json": schema}
    finally:
        llm_client._get_client = orig
        _restore(saved)


def test_structured_auto_falls_back_legacy_without_server():
    """auto mode with no live server (probe fails) must resolve to legacy — the safe default."""
    saved = _save_env()
    try:
        os.environ["LENSJUDGE_BASE_URL"] = "http://127.0.0.1:1"   # nothing listens
        assert llm_client._structured_mode() == "legacy"
        assert llm_client._STRUCTURED_RESOLVED == "legacy"        # cached
    finally:
        _restore(saved)


# ------------------------------------------------- v5 A0: grade-token logprobs
def _grade_lp(top=None, letter="A", letter_lp=-0.05):
    """Fake logprobs for text starting {"grade":"A" with the letter as its own token."""
    return _LP([
        _LPTok('{"'), _LPTok('grade'), _LPTok('":"'),
        _LPTok(letter, logprob=letter_lp, top=top),
        _LPTok('","criteria":{...}'),
    ])


def test_extract_grade_probs_from_top_logprobs():
    import math
    top = [_LPTop("A", math.log(0.70)), _LPTop('"B', math.log(0.20)),
           _LPTop(" D", math.log(0.05)), _LPTop("{", math.log(0.01))]
    gp = llm_client.extract_grade_probs(_grade_lp(top=top).content)
    assert abs(gp["A"] - 0.70) < 1e-4 and abs(gp["B"] - 0.20) < 1e-4 and abs(gp["D"] - 0.05) < 1e-4
    assert "C" not in gp
    assert abs(llm_client.logprob_p_lens(gp) - 0.90) < 1e-3   # P(A)+P(B)


def test_extract_grade_probs_fallback_and_missing():
    import math
    # no top_logprobs -> falls back to the sampled token's own logprob
    gp = llm_client.extract_grade_probs(_grade_lp(letter="D", letter_lp=math.log(0.9)).content)
    assert set(gp) == {"D"} and abs(gp["D"] - 0.9) < 1e-4
    # no "grade" key in the text -> None
    assert llm_client.extract_grade_probs([_LPTok("hello"), _LPTok(" world")]) is None
    assert llm_client.logprob_p_lens(None) is None


def test_chat_with_images_logprobs_default_on():
    saved = _save_env()
    orig = llm_client._get_client
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        import math
        top = [_LPTop("B", math.log(0.6)), _LPTop("A", math.log(0.3))]
        fake = _patch_client([_Resp(_Msg(content='{"grade":"B"}'), logprobs=_grade_lp(top=top, letter="B"))])
        res = asyncio.run(llm_client.chat_with_images(
            system="S", content=[{"type": "text", "text": "x"}], model="m"))
        req = fake.chat.completions.calls[0]
        assert req["logprobs"] is True and req["top_logprobs"] == 10   # default ON
        assert abs(res.grade_probs["B"] - 0.6) < 1e-4 and abs(res.grade_probs["A"] - 0.3) < 1e-4
        assert abs(llm_client.logprob_p_lens(res.grade_probs) - 0.9) < 1e-3
    finally:
        llm_client._get_client = orig
        _restore(saved)


def test_logprobs_opt_out():
    saved = _save_env()
    orig = llm_client._get_client
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        os.environ["LENSJUDGE_LOGPROBS"] = "0"
        fake = _patch_client([_Resp(_Msg(content="{}"))])
        res = asyncio.run(llm_client.chat_with_images(
            system="S", content=[{"type": "text", "text": "x"}], model="m"))
        assert "logprobs" not in fake.chat.completions.calls[0]
        assert res.grade_probs is None
    finally:
        llm_client._get_client = orig
        _restore(saved)


def test_run_tool_loop_grade_probs_from_final_turn():
    saved = _save_env()
    orig = llm_client._get_client
    try:
        os.environ["LENSJUDGE_BACKEND"] = "openai"
        os.environ["LENSJUDGE_BASE_URL"] = "http://localhost:8000/v1"
        import math
        tcall = _ToolCall("c1", "fetch_cutout", '{"name":"X"}')
        top = [_LPTop("A", math.log(0.8)), _LPTop("D", math.log(0.1))]
        _patch_client([
            _Resp(_Msg(content="", tool_calls=[tcall])),
            _Resp(_Msg(content='{"grade":"A","p_lens":0.8}'), logprobs=_grade_lp(top=top)),
        ])

        async def _exec(name, args):
            return ("ok", [])
        res = asyncio.run(llm_client.run_tool_loop(
            system="S", user_content=[{"type": "text", "text": "go"}],
            tools=[{"type": "function", "function": {"name": "fetch_cutout",
                    "parameters": {"type": "object", "properties": {}}}}],
            execute_tool=_exec, model="m", max_turns=4))
        assert abs(res.grade_probs["A"] - 0.8) < 1e-4
    finally:
        llm_client._get_client = orig
        _restore(saved)


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
