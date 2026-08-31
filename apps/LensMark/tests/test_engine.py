"""FixtureEngine, SdkEngine via a MockTransport (no subprocess), factory/doctor helpers."""
import asyncio
import json
from pathlib import Path

import pytest
from claude_agent_sdk import Transport

from lensmark.claude.engine import (DOCTOR_SCHEMA, FIXTURE_COST_USD, FixtureEngine, SdkEngine, bundled_cli_path,
                                    claude_version, cli_doctor, get_engine)
from lensmark.claude.engine_base import EngineRequest
from lensmark.claude.propose import PROPOSAL_SCHEMA

FIX = Path(__file__).parent / "fixtures"
DEFAULT = json.loads((FIX / "proposals" / "default.json").read_text())
DECK01 = json.loads((FIX / "proposals" / "deck-01.json").read_text())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LENSMARK_FIXTURE_DELAY", "0")
    monkeypatch.delenv("LENSMARK_TRACE_DIR", raising=False)
    monkeypatch.delenv("LENSMARK_FIXTURE_DIR", raising=False)
    monkeypatch.delenv("LENSMARK_ENGINE", raising=False)


def req(**kw) -> EngineRequest:
    base = dict(system="You annotate lens cutouts.", content=[{"type": "text", "text": "Annotate."}],
                schema=PROPOSAL_SCHEMA, model="claude-opus-5", effort="xhigh", max_budget_usd=0.5, max_turns=2)
    base.update(kw)
    return EngineRequest(**base)


# ----------------------------------------------------------------------------- fixture engine
async def test_fixture_engine_returns_named_fixture_and_events_in_order():
    events = []
    res = await FixtureEngine().run(req(fixture_key="deck-01"), on_event=events.append)
    assert res.structured == DECK01 and res.error is None
    assert res.cost_usd == FIXTURE_COST_USD and res.num_turns == 1 and res.usage["input_tokens"] > 0
    assert res.model == "claude-opus-5" and res.effort == "xhigh"
    assert [e["phase"] for e in events] == ["started", "thinking", "partial", "done"]
    assert events[1]["text"] and events[2]["text"].startswith("{") and events[3]["cost_usd"] == FIXTURE_COST_USD
    assert res.raw["fixture"].endswith("deck-01.json")
    res.structured["items"].clear()                      # result is a copy, not the cached doc
    assert (await FixtureEngine().run(req(fixture_key="deck-01"))).structured["items"]


async def test_fixture_engine_falls_back_to_default_and_patch_dir():
    res = await FixtureEngine().run(req(fixture_key="deck-07"))
    assert res.structured == DEFAULT
    patch = await FixtureEngine().run(req(purpose="patch", fixture_key="deck-01"))
    assert patch.structured["ops"][0]["op"] == "add" and patch.structured["ops"][0]["confidence"] == 0.88
    missing = await FixtureEngine().run(req(purpose="doctor"))
    assert missing.structured is None and "no fixture" in missing.error


async def test_fixture_dir_override(tmp_path, monkeypatch):
    (tmp_path / "proposals").mkdir()
    (tmp_path / "proposals" / "default.json").write_text(json.dumps({"system": {"verdict": "unclear", "description": "x"}, "items": []}))
    monkeypatch.setenv("LENSMARK_FIXTURE_DIR", str(tmp_path))
    res = await FixtureEngine().run(req(fixture_key="deck-01"))
    assert res.structured["items"] == []
    assert FixtureEngine(fixture_dir=FIX).resolve(req(fixture_key="deck-01")).name == "deck-01.json"


def test_get_engine_names(monkeypatch):
    assert get_engine("sdk").name == "sdk" and get_engine("fixture").name == "fixture"
    monkeypatch.setenv("LENSMARK_ENGINE", "fixture")
    assert isinstance(get_engine(), FixtureEngine)
    with pytest.raises(ValueError):
        get_engine("bogus")


# ----------------------------------------------------------------------------- mock transport
class MockTransport(Transport):
    """Replays canned NDJSON through the SDK's real Query/control protocol (no subprocess)."""

    def __init__(self, canned: list[dict]):
        self.canned = canned
        self.written: list[dict] = []
        self._q: asyncio.Queue = asyncio.Queue()
        self._ready = False
        self.closed = False
        self.input_ended = False

    async def connect(self) -> None:
        self._ready = True

    async def write(self, data: str) -> None:
        msg = json.loads(data)
        self.written.append(msg)
        if msg.get("type") == "control_request":
            await self._q.put({"type": "control_response",
                               "response": {"subtype": "success", "request_id": msg["request_id"],
                                            "response": {"commands": [], "output_style": "default"}}})
        elif msg.get("type") == "user":
            for m in self.canned:
                await self._q.put(m)

    async def read_messages(self):
        while True:
            m = await self._q.get()
            if m is None:
                return
            yield m

    async def close(self) -> None:
        self.closed = True
        await self._q.put(None)

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        self.input_ended = True
        await self._q.put(None)


class FailingTransport(MockTransport):
    async def connect(self) -> None:
        raise RuntimeError("cannot spawn")


def canned(*, text: str, structured=None, is_error=False, subtype="success", errors=None, stream=True):
    sid = "mock-session"
    msgs = [{"type": "system", "subtype": "init", "session_id": sid, "model": "claude-opus-5", "uuid": "u1", "tools": []}]
    if stream:
        msgs += [
            {"type": "stream_event", "uuid": "u2", "session_id": sid, "parent_tool_use_id": None,
             "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "read the grid"}}},
            {"type": "stream_event", "uuid": "u3", "session_id": sid, "parent_tool_use_id": None,
             "event": {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[:20]}}},
        ]
    msgs.append({"type": "assistant", "session_id": sid, "uuid": "u4", "parent_tool_use_id": None,
                 "message": {"id": "m1", "role": "assistant", "model": "claude-opus-5",
                             "content": [{"type": "thinking", "thinking": "read the grid", "signature": "sig"},
                                         {"type": "text", "text": text}],
                             "usage": {"input_tokens": 10, "output_tokens": 5}}})
    result = {"type": "result", "subtype": subtype, "duration_ms": 100, "duration_api_ms": 90, "is_error": is_error,
              "num_turns": 1, "session_id": sid, "total_cost_usd": 0.042, "usage": {"input_tokens": 10, "output_tokens": 5},
              "result": text, "stop_reason": "end_turn", "uuid": "u5"}
    if structured is not None:
        result["structured_output"] = structured
    if errors:
        result["errors"] = errors
    msgs.append(result)
    return msgs


async def test_sdk_engine_collects_via_mock_transport(tmp_path):
    tr = MockTransport(canned(text=json.dumps(DEFAULT), structured=DEFAULT))
    events = []
    r = req(cwd=tmp_path, content=[{"type": "text", "text": "go"},
                                   {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}])
    res = await SdkEngine(transport=tr).run(r, on_event=events.append)
    assert res.error is None and res.structured == DEFAULT
    assert res.cost_usd == pytest.approx(0.042) and res.num_turns == 1 and res.usage == {"input_tokens": 10, "output_tokens": 5}
    assert res.thinking == "read the grid" and res.text == json.dumps(DEFAULT)
    assert res.model == "claude-opus-5" and res.effort == "xhigh" and res.raw["session_id"] == "mock-session"
    assert res.raw["model_reported"] == "claude-opus-5" and res.duration_s is not None
    assert [e["phase"] for e in events] == ["started", "thinking", "partial", "done"]
    assert events[1]["text"] == "read the grid" and events[3]["cost_usd"] == pytest.approx(0.042)
    # the SDK handshake happened through the transport: initialize first, then ONE user message with our blocks
    assert tr.written[0]["type"] == "control_request" and tr.written[0]["request"]["subtype"] == "initialize"
    assert tr.written[0]["request"]["hooks"] is None                       # no LENSMARK_TRACE_DIR -> no hooks
    users = [m for m in tr.written if m.get("type") == "user"]
    assert len(users) == 1 and users[0]["message"]["content"] == r.content and users[0]["message"]["role"] == "user"
    assert tr.input_ended and tr.closed


async def test_sdk_engine_falls_back_to_parsing_prose_with_arcsec_marks():
    prose = ('Reasoning first: the arc sits 1.5" from the deflector and the star is 3" away. '
             '{"system": {"verdict": "possible", "description": "arc at 1.5\\" (cyan arrow)"}, '
             '"items": [{"type": "arrow", "head": [0.4, 0.5], "label": "arc 1.5\\""}]}')
    res = await SdkEngine(transport=MockTransport(canned(text=prose))).run(req())
    assert res.error is None
    assert res.structured["items"][0]["label"] == 'arc 1.5"' and res.structured["system"]["verdict"] == "possible"


async def test_sdk_engine_error_result_is_reported_not_raised():
    msgs = canned(text="", subtype="error_max_budget_usd", is_error=True, errors=["budget exceeded"], stream=False)
    events = []
    res = await SdkEngine(transport=MockTransport(msgs)).run(req(), on_event=events.append)
    assert res.error == "budget exceeded" and res.structured is None and res.cost_usd == pytest.approx(0.042)
    assert events[-1]["phase"] == "error"


async def test_sdk_engine_exception_becomes_error():
    events = []
    res = await SdkEngine(transport=FailingTransport([])).run(req(), on_event=events.append)
    assert res.error and "cannot spawn" in res.error and res.structured is None
    assert events == [{"phase": "error", "detail": res.error}]


async def test_sdk_engine_transport_kwarg_on_run():
    tr = MockTransport(canned(text="{}", structured={"ok": True}))
    res = await SdkEngine().run(req(schema=DOCTOR_SCHEMA), transport=tr)
    assert res.structured == {"ok": True}


async def test_sdk_engine_trace_hooks_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("LENSMARK_TRACE_DIR", str(tmp_path / "traces"))
    tr = MockTransport(canned(text=json.dumps(DEFAULT), structured=DEFAULT))
    res = await SdkEngine(transport=tr).run(req(fixture_key="deck-01"))
    init = tr.written[0]["request"]
    assert set(init["hooks"]) == {"PreToolUse", "PostToolUse", "SubagentStop"}
    trace_path = Path(res.raw["trace_path"])
    assert trace_path.exists() and trace_path.parent == tmp_path / "traces"
    events = [json.loads(l)["event"] for l in trace_path.read_text().splitlines()]
    assert events[0] == "request" and "thinking" in events and "assistant_text" in events and events[-1] == "result"


def test_build_options_matrix(monkeypatch):
    monkeypatch.setenv("LENSMARK_CLAUDE_BIN", "/x/claude")
    o = SdkEngine().build_options(req())
    assert o.model == "claude-opus-5" and o.effort == "xhigh" and o.setting_sources == [] and o.tools == []
    assert o.thinking == {"type": "adaptive", "display": "summarized"}
    assert o.output_format == {"type": "json_schema", "schema": PROPOSAL_SCHEMA}
    assert o.permission_mode == "bypassPermissions" and o.include_partial_messages and o.cli_path == "/x/claude"
    assert o.max_turns == 2 and o.max_budget_usd == 0.5 and o.hooks is None and o.system_prompt.startswith("You annotate")
    assert SdkEngine().build_options(req(model="claude-haiku-4-5")).effort is None    # haiku: no effort flag
    assert SdkEngine().build_options(req(effort=None)).effort is None


# ----------------------------------------------------------------------------- helpers / doctor
def test_claude_version_and_bundled(monkeypatch):
    fake = FIX / "fake_claude"
    monkeypatch.setenv("LENSMARK_CLAUDE_BIN", str(fake))
    assert claude_version() == "2.1.251 (Claude Code)"
    assert claude_version("/nonexistent/claude") is None
    b = bundled_cli_path()
    assert b is None or Path(b).is_file()


def test_cli_doctor_no_call(capsys, monkeypatch):
    monkeypatch.setenv("LENSMARK_CLAUDE_BIN", str(FIX / "fake_claude"))
    assert cli_doctor(None, call=False) == 0
    out = capsys.readouterr().out
    assert "claude-agent-sdk" in out and "fake_claude" in out and "LENSMARK_ENGINE" in out
