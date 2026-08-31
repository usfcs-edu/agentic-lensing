"""The executable fake CLI driven by the REAL SdkEngine -> real SDK subprocess transport + handshake."""
import json
from pathlib import Path

import pytest

from lensmark.claude.engine import SdkEngine
from lensmark.claude.engine_base import EngineRequest
from lensmark.claude.propose import PROPOSAL_SCHEMA

FIX = Path(__file__).parent / "fixtures"
FAKE = FIX / "fake_claude"
DEFAULT = json.loads((FIX / "proposals" / "default.json").read_text())
DECK02 = FIX / "proposals" / "deck-02.json"


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    argv = tmp_path / "argv.txt"
    monkeypatch.setenv("LENSMARK_CLAUDE_BIN", str(FAKE))
    monkeypatch.setenv("LENSMARK_FAKE_ARGV", str(argv))
    monkeypatch.delenv("LENSMARK_FAKE_MODE", raising=False)
    monkeypatch.delenv("LENSMARK_FAKE_STRUCTURED", raising=False)
    monkeypatch.delenv("LENSMARK_TRACE_DIR", raising=False)
    return argv


def req(tmp_path, **kw) -> EngineRequest:
    base = dict(system="You annotate strong-lens cutouts.\nSecond line.",
                content=[{"type": "text", "text": "Annotate deck-01."},
                         {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}],
                schema=PROPOSAL_SCHEMA, model="claude-opus-5", effort="xhigh", max_budget_usd=0.5, max_turns=2,
                cwd=tmp_path, fixture_key="deck-01")
    base.update(kw)
    return EngineRequest(**base)


def test_fake_is_executable_and_answers_version():
    import subprocess
    assert FAKE.stat().st_mode & 0o111
    assert subprocess.run([str(FAKE), "--version"], capture_output=True, text=True).stdout.startswith("2.1.251")


async def test_argv_and_structured_roundtrip(fake_env, tmp_path):
    events = []
    res = await SdkEngine().run(req(tmp_path), on_event=events.append)
    assert res.error is None, res.error
    assert res.structured == DEFAULT
    assert res.cost_usd == pytest.approx(0.0123) and res.num_turns == 1 and res.usage["output_tokens"] == 640
    assert res.raw["session_id"].startswith("fake-") and res.raw["model_reported"] == "claude-opus-5"
    assert res.thinking.startswith("Fake reasoning") and json.loads(res.text) == DEFAULT
    assert res.model == "claude-opus-5" and res.effort == "xhigh" and res.duration_s > 0
    lines = fake_env.read_text().strip().splitlines()
    assert len(lines) == 1                                  # exactly one run (the -v probe is not recorded)
    line = lines[0]
    for tok in ("--model claude-opus-5", "--effort xhigh", "--json-schema", "--setting-sources=",
                "--permission-mode bypassPermissions", "--thinking adaptive", "--thinking-display summarized",
                "--include-partial-messages", "--max-turns 2", "--max-budget-usd 0.5", "--tools ''",
                "--output-format stream-json", "--input-format stream-json", "--system-prompt"):
        assert tok in line, tok
    assert "--setting-sources=user" not in line and "--setting-sources=project" not in line   # isolation
    assert '"additionalProperties": false' in line          # the schema went through --json-schema
    phases = [e["phase"] for e in events]
    assert phases[0] == "started" and phases[-1] == "done" and "thinking" in phases and "partial" in phases


async def test_haiku_sends_no_effort(fake_env, tmp_path):
    res = await SdkEngine().run(req(tmp_path, model="claude-haiku-4-5"))
    assert res.error is None and res.effort is None and res.raw["model_reported"] == "claude-haiku-4-5"
    assert "--effort" not in fake_env.read_text() and "--model claude-haiku-4-5" in fake_env.read_text()


async def test_error_mode_sets_error(fake_env, monkeypatch, tmp_path):
    monkeypatch.setenv("LENSMARK_FAKE_MODE", "error")
    events = []
    res = await SdkEngine().run(req(tmp_path), on_event=events.append)
    assert res.error and "fake error" in res.error
    assert res.structured is None and res.cost_usd == pytest.approx(0.001)
    assert "error" in [e["phase"] for e in events]
    assert "ResultError" in res.raw.get("exception", "")     # the CLI's non-zero exit is folded, not raised


async def test_invalid_mode_keeps_text_without_structured(fake_env, monkeypatch, tmp_path):
    monkeypatch.setenv("LENSMARK_FAKE_MODE", "invalid")
    res = await SdkEngine().run(req(tmp_path))
    assert res.error is None and res.structured is None
    assert "prose" in res.text and '1.5"' in res.text


async def test_structured_override_and_trace(fake_env, monkeypatch, tmp_path):
    monkeypatch.setenv("LENSMARK_FAKE_STRUCTURED", str(DECK02))
    monkeypatch.setenv("LENSMARK_TRACE_DIR", str(tmp_path / "tr"))
    res = await SdkEngine().run(req(tmp_path, fixture_key="deck-02"))
    assert res.error is None and res.structured == json.loads(DECK02.read_text())
    tp = Path(res.raw["trace_path"])
    assert tp.is_file() and tp.name.startswith("propose.deck-02.")
    recs = [json.loads(l) for l in tp.read_text().splitlines()]
    assert recs[0]["event"] == "request" and "elided" in json.dumps(recs[0]["content"])   # images never logged
    assert [r["event"] for r in recs][-1] == "result" and recs[-1]["cost_usd"] == pytest.approx(0.0123)
