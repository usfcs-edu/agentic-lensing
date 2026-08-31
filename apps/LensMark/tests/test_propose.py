"""run_propose with the FixtureEngine on a throw-away copy of examples/nine; build_prompt; listing; CLI."""
import json
import shutil
from pathlib import Path

import pytest

from lensmark.claude.engine import FixtureEngine
from lensmark.claude.engine_base import EngineRequest, EngineResult
from lensmark.claude.propose import (PROMPT_PATH, ProposeRequest, build_prompt, cli_propose, list_runs, load_run,
                                     prompt_sha256, run_propose)
from lensmark.model import Arrow
from lensmark.store import Campaign

FIX = Path(__file__).parent / "fixtures"
DECK01 = json.loads((FIX / "proposals" / "deck-01.json").read_text())
DEFAULT = json.loads((FIX / "proposals" / "default.json").read_text())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LENSMARK_FIXTURE_DELAY", "0")
    monkeypatch.delenv("LENSMARK_TRACE_DIR", raising=False)
    monkeypatch.delenv("LENSMARK_FIXTURE_DIR", raising=False)
    monkeypatch.delenv("LENSMARK_ENGINE", raising=False)


class StubEngine:
    """Scripted results, one per call; records the requests it saw."""
    name = "stub"

    def __init__(self, results):
        self.results = list(results)
        self.requests: list[EngineRequest] = []

    async def run(self, req, on_event=None):
        self.requests.append(req)
        if on_event:
            on_event({"phase": "started", "detail": "stub"})
        return self.results.pop(0)


async def test_run_propose_writes_run_and_merges_items(nine):
    c = Campaign(nine)
    events = []
    run = await run_propose(c, "deck-01", ProposeRequest(model="opus", effort="xhigh"), engine=FixtureEngine(),
                            on_event=events.append)
    assert run.run_id.startswith("run-") and len(run.run_id) == len("run-20260830-120000-abcd")
    assert run.model == "claude-opus-5" and run.effort == "xhigh" and run.engine == "fixture"
    assert run.n_items_proposed == len(DECK01["items"]) and run.n_invalid == 0 and run.n_repaired == 0
    assert run.parse_ok and run.error is None and run.cost_usd == pytest.approx(0.0123) and run.num_turns == 1
    assert run.prompt_sha256 == prompt_sha256() and run.fewshot_sha256 is None and run.usage["input_tokens"] == 3210
    assert run.proposed_system["verdict"] == "likely_lens" and run.proposed_system["theta_e"]["value_arcsec"] == 1.5
    # immutable proposal file
    p = c.proposals_dir / f"deck-01.{run.run_id}.json"
    assert p.exists() and run.proposal_file == p.name
    d = json.loads(p.read_text())
    assert d["schema_version"] == "lensmark-proposal-run/1.0" and d["raw"]["structured"] == DECK01
    assert d["request"]["model_resolved"] == "claude-opus-5" and d["engine"] == "fixture" and d["n_image_blocks"] == 2
    assert len(d["items"]) == len(DECK01["items"]) and d["items"][0]["created_by"]["run_id"] == run.run_id
    assert d["parse_ok"] and d["repairs"] == [] and d["invalid"] == [] and d["cost_usd"] == pytest.approx(0.0123)
    # merged into the file: every item proposed, stamped with the run; system block untouched
    f = c.load("deck-01")
    assert len(f.items) == len(DECK01["items"])
    assert all(it.status == "proposed" and it.created_by.kind == "claude" and it.created_by.run_id == run.run_id
               and it.created_by.model == "claude-opus-5" and it.created_by.effort == "xhigh" for it in f.items)
    assert [it.type for it in f.items][:3] == ["arrow", "arrow", "arrow"] and f.items[0].color == "green"
    assert f.system.description == "" and f.system.theta_e.value_arcsec == 1.5 and f.system.theta_e.method == "reference"
    assert [r.run_id for r in f.provenance.proposal_runs] == [run.run_id]
    # append-only log: every event by claude
    log = c.read_log("deck-01")
    assert log and all(e["actor"] == "claude" and e["source"] == "claude" for e in log)
    assert sum(1 for e in log if e["op"] == "add") == len(DECK01["items"])
    # events: engine phases forwarded, pipeline validated -> done (with the run) last; no early "done"
    phases = [e["phase"] for e in events]
    assert phases[0] == "started" and "thinking" in phases and phases[-2:] == ["validated", "done"]
    assert phases.count("done") == 1 and events[-1]["run"]["run_id"] == run.run_id and events[-2]["n_items"] == len(f.items)


async def test_second_run_replaces_unreviewed_items_only(nine):
    c = Campaign(nine)
    r1 = await run_propose(c, "deck-02", ProposeRequest(), engine=FixtureEngine())
    assert r1.n_repaired == 1 and r1.n_invalid == 0            # deck-02 fixture: one centre clamped
    f = c.load("deck-02")
    first = f.items[0]
    first.status = "accepted"
    c.save("deck-02", f, actor="xhuang")
    r2 = await run_propose(c, "deck-02", ProposeRequest(), engine=FixtureEngine())
    f = c.load("deck-02")
    kept = [it for it in f.items if it.created_by.run_id == r1.run_id]
    assert [it.id for it in kept] == [first.id] and kept[0].status == "accepted"
    new = [it for it in f.items if it.created_by.run_id == r2.run_id]
    assert len(new) == r2.n_items_proposed and all(it.status == "proposed" for it in new)
    assert first.id not in {it.id for it in new}                # ids never collide with the kept item
    assert [r.run_id for r in f.provenance.proposal_runs] == [r1.run_id, r2.run_id]
    # the stale proposals were logged as replaced: ids are re-minted against the pruned file, so a
    # same-id replacement is an "update" (before = run 1 item, after = run 2 item) and a leftover a "delete"
    claude_ops = [e for e in c.read_log("deck-02")
                  if e.get("actor") == "claude" and e["op"] in ("update", "delete") and not e["item_id"].startswith("$")]
    assert claude_ops and all(e["before"]["created_by"]["run_id"] == r1.run_id for e in claude_ops)
    assert all(e["after"]["created_by"]["run_id"] == r2.run_id for e in claude_ops if e["op"] == "update")
    # listing / loading
    runs = list_runs(c, "deck-02")
    assert [r.run_id for r in runs] == [r1.run_id, r2.run_id]
    d = load_run(c, "deck-02", r2.run_id)
    assert d["run_id"] == r2.run_id and d["repairs"][0]["why"] == "clamped_to_image"
    with pytest.raises(FileNotFoundError):
        load_run(c, "deck-02", "run-00000000-000000-dead")


async def test_list_runs_recovers_orphan_proposal_files(nine):
    c = Campaign(nine)
    r = await run_propose(c, "deck-04", ProposeRequest(), engine=FixtureEngine())
    c.json_path("deck-04").unlink()                                  # provenance gone, file still there
    assert [x.run_id for x in list_runs(c, "deck-04")] == [r.run_id]
    assert list_runs(c, "deck-04")[0].n_items_proposed == r.n_items_proposed


async def test_default_fixture_and_haiku_has_no_effort(nine):
    c = Campaign(nine)
    run = await run_propose(c, "deck-05", ProposeRequest(model="haiku", effort="xhigh"), engine=FixtureEngine())
    assert run.model == "claude-haiku-4-5" and run.effort is None
    assert load_run(c, "deck-05", run.run_id)["raw"]["structured"] == DEFAULT
    assert all(it.created_by.effort is None for it in c.load("deck-05").items)


async def test_engine_error_is_recorded(nine):
    c = Campaign(nine)
    eng = StubEngine([EngineResult(structured=None, error="boom: CLI not logged in", cost_usd=0.0)])
    events = []
    run = await run_propose(c, "deck-03", ProposeRequest(), engine=eng, on_event=events.append)
    assert run.error == "boom: CLI not logged in" and run.n_items_proposed == 0 and run.engine == "stub"
    assert events[-1]["phase"] == "error" and events[-1]["run"]["run_id"] == run.run_id
    f = c.load("deck-03")
    assert f.items == [] and f.provenance.proposal_runs[0].error == run.error
    assert load_run(c, "deck-03", run.run_id)["error"] == run.error


async def test_parse_failure_triggers_one_repair_turn(nine):
    c = Campaign(nine)
    good = {"system": {"verdict": "possible", "description": "the cyan arrow marks an arc 2.0\" east"},
            "items": [{"type": "arrow", "head": [0.4, 0.5], "label": "arc", "color": "cyan"}]}
    eng = StubEngine([EngineResult(structured=None, text="Sorry, prose only 1.5\" here.", cost_usd=0.01, num_turns=1),
                      EngineResult(structured=good, text=json.dumps(good), cost_usd=0.02, num_turns=1)])
    run = await run_propose(c, "deck-06", ProposeRequest(), engine=eng)
    assert len(eng.requests) == 2 and eng.requests[1].max_turns == 1
    assert eng.requests[1].content[0]["type"] == "text" and "prose only" in eng.requests[1].content[0]["text"]
    assert run.parse_ok is False and run.error is None and run.cost_usd == pytest.approx(0.03) and run.num_turns == 2
    assert run.n_items_proposed == 1 and c.load("deck-06").items[0].label == "arc"
    assert load_run(c, "deck-06", run.run_id)["repair_turns"] == 1
    # repair also failing -> parse_failed error, nothing merged
    eng2 = StubEngine([EngineResult(structured=None, text="no json"), EngineResult(structured={"nope": 1}, text="{\"nope\": 1}")])
    run2 = await run_propose(c, "deck-07", ProposeRequest(), engine=eng2)
    assert run2.error.startswith("parse_failed") and run2.parse_ok is False and c.load("deck-07").items == []


def test_build_prompt_blocks_and_stable_sha(nine, tmp_path):
    c = Campaign(nine)
    f = c.new_file("deck-01")
    system, content, sha, fsha = build_prompt(c, "deck-01", f, ProposeRequest())
    assert fsha is None and len(sha) == 64 and sha == prompt_sha256()
    assert [b["type"] for b in content] == ["text", "image", "image"]
    assert content[1]["source"]["media_type"] == "image/png" and content[1]["source"]["type"] == "base64"
    assert f"{f.image.width}x{f.image.height}" in system and "16\"" in system and "East is LEFT" in system
    assert "{cutout_arcsec}" not in system and "{mask_cap}" not in system and "at most 12 masks" in system.lower()
    assert "deck-01" in content[0]["text"] and "no existing annotations" in content[0]["text"]
    system2, _, sha2, _ = build_prompt(c, "deck-01", f, ProposeRequest())
    assert (system2, sha2) == (system, sha)
    assert PROMPT_PATH.name == "propose_v1.md" and "self-verify" in PROMPT_PATH.read_text().lower()
    _, content3, _, _ = build_prompt(c, "deck-01", f, ProposeRequest(include_grid=False))
    assert [b["type"] for b in content3] == ["text", "image"]
    # existing accepted items are shown so the model does not duplicate them
    f.items.append(Arrow(id="ann-arrow-001", tail=[0.4, 0.7], head=[0.45, 0.6], label="tight arc", color="cyan"))
    _, content4, _, _ = build_prompt(c, "deck-01", f, ProposeRequest())
    assert "tight arc" in content4[0]["text"] and "do NOT duplicate" in content4[0]["text"]


def test_build_prompt_with_fewshot_bundle(nine, tmp_path):
    c = Campaign(nine)
    f = c.new_file("deck-01")
    bundle = tmp_path / "fewshot"
    bundle.mkdir()
    ex = c.new_file("deck-03")
    ex.items.append(Arrow(id="ann-arrow-001", tail=[0.62, 0.5], head=[0.53, 0.5], label="deflector", color="green"))
    ex.items.append(Arrow(id="ann-arrow-002", tail=[0.3, 0.3], head=[0.4, 0.4], label="rejected one", status="rejected"))
    ex.system.description = "The green arrow marks the deflector."
    shutil.copy(c.image_path("deck-03"), bundle / "001-deck-03.png")
    shutil.copy(c.image_path("deck-03"), bundle / "001-deck-03.annot.png")
    (bundle / "001-deck-03.lensmark.json").write_text(ex.to_json())
    (bundle / "001-deck-03.md").write_text("The green arrow marks the deflector; nothing else is lensed.\n")
    (bundle / "manifest.json").write_text(json.dumps({
        "schema_version": "lensmark-fewshot/1.0", "k": 1, "prompt_sha256": "a" * 64,
        "examples": [{"id": "deck-03", "png": "001-deck-03.png", "annot": "001-deck-03.annot.png",
                      "json": "001-deck-03.lensmark.json", "md": "001-deck-03.md"}]}))
    (bundle / "prompt.sha256").write_text("f" * 64 + "  fewshot\n")
    system, content, sha, fsha = build_prompt(c, "deck-01", f, ProposeRequest(fewshot=str(bundle)))
    assert fsha == "f" * 64 and sha == prompt_sha256()
    types = [b["type"] for b in content]
    assert types.count("image") == 4 and types[-3:] == ["text", "image", "image"] and types[0] == "text"
    texts = "\n".join(b["text"] for b in content if b["type"] == "text")
    assert '"label": "deflector"' in texts and "nothing else is lensed" in texts and "rejected one" not in texts
    # relative bundle path resolves against the campaign root
    shutil.copytree(bundle, nine / "exports" / "fewshot")
    _, _, _, fsha2 = build_prompt(c, "deck-01", f, ProposeRequest(fewshot="exports/fewshot"))
    assert fsha2 == fsha
    with pytest.raises(FileNotFoundError):
        build_prompt(c, "deck-01", f, ProposeRequest(fewshot=str(tmp_path / "nope")))


async def test_fewshot_sha_recorded_in_run(nine, tmp_path):
    c = Campaign(nine)
    bundle = tmp_path / "fs"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"schema_version": "lensmark-fewshot/1.0", "k": 0, "examples": []}))
    run = await run_propose(c, "deck-08", ProposeRequest(fewshot=str(bundle)), engine=FixtureEngine())
    assert run.fewshot_sha256 and len(run.fewshot_sha256) == 64
    assert load_run(c, "deck-08", run.run_id)["fewshot_sha256"] == run.fewshot_sha256


def test_cli_propose_all_images_then_nothing_left(nine, capsys):
    assert cli_propose(str(nine), engine="fixture", concurrency=3) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 9 and all("run-" in l and "cost=$0.0123" in l and l.endswith("ok") for l in out)
    c = Campaign(nine)
    assert all(len(list_runs(c, i)) == 1 for i in c.list_ids())
    assert cli_propose(str(nine), engine="fixture") == 0
    assert "nothing to do" in capsys.readouterr().out
    assert cli_propose(str(nine), image_id="deck-01", engine="fixture", model="sonnet", effort="low") == 0
    runs = list_runs(c, "deck-01")                                 # provenance (append) order
    assert [r.model for r in runs] == ["claude-opus-5", "claude-sonnet-5"] and runs[-1].effort == "low"
