import json

import pytest

from lensmark.claude.engine_base import EngineRequest, EngineResult
from lensmark.model import Patch
from lensmark.store import Campaign
from lensmark.voice import stt
from lensmark.voice.patch import apply_patch, build_request, make_patch, parse_patch
from synth_campaign import accepted_file, arrow, mask, ring, seed_run

UTTER_ADD = "put a dashed circle around the galaxy at upper left"
UTTER_UPDATE = "move the cyan arrow head a bit to the right and set theta E to 1.8"
UTTER_DELETE = "the magenta arrow is not an arc, remove it; and delete the note"


class StubEngine:
    name = "stub"

    def __init__(self, result: EngineResult):
        self.result = result
        self.requests: list[EngineRequest] = []

    async def run(self, req: EngineRequest, on_event=None) -> EngineResult:
        self.requests.append(req)
        return self.result


def seeded(nine) -> Campaign:
    c = Campaign(nine)
    seed_run(c, "deck-01", "r1", [arrow("ann-arrow-001", [0.40, 0.60], [0.30, 0.75], "arc", "cyan"),
                                  ring("ann-ring-001", [0.5, 0.5], 1.5)],
             human_items=[arrow("ann-arrow-002", [0.6, 0.3], [0.75, 0.15], "spiral arm", "magenta")])
    accepted_file(c, "deck-01", [], theta_e=1.5)
    from synth_campaign import note
    accepted_file(c, "deck-01", [note("ann-text-001", [0.1, 0.9], "seeing 1.1\"")])
    return c


# ----------------------------------------------------------------------------- apply_patch: the three worked utterances
def test_apply_add_galaxy_mask_upper_left(nine):
    c = seeded(nine)
    ops = [{"op": "add", "id": None, "item": {"type": "mask_circle", "center": [0.2, 0.25], "radius_arcsec": 1.0, "kind": "galaxy"},
            "confidence": 0.9, "rationale": "the galaxy at upper left"}]
    f = apply_patch(c, "deck-01", ops, transcript=UTTER_ADD)
    m = [it for it in f.items if it.type == "mask_circle"]
    assert len(m) == 1
    m = m[0]
    assert m.id == "ann-mask-001" and m.kind == "galaxy" and m.center[0] < 0.5 and m.center[1] < 0.5
    assert m.status == "accepted" and m.created_by.kind == "voice" and m.color == "mask_red" and m.radius_arcsec == 1.0
    assert m.notes == "the galaxy at upper left"
    assert c.load("deck-01").item("ann-mask-001") is not None                # persisted
    log = c.read_log("deck-01")
    patch_events = [e for e in log if e["op"] == "patch"]
    assert len(patch_events) == 1 and patch_events[0]["source"] == "voice" and patch_events[0]["actor"] == "voice"
    assert patch_events[0]["transcript"] == UTTER_ADD and patch_events[0]["ops"][0]["op"] == "add" and patch_events[0]["item_id"] == "$patch"
    assert any(e["op"] == "add" and e["item_id"] == "ann-mask-001" and e["source"] == "voice" for e in log)


def test_apply_update_head_theta_e_and_system(nine):
    c = seeded(nine)
    ops = [{"op": "update", "id": "ann-arrow-001", "set": {"head": [0.45, 0.60]}, "confidence": 0.8, "rationale": "a bit right = +0.05"},
           {"op": "update", "id": "ann-ring-001", "set": {"theta_e_arcsec": 1.8}, "confidence": 0.9, "rationale": "theta E 1.8"},
           {"op": "update", "id": "$system", "set": {"theta_e": {"value_arcsec": 1.8, "method": "human"}}, "confidence": 0.9, "rationale": "system"}]
    f = apply_patch(c, "deck-01", ops, transcript=UTTER_UPDATE)
    a = f.item("ann-arrow-001")
    assert a.head == [0.45, 0.60] and a.tail == [0.30, 0.75]
    assert a.status == "edited" and a.edit_of == {"head": [0.40, 0.60], "tail": [0.30, 0.75]}   # claude item -> critique signal
    assert f.item("ann-ring-001").theta_e_arcsec == 1.8
    assert f.system.theta_e.value_arcsec == 1.8 and f.system.theta_e.method == "human" and f.system.rank == 91
    # a second geometry edit keeps the ORIGINAL edit_of
    apply_patch(c, "deck-01", [{"op": "update", "id": "ann-arrow-001", "set": {"head": [0.5, 0.6]}}])
    assert c.load("deck-01").item("ann-arrow-001").edit_of["head"] == [0.40, 0.60]
    # human item: plain update, no edit_of
    g = apply_patch(c, "deck-01", [{"op": "update", "id": "ann-arrow-002", "set": {"label": "spiral arm - NOT an arc", "color": "yellow"}}])
    h = g.item("ann-arrow-002")
    assert h.label == "spiral arm - NOT an arc" and h.color == "yellow" and h.status == "accepted" and h.edit_of is None


def test_apply_delete_with_and_without_review(nine):
    c = seeded(nine)
    ops = [{"op": "delete", "id": "ann-arrow-002", "set": {"review": {"verdict": "spurious", "comment": "spiral arm, not an arc"}},
            "confidence": 0.9, "rationale": "user rejected it"},
           {"op": "delete", "id": "ann-text-001", "confidence": 0.9, "rationale": "delete the note"}]
    f = apply_patch(c, "deck-01", ops, transcript=UTTER_DELETE)
    a = f.item("ann-arrow-002")
    assert a is not None and a.status == "rejected"
    assert a.review.verdict == "spurious" and a.review.comment == "spiral arm, not an arc" and a.review.reviewer == "voice"
    assert f.item("ann-text-001") is None
    assert any(e["op"] == "delete" and e["item_id"] == "ann-text-001" for e in c.read_log("deck-01"))


def test_apply_rejects_bad_values_without_writing(nine):
    c = seeded(nine)
    before = c.json_path("deck-01").read_text()
    with pytest.raises(ValueError):
        apply_patch(c, "deck-01", [{"op": "update", "id": "ann-ring-001", "set": {"theta_e_arcsec": -1}}])
    with pytest.raises(ValueError):
        apply_patch(c, "deck-01", [{"op": "update", "id": "ann-nope", "set": {"label": "x"}}])
    with pytest.raises(ValueError):
        apply_patch(c, "deck-01", [{"op": "add", "item": {"type": "arrow", "head": [3.0, 3.0]}}])
    with pytest.raises(ValueError):
        apply_patch(c, "deck-01", [{"op": "update", "id": "ann-arrow-001", "set": {"color": "purple"}}])
    assert c.json_path("deck-01").read_text() == before
    assert not any(e["op"] == "patch" for e in c.read_log("deck-01"))


def test_apply_on_unsaved_image_creates_file(nine):
    c = Campaign(nine)
    f = apply_patch(c, "deck-02", [{"op": "add", "item": {"type": "arrow", "head": [0.5, 0.5], "label": "deflector"}}], transcript="mark the deflector")
    assert f.items[0].color == "green" and f.items[0].created_by.kind == "voice" and c.exists("deck-02")


# ----------------------------------------------------------------------------- make_patch with a stub engine
async def test_make_patch_with_stub_engine(nine):
    c = seeded(nine)
    canned = {"schema_version": "lensmark-patch/1.0", "transcript": UTTER_ADD,
              "ops": [{"op": "add", "id": None, "item": {"type": "mask_circle", "center": [0.2, 0.25], "radius_arcsec": 1.0, "kind": "galaxy"},
                       "set": None, "confidence": 0.9, "rationale": "galaxy at upper left"}],
              "clarification": None}
    eng = StubEngine(EngineResult(structured=canned, cost_usd=0.01, model="claude-opus-5"))
    patch = await make_patch(c, "deck-01", UTTER_ADD, engine=eng)
    assert isinstance(patch, Patch) and len(patch.ops) == 1 and patch.ops[0].op == "add"
    assert patch.ops[0].item["kind"] == "galaxy" and patch.clarification is None and patch.transcript == UTTER_ADD
    req = eng.requests[0]
    assert req.purpose == "patch" and req.fixture_key == "deck-01" and req.schema["title"] == "LensMarkPatch"
    assert req.model == "claude-opus-5" and req.effort == "low" and req.cwd == c.root
    assert "12 o'clock" in req.system and "$system" in req.system
    assert req.content[0]["type"] == "text" and "ann-arrow-001" in req.content[0]["text"] and '"cutout_arcsec":16.0' in req.content[0]["text"]
    assert sum(1 for b in req.content if b["type"] == "image") == 2
    assert req.content[-1]["type"] == "text" and UTTER_ADD in req.content[-1]["text"]
    # the ops round-trip into apply_patch
    f = apply_patch(c, "deck-01", [o.model_dump() for o in patch.ops], transcript=patch.transcript)
    assert f.item("ann-mask-001").kind == "galaxy"


async def test_make_patch_lenient_parsing(nine):
    c = seeded(nine)
    bad = StubEngine(EngineResult(structured=None, text="Sorry, I cannot do that."))
    p = await make_patch(c, "deck-01", "do something", engine=bad)
    assert p.ops == [] and p.clarification.startswith("model returned no valid patch")
    err = StubEngine(EngineResult(structured=None, text="", error="budget exceeded"))
    p = await make_patch(c, "deck-01", "do something", engine=err)
    assert p.ops == [] and "budget exceeded" in p.clarification
    prose = StubEngine(EngineResult(structured=None, text='Here you go: {"schema_version":"lensmark-patch/1.0","transcript":"x",'
                                                          '"ops":[{"op":"delete","id":"ann-text-001","confidence":0.8,"rationale":"r"},'
                                                          '{"op":"explode","id":"ann-text-001"}],"clarification":null}'))
    p = await make_patch(c, "deck-01", "delete the note", engine=prose)
    assert len(p.ops) == 1 and p.ops[0].op == "delete" and "op 1" in p.clarification
    q = parse_patch(EngineResult(structured={"ops": [], "clarification": "Which arrow: the cyan (ann-arrow-001) or the magenta (ann-arrow-002)?"}), "move the arrow")
    assert q.ops == [] and q.clarification.startswith("Which arrow")


def test_build_request_effort_and_model(nine):
    c = seeded(nine)
    f = c.load("deck-01")
    r = build_request(c, f, "x", model="haiku")
    assert r.model == "claude-haiku-4-5" and r.effort is None
    r = build_request(c, f, "x", model="sonnet", effort="medium")
    assert r.model == "claude-sonnet-5" and r.effort == "medium"
    state = json.loads(r.content[0]["text"].split("\n", 1)[1])
    assert {it["id"] for it in state["items"]} == {"ann-arrow-001", "ann-ring-001", "ann-arrow-002", "ann-text-001"}
    assert state["items"][0]["created_by"] == "claude" and state["system"]["theta_e"]["value_arcsec"] == 1.5


# ----------------------------------------------------------------------------- STT
def test_stt_raises_without_backend(monkeypatch):
    monkeypatch.setattr(stt, "available_backends", lambda: [])
    with pytest.raises(NotImplementedError, match="pip install mlx-whisper"):
        stt.transcribe(b"RIFF....", "audio/wav")
    assert isinstance(stt.available_backends(), list)


def test_patch_schema_has_no_dollar_schema_key():
    """claude --json-schema rejects a top-level $schema; the engine request must never carry one."""
    from lensmark.voice.patch import patch_schema
    assert "$schema" not in patch_schema()
    assert patch_schema()["additionalProperties"] is False
