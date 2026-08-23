#!/usr/bin/env python3
"""No-API tests for the evidence-first panel fan-out (WP-3): golden/views.py, golden/panel.py and
the grader seams (grader_jwst note=/schema=/extra_views=, grader_direct schema=, audit_traces
n_extra_views rule + --pi-only lexicon).

Pure logic on synthetic composites (752x562 canvases with coloured rectangles at the measured
panel slots) with grader_direct.grade_candidate monkeypatched to a stub that records
(system_prompt, content, schema) and returns canned pydantic records — NO network, NO API
spend. Run as a script or under pytest:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_panel.py
Checks that need the persona files (prompts/personas/jwst_v1, incumbent) skip when absent;
nothing here reads the JWST repo.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from lensjudge.common import jwst_fetch as jf  # noqa: E402
from lensjudge.common import llm_client  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.golden import _util, aggregate_v2, audit_traces, grader_jwst, panel, schemas_panel, views  # noqa: E402
from lensjudge.imaging import grader_direct  # noqa: E402
from lensjudge.imaging.grader_lean import GradeResult, _REPAIR  # noqa: E402

HERE = Path(__file__).resolve().parent
LENSJUDGE = HERE.parent
PERSONAS_OK = all((panel.PERSONA_DIR_V1 / f"{f}.md").exists() for f in
                  ("advocate", "critic_common", "critic_artifact", "critic_geometry", "critic_morphology", "arbitrator"))
INCUMBENT_OK = all((panel.PERSONA_DIR_INCUMBENT / f"{f}.md").exists() for f in ("wrapper", "artifact", "morphology", "geometry"))
NOTE_OK = panel.NOTE_V2_PATH.exists()
SLOT_RGB = {"a": (200, 30, 30), "b": (30, 200, 30), "c": (30, 30, 200),
            "d": (200, 200, 30), "e": (30, 200, 200), "f": (200, 30, 200)}


def _skip(msg: str) -> bool:
    print(f"  SKIP: {msg}")
    return True


# ------------------------------------------------------------------ synthetic composite
def synthetic_composite(footer: bool = True):
    """A 752x562 canvas drawn with render_cutout's conventions: background (12,12,16), a
    (70,70,80) 1-px outline around every measured panel box, each panel filled with its own
    colour, a title strip above each panel, a footer line. Nothing else."""
    from PIL import Image, ImageDraw
    W, H = jf.COMPOSITE_SIZE
    img = Image.new("RGB", (W, H), views.BACKGROUND_RGB)
    d = ImageDraw.Draw(img)
    for s, (x0, y0, x1, y1) in views.PANEL_BOXES["color"].items():
        d.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], fill=SLOT_RGB[s])
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=views.OUTLINE_RGB)
        d.text((x0 + 1, y0 - views.TH + 3), f"title {s}", fill=(205, 205, 215))
    if footer:
        d.text((8, H - 22 + 4), "FOOTER id ra dec", fill=(180, 180, 195))
    return img


def _dominant_rgb(img):
    a = np.asarray(img.convert("RGB")).reshape(-1, 3)
    vals, counts = np.unique(a, axis=0, return_counts=True)
    return tuple(int(x) for x in vals[counts.argmax()])


def make_kit(d: Path, name: str = "itemX"):
    """One footer-cropped q92 JPEG of the synthetic composite (what a truth kit serves)."""
    p = d / f"{name}.jpg"
    jf.crop_footer(synthetic_composite()).save(p, format="JPEG", quality=92)
    return p, _util.sha_file(p)


# ------------------------------------------------------------------ views: geometry
def test_panel_boxes_measured():
    """The hard-coded boxes reproduce the renderer's arithmetic, the gloss file's boxes (when
    present) and a synthetic composite painted at exactly those slots."""
    assert views.COL_X == (8, 256, 504) and views.ROW_Y == (26, 292)
    assert views.PANEL_BOXES["color"] == views.PANEL_BOXES["gray_sw_only"] == views.PANEL_BOXES["gray_lw_only"]
    assert views.PANEL_BOXES["color"]["a"] == (8, 26, 248, 266)
    assert views.PANEL_BOXES["color"]["f"] == (504, 292, 744, 532)
    assert views.TITLED_BOXES["color"]["d"] == (8, 274, 248, 532)
    assert all(b[3] <= jf.FOOTER_Y for b in views.PANEL_BOXES["color"].values())
    img = synthetic_composite()
    assert img.size == (752, 562)
    assert all(views.outline_ok(img).values()), views.outline_ok(img)
    # every crop is the one flat colour painted at that slot, 240 px at scale 1, 480 at 2x
    for s, rgb in SLOT_RGB.items():
        tile = views.crop_panel(img, s, "color", scale=1)
        assert tile.size == (240, 240) and _dominant_rgb(tile) == rgb, s
        assert views.crop_panel(img, s, "color", scale=2).size == (480, 480)
        titled = views.crop_panel(img, s, "gray_sw_only", scale=2, crop="boxes_titled")
        assert titled.size == (480, 516) and _dominant_rgb(titled) == rgb
    # the footer-cropped kit JPEG shares the panel coordinates
    kit = jf.crop_footer(img)
    assert kit.size == (752, 540) and _dominant_rgb(views.crop_panel(kit, "e", scale=1)) == SLOT_RGB["e"]
    # a shifted canvas fails the outline measurement; a wrong size is refused
    shifted = views._pil().new("RGB", (752, 562), views.BACKGROUND_RGB)
    shifted.paste(img.crop((0, 0, 748, 558)), (4, 4))
    assert not any(views.outline_ok(shifted).values())
    try:
        views.crop_panel(img.resize((700, 500)), "a"); raise AssertionError("no raise")
    except ValueError:
        pass
    if views.GLOSS_PATH.exists():
        g = json.loads(views.GLOSS_PATH.read_text())
        for s in views.SLOTS:
            assert tuple(g["composite"]["boxes"][s]) == views.PANEL_BOXES["color"][s]
            assert tuple(g["composite"]["boxes_titled"][s]) == views.TITLED_BOXES["color"][s]
    # a real composite, when a stamp is on this machine (gitignored; skipped otherwise)
    stamps = sorted((LENSJUDGE / "golden" / "stamps").glob("*/*_v1.jpg"))
    if stamps:
        real = views._pil().open(stamps[0]).convert("RGB")
        assert real.size == (752, 562) and all(views.outline_ok(real).values()), stamps[0]
    else:
        _skip("no golden/stamps composite on this machine")


def test_gloss_file_builtin_parity_and_validation():
    """The built-in gloss has the file's keys and role/panel map; a gloss that would show a
    subtraction panel to a crop role, or wrong boxes, is refused."""
    b = views.BUILTIN_GLOSS
    assert set(b["roles"]) == set(views.ROLES) and set(b["layouts"]) == {"color", "gray"}
    if views.GLOSS_PATH.exists():
        f = json.loads(views.GLOSS_PATH.read_text())
        assert f["roles"] == b["roles"] and f["layouts"] == b["layouts"] and f["headers"] == b["headers"]
        for k, vs in f["view_sets"].items():
            assert vs["panels"] == b["view_sets"][k]["panels"] and vs["crop"] == b["view_sets"][k]["crop"], k
        assert views.load_gloss() == f            # the file is what runs on this machine
    else:
        _skip("panel_gloss.json absent: built-in gloss in use")
    for kind in ("color", "gray"):
        for role in views.CROP_ROLES:
            assert not set(views.role_slots(role, kind)) & set(b["layouts"][kind]["subtracted"]), (role, kind)
    bad = json.loads(json.dumps(b))
    bad["view_sets"]["morphology_gray"]["panels"] = ["c", "d", "e"]     # (c) is a 10" subtraction in gray
    try:
        views._validate_gloss(bad, "bad"); raise AssertionError("no raise")
    except ValueError as e:
        assert "subtracted" in str(e)
    bad = json.loads(json.dumps(b))
    bad["composite"]["boxes"]["a"] = [8, 8, 248, 248]
    try:
        views._validate_gloss(bad, "bad"); raise AssertionError("no raise")
    except ValueError as e:
        assert "measured" in str(e)
    # a bad file on disk raises rather than silently degrading; an absent one falls back
    d = Path(tempfile.mkdtemp(prefix="gloss_"))
    try:
        (d / "g.json").write_text(json.dumps(bad))
        try:
            views.load_gloss(d / "g.json"); raise AssertionError("no raise")
        except ValueError:
            pass
        assert views.load_gloss(d / "absent.json") == b
        try:
            views.load_gloss(d / "absent.json", required=True); raise AssertionError("no raise")
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(d)
    assert re.fullmatch(r"[0-9a-f]{16}", views.gloss_sha16())
    assert views.layout_kind("gray_lw_only") == "gray" and views.layout_kind("color") == "color"
    # every gloss string is item-agnostic and registered as a known template sha
    known = audit_traces.known_template_shas()
    for t in views.gloss_strings():
        assert not re.search(r"J\d{6,}[+-]\d{5,}", t) and "RA " not in t
        assert _util.sha_text(t) in known


def test_role_views_withhold_subtraction_in_both_layouts():
    """Advocate / artifact / arbitrator see the whole footer-cropped composite; geometry and
    morphology see per-panel crops that never include a subtraction slot, in BOTH layouts
    (gray: (c) is a 10" subtraction and must not reach morphology); ctx20 only to geometry."""
    img = synthetic_composite()
    ctx = views._pil().new("RGB", (664, 354), (1, 2, 3))
    for layout in views.LAYOUTS:
        kind = views.layout_kind(layout)
        sub = set(views.BUILTIN_GLOSS["layouts"][kind]["subtracted"])
        for role in views.FULL_ROLES:
            rv = views.role_views(img, layout, role, ctx20=ctx)
            assert len(rv) == 1 and rv[0][1].size == (752, 540)
            assert rv[0][0] == views.view_text(role, layout)
            assert "(f)" in rv[0][0]                              # the full view describes (f)
        for role in views.CROP_ROLES:
            rv = views.role_views(img, layout, role, ctx20=ctx)
            slots = views.role_slots(role, layout)
            assert not set(slots) & sub, (role, layout, slots)
            n_ctx = 1 if role == "geometry" else 0
            assert len(rv) == len(slots) + n_ctx
            for (label, tile), s in zip(rv, slots):
                assert label.startswith(f"panel ({s})") and "subtracted" not in label
                assert tile.size == (480, 516) and _dominant_rgb(tile) == SLOT_RGB[s]
            if n_ctx:
                assert rv[-1][1] is ctx and "20\"" in rv[-1][0]
                assert views.view_text(role, layout, with_ctx20=True).endswith(rv[-1][0])
            assert "No deflector-subtracted panel is included" in views.view_text(role, layout)
            # without ctx20 the geometry critic gets the panels only, and the VIEW text omits it
            assert len(views.role_views(img, layout, role)) == len(slots)
            assert views.view_text(role, layout, with_ctx20=False) == views.view_set(role, layout)["text"]
    assert views.role_slots("morphology", "color") == ("c", "d", "e")
    assert "c" not in views.role_slots("morphology", "gray_sw_only")
    assert views.role_slots("geometry", "color") == views.role_slots("geometry", "gray_lw_only") == ("b", "d", "e")
    try:
        views.role_views(img, "color", "nobody"); raise AssertionError("no raise")
    except ValueError:
        pass


def test_ctx20_image_from_synthetic_stamps():
    d = Path(tempfile.mkdtemp(prefix="ctx20_"))
    try:
        rng = np.random.default_rng(2026)
        yy, xx = np.mgrid[:640, :640]
        gal = 50.0 * np.exp(-np.hypot(yy - 319.5, xx - 319.5) / 40.0)
        sw = (gal + rng.normal(0, 1.0, (640, 640))).astype(np.float32)
        lw = (1.5 * gal + rng.normal(0, 1.0, (640, 640))).astype(np.float32)
        assert views.ctx20_image(d) is None                       # nothing there
        hdr = jf.stamp_header("X", 10.0, -5.0, "SW", "F150W", "obs", "url", 20.0, 640, 0.99)
        jf.write_stamp_fits(d / "X_SW_20as.fits", sw, hdr)
        one = views.ctx20_image(d)
        assert one.size == (views.CTX_PX + 2 * views.GAP, views.CTX_PX + views.TH + 2 * views.GAP)   # deep tile only
        jf.write_stamp_fits(d / "X_LW_20as.fits", lw, jf.stamp_header("X", 10.0, -5.0, "LW", "F277W", "obs", "url", 20.0, 640, 0.99))
        two = views.ctx20_image(d)
        assert two.size == (2 * views.CTX_PX + 3 * views.GAP, views.CTX_PX + views.TH + 2 * views.GAP)
        assert views.find_20as(d)["SW"].name == "X_SW_20as.fits"
        # the centre of the deep tile is bright, a corner is dark (the stretch ran)
        a = np.asarray(two.convert("L"))
        cy, cx = views.GAP + views.TH + views.CTX_PX // 2, views.GAP + views.CTX_PX // 2
        assert a[cy, cx] > 200 and a[cy + 140, cx + 140] < 60
        # a channel below the finite gate is dropped (LW gated -> deep tile only)
        jf.write_stamp_fits(d / "X_LW_20as.fits", lw, jf.stamp_header("X", 10.0, -5.0, "LW", "F277W", "obs", "url", 20.0, 640, 0.2))
        assert views.ctx20_image(d).size == one.size
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------ grader seams
class _Rec:
    """Replaces grader_direct.grade_candidate: records every call, returns a canned reply."""
    def __init__(self, reply):
        self.calls = []
        self.reply = reply      # callable(call dict) -> (record | None, raw)

    async def __call__(self, cand, *, model=None, system_prompt=None, content=None, trace_path=None,
                       schema=ImageGrade, **kw):
        call = {"cand": cand, "model": model, "system_prompt": system_prompt, "content": content,
                "trace_path": trace_path, "schema": schema, "kw": kw,
                "n_images": sum(1 for b in content if b.get("type") == "image")}
        self.calls.append(call)
        rec, raw = self.reply(call)
        return GradeResult(rec, raw, cost_usd=0.01, num_turns=1, parse_ok=rec is not None,
                           error=None if rec is not None else "parse",
                           meta={"name": cand.get("name"), "mode": "direct", "n_images": call["n_images"],
                                 "wall_s": 0.0, "n_thinking_blocks": 0, "thinking_chars": 0})


def _patch(reply):
    orig = grader_direct.grade_candidate
    stub = _Rec(reply)
    grader_direct.grade_candidate = stub
    return stub, orig


def test_grader_jwst_note_schema_extra_views_seams():
    d = Path(tempfile.mkdtemp(prefix="seams_"))
    ig = ImageGrade(grade="B", p_lens=0.6, confidence=0.7, rationale="stub")
    stub, orig = _patch(lambda call: (ig, ig.model_dump_json()))
    try:
        p, sha = make_kit(d)
        cand = {"name": "itemX", "image_path": str(p), "render_sha": sha}
        # note: default unchanged (DIRECT_SYS_JWST); "" appends nothing; another note appended once
        asyncio.run(grader_jwst.grade_candidate(cand, model="sonnet"))
        assert stub.calls[-1]["system_prompt"] == grader_jwst.DIRECT_SYS_JWST
        assert stub.calls[-1]["schema"] is ImageGrade and stub.calls[-1]["n_images"] == 1
        asyncio.run(grader_jwst.grade_candidate(cand, system_prompt="PERSONA", note=""))
        assert stub.calls[-1]["system_prompt"] == "PERSONA"
        for given in ("PERSONA", "PERSONA" + "NOTE2"):
            asyncio.run(grader_jwst.grade_candidate(cand, system_prompt=given, note="NOTE2"))
            assert stub.calls[-1]["system_prompt"] == "PERSONANOTE2"
        assert grader_jwst.with_note("X", "") == "X" and grader_jwst.with_note("X", "N") == "XN"
        assert grader_jwst.with_note("XN", "N") == "XN" and grader_jwst.with_note("X") == "X" + grader_jwst.JWST_NOTE
        # schema forwarded; the result carries it
        res = asyncio.run(grader_jwst.grade_candidate(cand, system_prompt="P", note="", schema=schemas_panel.IncumbentVerdict))
        assert stub.calls[-1]["schema"] is schemas_panel.IncumbentVerdict and res.meta["schema"] == "IncumbentVerdict"
        # extra views: [label, image] pairs after the candidate blocks; audit counts them
        tiles = [("[view 2] panel (d)", views._pil().new("RGB", (480, 516), (1, 2, 3))),
                 ("[view 3] ctx", str(p))]                         # a PIL image and a JPEG path
        tp = d / "t.jsonl"
        long_lead = "LEAD " * 100
        res = asyncio.run(grader_jwst.grade_candidate(
            cand, system_prompt="P", note="", schema=schemas_panel.CriticRecord, trace_path=str(tp),
            content=[{"type": "text", "text": long_lead}] + grader_jwst.jwst_content(cand, gloss="G")[1:],
            extra_views=tiles, audit_full_text=True))
        c = stub.calls[-1]["content"]
        assert [b["type"] for b in c] == ["text", "image", "text", "image", "text", "image"]
        assert c[2]["text"] == "[view 2] panel (d)" and c[3]["source"]["media_type"] == "image/png"
        assert c[4]["text"] == "[view 3] ctx" and c[5]["source"]["media_type"] == "image/jpeg"
        assert base64.b64decode(c[5]["source"]["data"]) == p.read_bytes()
        assert res.meta["n_extra_views"] == 2 and res.meta["render_sha"] == sha
        ev = [json.loads(l) for l in tp.read_text().splitlines()][0]
        assert ev["event"] == "golden_content_audit" and ev["schema"] == "CriticRecord"
        assert ev["n_images"] == 3 and ev["n_exemplars"] == 0 and ev["n_extra_views"] == 2
        assert ev["candidate_image_sha"] == sha and len(ev["extra_view_shas"]) == 2
        assert ev["jwst_note_sha16"] is None and ev["full_text"] is True
        assert ev["text_blocks"][0]["head"] == long_lead and ev["text_blocks"][0]["n_chars"] == len(long_lead)
        # the audit accepts it (rule 4: 1 + 0 + 2) and refuses a mis-declared count
        rep = audit_traces.audit([ev], ["J0000000+000000"], set(), set())
        assert rep["passed"], rep["violations"]
        bad = dict(ev, n_extra_views=1)
        checks = {v["check"] for v in audit_traces.audit([bad], [], set(), set())["violations"]}
        assert {"n_images", "extra_view_count"} <= checks
        # without full text a long non-template block is unverifiable (unchanged rule 5)
        ev2 = grader_jwst.content_audit(c, "P", 0, n_extra_views=2, note="")
        assert ev2["text_blocks"][0]["head"] == long_lead[:200]
        assert any(v["check"] == "unverifiable_text_block" for v in audit_traces.audit([ev2], [], set(), set())["violations"])
        # a gloss string IS a known template even past 200 chars
        ev3 = grader_jwst.content_audit(grader_jwst.jwst_content(cand, gloss=views.view_text("advocate", "color")), "P", 0)
        assert audit_traces.audit([ev3], [], set(), set())["passed"]
        # exemplars still precede the candidate: candidate index = n_exemplars
        ex = grader_jwst.jwst_content(cand)[1]
        ev4 = grader_jwst.content_audit([ex, ex] + grader_jwst.jwst_content(cand, gloss="G") + grader_jwst.view_blocks(tiles[:1]),
                                        "P", 2, n_extra_views=1)
        assert ev4["candidate_image_sha"] == sha and len(ev4["exemplar_image_shas"]) == 2 and ev4["n_images"] == 4
    finally:
        grader_direct.grade_candidate = orig
        shutil.rmtree(d)


def test_grader_direct_schema_forwarding_openai_path():
    """The open backend path validates into the schema it is given (and repairs into it)."""
    assert grader_direct._repair_text(ImageGrade) is _REPAIR
    rt = grader_direct._repair_text(schemas_panel.IncumbentVerdict)
    assert "id, persona, verdict, alternative, notes" in rt and rt.endswith(":\n\n")
    good = '{"id":"item","persona":"geometry","verdict":"pass","alternative":"","notes":"ok"}'
    seen = {"schemas": [], "repairs": 0}

    class R:
        def __init__(self, text):
            self.text, self.cost_usd, self.grade_probs = text, 0.001, None
            self.input_tokens = self.output_tokens = 1
            self.finish_reason = "stop"

    async def chat_with_images(*, system, content, model, max_tokens=None, json_schema=None, temperature=None):
        seen["schemas"].append(json_schema)
        return R(seen["first"])

    async def chat_text(*, system, text, model, max_tokens=None, **kw):
        seen["repairs"] += 1
        assert text.startswith(rt)
        return R(good)

    orig = (llm_client.is_open, llm_client.chat_with_images, llm_client.chat_text, llm_client.get_backend)
    llm_client.is_open = lambda: True
    llm_client.chat_with_images, llm_client.chat_text = chat_with_images, chat_text
    llm_client.get_backend = lambda: "openai"
    try:
        content = [{"type": "text", "text": "x"}]
        seen["first"] = good
        res = asyncio.run(grader_direct.grade_candidate({"name": "n"}, model="m", system_prompt="S", content=content,
                                                        schema=schemas_panel.IncumbentVerdict))
        assert res.parse_ok and isinstance(res.grade, schemas_panel.IncumbentVerdict) and res.grade.verdict == "pass"
        assert seen["schemas"][-1] == schemas_panel.IncumbentVerdict.model_json_schema()
        # an ImageGrade-shaped reply is NOT coerced into the panel schema: repair runs, then parses
        seen["first"] = '{"grade":"B","p_lens":0.5,"confidence":0.5,"rationale":"r"}'
        res = asyncio.run(grader_direct.grade_candidate({"name": "n"}, model="m", system_prompt="S", content=content,
                                                        schema=schemas_panel.IncumbentVerdict))
        assert res.parse_ok and seen["repairs"] == 1 and res.raw == good
        # default schema unchanged: ImageGrade parses the ImageGrade reply with no repair
        res = asyncio.run(grader_direct.grade_candidate({"name": "n"}, model="m", system_prompt="S", content=content))
        assert res.parse_ok and isinstance(res.grade, ImageGrade) and seen["repairs"] == 1
        assert seen["schemas"][-1] == ImageGrade.model_json_schema()
    finally:
        llm_client.is_open, llm_client.chat_with_images, llm_client.chat_text, llm_client.get_backend = orig


# ------------------------------------------------------------------ panel fan-out
def _advocate(p_ev=0.7, items=2, nothing=""):
    its = [schemas_panel.EvidenceItem(k=1, what="arc", panel="d", r_arcsec=1.3, pa_deg_from=40, pa_deg_to=170,
                                      visible_in_direct=True, criteria=[3, 5]),
           schemas_panel.EvidenceItem(k=2, what="counter", panel="e", r_arcsec=1.1, pa_deg_from=230, pa_deg_to=260,
                                      visible_in_direct=True, criteria=[4])][:items]
    return schemas_panel.AdvocateRecord(
        id="item", persona="advocate",
        criteria=schemas_panel.CriteriaV2(source_contrast=7, low_surface_brightness=6, curvature=8, counter_image=6, arc_morphology=7),
        items=its, scale_class="galaxy", n_red_neighbours_10as=0, bcg_like_halo=False, deflector_is_centre=True,
        p_evidence=p_ev, nothing_because=nothing, notes="tangential arc east of the core")


def _critic(role, alternative=None, r=0.0, no_opinion=False, reason=None, accounts=(), location=True):
    loc = schemas_panel.LocationBox(r_arcsec_from=1.0, r_arcsec_to=1.6, pa_deg_from=30, pa_deg_to=180) if location else None
    return schemas_panel.CriticRecord(id="item", persona=role, no_opinion=no_opinion, no_opinion_reason=reason,
                                      alternative=alternative, alternative_desc="" if alternative is None else "desc",
                                      location=loc if alternative else None, accounts_for=list(accounts),
                                      leaves_standing=[], refutation_strength=r if alternative else None,
                                      notes=f"{role} note")


def _arbitrator(ruling="overruled"):
    return schemas_panel.ArbitratorRecord(id="item", persona="arbitrator",
                                          rulings=[schemas_panel.Ruling(persona="morphology", ruling=ruling, covers=[1], why="w")],
                                          surviving_items=[1, 2], letter_llm="B", scale_class_final="galaxy",
                                          needs_human=False, rationale="the arc stays")


def _role_of(call, sysp):
    """Which role a stubbed call served: the persona set text it starts with."""
    sp = call["system_prompt"]
    if call["schema"] is schemas_panel.IncumbentVerdict:
        return re.search(r'"persona":"(\w+)"', sp).group(1)
    for role, text in sysp.items():
        if sp.startswith(text):
            return role
    raise AssertionError("unknown role for system prompt " + sp[:60])


def _panel_stub(sysp, records):
    """records: role -> record | None (None = parse failure) | callable(call) -> record."""
    def reply(call):
        role = _role_of(call, sysp)
        rec = records[role]
        if callable(rec):
            rec = rec(call)
        return rec, (rec.model_dump_json() if rec is not None else "not json")
    return _patch(reply)


def _setup(tmp, layout="color"):
    p, sha = make_kit(tmp)
    cand = {"name": "J0000000+000000", "image_path": str(p), "render_sha": sha, "layout": layout, "catalog": "jwst"}
    note = "\n\nNOTE-V2 TEST TEXT\n\nRespond with ONLY the JSON object.\n"
    th = aggregate_v2.resolve_thresholds(json.loads((LENSJUDGE / "golden" / "thresholds_v2.json").read_text()), "sonnet_api") \
        if (LENSJUDGE / "golden" / "thresholds_v2.json").exists() else aggregate_v2.resolve_thresholds({}, "x")
    return cand, note, th


def test_grade_panel_full_stack_color_and_gray():
    if not (PERSONAS_OK and NOTE_OK):
        return _skip("prompts/personas/jwst_v1 or jwst_note_v2.md absent")
    d = Path(tempfile.mkdtemp(prefix="panel_"))
    try:
        for layout in ("color", "gray_sw_only"):
            cand, note, th = _setup(d, layout)
            sysp = panel.load_persona_set(panel.PERSONA_DIR_V1, note)
            records = {"advocate": _advocate(0.7), "artifact": _critic("artifact"),
                       "geometry": _critic("geometry", no_opinion=True, reason="outside_competence"),
                       "morphology": _critic("morphology", "spiral_arm", r=0.5, accounts=[1]),
                       "arbitrator": _arbitrator("upheld")}
            stub, orig = _panel_stub(sysp, records)
            try:
                tr = d / f"traces_{layout}"
                res = asyncio.run(panel.grade_panel(cand, model="sonnet", persona_dir=panel.PERSONA_DIR_V1, note_text=note,
                                                    thresholds=th, mode="full", trace_dir=tr))
            finally:
                grader_direct.grade_candidate = orig
            roles = [_role_of(c, sysp) for c in stub.calls]
            assert roles == ["advocate", "artifact", "geometry", "morphology", "arbitrator"], roles
            by = dict(zip(roles, stub.calls))
            # every persona system prompt = its brief then the note, note LAST, appended once
            for r, c in by.items():
                assert c["system_prompt"] == sysp[r] + note and c["system_prompt"].count("NOTE-V2 TEST TEXT") == 1
                assert not c["system_prompt"].endswith(grader_jwst.JWST_NOTE)
                assert c["schema"] is schemas_panel.SCHEMA_FOR_ROLE[r]
            # views: composite roles get the kit JPEG bytes; crop roles get PNG crops without (f) / (c in gray)
            kit_b64 = grader_jwst.jwst_content(cand)[1]["source"]["data"]
            for r in ("advocate", "artifact", "arbitrator"):
                imgs = [b for b in by[r]["content"] if b["type"] == "image"]
                assert len(imgs) == 1 and imgs[0]["source"]["data"] == kit_b64, r
                assert views.view_text(r, layout) in by[r]["content"][0]["text"]
            kind = views.layout_kind(layout)
            sub = set(views.BUILTIN_GLOSS["layouts"][kind]["subtracted"])
            for r in ("geometry", "morphology"):
                c = by[r]
                texts = [b["text"] for b in c["content"] if b["type"] == "text"]
                labels = [t for t in texts if t.startswith("[view ")]
                slots = views.role_slots(r, layout)
                assert c["n_images"] == len(slots) == len(labels), (r, c["n_images"], labels)
                for lbl, s in zip(labels, slots):
                    assert f"panel ({s})" in lbl and s not in sub and "subtracted" not in lbl
                assert "No deflector-subtracted panel is included" in texts[0]
                assert all(b["source"]["media_type"] == "image/png" for b in c["content"] if b["type"] == "image")
                # critics see the numbered items and the scale class, never p_evidence / criteria / notes
                assert '"k":1' in texts[0] and '"k":2' in texts[0] and "galaxy" in texts[0]
                assert "p_evidence" not in texts[0] and "0.7" not in texts[0] and "tangential arc east" not in texts[0]
                assert cand["name"] not in "".join(texts)
            if layout == "gray_sw_only":
                assert "SINGLE-BAND" in by["morphology"]["content"][0]["text"]
                assert "panel (a)" in [t for t in (b["text"] for b in by["morphology"]["content"] if b["type"] == "text")][1]
            # artifact critic sees everything (the full composite, (f) included)
            assert "(f)" in by["artifact"]["content"][0]["text"] and '"k":1' in by["artifact"]["content"][0]["text"]
            # arbitrator gets the image + every record as JSON
            at = by["arbitrator"]["content"][0]["text"]
            assert '"persona":"advocate"' in at and '"persona":"morphology"' in at and '"persona":"geometry"' in at
            assert '"p_evidence":0.7' in at
            # result
            assert isinstance(res, schemas_panel.PanelResult) and res.parse_ok and res.calls == 5
            assert abs(res.cost_usd - 0.05) < 1e-9 and res.parse_failures == []
            a_m = aggregate_v2.coverage_fraction(records["advocate"].items, records["morphology"])
            assert abs(res.S - 0.7 * (1 - 0.5 * a_m)) < 1e-9 and 0 < a_m <= 1
            assert abs(res.S_arb - res.S) < 1e-9            # upheld -> same term
            assert res.letter in ("A", "B", "C", "D") and res.letter_source == th["letter_source"]
            assert set(res.system_sha16s) == set(roles)
            assert res.system_sha16s["advocate"] == _util.sha_text(sysp["advocate"] + note)
            assert res.meta["layout"] == layout and res.meta["ctx20_used"] is False and res.meta["skipped"] == []
            assert res.meta["n_images_by_role"]["geometry"] == len(views.role_slots("geometry", layout))
            row = panel.to_row(res, cand)
            assert tuple(row) == schemas_panel.ROW_COLS
            assert row["p_lens"] == res.S and row["grade_pred"] == res.letter and row["confidence"] == 0.7
            assert row["contaminant"] == "spiral_arm" and row["no_opinion_geometry"] is True and row["ruling_morphology"] == "upheld"
            # traces: one per role + the panel summary; the audit passes on the trace dir
            names = sorted(p.name for p in tr.glob("*.jsonl"))
            assert names == sorted(f"{_util.safe_name(cand['name'])}_{r}.jsonl" for r in roles + ["panel"])
            evs = audit_traces.read_events(tr)
            assert len(evs) == 5 and all(e["full_text"] for e in evs)
            rep = audit_traces.audit(evs, ["needs a much closer look at the ring"], set(), set())
            assert rep["passed"], rep["violations"]
            # a planted banned phrase in the items text IS caught (full-text audit)
            evs[1]["text_blocks"][0]["head"] += " needs a much closer look at the ring"
            assert not audit_traces.audit(evs, ["needs a much closer look at the ring"], set(), set())["passed"]
            summ = [json.loads(l) for l in (tr / f"{_util.safe_name(cand['name'])}_panel.jsonl").read_text().splitlines()][0]
            assert summ["event"] == "golden_panel" and summ["roles"] == roles and summ["S"] == res.S
    finally:
        shutil.rmtree(d)


def test_grade_panel_gating_modes_and_ctx20():
    if not (PERSONAS_OK and NOTE_OK):
        return _skip("prompts/personas/jwst_v1 or jwst_note_v2.md absent")
    d = Path(tempfile.mkdtemp(prefix="panel_"))
    try:
        cand, note, th = _setup(d)
        sysp = panel.load_persona_set(panel.PERSONA_DIR_V1, note)
        # p_evidence < tau0: advocate only; S = p_ev; no items + nothing_because -> D
        low = {"advocate": _advocate(0.05, items=0, nothing="isolated elliptical")}
        stub, orig = _panel_stub(sysp, low)
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, mode="full"))
        finally:
            grader_direct.grade_candidate = orig
        assert len(stub.calls) == 1 and res.calls == 1 and res.S == 0.05 and res.letter == "D"
        assert res.critics == {} and res.arbitrator is None and any(s.startswith("critics:") for s in res.meta["skipped"])
        assert panel.to_row(res, cand)["a_geometry"] is None
        # no critic names an alternative -> arbitrator skipped
        none_named = {"advocate": _advocate(0.7), "artifact": _critic("artifact"),
                      "geometry": _critic("geometry"), "morphology": _critic("morphology", no_opinion=True, reason="image_quality")}
        stub, orig = _panel_stub(sysp, none_named)
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, mode="full"))
        finally:
            grader_direct.grade_candidate = orig
        assert len(stub.calls) == 4 and res.arbitrator is None and res.S == 0.7 and "arbitrator:no_alternative" in res.meta["skipped"]
        assert res.parse_ok and math.isfinite(res.S_arb)
        # advocate_only ignores tau0 entirely
        stub, orig = _panel_stub(sysp, {"advocate": _advocate(0.9)})
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, mode="advocate_only"))
        finally:
            grader_direct.grade_candidate = orig
        assert len(stub.calls) == 1 and res.S == 0.9 and res.letter in ("A", "B")
        # attr: same prompts, the SAME full composite to every role
        full = {"advocate": _advocate(0.7), "artifact": _critic("artifact", "psf_wing", r=0.3, accounts=[2]),
                "geometry": _critic("geometry"), "morphology": _critic("morphology"), "arbitrator": _arbitrator()}
        stub, orig = _panel_stub(sysp, full)
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, mode="attr"))
        finally:
            grader_direct.grade_candidate = orig
        assert len(stub.calls) == 5
        kit_b64 = grader_jwst.jwst_content(cand)[1]["source"]["data"]
        for c in stub.calls:
            assert c["n_images"] == 1 and [b for b in c["content"] if b["type"] == "image"][0]["source"]["data"] == kit_b64
            assert views.view_text("advocate", "color") in c["content"][0]["text"]
            assert c["system_prompt"] == sysp[_role_of(c, sysp)] + note
        assert res.meta["mode"] == "attr" and res.calls == 5
        # ctx20: a stamp dir with 20" FITS adds the context pair to the geometry critic only
        sd = d / "stamps"; sd.mkdir()
        arr = (50.0 * np.exp(-np.hypot(*np.mgrid[:640, :640] - 319.5) / 40.0)).astype(np.float32)
        for ch, f in (("SW", "F150W"), ("LW", "F277W")):
            jf.write_stamp_fits(sd / f"X_{ch}_20as.fits", arr, jf.stamp_header("X", 1.0, 1.0, ch, f, "o", "u", 20.0, 640, 0.99))
        stub, orig = _panel_stub(sysp, full)
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, mode="full", stamp_dir=sd))
        finally:
            grader_direct.grade_candidate = orig
        by = {_role_of(c, sysp): c for c in stub.calls}
        assert by["geometry"]["n_images"] == 4 and by["morphology"]["n_images"] == 3 and by["artifact"]["n_images"] == 1
        assert views.view_set("geometry", "color")["ctx20_text"] in by["geometry"]["content"][0]["text"]
        assert views.view_set("geometry", "color")["ctx20_text"] not in by["morphology"]["content"][0]["text"]
        assert res.meta["ctx20_used"] is True and res.meta["n_images_by_role"]["geometry"] == 4
        # a render the manifest does not carry is refused before any call
        try:
            asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, render="v2r")); raise AssertionError("no raise")
        except FileNotFoundError:
            pass
        # v2r: the image AND its description ship as one unit — the advocate's first text block
        # carries the v2r VIEW (chi (f), not "CIRCULAR") and the full render_v2_desc.md; a
        # crop role in a full v2r stack is untouched (it never sees (f)); a missing description
        # refuses before any call
        p2 = d / "itemX_v2r.jpg"
        jf.crop_footer(synthetic_composite()).save(p2, format="JPEG", quality=80)      # different bytes
        sha2 = _util.sha_file(p2)
        cand2 = {**cand, "image_path_v2r": str(p2), "render_sha_v2r": sha2}
        desc = "Panel (f) of this composite is a SIGNED-CHI residual (test description)."
        stub, orig = _panel_stub(sysp, full)
        try:
            res = asyncio.run(panel.grade_panel(cand2, model="sonnet", note_text=note, thresholds=th, mode="full",
                                                render="v2r", render_desc=desc))
        finally:
            grader_direct.grade_candidate = orig
        by = {_role_of(c, sysp): c for c in stub.calls}
        for r in ("advocate", "artifact", "arbitrator"):
            t0 = by[r]["content"][0]["text"]
            assert "SIGNED-CHI" in t0 and desc in t0 and "CIRCULAR" not in t0.split("Direct (un-subtracted)")[0].split("(f)")[1], r
            assert views.view_text(r, "color", render="v2r", render_desc=desc) in t0
            v2r_b64 = grader_jwst.jwst_content({**cand, "image_path": str(p2), "render_sha": sha2})[1]["source"]["data"]
            assert [b for b in by[r]["content"] if b["type"] == "image"][0]["source"]["data"] == v2r_b64 != kit_b64
        assert desc not in by["geometry"]["content"][0]["text"] and desc not in by["morphology"]["content"][0]["text"]
        assert res.meta["render"] == "v2r" and res.meta["render_desc_sha16"] == _util.sha_text(desc)
        # the real description file, when present, is read by default and is what the tuple hashes
        if panel.RENDER_V2_DESC_PATH.exists():
            stub, orig = _panel_stub(sysp, {"advocate": _advocate(0.9)})
            try:
                res = asyncio.run(panel.grade_panel(cand2, model="sonnet", note_text=note, thresholds=th,
                                                    mode="advocate_only", render="v2r"))
            finally:
                grader_direct.grade_candidate = orig
            t0 = stub.calls[0]["content"][0]["text"]
            assert panel.RENDER_V2_DESC_PATH.read_text().strip() in t0 and "do NOT treat blue" in t0
            assert _util.sha_text(t0.split("\n\n" + "Respond")[0]) in audit_traces.known_template_shas()
        try:
            views.view_text("advocate", "color", render="v2r", render_desc=""); raise AssertionError("no raise")
        except ValueError as e:
            assert "description" in str(e)
        try:
            asyncio.run(panel.grade_panel(cand2, model="sonnet", note_text=note, thresholds=th, render="v2r", render_desc="  ")); raise AssertionError("no raise")
        except ValueError:
            pass
        try:
            asyncio.run(panel.grade_panel(cand2, model="sonnet", note_text=note, thresholds=th, render="v3")); raise AssertionError("no raise")
        except ValueError:
            pass
        try:
            asyncio.run(panel.grade_panel({**cand, "layout": ""}, model="sonnet", note_text=note, thresholds=th)); raise AssertionError("no raise")
        except ValueError:
            pass
    finally:
        shutil.rmtree(d)


def test_grade_panel_parse_failures():
    if not (PERSONAS_OK and NOTE_OK):
        return _skip("prompts/personas/jwst_v1 or jwst_note_v2.md absent")
    d = Path(tempfile.mkdtemp(prefix="panel_"))
    try:
        cand, note, th = _setup(d)
        sysp = panel.load_persona_set(panel.PERSONA_DIR_V1, note)
        recs = {"advocate": _advocate(0.7), "artifact": _critic("artifact"), "geometry": None,
                "morphology": _critic("morphology", "spiral_arm", r=0.9, accounts=[1, 2]), "arbitrator": _arbitrator()}
        stub, orig = _panel_stub(sysp, recs)
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, mode="full", trace_dir=d / "tr"))
        finally:
            grader_direct.grade_candidate = orig
        assert len(stub.calls) == 4                         # arbitrator skipped: the row is already a parse failure
        assert res.parse_failures == ["geometry"] and math.isnan(res.S) and math.isnan(res.S_arb) and res.letter is None
        assert res.critics["geometry"] is None and res.critics["morphology"] is not None
        assert "arbitrator:parse_fail" in res.meta["skipped"] and res.meta["errors"]["geometry"] == "parse"
        row = panel.to_row(res, cand)
        assert math.isnan(row["p_lens"]) and row["grade_pred"] is None and row["parse_fail_roles"] == "geometry"
        assert row["error"] == "parse_fail:geometry" and row["parse_ok"] is False
        # advocate failure: one call, nothing else
        stub, orig = _panel_stub(sysp, {"advocate": None})
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", note_text=note, thresholds=th, mode="full"))
        finally:
            grader_direct.grade_candidate = orig
        assert len(stub.calls) == 1 and res.parse_failures == ["advocate"] and math.isnan(res.S) and res.advocate is None
        assert panel.to_row(res, cand)["confidence"] is None
    finally:
        shutil.rmtree(d)


def test_incumbent_mode_passcount_and_claims():
    if not INCUMBENT_OK:
        return _skip("prompts/personas/incumbent absent")
    d = Path(tempfile.mkdtemp(prefix="panel_"))
    try:
        cand, note, th = _setup(d)
        verdicts = {"artifact": schemas_panel.IncumbentVerdict(id="item", persona="artifact", verdict="pass", notes="clean"),
                    "morphology": schemas_panel.IncumbentVerdict(id="item", persona="morphology", verdict="fail", alternative="spiral arm", notes="arm"),
                    "geometry": schemas_panel.IncumbentVerdict(id="item", persona="geometry", verdict="uncertain", notes="?")}
        stub, orig = _panel_stub({}, verdicts)
        try:
            res = asyncio.run(panel.grade_panel(cand, model="sonnet", thresholds=th, mode="incumbent", trace_dir=d / "tr"))
        finally:
            grader_direct.grade_candidate = orig
        roles = [_role_of(c, {}) for c in stub.calls]
        assert roles == list(panel.INCUMBENT_ROLES)
        sysp = panel.load_incumbent_set(panel.PERSONA_DIR_INCUMBENT, claim_in_user=False)
        for c in stub.calls:
            r = _role_of(c, {})
            assert c["system_prompt"] == sysp[r]                                  # self-contained wrapper, no note
            assert not c["system_prompt"].endswith(grader_jwst.JWST_NOTE) and "{brief}" not in c["system_prompt"]
            assert panel.CLAIM_ABSENT in c["system_prompt"] and c["schema"] is schemas_panel.IncumbentVerdict
            assert c["n_images"] == 1 and c["content"][0]["text"] == "[composite]"
        assert isinstance(res, panel.IncumbentResult)
        assert (res.n_pass, res.n_fail, res.n_uncertain, res.letter) == (1, 1, 1, "C")
        assert aggregate_v2.passcount_incumbent(list(verdicts.values())) == (1, 1, 1, "C")
        assert res.calls == 3 and abs(res.cost_usd - 0.03) < 1e-9 and res.parse_ok
        row = panel.to_row(res, cand)
        assert set(row) == set(schemas_panel.ROW_COLS) | set(panel.INCUMBENT_EXTRA_COLS)
        assert row["grade_pred"] == "C" and abs(row["p_lens"] - 1 / 3) < 1e-9 and row["verdict_morphology"] == "fail"
        assert row["contaminant"] == "spiral arm" and row["alt_morphology"] == "spiral arm" and row["escalate"] is True
        assert row["letter_source"] == "passcount_incumbent" and math.isnan(row["S"])
        assert sorted(p.name for p in (d / "tr").glob("*.jsonl")) == sorted(
            f"{_util.safe_name(cand['name'])}_{x}.jsonl" for x in ["incumbent_" + r for r in roles] + ["panel"])
        # claim_mode=inspector: the claim rides in the USER message; the system prompt stays item-agnostic
        claim = {"lens_at_center": "yes", "quadrant_lens": "NE", "evidence": "faint arc to the NE " + "x" * 500}
        stub, orig = _panel_stub({}, verdicts)
        try:
            res2 = asyncio.run(panel.grade_panel(cand, model="sonnet", thresholds=th, mode="incumbent", claim=claim))
        finally:
            grader_direct.grade_candidate = orig
        sysp2 = panel.load_incumbent_set(panel.PERSONA_DIR_INCUMBENT, claim_in_user=True)
        for c in stub.calls:
            assert c["system_prompt"] == sysp2[_role_of(c, {})] and panel.CLAIM_IN_USER in c["system_prompt"]
            t = c["content"][0]["text"]
            assert t.startswith("[composite]") and '"claim_center": yes' in t and '"claim_quadrant": NE' in t
            assert "faint arc to the NE" in t and len(t) < 520            # evidence cut at 400 chars
            assert cand["name"] not in t
        assert res2.meta["claim_mode"] == "inspector" and res.meta["claim_mode"] == "none"
        assert res2.system_sha16s != res.system_sha16s               # the two claim modes are different tuples
        # the wrapper must say what the user message does: a pre-loaded CLAIM set on an item
        # without a claim is refused (C2: 220/282 holdout items have no claim), and
        # persona_set_noclaim is what such an item gets; the reverse mismatch is refused too
        try:
            asyncio.run(panel.grade_panel(cand, model="sonnet", thresholds=th, mode="incumbent", persona_set=sysp2)); raise AssertionError("no raise")
        except ValueError as e:
            assert "promises a claim" in str(e)
        try:
            asyncio.run(panel.grade_panel(cand, model="sonnet", thresholds=th, mode="incumbent", persona_set=sysp, claim=claim)); raise AssertionError("no raise")
        except ValueError as e:
            assert "denies a claim" in str(e)
        stub, orig = _panel_stub({}, verdicts)
        try:
            res4 = asyncio.run(panel.grade_panel(cand, model="sonnet", thresholds=th, mode="incumbent",
                                                 persona_set=sysp2, persona_set_noclaim=sysp))
        finally:
            grader_direct.grade_candidate = orig
        for c in stub.calls:
            assert c["system_prompt"] == sysp[_role_of(c, {})] and panel.CLAIM_ABSENT in c["system_prompt"]
            assert c["content"][0]["text"] == "[composite]"
        assert res4.system_sha16s == res.system_sha16s
        assert panel.claim_fields("just text")["claimed_evidence"] == "just text" and panel.claim_fields(None) is None
        assert panel.claim_fields({"lens_at_center": float("nan")})["claim_center"] == panel.CLAIM_ABSENT
        # a persona that fails to parse: letter None, p_lens NaN, counts over the rest
        stub, orig = _panel_stub({}, {**verdicts, "geometry": None})
        try:
            res3 = asyncio.run(panel.grade_panel(cand, model="sonnet", thresholds=th, mode="incumbent"))
        finally:
            grader_direct.grade_candidate = orig
        assert res3.parse_failures == ["geometry"] and res3.letter is None and math.isnan(res3.p_lens)
        assert (res3.n_pass, res3.n_fail, res3.n_uncertain) == (1, 1, 0)
        assert panel.to_row(res3, cand)["error"] == "parse_fail:geometry"
    finally:
        shutil.rmtree(d)


def test_persona_set_loading_conventions():
    d = Path(tempfile.mkdtemp(prefix="pset_"))
    try:
        for f in ("advocate", "critic_common", "critic_artifact", "critic_geometry", "critic_morphology"):
            (d / f"{f}.md").write_text(f"[{f}]\n")
        try:
            panel.load_persona_set(d); raise AssertionError("no raise")
        except FileNotFoundError as e:
            assert "arbitrator.md" in str(e)
        (d / "arbitrator.md").write_text("[arbitrator]\n")
        s = panel.load_persona_set(d, "NOTE")
        assert s["geometry"] == "[critic_common]\n" + panel.CRITIC_JOIN + "[critic_geometry]\n"
        assert s["advocate"] == "[advocate]\n" and set(s) == set(panel.ROLES)
        try:
            panel.load_persona_set(d, "[advocate]"); raise AssertionError("no raise")   # note already inside a brief
        except ValueError:
            pass
        sha = panel.persona_set_sha16(d)
        assert re.fullmatch(r"[0-9a-f]{16}", sha) and sha == panel.persona_set_sha16(d)
        (d / "advocate.md").write_text("[advocate] changed\n")
        assert panel.persona_set_sha16(d) != sha
        # incumbent wrapper filling
        w = "PRE\n{brief}\nclaims {claim_center}/{claim_quadrant}/{claimed_evidence}\n{\"id\":\"{item_id}\",\"persona\":\"{persona}\"}\n"
        (d / "wrapper.md").write_text(w)
        for r in panel.INCUMBENT_ROLES:
            (d / f"{r}.md").write_text(f"BRIEF {r}")
        inc = panel.load_incumbent_set(d)
        assert inc["artifact"] == 'PRE\nBRIEF artifact\nclaims not available/not available/not available\n{"id":"item","persona":"artifact"}\n'
        assert panel.CLAIM_IN_USER in panel.load_incumbent_set(d, claim_in_user=True)["geometry"]
        (d / "wrapper.md").write_text(w + "{unknown_token}")
        try:
            panel.load_incumbent_set(d); raise AssertionError("no raise")
        except ValueError:
            pass
        (d / "wrapper.md").write_text(w.replace("{brief}", ""))
        # a wrapper without {brief} still fills (the brief is simply not inserted) — the
        # prompt tests own that contract; here only unknown leftovers are refused
        assert "BRIEF" not in panel.load_incumbent_set(d)["artifact"]
        if PERSONAS_OK and NOTE_OK:
            real = panel.load_persona_set(panel.PERSONA_DIR_V1, panel.read_note())
            for r, t in real.items():
                assert t.rstrip().endswith(panel.RESPOND), r
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------ audit_traces: --pi-only
def test_parse_record_after_prose_with_arcsec_quotes():
    """common/parse.extract_json_block must return the RECORD, not its last nested object,
    when the model prefaces the JSON with prose carrying an odd number of double quotes
    (arcsecond marks). Seen on the first API smoke: a morphology critic wrote 'Panel (d)
    deep stretch 3.5"...' then a valid CriticRecord, and the old scanner returned its
    `measured` dict -> parse failure -> paid repair retry."""
    from lensjudge.common import parse
    rec = {"id": "item", "persona": "morphology", "no_opinion": False, "no_opinion_reason": None,
           "alternative": "merger", "alternative_desc": "two nuclei ~0.6\" apart",
           "location": {"r_arcsec_from": 0.0, "r_arcsec_to": 2.0, "pa_deg_from": 0.0, "pa_deg_to": 360.0},
           "accounts_for": [1, 2, 3], "leaves_standing": [4], "refutation_strength": 0.45,
           "measured": {"secondary_nucleus_offset_arcsec": 0.6, "envelope_elongation_pa": "east-west"},
           "scale_class": "galaxy", "notes": "constant radius ~1.2\" but a stellar bridge."}
    prose = ('**Panel (d) deep stretch 3.5"**: a bright core slightly west of the tick.\n'
             '**Panel (c) colour 10"**: irregular outer structure; a companion 0.6" east.\n\n')
    text = prose + json.dumps(rec, indent=1)
    got = parse.parse_model(text, schemas_panel.CriticRecord)
    assert got is not None and got.alternative == "merger" and got.accounts_for == [1, 2, 3]
    assert parse.extract_json_block(text) == rec
    # the same record with no prose, fenced, and with a brace inside a string still parse
    assert parse.extract_json_block(json.dumps(rec)) == rec
    assert parse.extract_json_block("```json\n" + json.dumps(rec) + "\n```") == rec
    assert parse.extract_json_block('{"s": "brace { in string }", "k": 1}') == {"s": "brace { in string }", "k": 1}
    # last balanced object still wins; junk stays None
    assert parse.extract_json_block('{"a": 1} then {"b": 2}') == {"b": 2}
    assert parse.extract_json_block('prose 3.5" only {broken') is None
    # an ImageGrade after prose with an odd quote count parses too (the DESI path)
    ig = {"grade": "C", "p_lens": 0.4, "confidence": 0.5, "criteria": {"blue_source": 3, "counter_images": 2, "curvature": 4,
                                                     "arc_morphology": 3, "low_surface_brightness": 2},
          "contaminant": "", "escalate_to_human": False, "rationale": "faint arc 1.1\" NE"}
    got2 = parse.parse_model('Looking at the 3.5" zoom first.\n' + json.dumps(ig), ImageGrade)
    assert got2 is not None and got2.grade == "C"


def test_audit_traces_pi_only_lexicon_and_extra_ids():
    d = Path(tempfile.mkdtemp(prefix="lex_"))
    try:
        import pandas as pd
        pd.DataFrame({"candidate_id": ["J1111111+111111", "J2222222-222222"], "half": ["holdout", "holdout"]}).to_csv(d / "ids.csv", index=False)
        (d / "ids.txt").write_text("# comment\nJ3333333+333333\n\n")
        assert audit_traces.read_ids_file(d / "ids.csv") == ["J1111111+111111", "J2222222-222222"]
        assert audit_traces.read_ids_file(d / "ids.txt") == ["J3333333+333333"]
        lex = audit_traces.build_lexicon(audit_traces.EMPTY_SPLITS, None, None, pi_comments=("the dissenting persona is right here",),
                                         extra_ids=["J1111111+111111", "J1111111+111111"])
        assert lex == ["the dissenting persona is right here", "J1111111+111111"]
        assert audit_traces.banned_hit("see J1111111+111111 there", lex) is not None
        # CLI: --pi-only needs no splits file; a full build without splits still refuses
        rc = audit_traces.main(["--build-lexicon", "--pi-only", "--banned", str(d / "lex.txt"), "--splits", str(d / "nosplits.csv"),
                                "--pi-comments", str(d / "absent.txt"), "--allow-missing-pi-comments",
                                "--extra-ids", str(d / "ids.csv"), "--extra-ids", str(d / "ids.txt")])
        assert rc == 0 and audit_traces.load_lexicon(d / "lex.txt") == ["J1111111+111111", "J2222222-222222", "J3333333+333333"]
        try:
            audit_traces.main(["--build-lexicon", "--banned", str(d / "lex2.txt"), "--splits", str(d / "nosplits.csv"),
                               "--pi-comments", str(d / "absent.txt"), "--allow-missing-pi-comments"])
            raise AssertionError("no raise")
        except SystemExit as e:
            assert "--pi-only" in str(e)
        try:
            audit_traces.main(["--pi-only", "--traces-dir", str(d)]); raise AssertionError("no raise")
        except SystemExit:
            pass
        # with the real PI comments on this machine the persona set is lexicon-clean
        pi = audit_traces.load_pi_comments(required=False)
        if pi and PERSONAS_OK and NOTE_OK:
            note = panel.read_note()
            for r, t in panel.load_persona_set(panel.PERSONA_DIR_V1, note).items():
                assert audit_traces.banned_hit(t + note, pi) is None, r
            for t in views.gloss_strings():
                assert audit_traces.banned_hit(t, pi) is None
        else:
            _skip("pi_comments.txt or persona set absent: real lexicon check not run")
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback; traceback.print_exc()
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
