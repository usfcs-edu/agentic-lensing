#!/usr/bin/env python3
"""No-API tests for the Part-2 prompt set (WP-1): the evidence-first JWST personas
(prompts/personas/jwst_v1), their layout-conditional view gloss (panel_gloss.json), the
role-neutral note (prompts/jwst_note_v2.md) and the byte-faithful incumbent briefs
(prompts/personas/incumbent).

What is asserted, and why:
  * every jwst_v1 file ends with the one-line "Respond with ONLY the JSON object." and
    spells out the record contract with the exact field names of golden/schemas_panel.py
    (checked against the pydantic models when that module exists, against a frozen list
    otherwise) — a renamed key is a parse failure, never a silent coercion;
  * the two removed priors ("bluer than the deflector", "theta_E roughly 0.3-3 arcsec;
    larger implies a group/cluster") and the veto sentences ("prefer fail", "mark it fail")
    are absent from every v1 prompt and from the v2 note, while the forbidden-grounds block,
    the symmetric mandate and the refutation_strength anchors are present;
  * the prompts are layout-neutral and the gloss is layout-conditional: no .md describes a
    colour panel, the gloss never shows a subtracted panel to geometry/morphology in either
    layout, and its crop boxes match common/jwst_fetch.render_cutout's geometry;
  * the incumbent briefs are byte-equal to LENS_BRIEF in J/scripts/verify_workflow.js
    (re-extracted from lines 33-72; skipped when the repo is absent) and pinned by sha16 so
    the check also runs without J; the wrapper keeps PANEL LAYOUT + STEP 2 verbatim and
    carries no tool/job-file text;
  * the gloss is render-conditional: `renders["v2r"]` maps every composite view set to a twin
    whose (f) sentence is the signed-chi description and which carries `render_desc: true`, so
    views.view_text ships golden/render_v2_desc.md with the v2r image (and refuses without
    it); the circular-subtraction caveat lives in the v1 composite texts, and neither the
    persona .md files nor the note say what model panel (f) removed (the VIEW does);
  * EMBARGO: no model-facing string (every .md, every gloss string, render_v2_desc.md, and
    the concatenations a runner would actually send) hits a 4-gram of golden/pi_comments.txt,
    contains a candidate id / coordinate pattern, or names a rank / the poster child; and no
    TRACKED file under golden/, prompts/ or tests/ (a coding agent reads those) shares a 4-gram
    with a PI comment either. The comment file is read as the lexicon source and NEVER
    printed — on a hit only the file name, window length and the window's sha16 are reported.

Run as a script or under pytest (no network, no API):
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_prompts.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lensjudge.golden import _util, audit_traces  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "prompts"
V1 = PROMPTS / "personas" / "jwst_v1"
INCUMBENT = PROMPTS / "personas" / "incumbent"
NOTE_V1 = PROMPTS / "jwst_note.md"
NOTE_V2 = PROMPTS / "jwst_note_v2.md"
GLOSS = V1 / "panel_gloss.json"
RENDER_V2_DESC = HERE.parent / "golden" / "render_v2_desc.md"
LENSJUDGE = HERE.parent

V1_ROLES = ("advocate", "critic_common", "critic_artifact", "critic_geometry",
            "critic_morphology", "arbitrator")
CRITICS = ("artifact", "geometry", "morphology")
INCUMBENT_ROLES = ("artifact", "morphology", "geometry")
RESPOND = "Respond with ONLY the JSON object."

# sha16 of the three LENS_BRIEF template strings in J/scripts/verify_workflow.js @ 7632b39
# (lines 33-72). Pinned so the byte-equality check has teeth on a machine without J.
INCUMBENT_SHA16 = {"artifact": "1339f2bfe6f2ef5c", "morphology": "65a3bc2c2a7bc023",
                   "geometry": "46b85ea02ddcb0a6"}
WRAPPER_PLACEHOLDERS = {"brief", "persona", "item_id", "claim_center", "claim_quadrant",
                        "claimed_evidence"}

# Record contracts (GOLDEN_CONTRACT_2 "Pydantic records"); the live schemas_panel models
# are preferred when importable, this frozen copy is the fallback.
ADVOCATE_FIELDS = ("id", "persona", "criteria", "items", "arc_radius_arcsec", "arc_pa_span_deg",
                   "counter_image_pos", "centre_of_curvature_offset_arcsec", "scale_class",
                   "n_red_neighbours_10as", "bcg_like_halo", "deflector_is_centre", "p_evidence",
                   "nothing_because", "notes")
CRITERIA = ("source_contrast", "low_surface_brightness", "curvature", "counter_image", "arc_morphology")
ITEM_FIELDS = ("k", "what", "panel", "r_arcsec", "pa_deg_from", "pa_deg_to", "visible_in_direct", "criteria")
CRITIC_FIELDS = ("id", "persona", "no_opinion", "no_opinion_reason", "alternative", "alternative_desc",
                 "location", "accounts_for", "leaves_standing", "refutation_strength", "measured",
                 "scale_class", "notes")
LOCATION_FIELDS = ("r_arcsec_from", "r_arcsec_to", "pa_deg_from", "pa_deg_to")
ALTERNATIVES = ("spiral_arm", "ring_galaxy", "shell_tidal", "merger", "edge_on_disk",
                "companion_projection", "star_forming_clump", "diffraction_spike",
                "detector_artifact", "subtraction_residual", "psf_wing", "scale_tension", "other")
NO_OPINION_REASONS = ("outside_competence", "feature_not_in_my_views", "image_quality")
ARBITRATOR_FIELDS = ("id", "persona", "rulings", "surviving_items", "letter_llm", "scale_class_final",
                     "needs_human", "rationale")
RULING_FIELDS = ("persona", "ruling", "covers", "why")
RULINGS = ("upheld", "partial", "overruled")
SCALE_CLASSES = ("galaxy", "group", "cluster", "none")

# synthetic stand-ins for the PI's comments on a machine without golden/pi_comments.txt
# (the real strings live only in that gitignored file and never in a test)
SYNTH_PI = ("needs a much closer look at the ring", "ask the tool to hyperlink every catalogue entry",
            "the dissenting persona is right here")


# ------------------------------------------------------------------ helpers
def read(role: str, base: Path = V1) -> str:
    return (base / f"{role}.md").read_text()


def fold(t: str) -> str:
    """Whitespace-folded text, so phrase assertions do not depend on line wrapping."""
    return re.sub(r"\s+", " ", t)


def gloss() -> dict:
    return json.loads(GLOSS.read_text())


def gloss_strings(g=None) -> dict[str, str]:
    """Every string leaf of the gloss, keyed by its JSON path (all are model-facing
    except `_what`, which is included anyway: it must be clean too)."""
    out: dict[str, str] = {}

    def walk(o, pre):
        if isinstance(o, str):
            out[pre] = o
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{pre}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{pre}[{i}]")
    walk(g if g is not None else gloss(), "panel_gloss.json")
    return out


def extract_lens_briefs(js_text: str) -> dict[str, str]:
    """The LENS_BRIEF template strings of verify_workflow.js (lines 33-72 at 7632b39), keyed
    by persona. Regex over the `const LENS_BRIEF = {...}` block; the strings contain no
    backticks and no ${} interpolation, so a backtick pair delimits each verbatim."""
    m = re.search(r"const LENS_BRIEF = \{(.*?)\n\}\n", js_text, re.S)
    assert m, "LENS_BRIEF block not found"
    briefs = dict(re.findall(r"(\w+): `([^`]*)`", m.group(1)))
    assert set(briefs) == set(INCUMBENT_ROLES), sorted(briefs)
    return briefs


def model_facing_strings() -> dict[str, str]:
    """Everything a runner can send to a model from this package: each .md, each gloss
    string, the v2 note, and the concatenations that actually form a system prompt
    (role [+ common] + note) — a 4-gram could straddle a file boundary."""
    s: dict[str, str] = {}
    for r in V1_ROLES:
        s[f"jwst_v1/{r}.md"] = read(r)
    s["jwst_note_v2.md"] = NOTE_V2.read_text()
    for r in INCUMBENT_ROLES + ("wrapper",):
        s[f"incumbent/{r}.md"] = read(r, INCUMBENT)
    s.update(gloss_strings())
    if RENDER_V2_DESC.exists():
        s["golden/render_v2_desc.md"] = RENDER_V2_DESC.read_text()
        g = gloss()
        for dst in g["renders"]["v2r"].values():       # the v2r VIEW paragraph + description, as sent
            s[f"concat/{dst}+render_v2_desc"] = g["view_sets"][dst]["text"] + "\n\n" + RENDER_V2_DESC.read_text().strip()
    note = NOTE_V2.read_text()
    s["concat/advocate+note"] = read("advocate") + note
    s["concat/arbitrator+note"] = read("arbitrator") + note
    for c in CRITICS:
        s[f"concat/critic_common+critic_{c}+note"] = read("critic_common") + "\n" + read(f"critic_{c}") + note
    for r in INCUMBENT_ROLES:
        s[f"concat/incumbent_wrapper[{r}]"] = fill_wrapper(read("wrapper", INCUMBENT), r, read(r, INCUMBENT))
    return s


def fill_wrapper(wrapper: str, persona: str, brief: str, item_id: str = "item",
                 claim_center: str = "not available", claim_quadrant: str = "not available",
                 claimed_evidence: str = "not available") -> str:
    """Fill the incumbent wrapper by token replacement (NOT str.format: the text carries
    literal JSON braces). This is the contract a runner should follow."""
    out = wrapper
    for k, v in (("brief", brief), ("persona", persona), ("item_id", item_id),
                 ("claim_center", claim_center), ("claim_quadrant", claim_quadrant),
                 ("claimed_evidence", claimed_evidence)):
        out = out.replace("{" + k + "}", v)
    return out


def pi_lexicon() -> tuple[list[str], bool]:
    """(lexicon entries, real) — the PI comments when the gitignored file is present (count +
    sha16 verified by the loader), else the synthetic stand-ins with real=False."""
    pi = audit_traces.load_pi_comments(required=False)
    return (pi, True) if pi else (list(SYNTH_PI), False)


def _hit_msg(name: str, hit) -> str:
    # never the comment text: only where, how long, and a hash of the shared window
    return (f"{name}: shares a {len(hit[1].split())}-word window ({len(hit[1])} chars, "
            f"sha16 {_util.sha_text(hit[1])}) with a PI comment")


# ------------------------------------------------------------------ jwst_v1 files
def test_v1_files_exist_and_end_with_respond_line():
    for r in V1_ROLES:
        t = read(r)
        assert t.strip(), r
        assert t.rstrip("\n").splitlines()[-1] == RESPOND, (r, t.rstrip().splitlines()[-1])
        assert t.count(RESPOND) == 1, r
        assert t.endswith("\n"), r
    # the note is appended AFTER a role prompt by the runner, so it ends with the same line
    note = NOTE_V2.read_text()
    assert note.rstrip("\n").splitlines()[-1] == RESPOND
    assert note.startswith("\n\n#"), "note_v2 keeps the v1 leading blank lines (appended to a prompt)"


def _schema_field_names():
    """Field names (and nested field names) from golden/schemas_panel.py when it exists."""
    try:
        from lensjudge.golden import schemas_panel as sp  # noqa: WPS433
    except Exception:  # noqa: BLE001 — another work package; frozen lists cover the contract
        return None
    out = {}
    for name in ("AdvocateRecord", "CriticRecord", "ArbitratorRecord", "IncumbentVerdict", "EvidenceItem"):
        cls = getattr(sp, name, None)
        if cls is not None and hasattr(cls, "model_fields"):
            out[name] = tuple(cls.model_fields)
    return out


def test_v1_record_contracts_name_every_field():
    adv, common, arb = read("advocate"), read("critic_common"), read("arbitrator")
    for f in ADVOCATE_FIELDS + CRITERIA + ITEM_FIELDS + SCALE_CLASSES:
        assert f'"{f}"' in adv, ("advocate", f)
    assert '"persona": "advocate"' in adv
    assert '"panel": "a|b|c|d|e|f|ctx"' in adv
    for f in CRITIC_FIELDS + LOCATION_FIELDS + ALTERNATIVES + NO_OPINION_REASONS + CRITICS:
        assert f'"{f}"' in common, ("critic_common", f)
    for c in CRITICS:                       # each critic states its own persona literal
        assert f'Your persona string is "{c}"' in fold(read(f"critic_{c}")), c
        assert "COMPETENCE:" in read(f"critic_{c}"), c
    for f in ARBITRATOR_FIELDS + RULING_FIELDS + RULINGS + CRITICS + SCALE_CLASSES:
        assert f'"{f}"' in arb, ("arbitrator", f)
    assert '"persona": "arbitrator"' in arb
    for letter in "ABCD":
        assert f'"{letter}"' in arb, letter
    live = _schema_field_names()
    if live is None:
        print("  (schemas_panel not importable yet: contract checked against the frozen field lists)")
        return
    for name, fields, text in (("AdvocateRecord", ADVOCATE_FIELDS, adv), ("CriticRecord", CRITIC_FIELDS, common),
                               ("ArbitratorRecord", ARBITRATOR_FIELDS, arb), ("EvidenceItem", ITEM_FIELDS, adv)):
        if name in live:
            assert set(live[name]) == set(fields), (name, sorted(set(live[name]) ^ set(fields)))
            for f in live[name]:
                assert f'"{f}"' in text, (name, f)


def test_v1_priors_and_veto_language_absent_mandates_present():
    note = NOTE_V2.read_text()
    texts = {r: read(r) for r in V1_ROLES}
    texts["note_v2"] = note
    texts.update(gloss_strings())
    for name, t in texts.items():
        low = t.lower()
        assert "bluer" not in low, name                                    # prior 1 (colour)
        assert not re.search(r"0\.3\s*[-–]\s*3\s*[\"″']", t), name          # prior 2 (theta_E range)
        assert "larger implies" not in low, name
        assert "prefer fail" not in low and 'prefer "fail"' not in low, name
        assert "mark it fail" not in low, name
        assert "default expectation" not in low, name
        assert "refute each claim" not in low, name
    common = fold(texts["critic_common"])
    # forbidden grounds (i) theta_E, (ii) colour alone, (iii) symmetric subtraction residual
    assert "You must NOT:" in common
    assert "implied Einstein radius as a refutation" in common
    assert '"scale_tension" with refutation_strength <= 0.4' in common
    assert "use colour alone as a refutation" in common
    assert "butterfly or bowtie residual" in common and "VIEW description says what model was removed" in common
    assert "circular" not in common.lower()          # the subtraction model is the VIEW's to state (render-conditional)
    assert 'counts as "subtraction_residual" ONLY if it is absent from every un-subtracted panel' in common
    # the PA-span direction is stated to the advocate and to every critic (the guard reads it)
    assert "350 -> 10" in common and "increasing-angle direction" in common
    assert "350 -> 10" in fold(texts["advocate"]) and "increasing-angle" in fold(texts["advocate"])
    # the circular caveat lives in the v1 composite VIEW texts, once, with the panel it describes
    g = gloss()
    for vs in ("composite_color", "composite_gray"):
        txt = g["view_sets"][vs]["text"]
        for must in ("CIRCULAR radial-profile", "azimuthally-averaged", "no ellipticity, bar, disc or off-centre",
                     "butterfly / bowtie", "concentric positive/negative rings", "dipole",
                     "evidence neither for nor against", "OFFSET, tangential", "traceable in (d) or (e)"):
            assert must in txt, (vs, must)
    adv_txt = texts["advocate"]
    assert "circular" not in adv_txt.lower() and "butterfly" not in adv_txt.lower()
    assert "VIEW description says which panels they are, what model" in fold(adv_txt)
    assert "Symmetric mandate:" in common and "alternative: null, refutation_strength 0" in common
    for anchor in ("0.0-0.2 possible", "0.3-0.6 about as likely as lensing", "0.7-0.9 the image clearly",
                   "1.0 unambiguous"):
        assert anchor in common, anchor
    assert "no_opinion" in common and "NOT a refutation and it is not a concession" in common
    # scale class is reported by the advocate and never penalised
    adv = fold(texts["advocate"])
    assert "never penalised" in adv and '"galaxy" (arc radius 0.3-2.5")' in adv
    assert "in either direction" in adv
    # the arbitrator names the three forbidden grounds and D only on an upheld alternative
    arb = fold(texts["arbitrator"])
    assert "forbidden ground" in arb and "Einstein-radius size, colour alone" in arb
    assert "ONLY when an upheld alternative accounts for every item" in arb
    assert "advisory" in arb


def test_v1_prompts_are_layout_neutral_and_gloss_is_layout_conditional():
    # .md files: never describe (c) as a colour panel, always acknowledge single-band layouts
    for r in V1_ROLES:
        t = read(r)
        assert "two-band colour" not in t, r
        assert not re.search(r"\(c\)\s*(is|=|the)?\s*(two-band|colour|color)", t), r
        assert "single-band" in t, r
    for c in CRITICS:          # critics are told they may not have every panel
        assert "VIEW description" in read("critic_common")
    assert "VIEW description" in read("advocate") and "VIEW description" in read("arbitrator")
    # gloss structure
    g = gloss()
    comp = g["composite"]
    assert (comp["width"], comp["height"], comp["footer_y"], comp["panel_px"]) == (752, 562, 540, 240)
    assert comp["x0"] == [8, 256, 504] and comp["y0"] == [26, 292] and comp["title_h"] == 18
    px, th = comp["panel_px"], comp["title_h"]
    slots = {"a": (0, 0), "b": (0, 1), "c": (0, 2), "d": (1, 0), "e": (1, 1), "f": (1, 2)}
    for p, (row, col) in slots.items():
        x0, y0 = comp["x0"][col], comp["y0"][row]
        assert comp["boxes"][p] == [x0, y0, x0 + px, y0 + px], p
        assert comp["boxes_titled"][p] == [x0, y0 - th, x0 + px, y0 + px], p
        assert comp["boxes"][p][3] <= comp["footer_y"]
    assert set(g["layout_aliases"]) == {"color", "gray_sw_only", "gray_lw_only"}
    assert set(g["layout_aliases"].values()) == set(g["layouts"]) == {"color", "gray"}
    for lay, L in g["layouts"].items():
        assert set(L["panels"]) == set("abcdef")
        assert set(L["direct"]) | set(L["subtracted"]) == set("abcdef")
        assert not set(L["direct"]) & set(L["subtracted"])
        assert "f" in L["subtracted"]
    assert g["layouts"]["color"]["subtracted"] == ["f"] and g["layouts"]["color"]["has_colour"]
    assert g["layouts"]["gray"]["subtracted"] == ["c", "f"] and not g["layouts"]["gray"]["has_colour"]
    assert "colour" in g["layouts"]["color"]["panels"]["c"]
    assert "subtracted" in g["layouts"]["gray"]["panels"]["c"] and "normal" in g["layouts"]["gray"]["panels"]["e"]
    # roles x layouts -> view sets; geometry/morphology never see a subtracted panel
    assert set(g["roles"]) == {"advocate", "artifact", "geometry", "morphology", "arbitrator"}
    for role, m in g["roles"].items():
        assert set(m) == {"color", "gray"}, role
        for lay, vs_name in m.items():
            vs = g["view_sets"][vs_name]
            L = g["layouts"][lay]
            text = vs["text"]
            assert text.startswith("VIEW:"), vs_name
            if role in ("geometry", "morphology"):
                assert isinstance(vs["panels"], list) and len(vs["panels"]) == 3, vs_name
                assert not set(vs["panels"]) & set(L["subtracted"]), (vs_name, vs["panels"])
                assert set(vs["panels"]) <= set(L["direct"]), vs_name
                assert "No deflector-subtracted panel is included" in text, vs_name
                assert vs["upscale"] == 2 and vs["crop"] == "boxes_titled", vs_name
                for p in vs["panels"]:
                    assert f"({p})" in text, (vs_name, p)
                assert "(f)" not in text, vs_name
            else:
                assert vs["panels"] == "composite" and vs["crop"] == "footer", vs_name
                for p in "abcdef":
                    assert f"({p})" in text, (vs_name, p)
                assert "Subtracted panels: " + ", ".join(L["subtracted"]) in text, vs_name
                assert "Direct (un-subtracted) panels: " + ", ".join(L["direct"]) in text, vs_name
            if L["has_colour"]:
                assert "two-band colour" in text, vs_name
            else:       # never describe a colour panel that is absent
                assert "colour" not in text.replace("no colour information", ""), vs_name
                assert "SINGLE" in text or "SINGLE-BAND" in text, vs_name
            # the wide 20" context is attached by default to geometry only, but every view
            # set carries the sentence so the gated ctx20 arm can attach it to another role
            assert vs["ctx20"] == (role == "geometry"), vs_name
            assert '20"' in vs["ctx20_text"], vs_name
            assert ("colour" in vs["ctx20_text"]) == L["has_colour"], vs_name
    assert g["roles"]["geometry"] == {"color": "geometry_color", "gray": "geometry_gray"}
    assert g["view_sets"]["geometry_color"]["panels"] == ["b", "d", "e"]
    assert g["view_sets"]["morphology_color"]["panels"] == ["c", "d", "e"]
    assert g["view_sets"]["morphology_gray"]["panels"] == ["a", "d", "e"]
    assert g["roles"]["advocate"] == g["roles"]["artifact"] == g["roles"]["arbitrator"]
    for k in ("items", "items_none", "scale_class", "arbitrator_texts"):
        assert g["headers"][k]
    # render-conditional: every composite view set has a v2r twin that describes (f) as the
    # signed-chi residual, never as a circular subtraction, and demands the description
    assert set(g["renders"]) == {"v2r"}
    assert g["renders"]["v2r"] == {"composite_color": "composite_color_v2r", "composite_gray": "composite_gray_v2r"}
    for src, dst in g["renders"]["v2r"].items():
        vs, base = g["view_sets"][dst], g["view_sets"][src]
        assert vs["panels"] == "composite" and vs["crop"] == "footer" and vs.get("render_desc") is True
        assert "SIGNED-CHI" in vs["text"] and "ELLIPTICAL" in vs["text"] and "(f)" in vs["text"]
        kind = "color" if "color" in src else "gray"
        assert "Subtracted panels: " + ", ".join(g["layouts"][kind]["subtracted"]) in vs["text"]
        assert "description follows this paragraph" in vs["text"]
        f_sentence = vs["text"].split("(f)", 1)[1].split("Direct (un-subtracted)")[0]
        assert "CIRCULAR" not in f_sentence and "circular subtraction" in f_sentence   # "NOT a circular subtraction"
        assert not base.get("render_desc")
        assert vs["text"].split("(f)")[0] == base["text"].split("(f)")[0]        # panels (a)-(e) described alike
    assert "(c) deflector-subtracted residual at 10\"" in g["view_sets"]["composite_gray_v2r"]["text"]   # (c) unchanged
    assert "single tile" in g["view_sets"]["composite_gray_v2r"]["text"]


# ------------------------------------------------------------------ note v2
def test_note_v2_rewrite():
    v1, v2 = NOTE_V1.read_text(), NOTE_V2.read_text()
    low = v2.lower()
    assert "bluer" not in low
    assert "default expectation" not in low and "remains c or d" not in low
    assert "c at best, usually d" not in low and "→ D" not in v2     # no grade-by-family rules
    assert "rubric" not in low and "get_photometry" not in low and "tractor" not in low
    assert "p_lens" not in low and "escalate_to_human" not in low and "blue_source" not in low
    # resolution facts kept from v1
    for must in ("0.031″/px", "~40×", "~1.3″ seeing", "θ_E ≈ 1″", "~32 px", "RESOLVED here",
                 "tangential curvature"):
        assert must in v2, must
        assert must in v1, must
    # layout facts, both layouts
    for must in ("6 panels, 2 rows", "N up, E left", "TWO-BAND composites", "SINGLE-BAND composites",
                 "(c) becomes", "(e) a normal-stretch zoom", "YELLOW ticks", "1″ white scale bar",
                 "two-band pseudo-COLOUR", "TWO-band only"):
        assert must in v2, must
    # the subtracted-panel caveat is RENDER-NEUTRAL: it defers to the VIEW description for the
    # model removed (circular profile on v1, elliptical signed residual on v2r) and keeps the
    # rules that hold for both; the circular specifics live in the gloss (tested above)
    sec = fold(v2.split("## The deflector-subtracted panels")[1].split("## ")[0])
    for must in ("VIEW description", "circular radial profile", "elliptical model", "butterfly / bowtie", "dipole",
                 "evidence neither for nor against", "OFFSET, tangential", "traceable in (d) or (e)",
                 "panel (c) in single-band", "Follow that description for the render in front of you"):
        assert must in sec, must
    assert "azimuthally-averaged" not in sec and "No ellipticity, bar, disc" not in sec
    # the JWST false-positive list, now keyed to the critic alternative enum
    fp = v2.split("## JWST-specific false positives")[1].split("## ")[0]
    for fam in ("6-spike diffraction", "Spiral arms", "Collisional / resonance rings", "shells",
                "Tidal features and mergers", "Edge-on discs", "Subtraction residuals",
                "Busy fields / chance alignments", "Reduction artefacts"):
        assert fam in fp, fam
    for alt in ALTERNATIVES:
        if alt not in ("scale_tension", "other"):
            assert f"`{alt}`" in fp, alt
    for must in ("snowballs", "persistence", "1/f striping", "stellar bridge"):
        assert must in fp, must
    # what does not exist; Huang scale for the arbitrator only; scale never rejects alone
    assert "NO CNN / ML scores" in v2 and "Do not call tools" in v2
    assert "Huang" in v2 and "NOT a pass count" in v2 and "critics never grade" in v2
    assert "never used on its own to reject" in fold(v2) and "2.5–10″" in v2
    assert "either direction" in v2 or "same hue" in v2


# ------------------------------------------------------------------ incumbent
def test_incumbent_briefs_pinned_and_carry_the_incumbent_priors():
    """The baseline must keep the priors the new scheme removes — that IS the comparison."""
    for r in INCUMBENT_ROLES:
        t = read(r, INCUMBENT)
        assert _util.sha_text(t) == INCUMBENT_SHA16[r], (r, _util.sha_text(t))
        assert not t.endswith("\n"), f"{r}.md must be the template string byte-for-byte (no trailing newline)"
    assert "bluer than the deflector" in read("morphology", INCUMBENT)
    assert 'roughly 0.3-3"' in read("geometry", INCUMBENT) and "larger implies a group/cluster" in read("geometry", INCUMBENT)
    assert read("artifact", INCUMBENT).endswith("mark it fail.")
    assert read("geometry", INCUMBENT).endswith("mark it fail.")


def test_incumbent_briefs_byte_equal_to_js():
    js = _util.JWST_REPO / "scripts" / "verify_workflow.js"
    if not js.exists():
        print(f"  (skip: {js} not present)")
        return
    text = js.read_text()
    lines = text.splitlines(keepends=True)
    block = "".join(lines[32:72])                       # lines 33-72, 1-indexed
    assert block.startswith("const LENS_BRIEF = {") and block.rstrip().endswith("}")
    briefs = extract_lens_briefs(text)
    assert briefs == extract_lens_briefs(block + "\n") or briefs == dict(re.findall(r"(\w+): `([^`]*)`", block))
    for r, s in briefs.items():
        assert read(r, INCUMBENT) == s, r
        assert read(r, INCUMBENT).encode() == s.encode(), r
    # the wrapper's verbatim blocks: PANEL LAYOUT (lines 87-91) and STEP 2 (lines 94-104)
    w = read("wrapper", INCUMBENT)
    layout = "".join(lines[86:91])
    step2 = "".join(lines[93:104])
    assert layout.startswith("PANEL LAYOUT (6 panels, 2 rows):") and layout in w
    assert step2.startswith("STEP 2.") and step2.rstrip().endswith("mundane explanation.") and step2 in w
    head = "".join(lines[75:79]).split("`", 1)[1]       # the preamble of prompt(), lines 76-79
    assert head.startswith("You are adversarially verifying") and head in w
    glossary = "".join(lines[110:115])                  # verdict / alternative / notes glossary
    assert glossary.lstrip().startswith("verdict") and glossary in w


def test_incumbent_wrapper_contract():
    w = read("wrapper", INCUMBENT)
    assert w.rstrip("\n").splitlines()[-1] == RESPOND
    found = set(re.findall(r"\{([a-z_]+)\}", w))
    assert found == WRAPPER_PLACEHOLDERS, found ^ WRAPPER_PLACEHOLDERS
    assert "The image is attached below. Respond with exactly one JSON object: {id, persona, verdict, alternative, notes}" in w
    for gone in ("STEP 1", "STEP 3", "Read tool", "Write tool", "job file", "${BASE}", ".jsonl",
                 "Bash + Python", "Every item must get exactly one line", "in job order"):
        assert gone not in w, gone
    assert "STEP 2." in w and 'prefer "fail"' in w      # the incumbent's veto sentence stays: it is the baseline
    filled = fill_wrapper(w, "geometry", read("geometry", INCUMBENT), item_id="item",
                          claim_center="yes", claim_quadrant="NE", claimed_evidence="faint arc to the NE")
    assert not re.search(r"\{[a-z_]+\}", filled), re.findall(r"\{[a-z_]+\}", filled)
    assert '"persona":"geometry"' in filled and '"id":"item"' in filled
    assert "You are a LENSING GEOMETRY specialist" in filled
    assert "claimed_evidence\": faint arc to the NE" in filled
    assert filled.index("LENSING GEOMETRY") < filled.index("PANEL LAYOUT") < filled.index("STEP 2.")


# ------------------------------------------------------------------ embargo / item-agnostic
def test_no_model_facing_string_hits_pi_lexicon_or_names_an_item():
    lex, real = pi_lexicon()
    strings = model_facing_strings()
    assert len(strings) >= 6 + 1 + 4 + 20
    bad = []
    for name, t in strings.items():
        hit = audit_traces.banned_hit(t, lex)
        if hit:
            bad.append(_hit_msg(name, hit))
    assert not bad, "\n".join(bad)
    id_re = re.compile(r"J\d{6,}[+-]\d+")
    coord_re = re.compile(r"\b\d{1,3}\.\d{4,}\b")                    # decimal RA/Dec
    for name, t in strings.items():
        low = t.lower()
        assert not id_re.search(t), name
        assert not coord_re.search(t), name
        for s in ("rank 15", "poster", "xiaosheng", "cowls", "rank-15", "rank15"):
            assert s not in low, (name, s)
        assert not re.search(r"\brank\s*\d", low), name
    if not real:
        print("  (golden/pi_comments.txt absent: lexicon check ran on synthetic stand-ins only)")
    else:
        print(f"  lexicon: {len(lex)} PI comments, {len(strings)} strings clean")


def tracked_text_files() -> list[Path]:
    """Every *.py / *.md under golden/, prompts/, tests/ plus GOLDEN_FINDINGS.md — the files a
    coding agent reads (embargo rule 1 extends to them); bulk/gitignored dirs and the
    generated patch tree are skipped (the patch tree is checked by the generator test
    through the embedded prompt files)."""
    skip = {"stamps", "kits", "kits_truth", "kits_truth_v2r", "__pycache__", "verifier_patch", "records"}
    out = []
    for top in ("golden", "prompts", "tests"):
        for p in sorted((LENSJUDGE / top).rglob("*")):
            if p.suffix not in (".py", ".md") or any(part in skip for part in p.relative_to(LENSJUDGE).parts):
                continue
            out.append(p)
    if (LENSJUDGE / "GOLDEN_FINDINGS.md").exists():
        out.append(LENSJUDGE / "GOLDEN_FINDINGS.md")
    return out


def test_no_tracked_module_quotes_a_pi_comment():
    """A verbatim PI 4-gram in a tracked module's docstring or a test fixture is a leak into a
    file coding agents read (audit_traces.py:38-40); golden/pi_comments.txt itself is
    gitignored and excluded (it IS the source)."""
    lex, real = pi_lexicon()
    bad = []
    for p in tracked_text_files():
        if p.name == "pi_comments.txt":
            continue
        hit = audit_traces.banned_hit(p.read_text(errors="replace"), lex)
        if hit:
            bad.append(_hit_msg(str(p.relative_to(LENSJUDGE)), hit))
    assert not bad, "\n".join(bad)
    n = len(tracked_text_files())
    assert n >= 30, n
    print(f"  {n} tracked files clean against {len(lex)} {'PI comments' if real else 'synthetic stand-ins'}")


def sha16_manifest() -> dict[str, str]:
    """sha16 of every prompt file in this package, keyed by path under prompts/ (the
    registry tuple and the verifier patch pin these)."""
    files = [V1 / f"{r}.md" for r in V1_ROLES] + [GLOSS, NOTE_V2] + \
            [INCUMBENT / f"{r}.md" for r in INCUMBENT_ROLES + ("wrapper",)]
    return {str(p.relative_to(PROMPTS)): _util.sha_file(p) for p in files}


def test_sha16_manifest():
    """Every prompt file exists, is non-empty and distinct; print the sha16 table."""
    shas = sha16_manifest()
    for rel, sha in shas.items():
        assert (PROMPTS / rel).stat().st_size > 0, rel
        print(f"  {sha}  {rel}")
    assert len(set(shas.values())) == len(shas)


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
