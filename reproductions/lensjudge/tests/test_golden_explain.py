#!/usr/bin/env python3
"""No-network tests for golden/explain.py (the rationale renderer).

Covers, on synthetic records only (no API, no real run):
  1. a full stack renders every traceable fact into the Markdown — headline letter and the
     threshold sentence, each item's what / panel / r / PA / criteria names, criteria scores,
     each critic's alternative / location / accounts_for / strength / ruling + why, the
     arbitrator's surviving items / letter_llm / needs_human / rationale VERBATIM, the score
     decomposition and the deploy letters + veto; the paragraph is <= 600 chars, plain text,
     and the facts dict is JSON-serialisable;
  2. `letter_reason` agrees with `aggregate_v2.assign_letter` on every branch (A, B via
     t_B, B with A blocked by criteria / by a strong critic, C, D by nothing_because, D by a
     full-coverage critic — upheld or not) and spells out the deciding comparison;
  3. advocate-only runs (no critic keys: "not called", S = p_ev, letter_final == letter_rank);
  4. parse failures (None records named as such, S NaN, no letter) and a record the votes
     lost (row parse_ok: decomposition falls back to the stored a_/r_/ruling_ columns);
  5. the decomposition arithmetic (records and row paths, the partial-ruling a', a stored
     mismatch annotation) and the R2 deploy sentence;
  6. missing thresholds (row's, explicit override, none at all) and a NaN-S row;
  7. paragraph truncation on an oversized record;
  8. the CLI on a synthetic preds + votes pair in tmp_path (md / csv / json outputs, pins,
     --thresholds/--model-key, --format subset).

Runs under pytest:
    cd reproductions/lensjudge && ~/.venvs/lensjudge/bin/python -m pytest tests/test_golden_explain.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from lensjudge.golden import _util, aggregate_v2  # noqa: E402
from lensjudge.golden import explain as E  # noqa: E402
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import schemas_panel as sp  # noqa: E402

THR = {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318, "letter_source": "sonnet_api_calibrated"}
PROV = {**aggregate_v2.PROVISIONAL, "letter_source": "provisional"}


# ------------------------------------------------------------------ synthetic records
def _advocate(p_ev=0.7, items=2, nothing="", crit=(7, 6, 8, 6, 7), notes="tangential arc east of the core"):
    its = [sp.EvidenceItem(k=1, what="blue arc", panel="d", r_arcsec=1.3, pa_deg_from=40, pa_deg_to=170,
                           visible_in_direct=True, criteria=[3, 5]),
           sp.EvidenceItem(k=2, what="counter knot", panel="e", r_arcsec=1.1, pa_deg_from=230, pa_deg_to=260,
                           visible_in_direct=False, criteria=[4]),
           sp.EvidenceItem(k=3, what="faint tail", panel="ctx", r_arcsec=2.5, pa_deg_from=300, pa_deg_to=20,
                           visible_in_direct=True, criteria=[])][:items]
    sc, lsb, cu, ci, am = crit
    return sp.AdvocateRecord(
        id="item", persona="advocate",
        criteria=sp.CriteriaV2(source_contrast=sc, low_surface_brightness=lsb, curvature=cu, counter_image=ci,
                               arc_morphology=am),
        items=its, arc_radius_arcsec=1.2, arc_pa_span_deg=(40.0, 170.0),
        counter_image_pos=sp.CounterImagePos(r_arcsec=1.1, pa_deg=245.0), centre_of_curvature_offset_arcsec=0.2,
        scale_class="galaxy", n_red_neighbours_10as=1, bcg_like_halo=False, deflector_is_centre=True,
        p_evidence=p_ev, nothing_because=nothing, notes=notes)


def _critic(role, alternative=None, r=0.0, no_opinion=False, reason=None, accounts=(), leaves=(),
            loc=(1.0, 1.6, 30, 180), desc="a tidal shell wrapping the east side", notes=""):
    location = sp.LocationBox(r_arcsec_from=loc[0], r_arcsec_to=loc[1], pa_deg_from=loc[2], pa_deg_to=loc[3])
    return sp.CriticRecord(id="item", persona=role, no_opinion=no_opinion, no_opinion_reason=reason,
                           alternative=alternative, alternative_desc=desc if alternative else "",
                           location=location if alternative else None, accounts_for=list(accounts),
                           leaves_standing=list(leaves), refutation_strength=r if alternative else None,
                           measured={"axis_ratio": 0.4} if alternative else None, scale_class="galaxy", notes=notes)


def _arbitrator(rulings=(("geometry", "partial", (1,), "the shell reaches item 1 only"),), surviving=(2,),
                letter="B", needs_human=False, rationale="the counter knot stays; the arc is partly a shell"):
    return sp.ArbitratorRecord(id="item", persona="arbitrator",
                               rulings=[sp.Ruling(persona=p, ruling=ru, covers=list(cv), why=w) for p, ru, cv, w in rulings],
                               surviving_items=list(surviving), letter_llm=letter, scale_class_final="galaxy",
                               needs_human=needs_human, rationale=rationale)


def _stack():
    """advocate (2 items, p_ev 0.3) + artifact abstains + geometry shell_tidal r 0.9 covering BOTH
    items (partial ruling on k1 only) + morphology names nothing + arbitrator. Under the sonnet
    thresholds: S = 0.3·(1 − 0.9) = 0.03 → stored grade_pred D (full-coverage rule, no
    arbitrator); S_arb = 0.3·(1 − 0.9·0.5) = 0.165 → B; deploy R1: letter_rank A → letter_final
    B, veto geometry:shell_tidal."""
    return {"advocate": _advocate(p_ev=0.3),
            "artifact": _critic("artifact", no_opinion=True, reason="feature_not_in_my_views", leaves=(1, 2)),
            "geometry": _critic("geometry", "shell_tidal", r=0.9, accounts=(1, 2), loc=(0.8, 1.6, 30, 270),
                                notes="shell-like"),
            "morphology": _critic("morphology"),
            "arbitrator": _arbitrator()}


def _critics(stack):
    return {r: stack[r] for r in R.CRITIC_ROLES if r in stack}


def _row(stack, thr=THR, **extra):
    """The stored preds row of a records dict (schemas_panel.to_row + the threshold columns)."""
    rec = {"x": stack}
    row = R.row_from_records("x", rec, thr)
    row.update({k: thr[k] for k in ("tau0", "t_A", "t_B", "letter_source")})
    row.update(extra)
    return row


def _j(rec, indent=2) -> str:
    return json.dumps(rec.model_dump(mode="json"), indent=indent)


def _fenced(rec) -> str:
    return "Reading the panels.\n\n```json\n" + _j(rec) + "\n```\n"


def _no_markdown(text: str) -> bool:
    return not any(line.startswith(("#", "- ", "* ", "|", "```")) or "**" in line for line in text.splitlines())


# ------------------------------------------------------------------ 1: every fact is in the markdown
def test_full_stack_markdown_is_traceable():
    stack = _stack()
    adv, crit, arb = stack["advocate"], _critics(stack), stack["arbitrator"]
    row = _row(stack)
    dep = aggregate_v2.deploy_letters(adv, crit, arb, THR, "R1")
    out = E.explain_item("itemA", stack, row, THR, dep)
    md, para, facts = out["markdown"], out["paragraph"], out["facts"]
    assert set(out) == {"markdown", "paragraph", "facts"}

    # S = 0.3·(1 − 0.9·1) = 0.03 (D by the full-coverage rule); S_arb = 0.3·(1 − 0.9·0.5) = 0.165 (B)
    S = aggregate_v2.score_S(adv, crit)
    assert math.isclose(S, 0.03) and math.isclose(row["S_arb"], 0.165) and row["grade_pred"] == "D"
    assert dep["letter_rank"] == "A" and dep["letter_final"] == "B" and dep["veto"] == "geometry:shell_tidal"
    # (1) headline: letter_final, p_evidence, S_arb, scale, the deciding comparison
    assert md.startswith("# itemA\n")
    assert "**Letter B** (deployed, rule R1) · p_evidence 0.30 · S_arb 0.165 · S 0.03 · scale galaxy" in md
    assert ("Letter: S_arb 0.165 >= t_B 0.132 (calibrated at 5% FPR on clean negatives); "
            "S_arb 0.165 < t_A 0.192 (calibrated at 1% FPR on clean negatives).") in md
    assert facts["letter"] == "B" and facts["letter_basis"] == "deployed, rule R1"
    # (2) advocate: criteria, every item with its located facts and criteria NAMES, flags, notes
    assert "- criteria: source_contrast 7, low_surface_brightness 6, curvature 8, counter_image 6, arc_morphology 7" in md
    assert ("- k1 — “blue arc” (panel d, r 1.3 arcsec, PA 40->170, visible in direct: yes; "
            "criteria: curvature, arc_morphology)") in md
    assert "- k2 — “counter knot” (panel e, r 1.1 arcsec, PA 230->260, visible in direct: no; criteria: counter_image)" in md
    assert ("scale galaxy; red neighbours within 10 arcsec 1; BCG-like halo no; deflector is centre yes; "
            "arc radius 1.2 arcsec; arc PA span 40->170; counter-image at r 1.1 arcsec PA 245; "
            "centre-of-curvature offset 0.2 arcsec") in md
    assert "- p_evidence: 0.30" in md and "- notes: “tangential arc east of the core”" in md
    assert "nothing_because" not in md.split("## Critics")[0]          # items exist → not shown
    # (3) critics in role order with location / accounts / strength / ruling + why
    i_art, i_geo, i_mor = md.index("### artifact"), md.index("### geometry"), md.index("### morphology")
    assert i_art < i_geo < i_mor
    assert "### artifact — no opinion (feature_not_in_my_views)" in md
    assert "- leaves standing: k1, k2" in md
    assert "### geometry — shell_tidal: “a tidal shell wrapping the east side”" in md
    assert "- location: r 0.8-1.6 arcsec, PA 30->270" in md
    assert ("- accounts for: k1, k2; leaves standing: none; strength r 0.90; coverage a 1.00 (geometric), "
            "arbitrated a' 0.50") in md
    assert "- measured: `{\"axis_ratio\": 0.4}`" in md and "- notes: “shell-like”" in md
    assert "- ruling: partial, covers k1 — “the shell reaches item 1 only”" in md
    assert "### morphology — no alternative named (nothing in its competence fits)" in md
    assert "- ruling: no ruling for this critic" in md
    # (4) arbitrator, rationale verbatim
    assert "- surviving items: k2 (1 of 2)" in md
    assert "- letter_llm: B; scale_class_final: galaxy; needs_human: no" in md
    assert "- rationale: “the counter knot stays; the arc is partly a shell”" in md
    # (5) score decomposition + letters + veto
    assert "- S = p_ev x prod(1 - r_i a_i) = 0.30 x (1 - 0.90x1.00) = 0.03" in md
    assert "  - terms: geometry shell_tidal r 0.90 a 1.00\n" in md
    assert "- S_arb = p_ev x prod_{upheld/partial}(1 - r_i a_i') = 0.30 x (1 - 0.90x0.50) = 0.165" in md
    assert "  - terms: geometry shell_tidal r 0.90 a' 0.50 (partial)\n" in md      # a' = the arbitrated coverage
    assert ("- stored grade_pred (on S): D — S 0.03 < t_B 0.132 (calibrated at 5% FPR on clean negatives) and "
            "geometry (shell_tidal) covers every item at r 0.90 >= 0.80\n") in md
    assert "differs" not in md
    assert "- stored letter_arb (on S_arb): B; letter_source: sonnet_api_calibrated" in md
    assert ("letter_rank (advocate only, on p_evidence) = A — p_evidence 0.30 >= t_A 0.192 (calibrated at 1% FPR on "
            "clean negatives); 3 of 3 configuration criteria >= 6 (curvature, counter_image, arc_morphology); "
            "no critic with r x a >= 0.80") in md
    assert ("letter_final = B — S_arb 0.165 >= t_B 0.132 (calibrated at 5% FPR on clean negatives); "
            "S_arb 0.165 < t_A 0.192 (calibrated at 1% FPR on clean negatives)") in md
    assert "- veto: geometry:shell_tidal" in md
    # section order
    idx = [md.index(h) for h in ("## Advocate", "## Critics", "## Arbitrator", "## Score")]
    assert idx == sorted(idx)
    # the paragraph: plain text, bounded, the same headline facts, quoted prose
    assert len(para) <= E.PARAGRAPH_MAX and _no_markdown(para)
    assert para.startswith("itemA: letter B (deployed, rule R1; letter_rank A, veto geometry:shell_tidal); "
                           "p_evidence 0.30, S_arb 0.165; scale galaxy.")
    assert "Letter: S_arb 0.165 >= t_B 0.132; S_arb 0.165 < t_A 0.192." in para and "calibrated at" not in para
    assert facts["letter_reason_short"] == "S_arb 0.165 >= t_B 0.132; S_arb 0.165 < t_A 0.192"
    assert "k1 “blue arc” (panel d, r 1.3 arcsec)" in para and "k2 “counter knot” (panel e, r 1.1 arcsec)" in para
    assert "geometry shell_tidal r 0.90 covering k1, k2 (partial)" in para and "artifact no opinion" in para
    assert "morphology no alternative named" in para
    assert "Arbitrator: surviving k2; letter_llm B; needs_human no." in para
    assert "Arbitrator rationale: “the counter knot stays; the arc is partly a shell”" in para
    assert facts["rationale_excerpted"] is False
    # the criteria scores are the last-priority filler: present iff they fit
    crit_sentence = "Criteria source_contrast 7, low_surface_brightness 6, curvature 8, counter_image 6, arc_morphology 7."
    assert (crit_sentence in para) == (not facts["paragraph_truncated"])
    # facts: JSON-safe and carrying the structured fields
    json.dumps(facts)
    assert facts["advocate"]["items"][0]["criteria"] == ["curvature", "arc_morphology"]
    assert facts["critics"]["geometry"]["ruling"] == {"ruling": "partial", "covers": [1], "why": "the shell reaches item 1 only"}
    assert facts["arbitrator"]["rationale"] == arb.rationale
    assert facts["score"]["source"] == "records" and facts["score"]["S_matches"] is True
    assert facts["deploy"]["veto"] == "geometry:shell_tidal" and facts["thresholds"]["t_A"] == 0.192
    assert facts["roles_called"] == list(R.ROLES) and facts["parse_fail_roles"] == []


def test_without_deploy_headline_is_stored_grade_pred_and_provisional_note():
    stack = _stack()
    row = _row(stack, PROV)
    out = E.explain_item("itemA", stack, row)                # thresholds from the row
    assert "**Letter D** (stored grade_pred, on S)" in out["markdown"]
    assert ("Letter: S 0.03 < t_B 0.50 (provisional, not calibrated) and geometry (shell_tidal) covers every item "
            "at r 0.90 >= 0.80.") in out["markdown"]
    assert out["facts"]["deploy"] is None and out["facts"]["thresholds"]["letter_source"] == "provisional"
    # the arbitrated letter on the same row is a C (partial ruling ⇒ no D rule; 0.165 < t_B 0.5)
    assert out["facts"]["stored"]["letter_arb"] == "C"
    assert "- stored letter_arb (on S_arb): C; letter_source: provisional" in out["markdown"]
    assert "deploy rule" not in out["markdown"] and "veto" not in out["markdown"]
    # an "uncalibrated" label is quoted as such
    out2 = E.explain_item("itemA", stack, row, {**THR, "letter_source": "opus_api_uncalibrated"})
    assert "(letter_source opus_api_uncalibrated)" in out2["markdown"]


# ------------------------------------------------------------------ 2: the letter sentence
def _reason(score, adv, crit, arb, thr=THR, label="S"):
    got = E.letter_reason(score, adv, crit, arb, thr, label)
    want, _ = aggregate_v2.assign_letter(score, adv, crit, thr, arb)
    assert got["letter"] == want, (got, want)
    assert got["sentence"] and not got["sentence"].endswith(".")
    return got


def test_letter_reason_every_branch():
    adv = _advocate(p_ev=0.9)
    # A
    g = _reason(0.9, adv, {}, None)
    assert g["letter"] == "A" and g["sentence"].startswith("S 0.90 >= t_A 0.192 (calibrated at 1% FPR on clean negatives); 3 of 3 configuration criteria >= 6 (curvature, counter_image, arc_morphology); no critic with r x a >= 0.80")
    # B by t_B only
    g = _reason(0.15, adv, {}, None)
    assert g["letter"] == "B" and g["sentence"] == ("S 0.15 >= t_B 0.132 (calibrated at 5% FPR on clean negatives); "
                                                    "S 0.15 < t_A 0.192 (calibrated at 1% FPR on clean negatives)")
    # B with A blocked by the criteria guard
    weak = _advocate(p_ev=0.9, crit=(9, 9, 5, 5, 6))
    g = _reason(0.9, weak, {}, None)
    assert g["letter"] == "B" and "but A is blocked: only 1 of 3 configuration criteria >= 6; S 0.90 >= t_B 0.132" in g["sentence"]
    assert g["n_strong"] == 1
    # B with A blocked by a strong critic (r·a ≥ 0.8): geometry covers both items at r 0.9 → S = 0.09 < t_B → D by the D rule;
    # so use a critic covering one of two items at r 1.0 → r·a = 0.5, not a blocker; make the blocker with one item
    one = _advocate(p_ev=0.9, items=1)
    strong = {"geometry": _critic("geometry", "merger", r=0.85, accounts=(1,))}
    s = aggregate_v2.score_S(one, strong)                   # 0.9·(1 − 0.85) = 0.135 ≥ t_B, < t_A
    g = _reason(s, one, strong, None)
    assert g["letter"] == "B" and "S 0.135 >= t_B" in g["sentence"]
    # force the blocker branch with a score above t_A but a critic at r·a ≥ 0.8 (assign_letter takes S as given)
    g = _reason(0.5, one, strong, None)
    assert g["letter"] == "B" and g["blockers"] == ["geometry"]
    assert "but A is blocked: geometry (merger) r x a = 0.85 >= 0.80; S 0.50 >= t_B 0.132" in g["sentence"]
    # C
    g = _reason(0.05, adv, {}, None)
    assert g["letter"] == "C" and g["sentence"] == ("S 0.05 < t_B 0.132 (calibrated at 5% FPR on clean negatives); "
                                                    "not D: 2 item(s) located; no named critic covers every item at r >= 0.80")
    # C when nothing located AND nothing_because empty (the D rule needs it named)
    bare = _advocate(p_ev=0.05, items=0, nothing="")
    g = _reason(0.05, bare, {}, None)
    assert g["letter"] == "C" and "no item located but nothing_because is empty" in g["sentence"]
    # D by nothing_because
    empty = _advocate(p_ev=0.05, items=0, nothing="a bare red elliptical")
    g = _reason(0.05, empty, {}, None)
    assert g["letter"] == "D" and g["nothing_located"] is True
    assert g["sentence"].endswith("and nothing located: nothing_because “a bare red elliptical”")
    # D by a full-coverage strong critic, with and without an arbitrator
    full = {"morphology": _critic("morphology", "spiral_arm", r=0.9, accounts=(1, 2), loc=(0.5, 3.0, 0, 360))}
    s = aggregate_v2.score_S(adv, full)                     # 0.9·0.1 = 0.09
    g = _reason(s, adv, full, None)
    assert g["letter"] == "D" and g["d_rule_roles"] == ["morphology"]
    assert g["sentence"].endswith("and morphology (spiral_arm) covers every item at r 0.90 >= 0.80")
    arb_up = _arbitrator(rulings=(("morphology", "upheld", (1, 2), "arms"),), surviving=(), letter="D")
    g = _reason(s, adv, full, arb_up)
    assert g["letter"] == "D" and g["sentence"].endswith("(upheld)")
    arb_over = _arbitrator(rulings=(("morphology", "overruled", (), "no bridge"),), surviving=(1, 2), letter="B")
    g = _reason(s, adv, full, arb_over)                     # overruled ⇒ no D rule ⇒ C on the same S
    assert g["letter"] == "C" and "no upheld critic covers every item" in g["sentence"]
    # NaN / no thresholds / no advocate
    assert E.letter_reason(float("nan"), adv, {}, None, THR, "S")["sentence"] == "S is NaN (parse failure): no letter"
    assert E.letter_reason(0.5, adv, {}, None, None, "S")["sentence"].startswith("S 0.50: thresholds unavailable")
    assert E.letter_reason(0.5, None, {}, None, THR, "S")["letter"] is None
    # the label follows the score that was lettered
    assert _reason(0.9, adv, {}, None, label="p_evidence")["sentence"].startswith("p_evidence 0.90 >= t_A")


# ------------------------------------------------------------------ 3: advocate-only
def test_advocate_only_run():
    adv = _advocate(p_ev=0.12, crit=(4, 3, 2, 2, 2))
    stack = {"advocate": adv}
    row = _row(stack)
    dep = aggregate_v2.deploy_letters(adv, {}, None, THR, "R1")
    out = E.explain_item("solo", stack, row, THR, dep)
    md, facts = out["markdown"], out["facts"]
    assert "- not called (p_evidence 0.12 < tau0 0.15)" in md
    assert "- no arbitrator record (not called)" in md
    assert "- S = p_ev = 0.12 (no critic term entered the product)" in md
    assert "- S_arb = p_ev = 0.12 (no critic term entered the product)" in md
    assert dep["letter_final"] == dep["letter_rank"] == "C" and "- veto: none (letter_final == letter_rank)" in md
    assert facts["critics"] == {} and facts["roles_called"] == ["advocate"]
    assert "Critics not called (p_evidence 0.12 < tau0 0.15)." in out["paragraph"]
    # a2-style: high p_evidence, no critic keys, without tau0 reason
    hi = {"advocate": _advocate(p_ev=0.8)}
    out2 = E.explain_item("solo2", hi, _row(hi))
    assert "- not called (no critic role in the records)" in out2["markdown"]
    assert "**Letter A** (stored grade_pred, on S)" in out2["markdown"]
    # critics called, none named an alternative, arbitrator not needed
    quiet = {"advocate": _advocate(p_ev=0.8), "artifact": _critic("artifact"), "geometry": _critic("geometry"),
             "morphology": _critic("morphology", no_opinion=True, reason="image_quality")}
    out3 = E.explain_item("quiet", quiet, _row(quiet))
    assert "- no arbitrator record (not called (no critic named an alternative))" in out3["markdown"]
    assert "- ruling: no arbitrator" in out3["markdown"]
    assert "### morphology — no opinion (image_quality)" in out3["markdown"]


# ------------------------------------------------------------------ 4: parse failures + lost raws
def test_parse_failures_are_named():
    # advocate failed: nothing to render but the failure
    stack = {"advocate": None}
    row = _row(stack)
    assert row["parse_fail_roles"] == "advocate" and math.isnan(row["S"])
    out = E.explain_item("bad", stack, row, THR, aggregate_v2.deploy_letters(None, {}, None, THR))
    md, para, facts = out["markdown"], out["paragraph"], out["facts"]
    assert "**Letter none** (no letter: advocate parse failure) · p_evidence NaN · S_arb NaN · S NaN · scale n/a" in md
    assert "Parse failure (run): advocate." in md and "- no advocate record (parse failure)" in md
    assert "- S = NaN (no p_evidence)" in md and facts["letter"] is None
    assert "letter none" in para and "Parse failure: advocate." in para
    json.dumps(facts)
    # a critic failed: the row is a parse failure (S NaN) though the advocate parsed
    stack = {"advocate": _advocate(p_ev=0.6), "artifact": None, "geometry": _critic("geometry"),
             "morphology": _critic("morphology")}
    row = _row(stack)
    assert row["parse_fail_roles"] == "artifact" and math.isnan(row["S"])
    out = E.explain_item("half", stack, row, THR, aggregate_v2.deploy_letters(stack["advocate"], _critics(stack), None, THR))
    md = out["markdown"]
    assert "### artifact — parse failure" in md and "**Letter none** (no letter: artifact parse failure)" in md
    assert "- k1 — “blue arc”" in md                      # the advocate's record still renders
    assert out["facts"]["score"]["source"] == "row" and out["facts"]["S"] is None and out["facts"]["S_arb"] is None
    assert "- S = NaN (parse failure: artifact)" in md and "- S_arb = NaN (parse failure: artifact)" in md
    assert "· p_evidence 0.60 · S_arb NaN · S NaN ·" in md
    assert "S_arb is NaN (parse failure): no letter" in md
    # the votes lost a critic raw but the run parsed it (row parse_ok): named as not rebuilt, row terms used
    full = _stack()
    row = _row(full)                                        # parse_ok, S stored
    lost = {**full, "geometry": None}
    out = E.explain_item("lost", lost, row, THR)
    md, facts = out["markdown"], out["facts"]
    assert "Record not rebuilt from the votes (raw missing; the stored row is parse_ok): geometry." in md
    assert "### geometry — record not rebuilt (raw missing from votes)" in md
    assert facts["records_missing"] == ["geometry"] and facts["parse_fail_roles"] == []
    assert facts["score"]["source"] == "row"
    assert "- S = p_ev x prod(1 - r_i a_i) = 0.30 x (1 - 0.90x1.00) = 0.03" in md      # from a_/r_ columns
    # the partial ruling's a' is not stored: the row path says so and shows the stored S_arb
    assert "(1 - 0.90xa') = not recomputable from the row (a' not stored); stored S_arb 0.165" in md
    assert facts["S_arb"] == pytest.approx(0.165)                       # the stored value heads the page
    # the stored D came from the lost critic: the recomputation on the known records says so
    assert "[recomputed C differs from stored D] [from the stored row; records not rebuilt: geometry]" in md
    assert "**Letter D** (stored grade_pred, on S)" in md


# ------------------------------------------------------------------ 5: decomposition arithmetic
def test_decomposition_arithmetic_and_r2():
    sc = aggregate_v2._scenarios()["rank15"]
    adv = sp.AdvocateRecord.model_validate(sc["advocate"])
    crit = {k: sp.CriticRecord.model_validate(v) for k, v in sc["critics"].items()}
    arb = sp.ArbitratorRecord.model_validate(sc["arbitrator"])
    stack = {"advocate": adv, **crit, "arbitrator": arb}
    row = _row(stack, PROV)
    dec = E.decomposition(adv, crit, arb, row)
    assert dec["source"] == "records"
    assert math.isclose(dec["S"]["value"], 0.8 * (1 - 0.4) * (1 - 0.6 / 3)) and dec["S_matches"] is True
    assert math.isclose(dec["S_arb"]["value"], 0.8 * (1 - 0.6 / 3)) and dec["S_arb_matches"] is True
    assert [t["role"] for t in dec["S"]["terms"]] == ["geometry", "morphology"]
    assert [t["role"] for t in dec["S_arb"]["terms"]] == ["morphology"]           # geometry overruled
    dep = aggregate_v2.deploy_letters(adv, crit, arb, PROV, "R1")
    out = E.explain_item("r15", stack, row, PROV, dep)
    md = out["markdown"]
    assert "- S = p_ev x prod(1 - r_i a_i) = 0.80 x (1 - 0.40x1.00) x (1 - 0.60x0.333) = 0.384" in md
    assert "- S_arb = p_ev x prod_{upheld/partial}(1 - r_i a_i') = 0.80 x (1 - 0.60x0.333) = 0.64" in md
    assert "terms: geometry scale_tension r 0.40 a 1.00; morphology spiral_arm r 0.60 a 0.333" in md
    assert "coverage a 1.00 (geometric), excluded from S_arb (overruled)" in md
    assert "### geometry — scale_tension (no description)" in md
    assert "- ruling: overruled — “group halo visible”" in md
    assert (dep["letter_rank"], dep["letter_final"], dep["veto"]) == ("A", "B", "morphology:spiral_arm")
    assert "- veto: morphology:spiral_arm" in md
    # a stored S that disagrees with the records is flagged, never silently replaced
    row_bad = {**row, "S": 0.5}
    md_bad = E.explain_item("r15", stack, row_bad, PROV)["markdown"]
    assert "= 0.384 (stored S 0.50 differs)" in md_bad
    # R2: the rank-13-like spiral makes a D by the D rule; the sentence names it
    sc13 = aggregate_v2._scenarios()["rank13"]
    adv13 = sp.AdvocateRecord.model_validate(sc13["advocate"])
    crit13 = {k: sp.CriticRecord.model_validate(v) for k, v in sc13["critics"].items()}
    arb13 = sp.ArbitratorRecord.model_validate(sc13["arbitrator"])
    st13 = {"advocate": adv13, **crit13, "arbitrator": arb13}
    dep2 = aggregate_v2.deploy_letters(adv13, crit13, arb13, PROV, "R2")
    md2 = E.explain_item("r13", st13, _row(st13, PROV), PROV, dep2)["markdown"]
    assert (dep2["letter_rank"], dep2["letter_final"]) == ("B", "D")
    assert "letter_final = D — D rule: morphology (spiral_arm) covers every item (a = 1) at r 0.90 >= 0.80 (upheld)" in md2
    assert "- veto: morphology:spiral_arm" in md2
    # R2 when the D rule does not fire: letter_rank stands
    dep3 = aggregate_v2.deploy_letters(adv, crit, arb, PROV, "R2")
    md3 = E.explain_item("r15", stack, row, PROV, dep3)["markdown"]
    assert dep3["letter_final"] == "A" and "letter_final = A — letter_rank stands (D rule not met); p_evidence 0.80 >= t_A" in md3
    # explain_run wraps it per row, in preds order
    preds = pd.DataFrame([{**row, "name": "r15"}, {**_row(st13, PROV), "name": "r13"}])
    res = E.explain_run(preds, {"r15": stack, "r13": st13}, PROV, "R1")
    assert [r["facts"]["name"] for r in res] == ["r15", "r13"]
    assert [r["facts"]["letter"] for r in res] == ["B", "D"]
    with pytest.raises(ValueError):
        E.explain_run(preds, {}, PROV, "R3")


# ------------------------------------------------------------------ 6: thresholds + NaN
def test_thresholds_fallbacks_and_nan_row():
    stack = _stack()
    row = _row(stack)
    # row thresholds when none given; explicit ones override
    assert E.item_thresholds(row) == {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318, "letter_source": "sonnet_api_calibrated"}
    assert E.item_thresholds(row, PROV)["t_A"] == 0.8
    assert E.item_thresholds({"tau0": 0.15}) is None and E.item_thresholds({}, {"t_A": 0.5}) is None
    assert E.item_thresholds({}, {"t_A": 0.5, "t_B": 0.2})["tau0"] == aggregate_v2.PROVISIONAL["tau0"]
    bare = {k: v for k, v in row.items() if k not in ("tau0", "t_A", "t_B", "letter_source")}
    out = E.explain_item("itemA", stack, bare)
    assert out["facts"]["thresholds"] is None
    assert "Letter: S 0.03: thresholds unavailable (no t_A / t_B), letter not derived." in out["markdown"]
    assert out["facts"]["letter"] == "D"                    # the stored grade_pred still heads the page
    out2 = E.explain_item("itemA", stack, bare, PROV)
    assert "S 0.03 < t_B 0.50 (provisional, not calibrated) and geometry (shell_tidal) covers every item" in out2["markdown"]
    # a NaN-S row with a parsed advocate (e.g. a voided row): S from the records shows, the stored NaN is reported
    nan_row = {**row, "S": float("nan"), "S_arb": float("nan"), "grade_pred": None, "letter_arb": None}
    out3 = E.explain_item("itemA", stack, nan_row)
    assert out3["facts"]["score"]["S_stored"] is None and out3["facts"]["S"] == pytest.approx(0.03)
    assert "- stored grade_pred (on S): none — S 0.03" in out3["markdown"]
    assert out3["facts"]["letter"] is None and "no letter: score NaN" in out3["markdown"]
    # pandas Series rows (NaN object columns) are read like dicts
    ser = pd.DataFrame([row]).iloc[0]
    out4 = E.explain_item("itemA", stack, ser)
    assert out4["markdown"] == E.explain_item("itemA", stack, row)["markdown"]
    # no row at all
    out5 = E.explain_item("itemA", stack, None, THR)
    assert "**Letter none**" in out5["markdown"] and "- S = p_ev x prod(1 - r_i a_i) = 0.30 x (1 - 0.90x1.00) = 0.03" in out5["markdown"]


# ------------------------------------------------------------------ 7: paragraph truncation
def test_paragraph_truncation():
    long_what = "a very long description of the located feature " * 6
    adv = _advocate(p_ev=0.7, items=3)
    adv = adv.model_copy(update={"items": [it.model_copy(update={"what": long_what + str(it.k)}) for it in adv.items]})
    arb = _arbitrator(rationale="rationale sentence. " * 60)
    stack = {**_stack(), "advocate": adv, "arbitrator": arb}
    row = _row(stack)
    out = E.explain_item("big", stack, row, THR, aggregate_v2.deploy_letters(adv, _critics(stack), arb, THR))
    para, facts, md = out["paragraph"], out["facts"], out["markdown"]
    assert len(para) <= E.PARAGRAPH_MAX
    assert _no_markdown(para)
    assert "…" in para                                       # an excerpt is marked, never silently cut
    assert facts["rationale_excerpted"] is True
    assert "+" not in para or "more" in para                 # item overflow is counted
    # the markdown keeps everything verbatim
    assert long_what + "1" in md and arb.rationale in md
    # an oversized head sentence is hard-capped with an ellipsis and still leads
    huge = E.render_paragraph({**facts, "name": "n" * 700})
    assert len(huge[0]) == E.PARAGRAPH_MAX and huge[0].endswith("…") and huge[0].startswith("nnn") and huge[1] is True
    # a short rationale is quoted whole; a long one is cut at PARAGRAPH_RATIONALE_MAX
    short = {**_stack(), "arbitrator": _arbitrator(rationale="short and sweet")}
    p_short = E.explain_item("s", short, _row(short))
    assert "Arbitrator rationale: “short and sweet”" in p_short["paragraph"] and p_short["facts"]["rationale_excerpted"] is False
    long_r = {**_stack(), "arbitrator": _arbitrator(rationale="word " * 100)}
    p_long = E.explain_item("l", long_r, _row(long_r))
    quoted = p_long["paragraph"].split("Arbitrator rationale: “")[1].split("”")[0]
    assert len(quoted) <= E.PARAGRAPH_RATIONALE_MAX and quoted.endswith("…") and p_long["facts"]["rationale_excerpted"] is True


# ------------------------------------------------------------------ 8: the CLI on a synthetic run
def _vote(name, role, raw, parse_ok=True, k=1):
    return {"name": name, "unit_id": "", "role": role, "k": k, "parse_ok": parse_ok, "raw": raw,
            "cost_usd": 0.01, "system_sha16": f"sha_{role}"}


def _write_run(tmp_path, records, thr=THR):
    rows, vrows = [], []
    for name, roles in records.items():
        r = R.row_from_records(name, records, thr, preds_row={"cost_usd": 0.1})
        r.update({"k": 1, "arm": "a1", "model": "sonnet", "layout": "color", **thr})
        rows.append(r)
        for role, rec in roles.items():
            vrows.append(_vote(name, role, _fenced(rec) if rec is not None else "garbled", parse_ok=rec is not None))
    preds_path = tmp_path / "preds_test_a1_sonnet.parquet"
    pd.DataFrame(rows).to_parquet(preds_path, index=False)
    pd.DataFrame(vrows, columns=list(R.VOTE_COLS)).to_parquet(R.votes_path_for(preds_path), index=False)
    return preds_path


def test_cli_on_synthetic_run(tmp_path, capsys):
    records = {"itemA": _stack(),
               "itemB": {"advocate": _advocate(p_ev=0.05, items=0, nothing="a bare elliptical", crit=(1, 1, 1, 1, 1))},
               "itemF": {"advocate": _advocate(p_ev=0.6), "artifact": None, "geometry": _critic("geometry"),
                         "morphology": _critic("morphology")},
               "item/G": {"advocate": _advocate(p_ev=0.9)}}
    preds_path = _write_run(tmp_path, records)
    out = tmp_path / "explain_out"
    assert E.main(["--preds", str(preds_path), "--rule", "R1", "--out-dir", str(out), "--format", "md,csv,json"]) == 0
    printed = capsys.readouterr().out
    assert "[explain] letters: {'A': 1, 'B': 1, 'D': 1, 'none': 1}" in printed
    assert "n_advocate_only: 2" in printed and "n_parse_fail: 1" in printed and "n_veto: 1" in printed
    assert "the counter knot stays" not in printed                # no rationale text on stdout
    # md: one file per item (safe names) + an index linking them
    names = sorted(p.name for p in out.glob("*.md"))
    assert names == ["index.md", "itemA.md", "itemB.md", "itemF.md", "item_G.md"]
    idx = (out / "index.md").read_text()
    assert "4 item(s)." in idx and "| itemA | B | 0.30 | 0.165 | geometry:shell_tidal | [itemA.md](itemA.md) |" in idx
    assert "| itemF | none | 0.60 | NaN |  | [itemF.md](itemF.md) |" in idx
    assert "| item/G | A |" in idx
    md_b = (out / "itemB.md").read_text()
    assert "**Letter D**" in md_b and "- nothing_because: “a bare elliptical”" in md_b
    assert "and nothing located: nothing_because “a bare elliptical”" in md_b
    # csv: pinned (sha sidecar verifies), the three columns
    df = _util.read_pinned(out / "explain.csv")
    assert list(df.columns) == list(E.CSV_COLS) and df["name"].tolist() == ["itemA", "itemB", "itemF", "item/G"]
    assert df["letter"].fillna("none").tolist() == ["B", "D", "none", "A"]
    assert df["paragraph"].str.len().max() <= E.PARAGRAPH_MAX
    # json: facts per name + sha
    facts = json.loads((out / "facts.json").read_text())
    assert set(facts) == set(records) and facts["itemA"]["deploy"]["veto"] == "geometry:shell_tidal"
    assert (out / "facts.json.sha").read_text().strip() == _util.sha_text((out / "facts.json").read_text())
    assert facts["itemF"]["parse_fail_roles"] == ["artifact"] and facts["itemF"]["letter"] is None
    # --thresholds/--model-key resolves the table; --format subset writes only what is asked
    table = {"sonnet_api": {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318}, "opus_api": None,
             "provisional": aggregate_v2.PROVISIONAL}
    tpath = tmp_path / "thresholds_v2.json"
    tpath.write_text(json.dumps(table))
    out2 = tmp_path / "explain_prov"
    assert E.main(["--preds", str(preds_path), "--thresholds", str(tpath), "--model-key", "opus_api",
                   "--out-dir", str(out2), "--format", "csv"]) == 0
    assert "thresholds provisional" in capsys.readouterr().out
    assert sorted(p.name for p in out2.iterdir()) == ["explain.csv", "explain.csv.sha"]
    df2 = pd.read_csv(out2 / "explain.csv")
    assert df2.set_index("name").loc["itemA", "letter"] == "D"         # stored grade_pred, re-explained on provisional
    assert ("Letter: S 0.03 < t_B 0.50 and geometry (shell_tidal) covers every item at r 0.90 >= 0.80."
            in df2.set_index("name").loc["itemA", "paragraph"])
    # a bad format refuses; --thresholds without --model-key refuses
    with pytest.raises(ValueError):
        E.write_outputs([], tmp_path / "x", ("md", "xml"))
    with pytest.raises(SystemExit):
        E.main(["--preds", str(preds_path), "--thresholds", str(tpath), "--out-dir", str(tmp_path / "y")])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))



def test_panel_marker_flags_a_radius_the_cited_panel_cannot_hold():
    """A zoom panel (3.5" field) cited at r > 1.75", or a 10" panel at r > 5", is a scale
    misreading; the record is printed verbatim with a marker (annotate.r_exceeds_panel)."""
    from lensjudge.golden import explain as E
    assert E.PANEL_FOV == {"a": 10.0, "b": 10.0, "c": 10.0, "d": 3.5, "e": 3.5, "f": 3.5}
    assert E.panel_marker("d", 3.4) == " [r exceeds panel d: its field is 3.5 arcsec]"
    assert E.panel_marker("e", 1.8) == " [r exceeds panel e: its field is 3.5 arcsec]"
    assert E.panel_marker("a", 5.5) == " [r exceeds panel a: its field is 10 arcsec]"
    assert E.panel_marker("d", 1.75) == "" and E.panel_marker("a", 4.9) == "" and E.panel_marker("ctx", 40) == ""
    assert E.panel_marker(None, 3.0) == "" and E.panel_marker("d", None) == ""
    it = {"k": 3, "what": "arc", "panel": "f", "r_arcsec": 3.6, "pa_deg_from": 10, "pa_deg_to": 40,
          "visible_in_direct": True, "criteria": ["curvature"]}
    line = E._item_line(it)
    assert line.startswith("k3 — “arc” (panel f, r 3.6 arcsec, PA 10->40, visible in direct: yes; criteria: curvature)")
    assert line.endswith(" [r exceeds panel f: its field is 3.5 arcsec]")
    assert not E._item_line({**it, "panel": "a"}).endswith("]")
