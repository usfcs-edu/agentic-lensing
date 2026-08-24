#!/usr/bin/env python3
"""No-network tests for golden/{schemas_panel,aggregate_v2}.py + thresholds_v2.json (Part-2 WP-2).

Synthetic records only (no API, no files outside the repo). The one J-dependent check (the
incumbent `grade()` table, executed from `J/scripts/09_rank_report.py` source) skips when the
JWST run repo is absent (`LENSJUDGE_JWST_REPO=/nonexistent`). Runs under pytest or directly:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_aggregate.py
"""
from __future__ import annotations

import ast
import copy
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from lensjudge.common import parse  # noqa: E402
from lensjudge.golden import _util, aggregate_v2 as ag, schemas_panel as sp  # noqa: E402

THR_JSON = _util.HERE / "thresholds_v2.json"
PROV = {**ag.PROVISIONAL, "letter_source": "provisional"}


# ------------------------------------------------------------------ fixtures
def item(k, r=1.0, pf=0.0, pt=0.0, panel="d", crit=(3,)):
    return {"k": k, "what": f"item {k}", "panel": panel, "r_arcsec": r, "pa_deg_from": pf,
            "pa_deg_to": pt, "visible_in_direct": True, "criteria": list(crit)}


def box(rf, rt, pf, pt):
    return {"r_arcsec_from": rf, "r_arcsec_to": rt, "pa_deg_from": pf, "pa_deg_to": pt}


def advocate(p_ev=0.8, items=None, criteria=None, nothing_because="", **kw):
    crit = {"source_contrast": 7, "low_surface_brightness": 6, "curvature": 8,
            "counter_image": 6, "arc_morphology": 7}
    crit.update(criteria or {})
    d = {"id": "x", "persona": "advocate", "criteria": crit,
         "items": [item(1, 1.0, 40, 120), item(2, 1.1, 230, 250), item(3, 0.9, 150, 170)]
         if items is None else items,
         "scale_class": "galaxy", "n_red_neighbours_10as": 0, "bcg_like_halo": False,
         "deflector_is_centre": True, "p_evidence": p_ev, "nothing_because": nothing_because,
         "notes": "advocate note"}
    d.update(kw)
    return d


def critic(role, alternative=None, r=None, loc=None, accounts_for=(), no_opinion=False, **kw):
    d = {"id": "x", "persona": role, "no_opinion": no_opinion, "alternative": alternative,
         "location": loc, "accounts_for": list(accounts_for), "leaves_standing": []}
    if r is not None:
        d["refutation_strength"] = r
    if no_opinion:
        d["no_opinion_reason"] = "outside_competence"
    d.update(kw)
    return d


def arbitrator(rulings, letter="B", **kw):
    d = {"id": "x", "persona": "arbitrator",
         "rulings": [{"persona": p, "ruling": ru, "covers": list(cov), "why": ""} for p, ru, cov in rulings],
         "surviving_items": [], "letter_llm": letter, "scale_class_final": "galaxy",
         "needs_human": False, "rationale": "arb rationale"}
    d.update(kw)
    return d


ALL = box(0.0, 5.0, 0, 360)   # a location covering every item


def models(adv, crits, arb=None):
    """The same synthetic case as pydantic records (what the lensjudge runner holds)."""
    a = sp.AdvocateRecord(**adv)
    c = {k: (sp.CriticRecord(**v) if v is not None else None) for k, v in crits.items()}
    b = sp.ArbitratorRecord(**arb) if arb is not None else None
    return a, c, b


# ------------------------------------------------------------------ geometry
def test_covers_pa_wraparound():
    # a 350 -> 10 box covers PA 5 (through north); the unpadded item sits at 5 exactly
    assert ag.covers(item(1, 1.0, 5, 5), box(0.5, 2.0, 350, 10), dpa=0.0)
    assert ag.covers(item(1, 1.0, 5, 5), box(0.5, 2.0, 350, 10))
    # ... and does not cover PA 180 even with the 20-degree pad
    assert not ag.covers(item(1, 1.0, 180, 180), box(0.5, 2.0, 350, 10))
    # the pad: item at 30, box ends at 10 -> 30-20 = 10 touches (inclusive)
    assert ag.covers(item(1, 1.0, 30, 30), box(0.5, 2.0, 350, 10))
    assert not ag.covers(item(1, 1.0, 31, 31), box(0.5, 2.0, 350, 10))
    # an item whose own span wraps (300 -> 30) vs a box at 350-10 and one at 100-200
    assert ag.covers(item(1, 1.0, 300, 30), box(0.5, 2.0, 350, 10), dpa=0.0)
    assert not ag.covers(item(1, 1.0, 300, 30), box(0.5, 2.0, 100, 200), dpa=0.0)
    # a near-ring written in wrap form (100 -> 80 = 340 deg) must stay a ring after the +-20 deg
    # padding (padding the endpoints first collapsed it to the 20 deg arc 80 -> 100): a critic
    # box on its far side covers it, with and without the pad; 90 -> 80 and 100 -> 60 likewise
    for pf, pt in ((100, 80), (90, 80), (100, 60), (350, 340)):
        assert ag.covers(item(1, 1.0, pf, pt), box(0.5, 2.0, 180, 220)), (pf, pt)
        assert ag.covers(item(1, 1.0, pf, pt), box(0.5, 2.0, 180, 220), dpa=0.0), (pf, pt)
    assert not ag.covers(item(1, 1.0, 100, 120), box(0.5, 2.0, 180, 220))      # the short arc does not
    assert ag.covers(item(1, 1.0, 100, 120), box(0.5, 2.0, 130, 140))          # ... but the pad reaches 140
    assert not ag.covers(item(1, 1.0, 100, 120), box(0.5, 2.0, 141, 150))
    # full circle box covers any PA; from == to + 360 is a full circle, from == to a point
    assert ag.covers(item(1, 1.0, 123, 123), box(0.5, 2.0, 0, 360), dpa=0.0)
    assert ag.covers(item(1, 1.0, 123, 123), box(0.5, 2.0, 90, 450), dpa=0.0)
    assert not ag.covers(item(1, 1.0, 123, 123), box(0.5, 2.0, 90, 90), dpa=0.0)
    # radius guard: item r 1.0 +- 0.3 vs box 1.4-2.0 (no), 1.3-2.0 (touch), 2.0-1.3 (swapped)
    assert not ag.covers(item(1, 1.0, 5, 5), box(1.4, 2.0, 0, 360))
    assert ag.covers(item(1, 1.0, 5, 5), box(1.3, 2.0, 0, 360))
    assert ag.covers(item(1, 1.0, 5, 5), box(2.0, 1.3, 0, 360))
    # pydantic records go through the same code path
    it = sp.EvidenceItem(**item(1, 1.0, 5, 5))
    lb = sp.LocationBox(**box(0.5, 2.0, 350, 10))
    assert ag.covers(it, lb) and not ag.covers(it, sp.LocationBox(**box(0.5, 2.0, 100, 200)))


def test_covers_location_none_or_incomplete():
    assert not ag.covers(item(1), None)
    assert not ag.covers(item(1), {"r_arcsec_from": 0.5})                       # missing fields
    assert not ag.covers(item(1), box(None, 2.0, 0, 360))                       # a None number
    assert not ag.covers({"k": 1}, box(0.0, 5.0, 0, 360))                       # item unlocated
    c = critic("morphology", "spiral_arm", 0.9, loc=None, accounts_for=[1, 2, 3])
    assert ag.coverage_fraction(advocate()["items"], c) == 0.0
    # no location => the critic never enters the product with a penalty
    assert ag.score_S(advocate(0.8), {"morphology": c}) == 0.8


def test_coverage_fraction_is_geometrically_guarded():
    items = advocate()["items"]                      # PA 40-120 @1.0, 230-250 @1.1, 150-170 @0.9
    # claims all three, its box only sits over item 1
    c = critic("morphology", "spiral_arm", 0.6, loc=box(0.7, 1.3, 30, 100), accounts_for=[1, 2, 3])
    assert abs(ag.coverage_fraction(items, c) - 1 / 3) < 1e-12
    # covers all geometrically but only CLAIMS item 2
    c2 = critic("geometry", "companion_projection", 0.5, loc=ALL, accounts_for=[2])
    assert abs(ag.coverage_fraction(items, c2) - 1 / 3) < 1e-12
    # claims an item number that does not exist
    c3 = critic("geometry", "companion_projection", 0.5, loc=ALL, accounts_for=[9])
    assert ag.coverage_fraction(items, c3) == 0.0
    assert ag.coverage_fraction([], c2) == 0.0


# ------------------------------------------------------------------ S
def test_excluded_critics():
    adv = advocate(0.8)
    crits = {"artifact": critic("artifact", no_opinion=True),                              # abstains
             "geometry": critic("geometry", None, 0.0, loc=ALL, accounts_for=[1, 2, 3]),  # no alternative
             "morphology": critic("morphology", "", 0.9, loc=ALL, accounts_for=[1, 2, 3])}  # "" == null
    assert ag.score_S(adv, crits) == 0.8
    assert ag.score_S(adv, {}) == 0.8 and ag.score_S(adv, None) == 0.8
    # list form (Nate's rows) behaves like the dict form
    assert ag.score_S(adv, list(crits.values())) == 0.8
    terms = ag.critic_terms(adv, crits)
    assert not any(t["included"] for t in terms.values())
    assert terms["artifact"]["no_opinion"] and terms["artifact"]["a"] == 0.0
    # the pydantic path: no_opinion record, "" alternative normalised to None
    a, c, _ = models(adv, crits)
    assert c["morphology"].alternative is None and ag.score_S(a, c) == 0.8


def test_score_S_formula_dicts_and_models():
    adv = advocate(0.8)
    crits = {"artifact": critic("artifact", no_opinion=True),
             "geometry": critic("geometry", "scale_tension", 0.4, loc=ALL, accounts_for=[1, 2, 3]),
             "morphology": critic("morphology", "spiral_arm", 0.6, loc=box(0.7, 1.3, 30, 100),
                                  accounts_for=[1, 2, 3])}
    want = 0.8 * (1 - 0.4 * 1.0) * (1 - 0.6 * (1 / 3))
    assert abs(ag.score_S(adv, crits) - want) < 1e-12
    a, c, _ = models(adv, crits)
    assert abs(ag.score_S(a, c) - want) < 1e-12
    # a called critic that failed to parse (None) => NaN; an absent advocate => NaN
    assert math.isnan(ag.score_S(adv, {**crits, "artifact": None}))
    assert math.isnan(ag.score_S(None, crits))
    # r and r*a are clamped to [0,1] on the dict path (Nate's rows are unvalidated)
    hot = {"geometry": critic("geometry", "companion_projection", 1.7, loc=ALL, accounts_for=[1, 2, 3])}
    assert ag.score_S(adv, hot) == 0.0


def test_score_S_arb_rulings():
    adv = advocate(0.8)
    crits = {"geometry": critic("geometry", "scale_tension", 0.4, loc=ALL, accounts_for=[1, 2, 3]),
             "morphology": critic("morphology", "spiral_arm", 0.6, loc=box(0.7, 1.3, 30, 100),
                                  accounts_for=[1, 2, 3])}
    S = ag.score_S(adv, crits)
    # no arbitrator (none was needed) => S_arb == S
    assert ag.score_S_arb(adv, crits, None) == S
    # upheld keeps the computed a; overruled drops the critic; partial uses |covers ∩ items|/|items|
    arb = arbitrator([("geometry", "overruled", []), ("morphology", "partial", [1, 2])])
    assert abs(ag.score_S_arb(adv, crits, arb) - 0.8 * (1 - 0.6 * (2 / 3))) < 1e-12
    arb2 = arbitrator([("geometry", "upheld", [1, 2, 3]), ("morphology", "overruled", [])])
    assert abs(ag.score_S_arb(adv, crits, arb2) - 0.8 * (1 - 0.4)) < 1e-12
    # partial covers ignore item numbers that do not exist; a critic not ruled on keeps its S term
    arb3 = arbitrator([("morphology", "partial", [1, 9])])
    assert abs(ag.score_S_arb(adv, crits, arb3) - 0.8 * (1 - 0.4) * (1 - 0.6 / 3)) < 1e-12
    # an upheld ruling does not lift the geometric guard: morphology still covers 1/3
    arb4 = arbitrator([("morphology", "upheld", [1, 2, 3])])
    assert abs(ag.score_S_arb(adv, crits, arb4) - S) < 1e-12
    a, c, b = models(adv, crits, arb)
    assert abs(ag.score_S_arb(a, c, b) - 0.8 * (1 - 0.6 * (2 / 3))) < 1e-12


# ------------------------------------------------------------------ letters
def test_letter_rules():
    full = {"geometry": critic("geometry", "companion_projection", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    # A: S >= t_A, >=2 of {curvature, counter_image, arc_morphology} >= 6, no r*a >= 0.8
    adv = advocate(0.9)
    assert ag.assign_letter(0.9, adv, {}, PROV) == ("A", "provisional")
    # only one strong configuration criterion => B
    weak = advocate(0.9, criteria={"curvature": 5, "counter_image": 5, "arc_morphology": 9})
    assert ag.assign_letter(0.9, weak, {}, PROV)[0] == "B"
    two = advocate(0.9, criteria={"curvature": 6, "counter_image": 5, "arc_morphology": 6})
    assert ag.assign_letter(0.9, two, {}, PROV)[0] == "A"
    # a critic with r*a >= 0.8 blocks A even at S >= t_A (S passed in is whatever the caller holds)
    assert ag.assign_letter(0.85, adv, full, PROV)[0] == "B"
    # exact boundaries are inclusive
    assert ag.assign_letter(0.80, adv, {}, PROV)[0] == "A"
    assert ag.assign_letter(0.50, adv, {}, PROV)[0] == "B"
    # below t_B: C unless nothing located (with nothing_because) or a full-coverage strong critic
    assert ag.assign_letter(0.3, adv, {}, PROV)[0] == "C"
    assert ag.assign_letter(0.05, advocate(0.05, items=[], nothing_because="isolated elliptical"), {}, PROV)[0] == "D"
    assert ag.assign_letter(0.05, advocate(0.05, items=[], nothing_because=""), {}, PROV)[0] == "C"
    assert ag.assign_letter(0.05, advocate(0.05, items=[], nothing_because="  "), {}, PROV)[0] == "C"
    assert ag.assign_letter(0.09, adv, full, PROV)[0] == "D"
    # r 0.8 exactly and a == 1 qualifies; r 0.79 does not; a < 1 does not
    r8 = {"geometry": critic("geometry", "companion_projection", 0.8, loc=ALL, accounts_for=[1, 2, 3])}
    assert ag.assign_letter(0.1, adv, r8, PROV)[0] == "D"
    r79 = {"geometry": critic("geometry", "companion_projection", 0.79, loc=ALL, accounts_for=[1, 2, 3])}
    assert ag.assign_letter(0.1, adv, r79, PROV)[0] == "C"
    part = {"geometry": critic("geometry", "companion_projection", 0.9, loc=ALL, accounts_for=[1, 2])}
    assert ag.assign_letter(0.1, adv, part, PROV)[0] == "C"
    # scale_tension alone can never make a D (r capped at 0.4 by the schema; even a=1)
    st = {"geometry": critic("geometry", "scale_tension", 0.4, loc=ALL, accounts_for=[1, 2, 3])}
    assert ag.assign_letter(0.1, adv, st, PROV)[0] == "C"
    # NaN S (parse failure) => no letter, source still reported
    assert ag.assign_letter(float("nan"), adv, {}, PROV) == (None, "provisional")
    assert ag.assign_letter(None, adv, {}, PROV) == (None, "provisional")
    # the source label travels with the thresholds dict
    assert ag.assign_letter(0.9, adv, {}, {**PROV, "letter_source": "sonnet_api_calibrated"})[1] == "sonnet_api_calibrated"


def test_letter_rules_arbitrated():
    adv = advocate(0.9)
    full = {"geometry": critic("geometry", "companion_projection", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    # arbitrated D needs the full-coverage critic UPHELD
    up = arbitrator([("geometry", "upheld", [1, 2, 3])])
    assert ag.assign_letter(0.09, adv, full, PROV, arbitrator=up)[0] == "D"
    over = arbitrator([("geometry", "overruled", [])])
    assert ag.assign_letter(0.09, adv, full, PROV, arbitrator=over)[0] == "C"
    partial = arbitrator([("geometry", "partial", [1, 2, 3])])
    assert ag.assign_letter(0.09, adv, full, PROV, arbitrator=partial)[0] == "C"
    silent = arbitrator([])
    assert ag.assign_letter(0.09, adv, full, PROV, arbitrator=silent)[0] == "C"
    # an overruled critic no longer blocks A
    assert ag.assign_letter(0.9, adv, full, PROV)[0] == "B"
    assert ag.assign_letter(0.9, adv, full, PROV, arbitrator=over)[0] == "A"
    assert ag.assign_letter(0.9, adv, full, PROV, arbitrator=up)[0] == "B"


# ------------------------------------------------------------------ thresholds
def test_thresholds_file_and_resolution():
    table = ag.load_thresholds(str(THR_JSON))
    # design freeze 2026-08-23: sonnet_api carries the t_A/t_B fit on the design negatives'
    # S (a1 full-stack pass, FPR <= 1% / <= 5%); calibration 2026-08-24 (v2-deploy item 1):
    # opus5_api fit on the a2-opus5 design negatives, inserted right after sonnet_api;
    # tau0 and the provisional block unchanged
    assert table == {"sonnet_api": {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318},
                     "opus5_api": {"tau0": 0.15, "t_A": 0.2, "t_B": 0.17},
                     "opus_claude_code": None,
                     "provisional": {"tau0": 0.15, "t_A": 0.80, "t_B": 0.50}}
    # the frozen entry resolves calibrated; null / unknown entries fall back to provisional
    t = ag.resolve_thresholds(table, "sonnet_api")
    assert (t["t_A"], t["t_B"], t["tau0"]) == (0.192, 0.1318, 0.15), t
    assert t["letter_source"] == "sonnet_api_calibrated" and t["thresholds_key"] == "sonnet_api"
    for key in ("opus_claude_code", "never_heard_of"):
        t = ag.resolve_thresholds(table, key)
        assert (t["t_A"], t["t_B"], t["tau0"]) == (0.80, 0.50, 0.15), t
        assert t["letter_source"] == "provisional" and t["thresholds_key"] == "provisional"
    # a calibrated entry is used and labelled "<model_key>_calibrated" (tau0 inherits provisional)
    cal = copy.deepcopy(table)
    cal["sonnet_api"] = {"tau0": None, "t_A": 0.71, "t_B": 0.42}
    t = ag.resolve_thresholds(cal, "sonnet_api")
    assert (t["t_A"], t["t_B"], t["tau0"], t["letter_source"]) == (0.71, 0.42, 0.15, "sonnet_api_calibrated")
    # Nate's case: opus null, fall back to sonnet's calibrated numbers with his label
    t = ag.resolve_thresholds(cal, "opus_claude_code", fallback_keys=("sonnet_api", "provisional"),
                              fallback_source="sonnet_thresholds_uncalibrated")
    assert (t["t_A"], t["letter_source"], t["thresholds_key"]) == (0.71, "sonnet_thresholds_uncalibrated", "sonnet_api")
    t = ag.resolve_thresholds(cal, "opus_claude_code", fallback_keys=("sonnet_api", "provisional"))
    assert t["letter_source"] == "sonnet_api_uncalibrated"
    # an empty table still yields the module's provisional numbers
    assert ag.resolve_thresholds({}, "x")["t_A"] == 0.80
    # end to end: letters under the file's resolution carry the source
    adv = advocate(0.9)
    assert ag.assign_letter(0.9, adv, {}, ag.resolve_thresholds(table, "sonnet_api")) == ("A", "sonnet_api_calibrated")


# ------------------------------------------------------------------ ranking
def test_model_keys_claude5_resolution():
    """opus5/sonnet5 route to their own thresholds keys. opus5_api was calibrated 2026-08-24
    (v2-deploy item 1: t_A 0.2 / t_B 0.17 on the a2-opus5 design negatives) and resolves
    calibrated; sonnet5_api still holds no frozen t_A/t_B and resolves provisional (its
    holdout run therefore still needs --allow-provisional-thresholds)."""
    assert ag.MODEL_KEYS["opus5"] == "opus5_api" and ag.MODEL_KEYS["sonnet5"] == "sonnet5_api"
    table = ag.load_thresholds(str(THR_JSON))
    t = ag.resolve_thresholds(table, "opus5_api")
    assert (t["t_A"], t["t_B"], t["tau0"]) == (0.2, 0.17, 0.15), t
    assert t["letter_source"] == "opus5_api_calibrated" and t["thresholds_key"] == "opus5_api"
    t = ag.resolve_thresholds(table, "sonnet5_api")
    assert (t["t_A"], t["t_B"], t["tau0"]) == (0.80, 0.50, 0.15), t
    assert t["letter_source"] == "provisional" and t["thresholds_key"] == "provisional"


def test_rank_key_u_below_examined():
    rows = [{"id": "u_hi", "S": None, "confidence": 95},             # never examined, loud inspector
            {"id": "u_nan", "S": float("nan"), "confidence": 40},
            {"id": "d", "S": 0.02, "p_evidence": 0.1},
            {"id": "c", "S": 0.30, "p_evidence": 0.5},
            {"id": "c2", "S": 0.30, "p_evidence": 0.7},                # tie on S -> p_evidence
            {"id": "a", "S": 0.91, "p_evidence": 0.9},
            {"id": "p_only", "p_lens": 0.5}]                          # a to_row-style row with p_lens
    order = [r["id"] for r in sorted(rows, key=ag.rank_key)]
    assert order == ["a", "p_only", "c2", "c", "d", "u_hi", "u_nan"]
    # rank_score: S for examined, -1 + conf/100 for U (strictly below every examined score)
    assert ag.rank_score(rows[0]) == -1 + 0.95 and ag.rank_score(rows[2]) == 0.02
    assert all(ag.rank_score(r) < 0 for r in rows[:2]) and all(ag.rank_score(r) >= 0 for r in rows[2:])
    # frame / truth-manifest rows carry pipe_inspector_conf on the run's 0-100 scale (15..95):
    # a U row must never reach an examined row's S, whatever the column name
    assert ag.rank_score({"S": None, "pipe_inspector_conf": 45}) == -1 + 0.45
    assert ag.rank_score({"S": None, "pipe_inspector_conf": 95}) < 0 < ag.rank_score({"S": 0.001})
    assert ag.rank_score({"S": None, "confidence": 250}) < 0                 # clamped, still below
    # rank_key on a pandas row (Series) works too, and a `name` column (to_row rows) is read
    # as the column, not as the Series' index label
    df = pd.DataFrame(rows)
    got = [df.loc[i, "id"] for i in sorted(df.index, key=lambda i: ag.rank_key(df.loc[i]))]
    assert got == order
    named = pd.DataFrame([{"name": "zz", "S": 0.5}, {"name": "aa", "S": 0.5}])
    keys = [ag.rank_key(named.loc[i]) for i in named.index]
    assert [k[3] for k in keys] == ["zz", "aa"] and sorted(keys)[0][3] == "aa"


# ------------------------------------------------------------------ the incumbent table
def _grade_table():
    """All 27 three-persona verdict combinations + the never-examined case, by the rule."""
    combos = {(): "U"}
    for vs in itertools.product(ag.VERDICTS, repeat=3):
        n_pass = vs.count("pass")
        combos[vs] = {3: "A", 2: "B", 1: "C"}.get(n_pass, "D")
    return combos


def test_passcount_incumbent_table():
    table = _grade_table()
    assert len(table) == 28
    for vs, letter in table.items():
        n_pass, n_fail, n_unc, got = ag.passcount_incumbent(list(vs))
        assert got == letter, (vs, got, letter)
        assert (n_pass, n_fail, n_unc) == (vs.count("pass"), vs.count("fail"), vs.count("uncertain"))
    # uncertain is a vote that is not a pass: pass/pass/uncertain = B, uncertain x3 = D (not U)
    assert ag.passcount_incumbent(["pass", "pass", "uncertain"])[3] == "B"
    assert ag.passcount_incumbent(["uncertain"] * 3) == (0, 0, 3, "D")
    assert ag.passcount_incumbent([]) == (0, 0, 0, "U")
    # records and dicts, case/whitespace like the incumbent loader; unknown -> uncertain unless strict
    recs = [sp.IncumbentVerdict(id="x", persona=p, verdict=v)
            for p, v in zip(("artifact", "morphology", "geometry"), ("pass", "fail", "pass"))]
    assert ag.passcount_incumbent(recs) == (2, 1, 0, "B")
    assert ag.passcount_incumbent([{"verdict": " PASS "}, {"verdict": "Fail"}]) == (1, 1, 0, "C")
    assert ag.passcount_incumbent(["pass", "maybe"]) == (1, 0, 1, "C")
    try:
        ag.passcount_incumbent(["pass", "maybe"], strict=True)
        raise SystemExit("strict did not raise")
    except ValueError:
        pass
    # n_pass > 3 (impossible after the incumbent's dedup) maps to D in both tables
    assert ag.passcount_incumbent(["pass"] * 4)[3] == "D"


def test_passcount_incumbent_matches_J_grade_source():
    """Execute grade() from J/scripts/09_rank_report.py itself (the module runs a pipeline at
    import, so the function is lifted from the AST) and compare all 28 cases."""
    src_path = _util.JWST_REPO / "scripts" / "09_rank_report.py"
    if not src_path.exists():
        print("  (skip: JWST run repo not present)")
        return
    tree = ast.parse(src_path.read_text())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "grade"]
    assert len(fn) == 1, "grade() not found in 09_rank_report.py"
    ns: dict = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(src_path), "exec"), ns)
    grade = ns["grade"]
    for vs, _ in _grade_table().items():
        n_pass, n_fail, n_unc, mine = ag.passcount_incumbent(list(vs))
        theirs = grade({"flagged": True, "n_votes": len(vs), "n_pass": n_pass})
        assert mine == theirs, (vs, mine, theirs)
    for n_pass in range(0, 6):   # the whole n_pass table, including the impossible tail
        theirs = grade({"flagged": True, "n_votes": max(1, n_pass), "n_pass": n_pass})
        assert ag.passcount_incumbent(["pass"] * n_pass + ["fail"] * (n_pass == 0))[3] == theirs


# ------------------------------------------------------------------ schemas
def _bad(model, d, what):
    try:
        model(**d)
    except ValidationError:
        return
    raise SystemExit(f"{model.__name__} accepted {what}")


def test_schemas_strict_never_coerce():
    adv, crits, arb = advocate(), {"geometry": critic("geometry", "scale_tension", 0.3, loc=ALL, accounts_for=[1])}, \
        arbitrator([("geometry", "upheld", [1])])
    a, c, b = models(adv, crits, arb)                         # the good records validate
    assert a.persona == "advocate" and c["geometry"].refutation_strength == 0.3 and b.letter_llm == "B"
    # extra="forbid" everywhere, including nested
    _bad(sp.AdvocateRecord, {**adv, "grade": "A"}, "an extra key")
    _bad(sp.AdvocateRecord, {**adv, "items": [{**item(1), "colour": "blue"}]}, "an extra item key")
    _bad(sp.AdvocateRecord, {**adv, "criteria": {**adv["criteria"], "blue_source": 5}}, "an extra criterion")
    _bad(sp.CriticRecord, {**crits["geometry"], "verdict": "fail"}, "an extra critic key")
    _bad(sp.ArbitratorRecord, {**arb, "p_lens": 0.5}, "an extra arbitrator key")
    _bad(sp.IncumbentVerdict, {"id": "x", "persona": "artifact", "verdict": "pass", "confidence": 3}, "an extra key")
    # unknown enums / letters => failure, never a default (ImageGrade would have made "E" a "D")
    _bad(sp.ArbitratorRecord, {**arb, "letter_llm": "E"}, "letter E")
    _bad(sp.ArbitratorRecord, {**arb, "letter_llm": "a"}, "lowercase letter")
    _bad(sp.ArbitratorRecord, {**arb, "rulings": [{"persona": "geometry", "ruling": "sustained", "covers": [], "why": ""}]}, "ruling sustained")
    _bad(sp.ArbitratorRecord, {**arb, "rulings": [{"persona": "advocate", "ruling": "upheld", "covers": [], "why": ""}]}, "ruling on the advocate")
    _bad(sp.CriticRecord, {**crits["geometry"], "alternative": "spiral"}, "alternative outside the enum")
    _bad(sp.CriticRecord, {**crits["geometry"], "persona": "advocate"}, "wrong persona")
    _bad(sp.AdvocateRecord, {**adv, "scale_class": "huge"}, "scale_class outside the enum")
    _bad(sp.AdvocateRecord, {**adv, "items": [{**item(1), "panel": "g"}]}, "panel g")
    _bad(sp.AdvocateRecord, {**adv, "persona": "critic"}, "wrong persona")
    _bad(sp.IncumbentVerdict, {"id": "x", "persona": "artifact", "verdict": "maybe"}, "verdict maybe")
    # ranges
    _bad(sp.AdvocateRecord, {**adv, "p_evidence": 1.2}, "p_evidence > 1")
    _bad(sp.AdvocateRecord, {**adv, "criteria": {**adv["criteria"], "curvature": 11}}, "criterion 11")
    _bad(sp.AdvocateRecord, {**adv, "items": [{**item(1), "criteria": [6]}]}, "criterion index 6")
    _bad(sp.CriticRecord, {**crits["geometry"], "refutation_strength": 1.5}, "strength > 1")
    # the brief's rules as validators
    _bad(sp.CriticRecord, {**crits["geometry"], "refutation_strength": 0.41}, "scale_tension at 0.41")
    assert sp.CriticRecord(**{**crits["geometry"], "refutation_strength": 0.4}).refutation_strength == 0.4
    _bad(sp.CriticRecord, critic("geometry", "spiral_arm", 0.5, loc=ALL, no_opinion=True), "no_opinion with an alternative")
    _bad(sp.CriticRecord, critic("geometry", "spiral_arm", None, loc=ALL), "a named alternative without strength")
    _bad(sp.CriticRecord, critic("geometry", "spiral_arm", 0.5, loc=None), "a named alternative without a location")
    _bad(sp.AdvocateRecord, {**adv, "items": [item(1), item(1)]}, "duplicate item k")
    _bad(sp.AdvocateRecord, {**adv, "items": [item(0)]}, "item k 0")
    # abstention / "nothing fits" records need no strength and read back as 0.0
    assert sp.CriticRecord(**critic("artifact", no_opinion=True)).refutation_strength == 0.0
    assert sp.CriticRecord(**critic("artifact")).refutation_strength == 0.0
    # the one normalisation: "" where the brief says null
    r = sp.CriticRecord(**critic("artifact", "", 0.0, no_opinion_reason=""))
    assert r.alternative is None and r.no_opinion_reason is None
    # through common.parse.parse_model (the grader seam): None on failure, never a repaired record
    assert parse.parse_model(json.dumps({**arb, "letter_llm": "E"}), sp.ArbitratorRecord) is None
    assert parse.parse_model(json.dumps({**adv, "extra": 1}), sp.AdvocateRecord) is None
    ok = parse.parse_model("```json\n" + json.dumps(adv) + "\n```", sp.AdvocateRecord)
    assert isinstance(ok, sp.AdvocateRecord) and [i.k for i in ok.items] == [1, 2, 3]
    assert sp.SCHEMA_FOR_ROLE["geometry"] is sp.CriticRecord and sp.SCHEMA_FOR_ROLE["incumbent"] is sp.IncumbentVerdict
    # the arc span tuple and the optional blocks round-trip
    full = sp.AdvocateRecord(**{**adv, "arc_pa_span_deg": [40, 170], "counter_image_pos": {"r_arcsec": 1.1, "pa_deg": 240},
                                "arc_radius_arcsec": 1.0, "centre_of_curvature_offset_arcsec": 0.2})
    assert full.arc_pa_span_deg == (40.0, 170.0) and full.counter_image_pos.pa_deg == 240


# ------------------------------------------------------------------ assemble + to_row
def _thr():
    # the provisional resolution (an empty table): these tests pin assemble/to_row mechanics
    # at the module's provisional letters (t_A 0.80 / t_B 0.50), independent of whether the
    # shipped thresholds_v2.json has been frozen; the file's content is pinned in
    # test_thresholds_file_and_resolution
    return ag.resolve_thresholds({}, "sonnet_api")


def test_assemble_nan_policy():
    adv = advocate(0.8)
    crits = {"artifact": critic("artifact", no_opinion=True),
             "geometry": critic("geometry", "scale_tension", 0.4, loc=ALL, accounts_for=[1, 2, 3]),
             "morphology": critic("morphology", "spiral_arm", 0.6, loc=box(0.7, 1.3, 30, 100), accounts_for=[1, 2, 3])}
    arb = arbitrator([("geometry", "overruled", []), ("morphology", "partial", [1])])
    a, c, b = models(adv, crits, arb)
    res = sp.assemble(a, c, b, _thr(), cost_usd=0.07, calls=5)
    assert res.parse_ok and res.parse_failures == [] and res.letter == "C" and res.letter_arb == "B"
    assert abs(res.S - 0.384) < 1e-12 and abs(res.S_arb - 0.64) < 1e-12
    assert res.a == {"artifact": 0.0, "geometry": 1.0, "morphology": 1 / 3}
    assert res.r == {"artifact": 0.0, "geometry": 0.4, "morphology": 0.6}
    # a failed critic: S, S_arb NaN, no letter, the role listed
    res = sp.assemble(a, {**c, "geometry": None}, b, _thr())
    assert math.isnan(res.S) and math.isnan(res.S_arb) and res.letter is None and res.letter_arb is None
    assert res.parse_failures == ["geometry"] and not res.parse_ok
    assert res.letter_source == "provisional"
    # a failed arbitrator (called, no record): ONE policy for every role — S and S_arb NaN,
    # no letter, parse_ok False (the row is excluded and counted, like any other parse failure)
    res = sp.assemble(a, c, None, _thr(), parse_failures=["arbitrator"])
    assert math.isnan(res.S) and math.isnan(res.S_arb) and res.letter is None and res.letter_arb is None
    assert res.parse_failures == ["arbitrator"] and not res.parse_ok
    assert sp.to_row(res, {"name": "x"})["parse_ok"] is False and math.isnan(sp.to_row(res, {"name": "x"})["p_lens"])
    # an arbitrator that was never needed: S_arb == S
    res = sp.assemble(a, c, None, _thr())
    assert res.S_arb == res.S and res.parse_failures == [] and res.letter_arb == "C"
    # a failed advocate: everything NaN/None, critics were never called
    res = sp.assemble(None, {}, None, _thr(), cost_usd=0.02, calls=1)
    assert math.isnan(res.S) and res.letter is None and res.parse_failures == ["advocate"]
    assert res.p_evidence is None and res.a == {}
    # advocate only (p_ev < tau0): S = p_ev, letter by the nothing-located rule
    lo = sp.AdvocateRecord(**advocate(0.04, items=[], nothing_because="star"))
    res = sp.assemble(lo, {}, None, _thr(), calls=1)
    assert res.S == 0.04 and res.letter == "D" and res.S_arb == 0.04
    # parse failures are ordered by role, deduplicated
    res = sp.assemble(a, {"morphology": None, "artifact": None}, None, _thr(), parse_failures=["morphology"])
    assert res.parse_failures == ["artifact", "morphology"]


def test_to_row_columns_and_consumers():
    from lensjudge.eval import lensbench_gate, score as escore
    adv = advocate(0.8)
    crits = {"artifact": critic("artifact", no_opinion=True),
             "geometry": critic("geometry", "scale_tension", 0.4, loc=ALL, accounts_for=[1, 2, 3]),
             "morphology": critic("morphology", "spiral_arm", 0.6, loc=box(0.7, 1.3, 30, 100), accounts_for=[1, 2, 3])}
    arb = arbitrator([("geometry", "overruled", []), ("morphology", "partial", [1])], needs_human=True)
    a, c, b = models(adv, crits, arb)
    shas = {r: f"{r[:4]:_<16}" for r in sp.ROLES}
    res = sp.assemble(a, c, b, _thr(), cost_usd=0.07, calls=5, system_sha16s=shas)
    cand = {"name": "J0001", "grade": "", "catalog": "jwst", "region": "1727", "p_meta": float("nan")}
    row = sp.to_row(res, cand)
    assert set(row) == set(sp.ROW_COLS) and list(row) == list(sp.ROW_COLS)
    # the _row_dict-style keys
    assert row["name"] == "J0001" and row["p_lens"] == res.S and row["grade_pred"] == "C"
    assert row["confidence"] == 0.8 and row["contaminant"] == "spiral_arm" and row["escalate"] is True
    assert row["rationale"] == "arb rationale" and row["parse_ok"] is True and row["error"] is None
    assert row["turns"] == 5 and row["calls"] == 5 and row["cost_usd"] == 0.07
    assert row["crit_curvature"] == 8 and row["crit_source_contrast"] == 7
    # the v2 keys
    assert row["S"] == res.S and abs(row["S_arb"] - 0.64) < 1e-12 and row["p_evidence"] == 0.8
    assert row["letter_llm"] == "B" and row["letter_arb"] == "B" and row["letter_source"] == "provisional"
    assert (row["a_artifact"], row["a_geometry"], row["a_morphology"]) == (0.0, 1.0, 1 / 3)
    assert (row["r_artifact"], row["r_geometry"], row["r_morphology"]) == (0.0, 0.4, 0.6)
    assert (row["no_opinion_artifact"], row["no_opinion_geometry"], row["no_opinion_morphology"]) == (True, False, False)
    assert (row["alt_geometry"], row["alt_morphology"], row["alt_artifact"]) == ("scale_tension", "spiral_arm", None)
    assert (row["ruling_geometry"], row["ruling_morphology"], row["ruling_artifact"]) == ("overruled", "partial", None)
    assert row["scale_class"] == "galaxy" and row["scale_class_final"] == "galaxy"
    assert row["n_items"] == 3 and row["n_surviving"] == 0 and row["needs_human"] is True
    assert row["parse_fail_roles"] == "" and row["system_sha16_geometry"] == shas["geometry"]
    # no arbitrator: rationale falls back to the advocate's notes; contaminant = largest r*a
    row2 = sp.to_row(sp.assemble(a, c, None, _thr()), cand)
    assert row2["rationale"] == "advocate note" and row2["contaminant"] == "scale_tension"
    assert row2["escalate"] is False and row2["letter_llm"] is None and row2["system_sha16_advocate"] is None
    # a parse failure row: kept, S NaN, grade None, roles named, same column set
    row3 = sp.to_row(sp.assemble(a, {**c, "morphology": None}, None, _thr(), calls=4), cand)
    assert list(row3) == list(sp.ROW_COLS) and math.isnan(row3["p_lens"]) and row3["grade_pred"] is None
    assert row3["parse_fail_roles"] == "morphology" and row3["error"] == "parse_fail:morphology"
    assert row3["a_morphology"] is None and row3["no_opinion_morphology"] is None and row3["alt_morphology"] is None
    row4 = sp.to_row(sp.assemble(None, {}, None, _thr(), calls=1), cand)
    assert list(row4) == list(sp.ROW_COLS) and row4["confidence"] is None and row4["parse_fail_roles"] == "advocate"
    # downstream consumers work unchanged on a frame of rows: recovery_at_fpr (masks NaN) and grade_flip_rate
    pos = [sp.to_row(sp.assemble(sp.AdvocateRecord(**advocate(p)), {}, None, _thr()), {"name": f"p{i}"})
           for i, p in enumerate((0.9, 0.85, 0.7, 0.6))]
    neg = [sp.to_row(sp.assemble(sp.AdvocateRecord(**advocate(p)), {}, None, _thr()), {"name": f"n{i}"})
           for i, p in enumerate((0.1, 0.2, 0.3, 0.55))]
    df = pd.DataFrame(pos + neg + [row3])
    y = [1] * 4 + [0] * 4 + [0]
    tpr, thr = escore.recovery_at_fpr(y, df["p_lens"], fpr_target=0.05)
    assert tpr == 1.0 and 0.55 < thr <= 0.6
    # flip rate on the parsed rows (a NaN grade_pred counts as a flip in pandas; the endpoints
    # exclude parse failures before this call)
    ok = df[df["grade_pred"].notna()]
    flips = lensbench_gate.grade_flip_rate(ok, ok.assign(grade_pred=ok["grade_pred"].where(ok["name"] != "p0", "B")))
    assert flips["n_shared"] == 8 and abs(flips["grade_flip_rate"] - 1 / 8) < 1e-9
    # the parquet round trip keeps the columns (object columns with None are fine)
    pq = df.to_parquet()
    assert list(pd.read_parquet(__import__("io").BytesIO(pq)).columns) == list(sp.ROW_COLS)


def test_self_test_and_stdlib_only():
    out = ag.self_test()
    assert abs(out["rank15"]["S"] - 0.8 * 0.6 * 0.8) < 1e-12 and out["rank15"]["letter"] == "C"
    assert out["rank13"]["letter"] == "D" and out["rank15"]["S"] > out["rank13"]["S"]
    assert ag.__version__ and ag.__version__.count(".") == 2
    # the module ships verbatim into Nate's repo: imports must be stdlib only
    tree = ast.parse(Path(ag.__file__).read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    assert mods <= {"json", "math", "typing", "__future__"}, mods
    assert not any(m in mods for m in ("lensjudge", "pydantic", "numpy", "pandas"))
    # the canned scenarios validate under the strict schemas (the two sides agree on the records)
    for case in ag._scenarios().values():
        a, c, b = models(case["advocate"], case["critics"], case["arbitrator"])
        assert abs(ag.score_S(a, c) - ag.score_S(case["advocate"], case["critics"])) < 1e-12
        assert abs(ag.score_S_arb(a, c, b) - ag.score_S_arb(case["advocate"], case["critics"], case["arbitrator"])) < 1e-12


# ------------------------------------------------------------------ deployment (v2-deploy)
DEPLOY_KEYS = ("letter_rank", "letter_final", "veto", "S", "S_arb", "p_evidence", "rule")


def _nanish(x):
    return isinstance(x, float) and math.isnan(x)


def _deploy_both(adv, crits, arb, thr):
    """deploy_letters under R1 and R2 on the dicts AND on the pydantic records (must agree)."""
    out = {}
    a, c, b = models(adv, crits, arb)
    for rule in ag.DEPLOY_RULES:
        d = ag.deploy_letters(adv, crits, arb, thr, rule)
        assert tuple(d) == DEPLOY_KEYS and d["rule"] == rule
        m = ag.deploy_letters(a, c, b, thr, rule)
        for k in DEPLOY_KEYS:
            assert (_nanish(d[k]) and _nanish(m[k])) or m[k] == d[k], (rule, k, d[k], m[k])
        out[rule] = d
    return out


def test_deploy_letters_rank15_like_overruled_keeps_letter_rank():
    """Strong located evidence, every named critic overruled: the stack certifies what the
    advocate ranked (letter_final == letter_rank, no veto) under both rules."""
    adv = advocate(0.85)                       # curvature 8 / counter_image 6 / arc_morphology 7, 3 items
    crits = {"artifact": critic("artifact", no_opinion=True),
             "geometry": critic("geometry", "scale_tension", 0.4, loc=ALL, accounts_for=[1, 2, 3]),
             "morphology": critic("morphology", "spiral_arm", 0.6, loc=box(0.7, 1.3, 30, 100), accounts_for=[1, 2, 3])}
    arb = arbitrator([("geometry", "overruled", []), ("morphology", "overruled", [])], letter="A")
    d = _deploy_both(adv, crits, arb, PROV)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_rank"], d[rule]["letter_final"], d[rule]["veto"]) == ("A", "A", ""), d
        assert d[rule]["p_evidence"] == 0.85 and d[rule]["S_arb"] == 0.85
        assert abs(d[rule]["S"] - 0.85 * 0.6 * 0.8) < 1e-12        # the PRIMARY S still carries the named terms
    # letter_rank is exactly the advocate-only letter (item 3), whatever the critics say
    assert d["R1"]["letter_rank"] == ag.assign_letter(0.85, adv, [], PROV)[0]
    # the module's own rank-15 scenario (morphology PARTIAL on item 1, S_arb 0.64) is the R1
    # demotion A -> B with the veto naming that critic; R2 keeps A
    sc = ag._scenarios()["rank15"]
    r1 = ag.deploy_letters(sc["advocate"], sc["critics"], sc["arbitrator"], PROV)
    assert (r1["letter_rank"], r1["letter_final"], r1["veto"]) == ("A", "B", "morphology:spiral_arm"), r1
    r2 = ag.deploy_letters(sc["advocate"], sc["critics"], sc["arbitrator"], PROV, "R2")
    assert (r2["letter_rank"], r2["letter_final"], r2["veto"]) == ("A", "A", ""), r2


def test_deploy_letters_rank13_like_spiral_upheld_is_D_under_both_rules():
    """The PI-confirmed spiral FP: an upheld spiral_arm covering every item at r 0.9 (a = 1)
    demotes to D under R1 (S_arb 0.07, full-coverage rule) and under R2 (the D rule)."""
    weak = {"source_contrast": 3, "low_surface_brightness": 5, "curvature": 5, "counter_image": 2, "arc_morphology": 4}
    adv = advocate(0.7, items=[item(1, 1.2, 300, 30), item(2, 1.4, 120, 200)], criteria=weak)
    crits = {"artifact": critic("artifact"),                                           # "nothing fits"
             "geometry": critic("geometry", no_opinion=True),
             "morphology": critic("morphology", "spiral_arm", 0.9, loc=box(0.8, 1.8, 0, 360), accounts_for=[1, 2])}
    arb = arbitrator([("morphology", "upheld", [1, 2])], letter="D")
    d = _deploy_both(adv, crits, arb, PROV)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_rank"], d[rule]["letter_final"], d[rule]["veto"]) == ("B", "D", "morphology:spiral_arm"), d
        assert abs(d[rule]["S_arb"] - 0.07) < 1e-12 and abs(d[rule]["S"] - 0.07) < 1e-12
    # R2's D rule does not care where letter_rank sits: a strong-criteria A is demoted to D too
    strong = advocate(0.95, items=adv["items"])
    d = _deploy_both(strong, crits, arb, PROV)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_rank"], d[rule]["letter_final"], d[rule]["veto"]) == ("A", "D", "morphology:spiral_arm"), d
    # ... but a PARTIAL ruling on the same critic is not the D rule: R2 keeps A, R1 re-scores
    # (0.95 * (1 - 0.9/2) = 0.5225 >= t_B, r*a 0.45 < 0.8 so no A blocker) to B with the veto
    part = arbitrator([("morphology", "partial", [1])])
    d = _deploy_both(strong, crits, part, PROV)
    assert (d["R2"]["letter_final"], d["R2"]["veto"]) == ("A", "")
    assert (d["R1"]["letter_final"], d["R1"]["veto"]) == ("B", "morphology:spiral_arm")
    assert abs(d["R1"]["S_arb"] - 0.95 * (1 - 0.45)) < 1e-12


def test_deploy_letters_partial_ruling_R1_demotes_R2_keeps():
    """A partial ruling that halves S_arb across t_B: R1 demotes (A -> C, veto), R2 keeps
    letter_rank (no upheld full-coverage critic)."""
    adv = advocate(0.8, items=[item(1, 1.0, 40, 120), item(2, 1.1, 230, 250)])    # p_ev == t_A: letter_rank A
    crits = {"geometry": critic("geometry", "companion_projection", 1.0, loc=ALL, accounts_for=[1, 2])}
    arb = arbitrator([("geometry", "partial", [1])])                               # a' = 1/2 -> S_arb 0.40 < t_B
    d = _deploy_both(adv, crits, arb, PROV)
    assert (d["R1"]["letter_rank"], d["R1"]["letter_final"], d["R1"]["veto"]) == ("A", "C", "geometry:companion_projection"), d
    assert abs(d["R1"]["S_arb"] - 0.4) < 1e-12 and d["R1"]["S"] == 0.0
    assert (d["R2"]["letter_rank"], d["R2"]["letter_final"], d["R2"]["veto"]) == ("A", "A", ""), d
    # item 4 verbatim: letter_final == assign_letter(S_arb, advocate, critics, thresholds, arbitrator)
    assert d["R1"]["letter_final"] == ag.assign_letter(d["R1"]["S_arb"], adv, crits, PROV, arbitrator=arb)[0]
    # with bars the halved score still clears (t_A 0.35 / t_B 0.20) nothing is demoted and no veto is written
    low = ag.resolve_thresholds({"opus5_api": {"tau0": 0.15, "t_A": 0.35, "t_B": 0.2}}, "opus5_api")
    assert low["letter_source"] == "opus5_api_calibrated"
    d = _deploy_both(adv, crits, arb, low)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_rank"], d[rule]["letter_final"], d[rule]["veto"]) == ("A", "A", ""), d


def test_deploy_letters_no_critics_equal():
    """No critics ({}, None, []), all abstaining / unnamed, or every named critic overruled:
    S_arb == p_evidence and letter_final == letter_rank, no veto, under both rules."""
    seen = []
    for adv in (advocate(0.9), advocate(0.6), advocate(0.3),
                advocate(0.1, items=[], nothing_because="isolated elliptical"),
                advocate(0.1, items=[], nothing_because="")):
        want = ag.assign_letter(adv["p_evidence"], adv, [], PROV)[0]
        seen.append(want)
        for crits in ({}, None, []):
            for rule in ag.DEPLOY_RULES:
                d = ag.deploy_letters(adv, crits, None, PROV, rule)
                assert tuple(d) == DEPLOY_KEYS
                assert d["letter_rank"] == d["letter_final"] == want and d["veto"] == "", (adv["p_evidence"], crits, rule, d)
                assert d["S"] == d["S_arb"] == d["p_evidence"] == adv["p_evidence"]
    assert seen == ["A", "B", "C", "D", "C"]                        # every letter reachable from the advocate alone
    adv = advocate(0.9)
    quiet = {"artifact": critic("artifact", no_opinion=True), "geometry": critic("geometry"),
             "morphology": critic("morphology", "", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    d = _deploy_both(adv, quiet, None, PROV)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_rank"], d[rule]["letter_final"], d[rule]["veto"]) == ("A", "A", "")
        assert d[rule]["S"] == d[rule]["S_arb"] == 0.9
    loud = {"geometry": critic("geometry", "companion_projection", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    d = _deploy_both(adv, loud, arbitrator([("geometry", "overruled", [])]), PROV)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_rank"], d[rule]["letter_final"], d[rule]["veto"]) == ("A", "A", "")
        assert d[rule]["S_arb"] == 0.9 and abs(d[rule]["S"] - 0.09) < 1e-12


def test_deploy_letters_provisional_and_calibrated_thresholds():
    """The same records under the provisional numbers and under a calibrated opus5_api entry:
    both resolve_thresholds dicts are accepted, and the veto follows the bars."""
    adv = advocate(0.7)
    crits = {"morphology": critic("morphology", "spiral_arm", 0.5, loc=box(0.7, 1.3, 30, 100), accounts_for=[1, 2, 3])}
    arb = arbitrator([("morphology", "upheld", [1])])                # a_geom 1/3 -> S_arb = 0.7 * (1 - 0.5/3)
    table = {"opus5_api": {"tau0": 0.15, "t_A": 0.6, "t_B": 0.3}, "provisional": dict(ag.PROVISIONAL)}
    prov, cal = ag.resolve_thresholds(table, "sonnet5_api"), ag.resolve_thresholds(table, "opus5_api")
    assert prov["letter_source"] == "provisional" and cal["letter_source"] == "opus5_api_calibrated"
    s_arb = 0.7 * (1 - 0.5 / 3)
    # provisional (t_A .80 / t_B .50): B stays B — the stack lowered S_arb without crossing a bar, so no veto
    d = _deploy_both(adv, crits, arb, prov)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_rank"], d[rule]["letter_final"], d[rule]["veto"]) == ("B", "B", ""), d
        assert abs(d[rule]["S_arb"] - s_arb) < 1e-12
    # calibrated (t_A .60 / t_B .30): A by R, B after the stack under R1 (veto = the upheld critic); R2 keeps A
    d = _deploy_both(adv, crits, arb, cal)
    assert (d["R1"]["letter_rank"], d["R1"]["letter_final"], d["R1"]["veto"]) == ("A", "B", "morphology:spiral_arm"), d
    assert (d["R2"]["letter_rank"], d["R2"]["letter_final"], d["R2"]["veto"]) == ("A", "A", ""), d
    # the shipped thresholds file resolves and is accepted too; the stack never promotes
    real = ag.resolve_thresholds(ag.load_thresholds(str(THR_JSON)), "sonnet_api")
    for rule in ag.DEPLOY_RULES:
        d = ag.deploy_letters(adv, crits, arb, real, rule)
        assert d["letter_rank"] in ag.LETTERS and d["letter_final"] in ag.LETTERS
        assert ag.letter_order(d["letter_final"]) >= ag.letter_order(d["letter_rank"])
    # a bare {t_A, t_B} dict (no letter_source) works as well
    assert ag.deploy_letters(adv, crits, arb, {"t_A": 0.6, "t_B": 0.3})["letter_final"] == "B"


def test_deploy_letters_edge_cases_and_veto_composition():
    # advocate None: everything None / NaN under both rules, the key set fixed
    for rule in ag.DEPLOY_RULES:
        d = ag.deploy_letters(None, {"geometry": critic("geometry", "merger", 0.9, loc=ALL, accounts_for=[1])}, None, PROV, rule)
        assert tuple(d) == DEPLOY_KEYS and d["rule"] == rule
        assert d["letter_rank"] is None and d["letter_final"] is None and d["veto"] == "" and d["p_evidence"] is None
        assert _nanish(d["S"]) and _nanish(d["S_arb"])
    # a called critic that failed to parse (None): S / S_arb NaN, letter_final None (parse-failure
    # row), letter_rank still the advocate's letter, no veto
    adv = advocate(0.9)
    broken = {"artifact": None, "geometry": critic("geometry", "companion_projection", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    for rule in ag.DEPLOY_RULES:
        d = ag.deploy_letters(adv, broken, arbitrator([("geometry", "upheld", [1, 2, 3])]), PROV, rule)
        assert d["letter_rank"] == "A" and d["letter_final"] is None and d["veto"] == ""
        assert _nanish(d["S"]) and _nanish(d["S_arb"]) and d["p_evidence"] == 0.9
    # an advocate without a usable p_evidence: no letters at all
    d = ag.deploy_letters({**adv, "p_evidence": float("nan")}, {}, None, PROV)
    assert d["letter_rank"] is None and d["letter_final"] is None and d["p_evidence"] is None and d["veto"] == ""
    # named critics but NO arbitrator record (none was needed): S_arb == S, R1 is the PRIMARY
    # letter (assign_letter on S with the critics), R2's D rule fires on the critic's own report
    loud = {"geometry": critic("geometry", "companion_projection", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    d = ag.deploy_letters(adv, loud, None, PROV)
    assert d["S_arb"] == d["S"] and abs(d["S"] - 0.09) < 1e-12
    assert d["letter_final"] == ag.assign_letter(d["S"], adv, loud, PROV)[0] == "D"
    assert (d["letter_rank"], d["veto"]) == ("A", "geometry:companion_projection")
    d2 = ag.deploy_letters(adv, loud, None, PROV, "R2")
    assert (d2["letter_final"], d2["veto"]) == ("D", "geometry:companion_projection")
    # list-form critics (Nate's rows) == dict form
    assert ag.deploy_letters(adv, list(loud.values()), None, PROV) == d
    # several responsible critics: joined with ";" in artifact, geometry, morphology order
    # (S_arb = 0.9 * 0.1 * 0.2 = 0.018; both upheld at a_geom 1, r >= 0.8 -> both are D-rule critics)
    many = {"morphology": critic("morphology", "spiral_arm", 0.8, loc=ALL, accounts_for=[1, 2, 3]),
            "geometry": critic("geometry", no_opinion=True),
            "artifact": critic("artifact", "detector_artifact", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    arb = arbitrator([("artifact", "upheld", [1, 2, 3]), ("morphology", "upheld", [1, 2, 3])])
    d = _deploy_both(adv, many, arb, PROV)
    for rule in ag.DEPLOY_RULES:
        assert (d[rule]["letter_final"], d[rule]["veto"]) == ("D", "artifact:detector_artifact;morphology:spiral_arm"), d
    # R1 blames every standing term with r*a > 0 (the partial one too); R2 only the D-rule critic
    arb2 = arbitrator([("artifact", "upheld", [1, 2, 3]), ("morphology", "partial", [1])])
    d = _deploy_both(adv, many, arb2, PROV)
    assert (d["R1"]["letter_final"], d["R1"]["veto"]) == ("D", "artifact:detector_artifact;morphology:spiral_arm")
    assert (d["R2"]["letter_final"], d["R2"]["veto"]) == ("D", "artifact:detector_artifact")
    # a critic the arbitrator did not rule on keeps its term (score_S_arb) and is blamed by R1
    arb3 = arbitrator([("artifact", "overruled", [])])
    d = _deploy_both(adv, many, arb3, PROV)
    assert (d["R1"]["letter_final"], d["R1"]["veto"]) == ("C", "morphology:spiral_arm")      # 0.9*0.2 = 0.18, no ruling -> no D
    assert (d["R2"]["letter_final"], d["R2"]["veto"]) == ("A", "")
    # the R1 veto is empty whenever no bar is crossed, even with a standing r*a > 0 term
    soft = {"geometry": critic("geometry", "companion_projection", 0.2, loc=ALL, accounts_for=[1, 2, 3])}
    d = ag.deploy_letters(advocate(0.95), soft, arbitrator([("geometry", "upheld", [1, 2, 3])]), PROV)
    assert (d["letter_rank"], d["letter_final"], d["veto"]) == ("A", "B", "geometry:companion_projection")   # 0.76 < t_A
    d = ag.deploy_letters(advocate(0.95), soft, arbitrator([("geometry", "upheld", [1, 2, 3])]), {"t_A": 0.7, "t_B": 0.3})
    assert (d["letter_rank"], d["letter_final"], d["veto"]) == ("A", "A", "")
    # rule validation and letter_order
    try:
        ag.deploy_letters(adv, {}, None, PROV, "R3")
        raise SystemExit("rule R3 accepted")
    except ValueError:
        pass
    assert [ag.letter_order(x) for x in ("A", "B", "C", "D", None, "U", "")] == [0, 1, 2, 3, 4, 4, 4]
    assert ag.LETTERS == ("A", "B", "C", "D") and ag.DEPLOY_RULES == ("R1", "R2")
    # the ranking note: letters certify, R orders — rank_key on rows whose S is p_evidence puts a
    # D with the higher R above a C with a lower one, and U strictly below both
    rows = [{"id": "c", "p_evidence": 0.3, "letter_final": "C"}, {"id": "d", "p_evidence": 0.6, "letter_final": "D"},
            {"id": "u", "p_evidence": None, "confidence": 95}]
    order = [r["id"] for r in sorted(rows, key=lambda r: ag.rank_key({**r, "S": r["p_evidence"]}))]
    assert order == ["d", "c", "u"]


def test_d_rule_roles_is_assign_letters_predicate():
    adv = advocate(0.9)
    full = {"geometry": critic("geometry", "companion_projection", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    assert ag.d_rule_roles(ag.critic_terms(adv, full)) == ["geometry"]              # no arbitrator: the report stands
    for arb, want in ((arbitrator([("geometry", "upheld", [1, 2, 3])]), ["geometry"]),
                      (arbitrator([("geometry", "partial", [1, 2, 3])]), []),
                      (arbitrator([("geometry", "overruled", [])]), []),
                      (arbitrator([]), [])):
        terms = ag.critic_terms(adv, full, arb)
        assert ag.d_rule_roles(terms, arb) == want, (arb["rulings"], want)
        # the same predicate decides assign_letter's D below t_B
        assert (ag.assign_letter(0.09, adv, full, PROV, arbitrator=arb)[0] == "D") is bool(want)
    # r 0.79, or geometric coverage < 1 (claims all, box over item 1 only), never qualifies
    r79 = {"geometry": critic("geometry", "companion_projection", 0.79, loc=ALL, accounts_for=[1, 2, 3])}
    assert ag.d_rule_roles(ag.critic_terms(adv, r79)) == []
    part = {"geometry": critic("geometry", "companion_projection", 0.9, loc=box(0.7, 1.3, 30, 100), accounts_for=[1, 2, 3])}
    assert ag.d_rule_roles(ag.critic_terms(adv, part)) == []
    # order is artifact, geometry, morphology whatever the input order
    two = {"morphology": critic("morphology", "spiral_arm", 0.8, loc=ALL, accounts_for=[1, 2, 3]),
           "artifact": critic("artifact", "psf_wing", 0.9, loc=ALL, accounts_for=[1, 2, 3])}
    assert ag.d_rule_roles(ag.critic_terms(adv, two)) == ["artifact", "morphology"]


def main() -> None:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        print(f"{name} ...", flush=True)
        fn()
        print(f"  PASS {name}")
    print(f"\nPASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
