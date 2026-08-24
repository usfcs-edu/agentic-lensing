#!/usr/bin/env python3
"""No-network tests for golden/records.py (per-role records rebuilt from a run's votes).

Covers, on synthetic records / raws / parquets only (no API, no real key file):
  1. `parse_role_raw` is the run-time parse path (common.parse.parse_model on the role's
     schemas_panel record): ```json fences, a prose preamble with arcsecond quotes, trailing
     text and a stray brace object all yield the SAME record parse_model yields, and the
     record the raw was serialised from;
  2. a None / NaN / blank / garbage raw is None; an extra key or an out-of-range field is
     None (extra=forbid, nothing coerced); an unknown role refuses;
  3. role mismatch: an advocate raw under a critic or arbitrator role (and every other
     cross) is None; a geometry critic raw under the artifact role parses exactly as at run
     time, and `persona_matches` is the separate flag;
  4. a tiny votes frame round-trips (absent roles are absent keys, failures None, the
     stored parse_ok honoured by default, k selection, duplicates refused, the trace
     fallback for a missing raw, the parity count table);
  5. `panel_result_from_records` reproduces aggregate_v2 / assemble arithmetic, and a
     preds + votes parquet pair written to tmp_path rebuilds column-for-column
     (`load_run` → `rebuild_rows` → `compare_rebuild` with zero mismatches);
  6. the real scrambled-100 dev run, when present on this machine (skipped otherwise).

Runs under pytest:
    cd reproductions/lensjudge && ~/.venvs/lensjudge/bin/python -m pytest tests/test_golden_records.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from lensjudge.common import parse  # noqa: E402
from lensjudge.golden import aggregate_v2  # noqa: E402
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import schemas_panel as sp  # noqa: E402

LENSJUDGE = Path(__file__).resolve().parents[1]
DEV_DIR = LENSJUDGE / "outputs" / "scrambled100"
DEV_PREDS = DEV_DIR / "preds_scrambled100_a1_sonnet.parquet"
DEV_TRACES = DEV_DIR / "traces_scrambled100_blind"
THR = {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318, "letter_source": "test_calibrated"}


# ------------------------------------------------------------------ synthetic records
def _advocate(p_ev=0.7, items=2, nothing="", span=None):
    its = [sp.EvidenceItem(k=1, what="arc", panel="d", r_arcsec=1.3, pa_deg_from=40, pa_deg_to=170,
                           visible_in_direct=True, criteria=[3, 5]),
           sp.EvidenceItem(k=2, what="counter", panel="e", r_arcsec=1.1, pa_deg_from=230, pa_deg_to=260,
                           visible_in_direct=True, criteria=[4])][:items]
    return sp.AdvocateRecord(
        id="item", persona="advocate",
        criteria=sp.CriteriaV2(source_contrast=7, low_surface_brightness=6, curvature=8, counter_image=6,
                               arc_morphology=7),
        items=its, arc_pa_span_deg=span, scale_class="galaxy", n_red_neighbours_10as=0, bcg_like_halo=False,
        deflector_is_centre=True, p_evidence=p_ev, nothing_because=nothing, notes="tangential arc east of the core")


def _critic(role, alternative=None, r=0.0, no_opinion=False, reason=None, accounts=(), location=True):
    loc = sp.LocationBox(r_arcsec_from=1.0, r_arcsec_to=1.6, pa_deg_from=30, pa_deg_to=180) if location else None
    return sp.CriticRecord(id="item", persona=role, no_opinion=no_opinion, no_opinion_reason=reason,
                           alternative=alternative, alternative_desc="" if alternative is None else "desc",
                           location=loc if alternative else None, accounts_for=list(accounts),
                           leaves_standing=[], refutation_strength=r if alternative else None,
                           notes=f"{role} note")


def _arbitrator(ruling="overruled", persona="geometry"):
    return sp.ArbitratorRecord(id="item", persona="arbitrator",
                               rulings=[sp.Ruling(persona=persona, ruling=ruling, covers=[1], why="w")],
                               surviving_items=[1, 2], letter_llm="B", scale_class_final="galaxy",
                               needs_human=False, rationale="the arc stays")


def _j(rec, indent=2) -> str:
    return json.dumps(rec.model_dump(mode="json"), indent=indent)


def _fenced(rec) -> str:
    return "Here is my reading of the panels.\n\n```json\n" + _j(rec) + "\n```\n"


def _full_stack() -> dict:
    """itemA: advocate + three critics (one named, one abstaining, one nothing) + arbitrator."""
    return {"advocate": _advocate(span=(40.0, 170.0)),
            "artifact": _critic("artifact", no_opinion=True, reason="feature_not_in_my_views"),
            "geometry": _critic("geometry", "shell_tidal", r=0.6, accounts=(1,)),
            "morphology": _critic("morphology"),
            "arbitrator": _arbitrator("partial")}


def _votes_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(R.VOTE_COLS))


def _vote(name, role, raw, parse_ok=True, k=1):
    return {"name": name, "unit_id": "", "role": role, "k": k, "parse_ok": parse_ok, "raw": raw,
            "cost_usd": 0.01, "system_sha16": f"sha_{role}"}


# ------------------------------------------------------------------ 1 + 2: one raw → one record
def test_parse_role_raw_is_the_runtime_parse():
    adv = _advocate(span=(40.0, 170.0))
    clean = _j(adv)
    variants = {
        "bare": clean,
        "fenced": _fenced(adv),
        "prose_arcsec": ('The arc sits 1.3" east of the core and a 0.8" knot lies south; "faint" but real.\n'
                         "My record:\n" + clean + "\nThat is the complete evidence set."),
        "fenced_trailing_brace": "```json\n" + clean + "\n```\nNote: the {counter} image is marginal. {not: json}",
        "compact_no_fence_label": "```\n" + _j(adv, indent=None) + "\n```",
    }
    for label, raw in variants.items():
        rec = R.parse_role_raw("advocate", raw)
        assert rec is not None, label
        assert rec == adv, label                                    # the record it was serialised from
        assert rec == parse.parse_model(raw, sp.AdvocateRecord), label   # == the runner's parse
        assert rec.arc_pa_span_deg == (40.0, 170.0)
    for role, rec in (("artifact", _critic("artifact", "psf_wing", r=0.3, accounts=(1,))),
                      ("geometry", _critic("geometry", no_opinion=True, reason="image_quality")),
                      ("morphology", _critic("morphology")),
                      ("arbitrator", _arbitrator())):
        assert R.parse_role_raw(role, _fenced(rec)) == rec, role
        assert R.parse_role_raw(role, "preamble...\n" + _j(rec) + "\ntrailing") == rec, role
    # the incumbent arm: plain role names, IncumbentVerdict schema under arm="a0" only
    verdict = sp.IncumbentVerdict(id="item", persona="artifact", verdict="fail", alternative="spike", notes="n")
    assert R.parse_role_raw("artifact", _fenced(verdict), arm="a0") == verdict
    assert R.parse_role_raw("artifact", _fenced(verdict)) is None          # CriticRecord forbids `verdict`
    assert R.schema_for("advocate") is sp.AdvocateRecord and R.schema_for("geometry") is sp.CriticRecord
    assert R.schema_for("arbitrator", arm="a0") is sp.IncumbentVerdict


def test_parse_role_raw_missing_and_garbage():
    for raw in (None, float("nan"), "", "   \n", "no json here", "{'bad': json}", b"{}", 3.5):
        assert R.parse_role_raw("advocate", raw) is None, repr(raw)
        assert R.parse_role_raw("artifact", raw) is None, repr(raw)
    adv = _advocate().model_dump(mode="json")
    assert R.parse_role_raw("advocate", json.dumps({**adv, "extra_key": 1})) is None    # extra=forbid
    assert R.parse_role_raw("advocate", json.dumps({**adv, "p_evidence": 1.5})) is None  # out of range
    assert R.parse_role_raw("advocate", json.dumps({**adv, "scale_class": "field"})) is None
    crit = _critic("artifact", "merger", r=0.5).model_dump(mode="json")
    assert R.parse_role_raw("artifact", json.dumps({**crit, "location": None})) is None   # brief rule
    assert R.parse_role_raw("artifact", json.dumps({**crit, "alternative": "bar"})) is None
    with pytest.raises(ValueError):
        R.parse_role_raw("inspector", _j(_advocate()))
    with pytest.raises(ValueError):
        R.schema_for("critic")


# ------------------------------------------------------------------ 3: role mismatch
def test_role_mismatch():
    adv, crit, arb = _fenced(_advocate()), _fenced(_critic("geometry", "merger", r=0.5, accounts=(1,))), _fenced(_arbitrator())
    for role in ("artifact", "geometry", "morphology", "arbitrator"):
        assert R.parse_role_raw(role, adv) is None, role
    for role in ("advocate", "arbitrator"):
        assert R.parse_role_raw(role, crit) is None, role
    for role in ("advocate", "artifact", "geometry", "morphology"):
        assert R.parse_role_raw(role, arb) is None, role
    # run-time parity: CriticRecord's persona Literal spans the three critics, so a geometry
    # reply stored under the artifact role parses; persona_matches is the separate flag
    rec = R.parse_role_raw("artifact", crit)
    assert rec is not None and rec.persona == "geometry"
    assert R.persona_matches("artifact", rec) is False
    assert R.persona_matches("geometry", rec) is True
    assert R.persona_matches("geometry", None) is False
    assert R.persona_matches("advocate", R.parse_role_raw("advocate", adv)) is True


# ------------------------------------------------------------------ 4: votes → records
def test_records_from_votes_roundtrip(tmp_path):
    stack = _full_stack()
    rows = [_vote("itemA", role, _fenced(rec) if role != "morphology" else _j(rec)) for role, rec in stack.items()]
    rows.append(_vote("itemB", "advocate", _j(_advocate(p_ev=0.05, items=0, nothing="a bare elliptical"))))
    rows.append(_vote("itemC", "advocate", "I cannot produce a record for this one.", parse_ok=False))
    rows.append(_vote("itemD", "advocate", _j(_advocate()), parse_ok=False))     # parses today, failed then
    rows.append(_vote("itemE", "advocate", float("nan")))                        # raw lost, parse_ok True
    votes = _votes_frame(rows)

    recs = R.records_from_votes(votes)
    assert list(recs) == ["itemA", "itemB", "itemC", "itemD", "itemE"]
    assert set(recs["itemA"]) == set(R.ROLES)
    for role, rec in stack.items():
        assert recs["itemA"][role] == rec, role
    assert set(recs["itemB"]) == {"advocate"} and recs["itemB"]["advocate"].p_evidence == 0.05
    assert recs["itemC"] == {"advocate": None}
    assert recs["itemD"] == {"advocate": None}                    # stored parse_ok honoured
    assert recs["itemE"] == {"advocate": None}                    # missing raw, no trace
    assert R.records_from_votes(votes, respect_parse_ok=False)["itemD"]["advocate"] == _advocate()
    assert R.records_from_votes(votes, k=1)["itemA"]["geometry"] == stack["geometry"]
    assert R.records_from_votes(votes, k=2) == {}
    assert R.records_from_votes(votes.iloc[0:0]) == {}

    # the trace fallback: a per-role trace with the run's direct_response text
    tdir = tmp_path / "traces"
    tdir.mkdir()
    tp = R.trace_path_for(tdir, "itemE", "advocate")
    assert tp.name == "itemE_advocate.jsonl"
    events = [{"t": 1.0, "event": "golden_content_audit", "system_sha16": "x"},
              {"t": 2.0, "event": "direct_request", "model": "m"},
              {"t": 3.0, "event": "direct_response", "text": "not a record", "cost_usd": 0.01},
              {"t": 4.0, "event": "direct_repair", "parse_ok": True, "text": _fenced(_advocate(p_ev=0.42))}]
    tp.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    assert R.raw_from_trace(tp) == _fenced(_advocate(p_ev=0.42))       # the repaired text wins
    assert R.raw_from_trace(tdir / "nope_advocate.jsonl") is None
    # a failed repair leaves the response text; a later response starts over
    (tdir / "x_artifact.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event": "direct_response", "text": "first"}, {"event": "direct_repair", "parse_ok": False, "text": "bad"},
        {"event": "direct_response", "text": "second"}]) + "\n")
    assert R.raw_from_trace(tdir / "x_artifact.jsonl") == "second"
    recs_t = R.records_from_votes(votes, trace_dir=tdir)
    assert recs_t["itemE"]["advocate"] == _advocate(p_ev=0.42)
    assert recs_t["itemA"]["advocate"] == stack["advocate"]           # a present raw is never overridden

    # parity count table (no ids)
    par = R.parse_parity(votes)
    assert list(par.columns) == list(R.PARITY_COLS) and int(par["n"].sum()) == len(votes)
    key = par.set_index(["role", "parse_ok", "raw_missing", "parsed_now"])["n"]
    assert int(key.loc[("advocate", True, False, True)]) == 2           # itemA, itemB
    assert int(key.loc[("advocate", False, False, False)]) == 1         # itemC
    assert int(key.loc[("advocate", False, False, True)]) == 1          # itemD parses today
    assert int(key.loc[("advocate", True, True, False)]) == 1           # itemE raw missing
    par_t = R.parse_parity(votes, trace_dir=tdir)
    assert int(par_t.set_index(["role", "parse_ok", "raw_missing", "parsed_now"])["n"].loc[("advocate", True, True, True)]) == 1
    assert par["role"].tolist()[0] == "advocate" and par["role"].tolist()[-1] == "arbitrator"
    assert bool(par[par["role"] == "arbitrator"]["persona_ok"].iloc[0]) is True

    # refusals: several replicates without k, duplicate (name, role), missing columns
    two_k = pd.concat([votes, _votes_frame([_vote("itemA", "advocate", _j(_advocate()), k=2)])], ignore_index=True)
    with pytest.raises(ValueError):
        R.records_from_votes(two_k)
    assert set(R.records_from_votes(two_k, k=2)) == {"itemA"}
    dup = pd.concat([votes, votes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        R.records_from_votes(dup)
    with pytest.raises(ValueError):
        R.records_from_votes(votes.drop(columns=["raw"]))
    with pytest.raises(ValueError):
        R.records_from_votes(votes.drop(columns=["k"]), k=1)


# ------------------------------------------------------------------ 5: records → S, letters, rows
def test_panel_result_from_records_matches_aggregate():
    stack = _full_stack()
    records = {"itemA": stack,
               "itemB": {"advocate": _advocate(p_ev=0.05, items=0, nothing="a bare elliptical")},
               "itemF": {"advocate": _advocate(p_ev=0.6), "artifact": None,
                         "geometry": _critic("geometry"), "morphology": _critic("morphology")},
               "itemG": {"advocate": None}}
    adv, critics, arb = stack["advocate"], {r: stack[r] for r in R.CRITIC_ROLES}, stack["arbitrator"]
    res = R.panel_result_from_records("itemA", records, THR)
    assert isinstance(res, sp.PanelResult) and res.parse_ok and res.parse_failures == []
    assert math.isclose(res.S, aggregate_v2.score_S(adv, critics))
    assert math.isclose(res.S_arb, aggregate_v2.score_S_arb(adv, critics, arb))
    assert res.S < adv.p_evidence and res.S_arb <= adv.p_evidence          # the named critic demotes
    assert (res.letter, res.letter_source) == aggregate_v2.assign_letter(res.S, adv, critics, THR)
    assert res.letter_arb == aggregate_v2.assign_letter(res.S_arb, adv, critics, THR, arbitrator=arb)[0]
    assert res.letter_source == "test_calibrated" and res.calls == 5 and res.cost_usd == 0.0
    assert set(res.critics) == set(R.CRITIC_ROLES) and res.arbitrator == arb
    assert res.meta["rebuilt_from"] == "votes"
    # advocate-only item: S = p_evidence, no critics, no arbitrator
    resB = R.panel_result_from_records("itemB", records, THR)
    assert resB.S == 0.05 and resB.critics == {} and resB.arbitrator is None and resB.letter == "D"
    # a failed critic ⇒ parse failure row (S NaN, letter None), the arbitrator not called
    resF = R.panel_result_from_records("itemF", records, THR)
    assert resF.parse_failures == ["artifact"] and math.isnan(resF.S) and resF.letter is None
    assert not resF.parse_ok and resF.calls == 4
    resG = R.panel_result_from_records("itemG", records, THR)
    assert resG.parse_failures == ["advocate"] and math.isnan(resG.S) and resG.p_evidence is None
    # refusals
    with pytest.raises(KeyError):
        R.panel_result_from_records("itemZ", records, THR)
    with pytest.raises(ValueError):
        R.panel_result_from_records("itemA", records)                  # no thresholds, no row
    inc = {"itemI": {"artifact": sp.IncumbentVerdict(id="item", persona="artifact", verdict="pass")}}
    with pytest.raises(ValueError):
        R.panel_result_from_records("itemI", inc, THR)
    # thresholds from a stored row
    row = {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318, "letter_source": "test_calibrated", "S": 0.3}
    assert R.thresholds_from_row(row) == THR
    assert R.thresholds_from_row(pd.Series(row)) == THR
    assert R.thresholds_from_row({"t_A": 0.8, "t_B": 0.5})["letter_source"] == "provisional"
    with pytest.raises(ValueError):
        R.thresholds_from_row({"tau0": 0.15, "t_A": float("nan"), "t_B": 0.5})
    # the row shape
    row_a = R.row_from_records("itemA", records, THR, preds_row={"grade_truth": "A", "catalog": "c", "region": "r"})
    assert list(row_a) == list(sp.ROW_COLS)
    assert row_a["grade_truth"] == "A" and row_a["p_lens"] == res.S and row_a["grade_pred"] == res.letter
    assert row_a["calls"] == 5 and row_a["alt_geometry"] == "shell_tidal" and row_a["ruling_geometry"] == "partial"
    assert set(R.REBUILD_COLS) <= set(sp.ROW_COLS)


def test_load_run_and_rebuild_roundtrip(tmp_path):
    stack = _full_stack()
    records = {"itemA": stack,
               "itemB": {"advocate": _advocate(p_ev=0.05, items=0, nothing="a bare elliptical")},
               "itemF": {"advocate": _advocate(p_ev=0.6), "artifact": None,
                         "geometry": _critic("geometry"), "morphology": _critic("morphology")}}
    # the preds parquet as the runner would write it: to_row rows + the run-tuple columns
    rows = []
    for name, roles in records.items():
        r = R.row_from_records(name, records, THR, preds_row={"cost_usd": 0.1 * len(roles),
                                                             **{f"system_sha16_{k}": f"sha_{k}" for k in roles}})
        r.update({"k": 1, "arm": "a1", "model": "sonnet", **THR})
        rows.append(r)
    preds = pd.DataFrame(rows)
    preds_path = tmp_path / "preds_test_a1_sonnet.parquet"
    preds.to_parquet(preds_path, index=False)
    assert R.votes_path_for(preds_path) == tmp_path / "preds_test_a1_sonnet_votes.parquet"
    vrows = []
    for name, roles in records.items():
        for role, rec in roles.items():
            raw = _fenced(rec) if rec is not None else "garbled reply"
            vrows.append(_vote(name, role, raw, parse_ok=rec is not None))
    _votes_frame(vrows).to_parquet(R.votes_path_for(preds_path), index=False)

    preds2, records2 = R.load_run(preds_path)
    assert len(preds2) == 3 and set(records2) == set(records)
    for name, roles in records.items():
        assert records2[name] == roles, name
    rebuilt = R.rebuild_rows(preds2, records2)
    assert list(rebuilt.columns) == list(sp.ROW_COLS) and rebuilt["name"].tolist() == ["itemA", "itemB", "itemF"]
    check = R.compare_rebuild(preds2, rebuilt)
    assert list(check.columns) == list(R.REBUILD_CHECK_COLS) and len(check) == len(R.REBUILD_COLS)
    assert (check["n_compared"] == 3).all()
    assert int(check["n_mismatch"].sum()) == 0, check[check["n_mismatch"] > 0]
    # a deliberate change is counted (one letter, one S)
    tweaked = rebuilt.copy()
    tweaked.loc[0, "grade_pred"] = "D"
    tweaked.loc[1, "S"] = 0.99
    bad = R.compare_rebuild(preds2, tweaked).set_index("col")["n_mismatch"]
    assert int(bad["grade_pred"]) == 1 and int(bad["S"]) == 1 and int(bad["S_arb"]) == 0
    # NaN agrees with NaN / None (the parse-failure row), names align by string
    assert math.isnan(preds2.set_index("name").loc["itemF", "S"]) and int(bad["p_lens"]) == 0
    # a stray votes name refuses; an explicit votes path is honoured
    stray = _votes_frame(vrows + [_vote("itemQ", "advocate", _j(_advocate()))])
    stray_path = tmp_path / "stray_votes.parquet"
    stray.to_parquet(stray_path, index=False)
    with pytest.raises(ValueError):
        R.load_run(preds_path, stray_path)
    _, records3 = R.load_run(preds_path, R.votes_path_for(preds_path), respect_parse_ok=False)
    assert records3["itemF"]["artifact"] is None                        # garbled stays None either way
    # the CLI runs zero-API on the pair and pins the two count tables in a NEW directory
    out = tmp_path / "records_check"
    assert R.main(["--preds", str(preds_path), "--out", str(out)]) == 0
    assert (out / "parse_parity.csv.sha").exists() and (out / "rebuild_check.csv.sha").exists()
    assert int(pd.read_csv(out / "rebuild_check.csv")["n_mismatch"].sum()) == 0


# ------------------------------------------------------------------ 6: the real dev run
@pytest.mark.skipif(not (DEV_PREDS.exists() and R.votes_path_for(DEV_PREDS).exists()),
                    reason="scrambled-100 dev parquets not on this machine")
def test_real_scrambled100_dev_run():
    votes = pd.read_parquet(R.votes_path_for(DEV_PREDS))
    par = R.parse_parity(votes, arm="a1")
    stored_ok = par[par["parse_ok"] == True]                                        # noqa: E712
    # every parse_ok row whose raw is stored still parses to a record of its own persona
    present = stored_ok[~stored_ok["raw_missing"]]
    assert int(present.loc[~present["parsed_now"], "n"].sum()) == 0, par
    assert int(present.loc[~present["persona_ok"], "n"].sum()) == 0, par
    # a missing raw is the only way a parse_ok row fails to rebuild
    assert int(stored_ok.loc[~stored_ok["parsed_now"], "n"].sum()) == int(stored_ok.loc[stored_ok["raw_missing"], "n"].sum())
    trace_dir = DEV_TRACES if DEV_TRACES.exists() else None
    preds, records = R.load_run(DEV_PREDS, trace_dir=trace_dir)
    assert len(records) == len(preds)
    rebuilt = R.rebuild_rows(preds, records)
    check = R.compare_rebuild(preds, rebuilt).set_index("col")
    if trace_dir is not None:   # with the traces every row rebuilds column-for-column
        par_t = R.parse_parity(votes, arm="a1", trace_dir=trace_dir)
        assert int(par_t.loc[~par_t["parsed_now"], "n"].sum()) == 0, par_t
        assert int(check["n_mismatch"].sum()) == 0, check[check["n_mismatch"] > 0]
    else:                       # without them only the raw-less items differ
        n_missing_items = votes.loc[votes["raw"].isna(), "name"].nunique()
        assert int(check["n_mismatch"].max()) <= n_missing_items, check


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


def _adv(p_ev=0.95):
    crit = {"source_contrast": 7, "low_surface_brightness": 6, "curvature": 8, "counter_image": 6, "arc_morphology": 7}
    items = [{"k": 1, "what": "item 1", "panel": "d", "r_arcsec": 1.2, "pa_deg_from": 300, "pa_deg_to": 30,
              "visible_in_direct": True, "criteria": [3]},
             {"k": 2, "what": "item 2", "panel": "d", "r_arcsec": 1.4, "pa_deg_from": 120, "pa_deg_to": 200,
              "visible_in_direct": True, "criteria": [3]}]
    return {"id": "x", "persona": "advocate", "criteria": crit, "items": items, "scale_class": "galaxy",
            "n_red_neighbours_10as": 0, "bcg_like_halo": False, "deflector_is_centre": True,
            "p_evidence": p_ev, "nothing_because": "", "notes": "n"}


def _crit(role, alternative=None, r=None, no_opinion=False):
    d = {"id": "x", "persona": role, "no_opinion": no_opinion, "alternative": alternative,
         "location": {"r_arcsec_from": 0.8, "r_arcsec_to": 1.8, "pa_deg_from": 0, "pa_deg_to": 360} if alternative else None,
         "accounts_for": [1, 2] if alternative else [], "leaves_standing": []}
    if r is not None:
        d["refutation_strength"] = r
    if no_opinion:
        d["no_opinion_reason"] = "outside_competence"
    return d


def test_deploy_from_roles_voids_a_called_but_failed_arbitrator():
    """records_from_votes marks a called role that failed to parse as a PRESENT key with a
    None record. deploy_letters treats a None arbitrator as 'not called' (the critics'
    reports stand -> demotion), which is wrong for a failed one: at run time that row is a
    parse failure (S NaN, letter None). deploy_from_roles voids letter_final / veto / S_arb
    and keeps letter_rank; an absent arbitrator key and a full stack pass through unchanged."""
    thr = {**aggregate_v2.PROVISIONAL, "letter_source": "provisional"}
    adv = _adv(0.95)
    crits = {"artifact": _crit("artifact"), "geometry": _crit("geometry", no_opinion=True),
             "morphology": _crit("morphology", "spiral_arm", 0.9)}
    arb = {"id": "x", "persona": "arbitrator",
           "rulings": [{"persona": "morphology", "ruling": "upheld", "covers": [1, 2], "why": ""}],
           "surviving_items": [], "letter_llm": "D", "scale_class_final": "galaxy", "needs_human": False,
           "rationale": "r"}
    full = {"advocate": adv, **crits, "arbitrator": arb}
    for rule in aggregate_v2.DEPLOY_RULES:
        ref = aggregate_v2.deploy_letters(adv, crits, arb, thr, rule)
        assert R.deploy_from_roles(full, thr, rule) == ref
        assert (ref["letter_rank"], ref["letter_final"], ref["veto"]) == ("A", "D", "morphology:spiral_arm")
        # the arbitrator was called and failed: no certified letter, letter_rank kept
        d = R.deploy_from_roles({**full, "arbitrator": None}, thr, rule)
        assert d["letter_rank"] == "A" and d["letter_final"] is None and d["veto"] == ""
        assert math.isnan(d["S_arb"]) and abs(d["p_evidence"] - 0.95) < 1e-12 and d["rule"] == rule
        # not called (key absent): deploy_letters' own semantics (critic reports stand)
        absent = {k: v for k, v in full.items() if k != "arbitrator"}
        assert R.deploy_from_roles(absent, thr, rule) == aggregate_v2.deploy_letters(adv, crits, None, thr, rule)
        # a failed advocate / no records: every letter None
        assert R.deploy_from_roles({"advocate": None}, thr, rule)["letter_rank"] is None
        assert R.deploy_from_roles(None, thr, rule)["letter_final"] is None
    # advocate-only stack (critics not called): letter_final == letter_rank
    d = R.deploy_from_roles({"advocate": _adv(0.1)}, thr, "R1")
    assert d["letter_final"] == d["letter_rank"] and d["veto"] == ""
