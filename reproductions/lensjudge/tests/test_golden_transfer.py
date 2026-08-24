#!/usr/bin/env python3
"""No-network tests for golden/transfer_check.py (REGISTRY "Deployment rule v2-deploy" items
6–7): the derived holdout endpoints of letter_rank / R1 / R2 and the pre-stated selection.

Everything runs on synthetic a1 / a2 run directories written to tmp_path — tiny votes
parquets holding fenced synthetic raws plus `to_row` preds rows with truth columns — whose
letters are known by construction, so FPR / recall / stress_D counts and their
Clopper–Pearson CIs are asserted as exact numbers. Covered: the CP formula against
analyze_truth's; provisional vs calibrated threshold resolution and the tuple sha; the three
letter tables row by row; NaN-S exclusion (a1 row with a failed advocate); both branches of
the selection rule (pure function AND end to end under two thresholds tables); the
crit-column fallback for letter_rank when the a2 votes are absent; the CLI (files, PROVISIONAL
label, pinned csv, selected_rule.json shape, --rule-select refusal, --overwrite); and a
read-only smoke on the real holdout parquets (skipped when absent — never writes into
outputs/).

    cd reproductions/lensjudge && ~/.venvs/lensjudge/bin/python -m pytest tests/test_golden_transfer.py -q
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
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import run_truth_eval as rte  # noqa: E402
from lensjudge.golden import schemas_panel as sp  # noqa: E402
from lensjudge.golden import transfer_check as T  # noqa: E402

LENSJUDGE = Path(__file__).resolve().parents[1]
REAL_A2 = LENSJUDGE / "outputs" / "preds_truth_a2_opus5_holdout_k1_r1.parquet"
REAL_A1 = LENSJUDGE / "outputs" / "preds_truth_a1_opus5_holdout_k1_r1.parquet"
REAL_THRESHOLDS = _util.HERE / "thresholds_v2.json"

PROV_TABLE = {"sonnet_api": {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318}, "opus5_api": None,
              "provisional": {"tau0": 0.15, "t_A": 0.8, "t_B": 0.5}}
CAL_TABLE = {"sonnet_api": {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318},
             "opus5_api": {"tau0": 0.15, "t_A": 0.5, "t_B": 0.3},
             "provisional": {"tau0": 0.15, "t_A": 0.8, "t_B": 0.5}}


# ------------------------------------------------------------------ synthetic records
def _advocate(p_ev, items=2, nothing="", strong=True):
    its = [sp.EvidenceItem(k=1, what="arc", panel="d", r_arcsec=1.3, pa_deg_from=40, pa_deg_to=170,
                           visible_in_direct=True, criteria=[3, 5]),
           sp.EvidenceItem(k=2, what="counter", panel="e", r_arcsec=1.1, pa_deg_from=230, pa_deg_to=260,
                           visible_in_direct=True, criteria=[4])][:items]
    crit = (dict(curvature=8, counter_image=6, arc_morphology=7) if strong
            else dict(curvature=3, counter_image=2, arc_morphology=4))
    return sp.AdvocateRecord(
        id="item", persona="advocate", criteria=sp.CriteriaV2(source_contrast=7, low_surface_brightness=6, **crit),
        items=its, scale_class="galaxy", n_red_neighbours_10as=0, bcg_like_halo=False, deflector_is_centre=True,
        p_evidence=p_ev, nothing_because=nothing, notes="an arc east of the core")


def _critic(role, alternative=None, r=0.0, no_opinion=False, reason=None, accounts=()):
    # the box covers item 1 (r 1.3, PA 40–170) and not item 2 (r 1.1, PA 230–260)
    loc = sp.LocationBox(r_arcsec_from=1.0, r_arcsec_to=1.6, pa_deg_from=30, pa_deg_to=180)
    return sp.CriticRecord(id="item", persona=role, no_opinion=no_opinion, no_opinion_reason=reason,
                           alternative=alternative, alternative_desc="desc" if alternative else "",
                           location=loc if alternative else None, accounts_for=list(accounts),
                           leaves_standing=[], refutation_strength=r if alternative else None, notes=f"{role}")


def _arbitrator(rulings=None):
    rl = [sp.Ruling(persona=p, ruling=v, covers=[1], why="w") for p, v in (rulings or {}).items()]
    return sp.ArbitratorRecord(id="item", persona="arbitrator", rulings=rl, surviving_items=[1, 2],
                               letter_llm="C", scale_class_final="galaxy", needs_human=False, rationale="r")


def _stack(adv, named=None, ruling=None, abstain=False):
    """advocate + three critics (`named` = (role, alternative, r, accounts)) + arbitrator."""
    roles = {"advocate": adv, "artifact": _critic("artifact", no_opinion=abstain,
                                                  reason="feature_not_in_my_views" if abstain else None),
             "geometry": _critic("geometry"), "morphology": _critic("morphology")}
    if named:
        role, alt, r, accounts = named
        roles[role] = _critic(role, alt, r, accounts=accounts)
        roles["arbitrator"] = _arbitrator({role: ruling})
    else:
        roles["arbitrator"] = _arbitrator()
    return roles


def _fenced(rec) -> str:
    return "Reading the panels.\n\n```json\n" + json.dumps(rec.model_dump(mode="json"), indent=2) + "\n```\n"


# ------------------------------------------------------------------ the synthetic holdout
# name: (truth_class, is_positive, advocate) — the same advocate record in a2 and a1
ITEMS = {
    "neg1": ("negative", False, _advocate(0.85)),                       # rank A
    "neg2": ("negative", False, _advocate(0.60, items=1)),              # rank B
    "neg3": ("negative", False, _advocate(0.30)),                       # rank C
    "neg4": ("negative", False, _advocate(0.10, items=0, nothing="a bare elliptical")),   # rank D, no critics
    "neg5": ("negative", False, _advocate(0.20)),                       # rank C
    "neg6": ("negative", False, _advocate(0.05, items=0, nothing="empty field")),         # rank D, no critics
    "neg7": ("negative", False, _advocate(0.55)),                       # rank B
    "neg8": ("negative", False, _advocate(0.85, strong=False)),         # rank B (A blocked by criteria)
    "neg9": ("negative", False, _advocate(0.30)),                       # rank C; a1: advocate parse failure
    "pos1": ("cowls", True, _advocate(0.90)),                           # rank A
    "pos2": ("cowls", True, _advocate(0.70)),                           # rank A (0.7 < 0.8 => B under provisional)
    "pos3": ("lit_galaxy", True, _advocate(0.40)),                      # rank C
    "pos4": ("lit_cluster", True, _advocate(0.90, items=1)),            # rank A
    "sd1": ("stress_D", False, _advocate(0.60, items=1)),               # rank B
    "sd2": ("stress_D", False, _advocate(0.30)),                        # rank C
}
# a1: the critic stack per item (None = advocate only, p_evidence < tau0)
A1_STACKS = {
    "neg1": _stack(ITEMS["neg1"][2], ("morphology", "spiral_arm", 0.6, (1,)), "partial"),   # S_arb .595
    "neg2": _stack(ITEMS["neg2"][2], ("geometry", "shell_tidal", 0.9, (1,)), "upheld"),     # S_arb .06, D rule
    "neg3": _stack(ITEMS["neg3"][2]),
    "neg4": None,
    "neg5": _stack(ITEMS["neg5"][2], abstain=True),
    "neg6": None,
    "neg7": _stack(ITEMS["neg7"][2], ("morphology", "spiral_arm", 0.9, (1,)), "upheld"),    # S_arb .3025, a=.5
    "neg8": _stack(ITEMS["neg8"][2], ("geometry", "shell_tidal", 0.9, (1,)), "overruled"),  # S_arb .85
    "neg9": "FAIL",
    "pos1": _stack(ITEMS["pos1"][2]),
    "pos2": _stack(ITEMS["pos2"][2], ("morphology", "spiral_arm", 0.6, (1,)), "partial"),   # S_arb .49
    "pos3": _stack(ITEMS["pos3"][2]),
    "pos4": _stack(ITEMS["pos4"][2], ("geometry", "shell_tidal", 0.9, (1,)), "upheld"),     # S_arb .09, D rule
    "sd1": _stack(ITEMS["sd1"][2], ("geometry", "shell_tidal", 0.9, (1,)), "upheld"),       # D rule
    "sd2": _stack(ITEMS["sd2"][2]),
}
# expected letters under the PROVISIONAL thresholds (t_A .8, t_B .5)
EXPECT_PROV = {
    #         rank  R1   R2
    "neg1": ("A", "B", "A"), "neg2": ("B", "D", "D"), "neg3": ("C", "C", "C"), "neg4": ("D", "D", "D"),
    "neg5": ("C", "C", "C"), "neg6": ("D", "D", "D"), "neg7": ("B", "C", "B"), "neg8": ("B", "B", "B"),
    "neg9": ("C", None, None),
    "pos1": ("A", "A", "A"), "pos2": ("B", "C", "B"), "pos3": ("C", "C", "C"), "pos4": ("A", "D", "D"),
    "sd1": ("B", "D", "D"), "sd2": ("C", "C", "C"),
}


def _vote(name, role, raw, parse_ok=True):
    return {"name": name, "unit_id": "", "role": role, "k": 1, "parse_ok": parse_ok, "raw": raw,
            "cost_usd": 0.01, "system_sha16": f"sha_{role}"}


def _write_run(dir_: Path, arm: str, thr: dict, stacks: dict) -> Path:
    """A preds + votes parquet pair the runner would have written, lettered under `thr`."""
    thr_row = {**{k: thr[k] for k in ("tau0", "t_A", "t_B", "letter_source")}, "thresholds_sha16": thr["thresholds_sha16"]}
    rows, votes = [], []
    for name, (cls, pos, _adv) in ITEMS.items():
        roles = stacks[name]
        if roles is None:                                   # p_evidence < tau0: advocate only
            roles = {"advocate": _adv}
        if roles == "FAIL":
            roles = {"advocate": None}
            votes.append(_vote(name, "advocate", "garbled reply", parse_ok=False))
        else:
            for role, rec in roles.items():
                votes.append(_vote(name, role, _fenced(rec)))
        row = R.row_from_records(name, {name: roles}, T._bare(thr), preds_row={"cost_usd": 0.01 * len(roles)})
        row.update({"truth_class": cls, "is_positive": pos, "is_anchor": False, "k": 1, "arm": arm,
                    "model": "opus5", **thr_row})
        rows.append(row)
    preds = dir_ / f"preds_truth_{arm}_opus5_holdout_k1_r1.parquet"
    pd.DataFrame(rows).to_parquet(preds, index=False)
    pd.DataFrame(votes, columns=list(R.VOTE_COLS)).to_parquet(R.votes_path_for(preds), index=False)
    return preds


def _runs(tmp_path: Path, table: dict = PROV_TABLE) -> tuple:
    """(a2_path, a1_path, thresholds_path, thr) for one thresholds table."""
    tpath = tmp_path / "thresholds_v2.json"
    tpath.write_text(json.dumps(table, indent=2) + "\n")
    thr = T.load_thresholds(tpath, "opus5_api")
    a2 = _write_run(tmp_path, "a2", thr, {n: {"advocate": adv} for n, (_c, _p, adv) in ITEMS.items()})
    a1 = _write_run(tmp_path, "a1", thr, A1_STACKS)
    return a2, a1, tpath, thr


def _cp(k, n):
    return T.clopper_pearson(k, n)


# ------------------------------------------------------------------ 1: the CI
def test_clopper_pearson_hand_values_and_analyze_truth():
    lo, hi = _cp(0, 8)
    assert lo == 0.0 and math.isclose(hi, 1 - 0.025 ** (1 / 8), rel_tol=1e-9)
    lo, hi = _cp(8, 8)
    assert hi == 1.0 and math.isclose(lo, 0.025 ** (1 / 8), rel_tol=1e-9)
    assert all(math.isnan(x) for x in _cp(0, 0))
    from lensjudge.golden import analyze_truth
    for k, n in ((1, 9), (4, 9), (3, 4), (2, 231), (0, 200)):
        a, b = _cp(k, n)
        c, d = analyze_truth.clopper_pearson(k, n)
        assert math.isclose(a, c, abs_tol=1e-12) and math.isclose(b, d, abs_tol=1e-12)
    # monotone: a wider n at the same rate narrows
    assert _cp(10, 100)[1] - _cp(10, 100)[0] < _cp(1, 10)[1] - _cp(1, 10)[0]


# ------------------------------------------------------------------ 2: thresholds
def test_load_thresholds_provisional_and_calibrated(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(PROV_TABLE))
    thr = T.load_thresholds(p, "opus5_api")
    assert thr["provisional"] and thr["thresholds_key"] == "provisional" and thr["letter_source"] == "provisional"
    assert (thr["t_A"], thr["t_B"], thr["tau0"]) == (0.8, 0.5, 0.15)
    # the tuple sha a run carries: run_truth_eval's on the same resolved dict
    want = rte.thresholds_sha({**aggregate_v2.resolve_thresholds(PROV_TABLE, "opus5_api"), "model_key": "opus5_api"})
    assert thr["thresholds_sha16"] == want == "a40ae6e201a03e65"
    assert thr["table_sha16"] == _util.sha_text(p.read_text())
    p.write_text(json.dumps(CAL_TABLE))
    thr = T.load_thresholds(p, "opus5_api")
    assert not thr["provisional"] and thr["letter_source"] == "opus5_api_calibrated"
    assert (thr["t_A"], thr["t_B"]) == (0.5, 0.3) and thr["thresholds_sha16"] != want
    assert set(T._bare(thr)) == {"tau0", "t_A", "t_B", "letter_source"}


# ------------------------------------------------------------------ 3: letters + endpoints, exact
def test_letter_tables_endpoints_and_nan_exclusion(tmp_path):
    a2, a1, _tpath, thr = _runs(tmp_path)
    res = T.run(a2, a1, thr)
    assert res.letter_rank_source == T.SOURCE_VOTES and set(res.tables) == set(T.RULES)
    for rule, col in (("letter_rank", 0), ("R1", 1), ("R2", 2)):
        tab = res.tables[rule].set_index("name")
        assert list(tab.columns) == [c for c in T.LETTER_TABLE_COLS if c != "name"]
        got = {n: (None if pd.isna(tab.at[n, "letter"]) else tab.at[n, "letter"]) for n in ITEMS}
        assert got == {n: v[col] for n, v in EXPECT_PROV.items()}, rule
    rank, r1, r2 = (res.tables[r].set_index("name") for r in T.RULES)
    # NaN-S exclusion: neg9's advocate failed in a1 (S NaN) — excluded from R1 / R2, scored in a2
    assert bool(r1.at["neg9", "excluded"]) and r1.at["neg9", "exclude_reason"] == "nan_S"
    assert math.isnan(r1.at["neg9", "S_stored"]) and not bool(rank.at["neg9", "excluded"])
    assert int(r1["excluded"].sum()) == 1 and int(rank["excluded"].sum()) == 0
    # vetoes name the demoting critic
    assert r1.at["neg1", "veto"] == "morphology:spiral_arm" and r2.at["neg1", "veto"] == ""
    assert r2.at["neg2", "veto"] == "geometry:shell_tidal" and r1.at["neg8", "veto"] == ""
    assert math.isclose(r1.at["neg1", "S_arb"], 0.595) and math.isclose(r1.at["neg7", "S_arb"], 0.3025)
    rows = res.rows
    s = lambda stat, rule: T.stat(rows, stat, rule)  # noqa: E731

    def check(stat_, rule, k, n):
        r = s(stat_, rule)
        lo, hi = _cp(k, n)
        assert r["n"] == n and math.isclose(r["value"], k / n) and math.isclose(r["ci_lo"], lo) \
            and math.isclose(r["ci_hi"], hi), (stat_, rule, r)
    # counts
    assert (s("n_rows", "letter_rank")["value"], s("n_excluded_nan", "letter_rank")["value"]) == (15, 0)
    assert (s("n_rows", "R1")["value"], s("n_excluded_nan", "R1")["value"], s("n_scored", "R1")["value"]) == (15, 1, 14)
    assert s("n_unlettered", "R1")["value"] == 0 and s("n_anchor_excluded", "R1")["value"] == 0
    assert (s("n_neg", "letter_rank")["value"], s("n_neg", "R1")["value"], s("n_neg", "R2")["value"]) == (9, 8, 8)
    assert all(s("n_pos", r)["value"] == 4 and s("n_stress_D", r)["value"] == 2 for r in T.RULES)
    # letter_rank on 9 negatives: A = {neg1}; A∪B = {neg1, neg2, neg7, neg8}
    check("fpr_A", "letter_rank", 1, 9)
    check("fpr_AB", "letter_rank", 4, 9)
    check("recall_A", "letter_rank", 2, 4)          # pos1, pos4
    check("recall_AB", "letter_rank", 3, 4)         # + pos2
    assert s("stress_D_AB_count", "letter_rank")["value"] == 1 and s("stress_D_AB_count", "letter_rank")["n"] == 2
    check("stress_D_AB_rate", "letter_rank", 1, 2)
    # R1 on 8 negatives: A = {}; A∪B = {neg1, neg8}
    check("fpr_A", "R1", 0, 8)
    check("fpr_AB", "R1", 2, 8)
    check("recall_A", "R1", 1, 4)
    check("recall_AB", "R1", 1, 4)
    assert s("stress_D_AB_count", "R1")["value"] == 0
    # R2: A = {neg1}; A∪B = {neg1, neg7, neg8}
    check("fpr_A", "R2", 1, 8)
    check("fpr_AB", "R2", 3, 8)
    check("recall_A", "R2", 1, 4)
    check("recall_AB", "R2", 2, 4)
    assert s("stress_D_AB_count", "R2")["value"] == 0
    # P2: upper CIs on n = 8 / 9 never clear 2.5 % / 7.5 %; the verdicts follow the CI arithmetic
    for rule in T.RULES:
        fa, fab = s("fpr_A", rule), s("fpr_AB", rule)
        assert s("P2_A", rule)["value"] == float(fa["ci_hi"] <= 0.025)
        assert s("P2_AB", rule)["value"] == float(fab["ci_hi"] <= 0.075)
        assert s("P2", rule)["value"] == 0.0
    # letter distribution rows and table
    assert s("letter_dist_negative_A", "letter_rank")["value"] == 1 and s("letter_dist_negative_A", "letter_rank")["n"] == 9
    assert s("letter_dist_negative_D", "R1")["value"] == 3 and s("letter_dist_negative_D", "R1")["n"] == 8
    dist = T.letter_distribution(res.tables["R1"])       # neg1 B, neg8 B; neg3/5/7 C; neg2/4/6 D; neg9 excluded
    assert dist.loc["negative", ["A", "B", "C", "D", "None", "excluded", "n"]].tolist() == [0, 2, 3, 3, 0, 1, 9]
    assert dist.loc["cowls", ["A", "B", "C", "D"]].tolist() == [1, 0, 1, 0]
    assert dist.loc["stress_D", ["C", "D"]].tolist() == [1, 1]
    # rebuild parity: every parquet reproduces its stored S / S_arb / letters
    assert set(res.parity) == {"a2", "a1"}
    for key, tab in res.parity.items():
        assert list(tab["col"]) == list(T.REBUILD_COLS) and int(tab["n_mismatch"].sum()) == 0, key
    assert s("rebuild_mismatch_S_arb", "R1")["value"] == 0 and s("rebuild_mismatch_S_arb", "R1")["n"] == 15
    assert s("rebuild_mismatch_letter_arb", "letter_rank")["n"] == 15
    # selection: recall_AB(R1) = .25 < .5 x .75 => R2
    assert res.selected.rule == "R2" and res.selected.provisional
    assert res.selected.thresholds_sha16 == thr["thresholds_sha16"]
    assert s("selected_rule_is_R2", "selection")["value"] == 1.0 and math.isclose(s("selection_bar", "selection")["value"], 0.375)
    num = res.selected.numbers
    assert math.isclose(num["recall_AB_letter_rank"], 0.75) and math.isclose(num["recall_AB_R1"], 0.25)
    assert num["n_excluded_nan_a1"] == 1 and num["n_excluded_nan_a2"] == 0 and num["letter_rank_source"] == "votes"
    assert num["fpr_AB_R1_ci"] == list(_cp(2, 8)) and num["stress_D_AB_count_R2"] == 0
    assert set(num["inputs"]) == {"a2", "a1", "a2_votes", "a1_votes"}
    # the csv frame has exactly the six columns and one row per (statistic, rule)
    df = T.rows_frame(rows)
    assert list(df.columns) == list(T.CSV_COLS) and not df.duplicated(["statistic", "rule"]).any()
    # the markdown carries the tables and the PROVISIONAL banner
    md = T.render_md(res)
    assert "PROVISIONAL" in md and "Selected rule: R2" in md and "morphology:spiral_arm" in md


# ------------------------------------------------------------------ 4: the selection rule, both branches
def test_select_rule_pure_and_end_to_end(tmp_path):
    assert T.select_rule(0.75, 0.25)["rule"] == "R2"
    assert T.select_rule(0.75, 0.375)["rule"] == "R1"            # equality keeps R1 (strictly below demotes)
    assert T.select_rule(0.75, 0.40)["rule"] == "R1"
    assert T.select_rule(0.0, 0.0)["rule"] == "R1"
    out = T.select_rule(0.5, 0.2, fraction=0.3)
    assert out["rule"] == "R1" and math.isclose(out["numbers"]["bar"], 0.15)
    assert T.select_rule(float("nan"), 0.2)["rule"] == "R1" and "undefined" in T.select_rule(float("nan"), 0.2)["reason"]
    # end to end under the calibrated table (t_A .5, t_B .3): rank A∪B = 4/4, R1 = {pos1 A, pos2 .49 B, pos3 .4 B} = 3/4
    a2, a1, _tpath, thr = _runs(tmp_path, CAL_TABLE)
    assert not thr["provisional"]
    res = T.run(a2, a1, thr)
    s = lambda stat, rule: T.stat(res.rows, stat, rule)  # noqa: E731
    assert math.isclose(s("recall_AB", "letter_rank")["value"], 1.0) and math.isclose(s("recall_AB", "R1")["value"], 0.75)
    assert res.selected.rule == "R1" and not res.selected.provisional
    assert res.selected.thresholds_sha16 == thr["thresholds_sha16"] != "a40ae6e201a03e65"
    rank = res.tables["letter_rank"].set_index("name")["letter"]
    assert rank["pos2"] == "A" and rank["neg8"] == "B" and rank["neg3"] == "B"     # .3 ≥ t_B .3
    r1 = res.tables["R1"].set_index("name")["letter"]
    assert r1["pos2"] == "B" and r1["neg7"] == "B" and r1["neg2"] == "D"
    for key, tab in res.parity.items():
        assert int(tab["n_mismatch"].sum()) == 0, key


# ------------------------------------------------------------------ 5: the crit-column fallback
def test_letter_rank_fallback_from_crit_columns(tmp_path):
    a2, a1, _tpath, thr = _runs(tmp_path)
    with_votes = T.run(a2, a1, thr).tables["letter_rank"].set_index("name")["letter"]
    R.votes_path_for(a2).unlink()
    res = T.run(a2, a1, thr)
    assert res.letter_rank_source == T.SOURCE_CRIT_COLUMNS and "a2" not in res.parity and "a2_votes" not in res.inputs
    fallback = res.tables["letter_rank"].set_index("name")["letter"]
    assert fallback.to_dict() == with_votes.to_dict()          # exact here: stored letters were C/D where n_items == 0
    assert T.stat(res.rows, "letter_rank_from_votes", "letter_rank")["value"] == 0.0
    assert res.selected.numbers["letter_rank_source"] == T.SOURCE_CRIT_COLUMNS
    assert "crit_*" in T.render_md(res)
    # the fallback advocate: criteria + placeholder items + recovered nothing_because
    row = pd.read_parquet(a2).set_index("name").loc["neg4"]
    adv = T.advocate_from_row(row)
    assert adv["items"] == [] and adv["nothing_because"] and adv["criteria"]["curvature"] == 8
    assert T.advocate_from_row({"p_evidence": None}) is None
    # R1 / R2 need records: the fallback refuses them; a1 without votes refuses
    with pytest.raises(ValueError):
        T.letter_table(pd.read_parquet(a1), None, thr, "R1")
    with pytest.raises(ValueError):
        T.letter_table(pd.read_parquet(a1), {}, thr, "R3")
    R.votes_path_for(a1).unlink()
    with pytest.raises(FileNotFoundError):
        T.run(a2, a1, thr)


# ------------------------------------------------------------------ 6: the CLI
def test_cli_writes_pinned_outputs_and_guards(tmp_path, capsys):
    a2, a1, tpath, thr = _runs(tmp_path)
    out = tmp_path / "transfer_dev"
    args = ["--thresholds", str(tpath), "--model-key", "opus5_api", "--a2", str(a2), "--a1", str(a1),
            "--out-dir", str(out)]
    # --rule-select refuses provisional thresholds before writing anything
    assert T.main(args + ["--rule-select"]) == 2 and not out.exists()
    assert "PROVISIONAL" in capsys.readouterr().out
    assert T.main(args) == 0
    printed = capsys.readouterr().out
    assert "PROVISIONAL" in printed and "selected rule: R2 [PROVISIONAL" in printed
    md, csv, js = out / T.MD_NAME, out / T.CSV_NAME, out / T.JSON_NAME
    assert md.exists() and csv.exists() and js.exists() and (out / (T.CSV_NAME + ".sha")).exists()
    df = _util.read_pinned(csv)                                    # sha sidecar verifies
    assert list(df.columns) == list(T.CSV_COLS)
    assert float(df[(df.statistic == "fpr_AB") & (df.rule == "R1")]["value"].iloc[0]) == 0.25
    assert float(df[(df.statistic == "provisional") & (df.rule == "thresholds")]["value"].iloc[0]) == 1.0
    sel = json.loads(js.read_text())
    assert list(sel) == ["rule", "reason", "numbers", "thresholds_sha16", "provisional"]
    assert sel["rule"] == "R2" and sel["provisional"] is True and sel["thresholds_sha16"] == thr["thresholds_sha16"]
    assert "PROVISIONAL THRESHOLDS" in md.read_text()
    # existing outputs refuse without --overwrite
    with pytest.raises(FileExistsError):
        T.main(args)
    assert T.main(args + ["--overwrite"]) == 0
    capsys.readouterr()
    # calibrated table: --rule-select runs, nothing is provisional
    tpath.write_text(json.dumps(CAL_TABLE, indent=2) + "\n")
    out2 = tmp_path / "transfer_cal"
    assert T.main(args[:-1] + [str(out2), "--rule-select"]) == 0
    sel2 = json.loads((out2 / T.JSON_NAME).read_text())
    assert sel2["rule"] == "R1" and sel2["provisional"] is False
    assert "PROVISIONAL" not in (out2 / T.MD_NAME).read_text()
    assert "PROVISIONAL" not in capsys.readouterr().out


# ------------------------------------------------------------------ 7: the real holdout, read-only
@pytest.mark.skipif(not (REAL_A2.exists() and REAL_A1.exists() and R.votes_path_for(REAL_A1).exists()),
                    reason="real opus5 holdout parquets not on this machine")
def test_real_holdout_read_only(tmp_path):
    thr = T.load_thresholds(REAL_THRESHOLDS, "opus5_api")
    before = {p: p.stat().st_mtime_ns for p in (REAL_A2, REAL_A1, R.votes_path_for(REAL_A2), R.votes_path_for(REAL_A1))
              if p.exists()}
    res = T.run(REAL_A2, REAL_A1, thr)
    s = lambda stat, rule: T.stat(res.rows, stat, rule)  # noqa: E731
    assert s("n_rows", "letter_rank")["value"] == s("n_rows", "R1")["value"]
    assert s("n_unlettered", "R1")["value"] == 0 and s("n_unlettered", "letter_rank")["value"] == 0
    assert s("n_scored", "R1")["value"] + s("n_excluded_nan", "R1")["value"] == s("n_rows", "R1")["value"]
    assert s("n_neg", "R1")["value"] <= s("n_neg", "letter_rank")["value"]
    for key, tab in res.parity.items():
        assert int(tab["n_mismatch"].sum()) == 0, (key, tab)
    for rule in T.RULES:
        r = s("fpr_AB", rule)
        assert 0.0 <= r["ci_lo"] <= r["value"] <= r["ci_hi"] <= 1.0
    assert res.selected.rule in aggregate_v2.DEPLOY_RULES
    # the CLI into a tmp directory; the registered parquets are untouched
    assert T.main(["--thresholds", str(REAL_THRESHOLDS), "--a2", str(REAL_A2), "--a1", str(REAL_A1),
                   "--out-dir", str(tmp_path / "real")]) == 0
    assert (tmp_path / "real" / T.JSON_NAME).exists()
    assert {p: p.stat().st_mtime_ns for p in before} == before
