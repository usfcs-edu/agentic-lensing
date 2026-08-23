#!/usr/bin/env python3
"""No-network tests for golden/{stats,drift_report,split_halves,registry}.py (work package D).

Synthetic frame / labels / grades only (no API, no files outside a scratch dir). Runs under
pytest or directly:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_stats.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util, analyze_golden, drift_report, registry, split_halves, stats  # noqa: E402

STRATA = ["T_verified", "T_U", "K_cowls", "L_known", "D_refuted", "U_tail", "N_unflagged"]
N_BOOT = 150   # tests only; the scripts default to 2,000


# ------------------------------------------------------------------ synthetic fixtures
def make_frame(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strata = rng.choice(STRATA, size=n, p=[.09, .30, .12, .12, .16, .12, .09])
    ra = 270.0 + rng.uniform(0, 0.5, n)          # ~30' field, units well separated
    dec = 23.0 + rng.uniform(0, 0.5, n)
    sid = np.arange(n)
    # two duplicate systems: units 5/6 and 40/41 share pixels (8.8" apart, same system_id)
    for a, b in ((5, 6), (40, 41)):
        ra[b] = ra[a] + 8.8 / 3600 / np.cos(np.deg2rad(dec[a])); dec[b] = dec[a]; sid[b] = sid[a]
    prior = np.zeros(n, int)
    tv = np.where(strata == "T_verified")[0]
    prior[tv[:6]] = 2                            # "ranks 1-15" analogue: forced into align
    prior[tv[6:]] = 1
    pc = np.where(strata == "T_verified", rng.choice(list("ABC"), n),
                  np.where(strata == "D_refuted", "D", np.where(np.isin(strata, ["T_U", "U_tail"]), "U", "")))
    conf = np.where(strata == "N_unflagged", np.nan, rng.uniform(0.3, 1.0, n))
    return pd.DataFrame({
        "unit_id": [f"u{i:04d}" for i in range(n)], "candidate_id": [f"J{i:08d}" for i in range(n)],
        "system_id": sid, "ra_deg": ra, "dec_deg": dec, "stratum": strata,
        "pipe_grade_passcount": pc, "pipe_inspector_conf": conf, "prior_exposure": prior,
        "lit_known": np.isin(strata, ["K_cowls", "L_known"]),
        "layout": rng.choice(["color", "gray_sw_only"], n, p=[.9, .1]),
        "desi_pool_overlap": rng.random(n) < 0.2,
    })


def make_labels(frame: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # lens-ish strata score high; score correlates with inspector conf so the AUC is > 0.5
    base = frame["stratum"].map({"T_verified": 3.4, "T_U": 2.3, "K_cowls": 3.3, "L_known": 3.0,
                                 "D_refuted": 1.4, "U_tail": 1.8, "N_unflagged": 1.3}).to_numpy()
    conf = frame["pipe_inspector_conf"].fillna(0.5).to_numpy()
    s = np.clip(np.round(base + 2.0 * (conf - 0.6) + rng.normal(0, 0.6, len(frame))), 1, 4).astype(int)
    c = rng.choice(["L", "M", "H"], len(frame), p=[.2, .4, .4])
    lab = pd.DataFrame({
        "unit_id": frame["unit_id"], "candidate_id": frame["candidate_id"],
        "ra_deg": frame["ra_deg"], "dec_deg": frame["dec_deg"], "stratum": frame["stratum"],
        "score_1_4": s, "grade_letter": [_util.score_to_letter(x) for x in s], "confidence_lmh": c,
        "confidence01": [_util.CONF_TO_01[x] for x in c], "pass2_score_1_4": np.nan,
        "pass2_confidence_lmh": "", "n_passes": 1, "label_stable": np.nan,
        "render_sha": [_util.sha_text(u) for u in frame["unit_id"]],
        "grade_scale": _util.GRADE_SCALE, "grader_id": "xh",
    })
    return lab


def make_grades(labels: pd.DataFrame, n_repeat: int = 60, p_exact: float = 0.73,
                seed: int = 13, mean_seconds: float = 40.0) -> pd.DataFrame:
    """One row per (unit, pass): pass 1 for all, pass 2 for n_repeat units with a known
    exact-agreement rate (the rest move one step)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    order = rng.permutation(n)
    rows = []
    for k, i in enumerate(order):
        r = labels.iloc[i]
        rows.append(dict(unit_id=r["unit_id"], candidate_id=r["candidate_id"], kit_id="kit01",
                         item_id=f"{k:03d}", pass_=1, session_id=f"s{1 + k // 50}",
                         presentation_index=k, score_1_4=int(r["score_1_4"]),
                         grade_letter=r["grade_letter"], confidence_lmh=r["confidence_lmh"],
                         confidence01=r["confidence01"],
                         seconds=float(rng.gamma(4, mean_seconds / 4)), revision_count=1, flag=False,
                         render_sha=r["render_sha"], manifest_sha="m", render_version="jwst_v1",
                         grade_scale=_util.GRADE_SCALE, timestamp="2026-09-01T00:00:00Z", grader_id="xh"))
    rep = labels.iloc[order[:n_repeat]]
    n_same = int(round(p_exact * n_repeat))
    same = np.r_[np.ones(n_same, bool), np.zeros(n_repeat - n_same, bool)]
    for k, (_, r) in enumerate(rep.iterrows()):
        s1 = int(r["score_1_4"])
        s2 = s1 if same[k] else int(np.clip(s1 + (1 if s1 < 4 else -1), 1, 4))
        rows.append(dict(unit_id=r["unit_id"], candidate_id=r["candidate_id"], kit_id="kit01",
                         item_id=f"{n + k:03d}", pass_=2, session_id=f"s{1 + (n + k) // 50}",
                         presentation_index=n + k, score_1_4=s2, grade_letter=_util.score_to_letter(s2),
                         confidence_lmh=r["confidence_lmh"] if same[k] else "M",
                         confidence01=r["confidence01"],
                         seconds=float(rng.gamma(4, mean_seconds / 4)), revision_count=1, flag=False,
                         render_sha=r["render_sha"], manifest_sha="m", render_version="jwst_v1",
                         grade_scale=_util.GRADE_SCALE, timestamp="2026-09-02T00:00:00Z", grader_id="xh"))
    g = pd.DataFrame(rows).rename(columns={"pass_": "pass"})
    return g


FRAME = make_frame()
LABELS = make_labels(FRAME)
GRADES = make_grades(LABELS)


# ----------------------------------------------------------------------------- stats
def test_pivot_pairs_and_intrarater_table():
    pairs = stats.pivot_pairs(GRADES, FRAME)
    assert len(pairs) == 60 and pairs["unit_id"].is_unique
    assert set(stats.SUBGROUPS) <= set(pairs.columns)
    t = stats.intrarater_table(pairs, n_boot=N_BOOT)
    assert list(t.columns) == ["statistic", "value", "ci_lo", "ci_hi"]
    by = t.set_index("statistic")
    # the known agreement rate is recovered exactly, by the reused and the added paths alike
    assert abs(by.loc["exact_score_agree", "value"] - 0.733) < 1e-6
    assert abs(by.loc[f"{stats.PAIR_LABEL} | exact_agree", "value"] - 0.733) < 1e-6
    assert by.loc["one_step", "value"] + by.loc["exact_score_agree", "value"] == 1.0   # all moves are 1 step
    # finite point estimates and CIs on the headline rows, CI brackets the value
    for k in ("sym_qwk_scores", "kripp_alpha_interval", "qwk_pass1_vs_pass2", "drift_mean_s2_minus_s1",
              "conf_exact_agree", "binary_ge3_exact_agree", "binary_ge3_kappa",
              f"{stats.PAIR_LABEL} | qwk"):
        r = by.loc[k]
        assert np.isfinite(r.value) and np.isfinite(r.ci_lo) and np.isfinite(r.ci_hi), k
        assert r.ci_lo - 1e-9 <= r.value <= r.ci_hi + 1e-9, (k, r.to_dict())
    # the letter QWK (human_ceiling.pair) equals the score QWK (affine-invariant weights)
    assert abs(by.loc[f"{stats.PAIR_LABEL} | qwk", "value"] - by.loc["qwk_pass1_vs_pass2", "value"]) < 2e-3
    assert 0 <= by.loc["drift_sign_test_p", "value"] <= 1
    assert "qwk_grader_vs_consensus" not in by.index          # degenerate for a self-pair: dropped
    assert "exact_agree_within_D" in by.index
    assert "p_one_grader_reject" not in by.index and "p_either_pass_score1" in by.index
    assert abs(by.loc["p_either_pass_score1", "value"] - float((np.minimum(pairs.s1, pairs.s2) == 1).mean())) < 1e-3
    # ipw rows exist with CIs; per-s1-level rows exist for every observed level
    for k in ("ipw_exact_score_agree", "ipw_drift_mean_s2_minus_s1", "ipw_within1_step"):
        r = by.loc[k]
        assert np.isfinite(r.value) and r.ci_lo - 1e-9 <= r.value <= r.ci_hi + 1e-9, k
    for lvl in sorted(pairs.s1.unique()):
        assert f"s1={lvl}/drift_mean_s2_minus_s1" in by.index and f"s1={lvl}/n_pairs" in by.index
    # layout / lit_known are no longer subgroups (constant on repeats by construction)
    assert not any(s.startswith(("layout=", "lit_known=")) for s in by.index)
    assert stats.SUBGROUPS == ("prior_exposure", "stratum")
    # subgroup rows exist for the big cells and are skipped under n < 8
    subs = [s for s in by.index if "/" in s and not s.startswith("s1=")]   # s1= rows are per-level, unfiltered
    assert any(s.startswith("stratum=T_U/") for s in subs)
    for s in subs:
        if s.endswith("/n_pairs"):
            assert by.loc[s, "value"] >= stats.MIN_SUBGROUP_N
    # deterministic under the seed
    t2 = stats.intrarater_table(pairs, n_boot=N_BOOT)
    pd.testing.assert_frame_equal(t, t2)


def test_sign_test_and_stats_frame():
    assert stats.sign_test_p(np.array([0, 0, 0])) == 1.0
    assert stats.sign_test_p(np.array([1, 1, 1, 1, 1, 1, 0])) < 0.05
    pairs = pd.DataFrame({"s1": [4, 1, 2], "s2": [3, 2, 2], "l1": ["A", "D", "C"]})
    ig = stats._as_intergrader_frame(pairs)
    assert (ig["s_lo"] <= ig["s_hi"]).all() and list(ig["Q"]) == ["A", "D", "C"]


def test_ipw_removes_regression_to_the_mean():
    """A stationary rater (latent truth + symmetric +-1 noise on BOTH passes, so the two
    passes are exchangeable and the population drift is exactly 0) whose repeats are drawn
    BALANCED across observed pass-1 scores shows a spurious raw drift: negative when the
    queue is skewed low (a high observed s1 is often noise-inflated), positive when skewed
    high. The ipw rows recover ~0 either way (the review's simulation). Weights are mean-1."""
    rng = np.random.default_rng(2026)
    levels = np.array([1, 2, 3, 4])

    def observe(t, p_err=0.5):
        e = rng.choice([-1, 0, 1], size=len(t), p=[p_err / 2, 1 - p_err, p_err / 2])
        return np.clip(t + e, 1, 4)

    for marginal, sign in (((.40, .30, .18, .12), -1), ((.12, .18, .30, .40), +1)):
        raw, ipw = [], []
        for _ in range(80):
            truth = rng.choice(levels, size=400, p=marginal)
            s1, s2 = observe(truth), observe(truth)
            idx = np.concatenate([rng.choice(np.where(s1 == k)[0], 10, replace=False) for k in levels])
            pairs = pd.DataFrame({"s1": s1[idx], "s2": s2[idx], "c1": "H", "c2": "H"})
            pairs["w_ipw"] = stats.ipw_weights(s1, pairs["s1"].to_numpy())
            assert abs(pairs["w_ipw"].mean() - 1.0) < 1e-9
            st = stats._added_stats(pairs)
            raw.append(st["drift_mean_s2_minus_s1"]); ipw.append(st["ipw_drift_mean_s2_minus_s1"])
        raw_m, ipw_m = float(np.mean(raw)), float(np.mean(ipw))
        assert sign * raw_m > 0.05, (marginal, raw_m)        # the artefact is there and signed
        assert abs(ipw_m) < 0.04 < abs(raw_m), (marginal, raw_m, ipw_m)
    # a second kit in the grades file must be selected explicitly
    g2 = GRADES.copy(); g2["kit_id"] = "kit02"
    both = pd.concat([GRADES, g2], ignore_index=True)
    try:
        stats.pivot_pairs(both, FRAME); raise AssertionError("no raise")
    except ValueError as e:
        assert "pass kit_id" in str(e)
    assert len(stats.pivot_pairs(both, FRAME, kit_id="kit02")) == len(stats.pivot_pairs(GRADES, FRAME))


def test_agreement_table():
    t = stats.agreement_table(LABELS, FRAME, n_boot=N_BOOT)
    by = t.set_index("statistic")
    a = by.loc["auc_pipe_inspector_conf_vs_xh_ge3[flagged_only]"]
    assert 0.5 < a.value <= 1.0 and a.ci_lo <= a.value <= a.ci_hi
    assert by.loc["n_auc_pipe_inspector_conf_vs_xh_ge3[flagged_only]", "value"] == FRAME["pipe_inspector_conf"].notna().sum()
    # [all] charges the incumbent for its unflagged rows (confidence 0): every unit counts
    b = by.loc["auc_pipe_inspector_conf_vs_xh_ge3[all]"]
    assert by.loc["n_auc_pipe_inspector_conf_vs_xh_ge3[all]", "value"] == len(FRAME)
    assert 0.5 < b.value <= 1.0 and b.ci_lo <= b.value <= b.ci_hi
    assert "auc_pipe_inspector_conf_vs_xh_ge3" not in by.index          # no untagged headline
    tag = "[verified rows only; different semantics - not a QWK headline]"
    n_ver = int(FRAME["pipe_grade_passcount"].isin(list("ABCD")).sum())
    assert by.loc["n_verified_passcount" + tag, "value"] == n_ver
    xt = [s for s in by.index if s.startswith("xtab_verified[")]
    assert sum(by.loc[s, "value"] for s in xt) == n_ver
    assert "p_xh_ge3[pipe_grade_passcount=U]" in by.index          # U stays U, never coerced
    assert "p_xh_ge3[pipe_grade_passcount=none]" in by.index       # unflagged ('' / NaN) is its own level
    for st in ("K_cowls", "L_known"):
        r = by.loc[f"recall_xh_ge3[stratum={st}]"]
        assert 0 <= r.ci_lo <= r.value <= r.ci_hi <= 1


def test_analyze_golden_endpoints():
    """The registered endpoints on a synthetic validate half: purity@recall by hand, a
    strongly-informative E2 beats a weak E1 (dAUC > 0, CI above 0, DeLong p small), the
    incumbent is scored on flagged rows AND on all rows, per-replicate rows pair by k, and
    a manifest whose binary_label is not the score>=3 cut is refused."""
    rng = np.random.default_rng(5)
    n = 80
    score = rng.choice([1, 2, 3, 4], size=n, p=[.3, .3, .25, .15])
    y = (score >= 3).astype(int)
    names = [f"J{i:07d}+0000000" for i in range(n)]
    flagged = rng.random(n) < 0.7
    man = pd.DataFrame({"name": names, "score_1_4": score, "binary_label": np.where(y == 1, "lens", "nonlens"),
                        "split": "validate", "stratum": rng.choice(["T_U", "K_cowls", "D_refuted"], n),
                        "p_pipeline": np.where(flagged, np.clip(0.5 * y + rng.normal(0, .3, n), 0, 1), 0.0),
                        "pipe_flagged": flagged})
    tmp = Path(tempfile.mkdtemp(prefix="golden_an_"))
    try:
        _util.pin(man, tmp / "man.csv")
        def rep(arm, strength, k):
            p = np.clip(0.5 + strength * (y - 0.5) + rng.normal(0, .25, n), 0, 1)
            letter = np.where(p > .75, "A", np.where(p > .5, "B", np.where(p > .25, "C", "D")))
            df = pd.DataFrame({"name": names, "p_lens": p, "grade_pred": letter, "parse_ok": True})
            df.loc[0, "parse_ok"] = False; df.loc[0, "p_lens"] = np.nan     # one parse failure -> 0
            df.to_parquet(tmp / f"preds_golden_{arm}_x_validate_r{k}.parquet", index=False)
        for k in (1, 2):
            rep("e1", 0.2, k); rep("e2", 0.9, k)
        m = analyze_golden.load_manifest(tmp / "man.csv", "validate")
        arms = {a: analyze_golden.load_replicates([str(tmp / f"preds_golden_{a}_x_validate_r*.parquet")], "p_lens")
                for a in ("e1", "e2")}
        reps1 = arms["e1"]
        assert len(reps1) == 2 and reps1[0].loc[0, "score"] == 0.0 and not reps1[0].loc[0, "parse_ok"]
        res, extra = analyze_golden.analyze(m, arms, n_boot=150)
        by = res.set_index("statistic")
        assert list(res.columns) == ["statistic", "value", "ci_lo", "ci_hi"]
        d = by.loc["e2_minus_e1/dAUC_ge3[pooled]"]
        assert d.value > 0.1 and d.ci_lo > 0 and by.loc["e2_minus_e1/dAUC_ge3_delong_p[pooled]", "value"] < 0.01
        assert by.loc["e2/auc_ge3[pooled]", "value"] > by.loc["e1/auc_ge3[pooled]", "value"]
        assert "e2_minus_e1/dAUC_ge3[r1]" in by.index and "e2_minus_e1/dAUC_ge3[r2]" in by.index
        assert "e2_minus_e1/dPurity_at_recall0.8[pooled]" in by.index
        assert by.loc["e2_minus_e1/dPurity_at_recall0.8[pooled]", "value"] > 0
        assert by.loc["incumbent/n[flagged_only]", "value"] == int(flagged.sum())
        assert by.loc["incumbent/n[all]", "value"] == n
        assert "e1_minus_incumbent/dAUC_ge3[all]" in by.index
        assert 0 <= by.loc["e1/parse_rate", "value"] < 1 and np.isfinite(by.loc["e1/qwk_letter_vs_xh[pooled]", "value"])
        assert np.isfinite(by.loc["e2/qwk_self_consistency[mean over replicate pairs]", "value"])
        assert extra["e2_minus_e1"]["n"] == n
        # purity at recall by hand: scores rank positives 1,1,0,1,0,0 -> recall 0.8 of 3 pos
        # needs 3 of them -> top set of 4 -> purity 3/4
        assert abs(analyze_golden.purity_at_recall(np.array([1, 1, 0, 1, 0, 0]), np.array([.9, .8, .7, .6, .5, .4])) - 0.75) < 1e-9
        assert np.isnan(analyze_golden.purity_at_recall(np.zeros(3, int), np.ones(3)))
        # a manifest whose binary_label disagrees with the score>=3 cut is refused
        bad = man.copy(); bad["binary_label"] = np.where(score >= 2, "lens", "nonlens")
        _util.pin(bad, tmp / "bad.csv")
        try:
            analyze_golden.load_manifest(tmp / "bad.csv", "validate"); raise AssertionError("no raise")
        except AssertionError as e:
            assert "score>=3" in str(e)
    finally:
        shutil.rmtree(tmp)


def test_simulate_ci_width_documents_the_stratification_argument():
    bal = stats.simulate_ci_width(40, (0.25, 0.25, 0.25, 0.25), p_exact=0.73, n_rep=12, n_boot=150)
    skew = stats.simulate_ci_width(40, (0.70, 0.10, 0.10, 0.10), p_exact=0.73, n_rep=12, n_boot=150)
    print(f"  n=40 balanced marginal : QWK {bal['qwk_median']}  CI width {bal['qwk_ci_width']}  "
          f"exact-agree CI width {bal['exact_ci_width']}")
    print(f"  n=40 70%-one-level     : QWK {skew['qwk_median']}  CI width {skew['qwk_ci_width']}  "
          f"exact-agree CI width {skew['exact_ci_width']}")
    assert np.isfinite(bal["qwk_ci_width"]) and np.isfinite(skew["qwk_ci_width"])
    assert skew["qwk_ci_width"] > bal["qwk_ci_width"]           # why the repeats are stratified


# ----------------------------------------------------------------------------- drift
def test_drift_table_and_pilot_verdict():
    t = drift_report.drift_table(GRADES, n_boot=N_BOOT, queue_total=len(GRADES) + 100)
    by = t.set_index("statistic")
    assert by.loc["n_exposures", "value"] == len(GRADES) and by.loc["n_pass1", "value"] == len(LABELS)
    for k in ("spearman_score_vs_presentation_index[pass1]", "spearman_seconds_vs_presentation_index[all]",
              "q1/score_mean[pass1]", "q4/p_score1[pass1]", "session=s1/score_mean[pass1]"):
        r = by.loc[k]
        assert np.isfinite(r.value) and r.ci_lo <= r.value <= r.ci_hi, k
    assert by.loc["pilot_n", "value"] == 30
    assert by.loc["queue_remaining", "value"] == 100
    assert abs(by.loc["pilot_projected_hours_remaining", "value"]
               - by.loc["queue_remaining", "value"] * by.loc["pilot_seconds_mean", "value"] / 3600) < 2e-3
    assert "KEEP" in drift_report.pilot_verdict(40.0) and "DROP" in drift_report.pilot_verdict(90.0)
    assert "no verdict" in drift_report.pilot_verdict(np.nan)
    slow = make_grades(LABELS, mean_seconds=120.0)
    assert "DROP" in drift_report.pilot_summary(slow)["verdict"]
    assert "KEEP" in drift_report.pilot_summary(GRADES)["verdict"]


# ---------------------------------------------------------------------------- splits
def test_split_halves_invariants_and_determinism():
    sp = split_halves.assign_halves(LABELS, FRAME, seed=2026)
    assert list(sp.columns) == split_halves.SPLIT_COLS and len(sp) == len(LABELS)
    d = sp.merge(FRAME[["unit_id", "prior_exposure"]], on="unit_id").merge(
        LABELS[["unit_id", "score_1_4"]], on="unit_id")
    forced = d[d["prior_exposure"] == 2]
    assert len(forced) == 6 and (forced["split"] == "align").all() and forced["forced"].all()
    # forced is a SYSTEM-level flag: a dup partner of a forced unit is dragged along and flagged
    forced_sys = set(d.loc[d["prior_exposure"] == 2, "system_id"])
    assert set(d.loc[d["forced"], "system_id"]) == forced_sys
    assert (d.loc[d["forced"], "split"] == "align").all()
    for a, b in (("u0005", "u0006"), ("u0040", "u0041")):          # dup systems move together
        assert d.set_index("unit_id").loc[a, "split"] == d.set_index("unit_id").loc[b, "split"]
    ge3 = d[d["score_1_4"] >= 3].groupby("split").size().reindex(split_halves.HALVES, fill_value=0)
    assert abs(int(ge3["align"]) - int(ge3["validate"])) <= 1, ge3.to_dict()
    tot = d.groupby("split").size()
    assert abs(int(tot["align"]) - int(tot["validate"])) <= 8, tot.to_dict()   # forced block skews it a little
    # per-(stratum x letter) cells are balanced to within 1 outside the forced stratum
    cells = d[d["stratum"] != "T_verified"].groupby(["stratum", "grade_letter", "split"]).size().unstack(fill_value=0)
    cells = cells.reindex(columns=split_halves.HALVES, fill_value=0)
    assert (cells["align"] - cells["validate"]).abs().max() <= 2, cells
    # seed-reproducible; another seed draws differently
    pd.testing.assert_frame_equal(sp, split_halves.assign_halves(LABELS, FRAME, seed=2026))
    other = split_halves.assign_halves(LABELS, FRAME, seed=1)
    assert not sp["split"].equals(other["split"])


def test_split_firewall_reports_pool_overlap_without_excluding():
    sp = split_halves.assign_halves(LABELS, FRAME, seed=2026)
    tmp = Path(tempfile.mkdtemp(prefix="golden_fw_"))
    try:
        # a fake DESI pool: 5 golden positions (1" off) + 50 unrelated rows
        hit = LABELS.iloc[[0, 10, 20, 30, 50]]
        pool = pd.DataFrame({"name": [f"DESI-{i}" for i in range(55)],
                             "ra": np.r_[hit["ra_deg"].to_numpy() + 1 / 3600, 10 + np.arange(50) * 0.01],
                             "dec": np.r_[hit["dec_deg"].to_numpy(), np.full(50, -30.0)],
                             "split": ["train"] * 55, "label_source": ["random_neg"] * 55})
        pool.to_csv(tmp / "parity_train_pool.csv", index=False)
        ov = split_halves.firewall(sp, LABELS, pool_dir=tmp)
        assert sorted(ov["unit_id"]) == sorted(hit["unit_id"]) and (ov["sep_arcsec"] < 2).all()
        assert set(ov["pool_file"]) == {"parity_train_pool.csv"}
        assert len(sp) == len(LABELS)                               # reported, never excluded
        # a validate unit planted 1" from an align unit trips the between-halves assertion
        bad = LABELS.copy()
        a_unit = sp.loc[sp["split"] == "align", "unit_id"].iloc[0]
        v_unit = sp.loc[sp["split"] == "validate", "unit_id"].iloc[0]
        bad.loc[bad["unit_id"] == v_unit, ["ra_deg", "dec_deg"]] = \
            bad.loc[bad["unit_id"] == a_unit, ["ra_deg", "dec_deg"]].to_numpy() + np.array([[0, 1 / 3600]])
        try:
            split_halves.firewall(sp, bad, pool_dir=tmp)
            raise SystemExit("between-halves firewall did not fire")
        except AssertionError as e:
            assert "leak audit FAILED" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -------------------------------------------------------------------------- registry
def test_registry_sync_mark_assert():
    tmp = Path(tempfile.mkdtemp(prefix="golden_reg_"))
    try:
        path = tmp / "golden_registry.csv"
        sp = split_halves.assign_halves(LABELS, FRAME, seed=2026)
        reg = registry.sync_from(LABELS, sp, frame=FRAME, grades=GRADES, path=path)
        assert list(reg.columns) == registry.COLS and len(reg) == len(LABELS)
        assert (path.with_suffix(".csv.sha")).exists()
        assert set(reg["split"]) == {"align", "validate"} and (reg["kit_ids"] == "kit01").all()
        assert set(reg["leak"]) == {"desi_train", "no"}
        assert (reg["leak"] == "desi_train").sum() == int(FRAME["desi_pool_overlap"].sum())
        assert (reg["render_version"] == "jwst_v1").all() and (reg[list(registry.EXPOSURE_COLS)] == "").all().all()
        sha1 = path.with_suffix(".csv.sha").read_text()
        reg2 = registry.sync_from(LABELS, sp, frame=FRAME, grades=GRADES, path=path)   # idempotent
        pd.testing.assert_frame_equal(reg, reg2)
        assert path.with_suffix(".csv.sha").read_text() == sha1

        align = reg.loc[reg["split"] == "align", "unit_id"].tolist()[:3]
        val = reg.loc[reg["split"] == "validate", "unit_id"].tolist()[:3]
        registry.assert_unexposed(align + val, path=path)                        # clean ledger
        registry.mark_exposed(align, "e2_sonnet_r1", "fewshot", path=path)
        registry.mark_exposed(align, "e2_sonnet_r1", "fewshot", path=path)       # no duplicate tags
        registry.mark_exposed(align[:1], "e4_sft_v1", "sft", path=path)
        registry.mark_exposed(val, "e1_sonnet_r1", "eval", path=path)            # eval never blocks
        r = registry.load(path).set_index("unit_id")
        assert r.loc[align[0], "in_fewshot"] == "e2_sonnet_r1" and r.loc[align[0], "in_sft"] == "e4_sft_v1"
        assert r.loc[val[0], "exposed_runs"] == "e1_sonnet_r1"
        registry.assert_unexposed(val, path=path)
        try:
            registry.assert_unexposed(val + align, path=path)
            raise SystemExit("assert_unexposed did not raise")
        except registry.ExposureError as e:
            assert all(u in str(e) for u in align) and "e2_sonnet_r1" in str(e)
        try:
            registry.assert_unexposed(["nope"], path=path)
            raise SystemExit("unknown unit did not raise")
        except KeyError:
            pass
        # a re-sync keeps the exposure ledger
        reg3 = registry.sync_from(LABELS, sp, frame=FRAME, grades=GRADES, path=path).set_index("unit_id")
        assert reg3.loc[align[0], "in_fewshot"] == "e2_sonnet_r1"
        try:
            registry.mark_exposed(align, "x", "bogus", path=path)
            raise SystemExit("bad kind did not raise")
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        print(f"{name} ...", flush=True)
        fn()
        print(f"  PASS {name}")
    print(f"\nPASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
