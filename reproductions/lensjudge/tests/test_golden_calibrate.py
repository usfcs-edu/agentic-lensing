#!/usr/bin/env python3
"""No-network tests for golden/calibrate_thresholds.py (REGISTRY "Deployment rule v2-deploy",
item 1): the "smallest negative S with FPR <= target" threshold on a known score distribution
(ties, +inf, NaN), agreement of the achieved FPR with analyze_truth.threshold_at_fpr and the
documented difference when a positive sits in the gap, the non-anchor filter, refusal below
--min-neg and on non-design rows, --write on a tmp copy of the real thresholds_v2.json (other
keys + order preserved byte-for-byte, archive naming _2/_3, calibration record, the
thresholds_sha16 that run_truth_eval would carry), and a read-only smoke on the real holdout
a2 parquet (skipped when absent; never --write).

    cd reproductions/lensjudge && ~/.venvs/lensjudge/bin/python -m pytest tests/test_golden_calibrate.py -q
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from lensjudge.golden import _util, aggregate_v2, analyze_truth  # noqa: E402
from lensjudge.golden import calibrate_thresholds as ct  # noqa: E402
from lensjudge.golden import run_truth_eval as rte  # noqa: E402

REAL_THRESHOLDS = _util.HERE / "thresholds_v2.json"
HOLDOUT_A2 = _util.LENSJUDGE / "outputs" / "preds_truth_a2_opus5_holdout_k1_r1.parquet"
SCRIPT = _util.HERE / "calibrate_thresholds.py"


# ------------------------------------------------------------------ synthetic frames
def _frame(neg_scores, pos_scores=(), anchor_neg=(), anchor_pos=(), half="design", model="opus5",
           arm="a2", tau0=0.15, stress_scores=()) -> pd.DataFrame:
    """A preds-like frame with the columns calibrate_thresholds reads. Anchors are extra rows
    with is_anchor=True (negative-class or positive)."""
    rows = []
    for s in neg_scores:
        rows.append(dict(truth_class="negative", is_positive=False, is_anchor=False, S=s))
    for s in pos_scores:
        rows.append(dict(truth_class="cowls", is_positive=True, is_anchor=False, S=s))
    for s in anchor_neg:
        rows.append(dict(truth_class="negative", is_positive=False, is_anchor=True, S=s))
    for s in anchor_pos:
        rows.append(dict(truth_class="cowls", is_positive=True, is_anchor=True, S=s))
    for s in stress_scores:
        rows.append(dict(truth_class="stress_D", is_positive=False, is_anchor=False, S=s))
    df = pd.DataFrame(rows)
    df["S"] = df["S"].astype(float)
    df["half"], df["model"], df["arm"], df["tau0"] = half, model, arm, float(tau0)
    df["run_tag"], df["thinking"], df["effort"] = f"truth_{arm}_{model}_{half}_k1_r1", "adaptive", "xhigh"
    df["thresholds_sha16"] = "a40ae6e201a03e65"
    return df


# 200 distinct negatives: i/1000 for i = 1..200 -> 2nd-highest 0.199, 10th-highest 0.191
NEG200 = [i / 1000 for i in range(1, 201)]


def _preds(path: Path, df: pd.DataFrame, meta: bool = True) -> Path:
    """Write a preds parquet and (by default) the `.meta.json` run_golden_eval writes on
    completion — the file the --write completeness blocker looks for."""
    df.to_parquet(path)
    mp = path.with_name(path.stem + ".meta.json")
    if meta:
        mp.write_text(json.dumps({"tuple": {"arm": "a2", "model": "opus5"}, "n": int(len(df))}))
    elif mp.exists():
        mp.unlink()
    return path


def _tmp_thresholds(tmp_path: Path, table=None) -> Path:
    p = tmp_path / "thresholds_v2.json"
    if table is None:
        shutil.copyfile(REAL_THRESHOLDS, p)
    else:
        p.write_text(ct.render_table(table))
    return p


# ------------------------------------------------------------------ threshold semantics
def test_threshold_distinct_scores_exact():
    t, fpr, n = ct.threshold_at_fpr_neg(NEG200, 0.01)
    assert (t, n) == (0.199, 2) and abs(fpr - 0.01) < 1e-12
    t, fpr, n = ct.threshold_at_fpr_neg(NEG200, 0.05)
    assert (t, n) == (0.191, 10) and abs(fpr - 0.05) < 1e-12
    # order does not matter, NaNs are dropped
    rng = np.random.default_rng(0)
    shuffled = list(rng.permutation(NEG200)) + [float("nan")] * 3
    assert ct.threshold_at_fpr_neg(shuffled, 0.05)[:2] == (0.191, 0.05)


def test_threshold_ties_push_up_and_inf():
    # top three tied at 0.9: count(>=0.9) = 3 > 2 allowed -> nothing qualifies -> +inf
    neg = [0.9, 0.9, 0.9] + [i / 1000 for i in range(1, 198)]
    t, fpr, n = ct.threshold_at_fpr_neg(neg, 0.01)
    assert math.isinf(t) and fpr == 0.0 and n == 0
    # 0.95 then a tie at 0.9: count(>=0.9) = 3 -> t_A = 0.95 (FPR 0.5 %)
    neg = [0.95, 0.9, 0.9] + [i / 1000 for i in range(1, 198)]
    t, fpr, n = ct.threshold_at_fpr_neg(neg, 0.01)
    assert (t, n) == (0.95, 1) and abs(fpr - 0.005) < 1e-12
    # tie exactly at the allowed count is fine: 0.9, 0.9 then lower -> t_A = 0.9, 2/200
    neg = [0.9, 0.9] + [i / 1000 for i in range(1, 199)]
    assert ct.threshold_at_fpr_neg(neg, 0.01)[0] == 0.9
    # fewer negatives than 1/target: 50 negatives at 1 % allow 0 -> +inf; at 5 % allow 2
    neg50 = [i / 100 for i in range(1, 51)]
    assert math.isinf(ct.threshold_at_fpr_neg(neg50, 0.01)[0])
    assert ct.threshold_at_fpr_neg(neg50, 0.05) == (0.49, 0.04, 2)
    with pytest.raises(ValueError):
        ct.threshold_at_fpr_neg([float("nan")], 0.05)


def test_matches_analyze_truth_fpr_and_documents_the_gap_difference():
    """analyze_truth.threshold_at_fpr takes candidate thresholds from the union of both
    classes: same achieved FPR, but a positive in the gap below the k-th negative becomes the
    threshold there — never here."""
    pos_outside = [0.5, 0.6, 0.05, 0.001]                       # none between 0.191 and 0.192
    y = np.array([0] * 200 + [1] * len(pos_outside))
    s = np.array(NEG200 + pos_outside)
    rec, thr, fpr = analyze_truth.threshold_at_fpr(y, s, 0.05)
    t, fpr_ours, _ = ct.threshold_at_fpr_neg(NEG200, 0.05)
    assert thr == t == 0.191 and abs(fpr - fpr_ours) < 1e-12
    pos_in_gap = [0.1905, 0.6]                                  # between the 10th (0.191) and 11th (0.190) negative
    y = np.array([0] * 200 + [1] * 2)
    s = np.array(NEG200 + pos_in_gap)
    rec, thr, fpr = analyze_truth.threshold_at_fpr(y, s, 0.05)
    assert thr == 0.1905 and abs(fpr - 0.05) < 1e-12          # sklearn picks the positive's score
    assert ct.threshold_at_fpr_neg(NEG200, 0.05)[0] == 0.191    # we stay on the negatives


# ------------------------------------------------------------------ calibrate() on frames
def test_calibrate_filters_anchors_nan_and_reports_recall():
    df = _frame(NEG200 + [float("nan")] * 2, pos_scores=[0.5, 0.195, 0.192, 0.1, float("nan")],
                anchor_neg=[0.99, 0.98], anchor_pos=[0.97], stress_scores=[0.95])
    fit = ct.calibrate(df, 0.01, 0.05)
    assert fit["n_neg"] == 200 and fit["n_neg_nan"] == 2 and fit["n_anchor_excluded"] == 3
    assert fit["n_pos"] == 4 and fit["n_pos_nan"] == 1
    assert fit["t_A"] == 0.199 and fit["t_B"] == 0.191            # anchors at 0.99/0.98 ignored
    assert fit["n_neg_ge_tA"] == 2 and fit["n_neg_ge_tB"] == 10
    assert fit["n_pos_ge_tA"] == 1 and abs(fit["recall_A"] - 0.25) < 1e-12
    assert fit["n_pos_ge_tB"] == 3 and abs(fit["recall_B"] - 0.75) < 1e-12
    # with the anchors counted as negatives the thresholds would have moved up
    df2 = df.copy()
    df2["is_anchor"] = False
    assert ct.calibrate(df2, 0.01, 0.05)["t_A"] == 0.98         # 0.99, 0.98 are now the top two of 202


def test_calibrate_no_positives_and_missing_columns():
    fit = ct.calibrate(_frame(NEG200), 0.01, 0.05)
    assert fit["n_pos"] == 0 and fit["recall_A"] is None and fit["recall_B"] is None
    with pytest.raises(KeyError):
        ct.calibrate(_frame(NEG200).drop(columns=["is_anchor"]), 0.01, 0.05)
    with pytest.raises(ValueError):
        ct.calibrate(_frame([float("nan")] * 5, pos_scores=[0.5]), 0.01, 0.05)


# ------------------------------------------------------------------ table editing
def test_updated_table_order_and_render():
    table = json.loads(REAL_THRESHOLDS.read_text())
    assert ct.render_table(table) == REAL_THRESHOLDS.read_text()     # the real file is canonical
    entry = {"tau0": 0.15, "t_A": 0.2, "t_B": 0.1}
    new = ct.updated_table(table, "opus5_api", entry)
    keys = list(new)
    assert keys.index("opus5_api") == keys.index("sonnet_api") + 1
    assert [k for k in keys if k != "opus5_api"] == list(table)
    assert all(new[k] == table[k] for k in table) and new["opus5_api"] == entry
    assert table.get("opus5_api", "absent") == "absent"               # input untouched
    # present key (null or dict) is replaced in place
    t2 = {"a": 1, "opus5_api": None, "sonnet_api": {"t_A": 1}, "z": 2}
    assert list(ct.updated_table(t2, "opus5_api", entry)) == ["a", "opus5_api", "sonnet_api", "z"]
    assert ct.updated_table(t2, "opus5_api", entry)["opus5_api"] == entry
    # no sonnet_api: appended at the end
    assert list(ct.updated_table({"a": 1}, "opus5_api", entry)) == ["a", "opus5_api"]


def test_archive_path_suffixes(tmp_path):
    p = ct.archive_path(tmp_path, "opus5_api")
    assert p == tmp_path / "thresholds_v2.pre_opus5.json"
    p.write_text("x")
    p2 = ct.archive_path(tmp_path, "opus5_api")
    assert p2 == tmp_path / "thresholds_v2.pre_opus5_2.json"
    p2.write_text("x")
    assert ct.archive_path(tmp_path, "opus5_api") == tmp_path / "thresholds_v2.pre_opus5_3.json"
    assert ct.archive_path(tmp_path, "opus_claude_code") == tmp_path / "thresholds_v2.pre_opus_claude_code.json"


# ------------------------------------------------------------------ run() end to end
def test_run_report_only_writes_nothing(tmp_path):
    preds = _preds(tmp_path / "preds.parquet", _frame(NEG200, pos_scores=[0.5, 0.05]))
    thr = _tmp_thresholds(tmp_path)
    before = thr.read_text()
    res = ct.run(preds, "opus5_api", thr, archive_dir=tmp_path / "arch")
    assert not res.written and res.write_blockers == [] and res.archive_path is None
    assert res.meta_present and res.n_neg_expected == 200 and not res.allow_nan_neg
    assert thr.read_text() == before and not (tmp_path / "arch").exists()
    assert (res.t_A, res.t_B, res.n_neg, res.n_pos) == (0.199, 0.191, 200, 2)
    assert res.key_present is False and res.existing_entry is None
    assert res.letter_source_before == "provisional" and res.letter_source_after == "opus5_api_calibrated"
    assert res.file_sha16_before == _util.sha_text(before)
    assert res.tuple_sha16_before == rte.thresholds_sha(rte.load_thresholds(thr, "opus5_api"))
    assert res.tuple_sha16_after == _util.sha_json({"tau0": 0.15, "t_A": 0.199, "t_B": 0.191,
                                                    "letter_source": "opus5_api_calibrated",
                                                    "thresholds_key": "opus5_api"})
    assert res.run_tag == "truth_a2_opus5_design_k1_r1" and res.effort == "xhigh"
    ct.summary(res)                                                 # renders without error
    json.dumps(res.model_dump())


def test_run_write_preserves_other_keys_and_archives(tmp_path):
    preds = _preds(tmp_path / "preds.parquet", _frame(NEG200, pos_scores=[0.5, 0.195]))
    thr = _tmp_thresholds(tmp_path)
    original = thr.read_text()
    arch = tmp_path / "outputs"
    res = ct.run(preds, "opus5_api", thr, write=True, archive_dir=arch)
    assert res.written and res.write_blockers == []
    # archive is a byte copy of the previous file, named per the protocol
    assert res.archive_path == str(arch / "thresholds_v2.pre_opus5.json")
    assert Path(res.archive_path).read_text() == original
    # the new file: canonical form, the new key right after sonnet_api, every other key intact
    text = thr.read_text()
    new = json.loads(text)
    old = json.loads(original)
    assert text == ct.render_table(new) and text.endswith("}\n")
    assert list(new) == ["sonnet_api", "opus5_api", "opus_claude_code", "provisional"]
    assert {k: v for k, v in new.items() if k != "opus5_api"} == old
    assert new["opus5_api"] == {"tau0": 0.15, "t_A": 0.199, "t_B": 0.191}
    # untouched parts byte-for-byte: the original text minus its first line is a suffix of the
    # new text, and the prefix up to (and including) the sonnet_api block is identical
    cut = original.index('  "opus_claude_code"')
    assert text.endswith(original[cut:]) and text.startswith(original[:cut])
    assert res.file_sha16_after == _util.sha_text(text) == _util.sha_file(thr)
    # the tuple sha later runs will carry, via the runner's own resolution
    resolved = rte.load_thresholds(thr, "opus5_api")
    assert resolved["letter_source"] == "opus5_api_calibrated" and resolved["t_A"] == 0.199
    assert rte.thresholds_sha(resolved) == res.tuple_sha16_after
    assert aggregate_v2.resolve_thresholds(new, "sonnet_api")["t_A"] == old["sonnet_api"]["t_A"]
    # the calibration record
    cal = arch / "opus5_api_calibration.json"
    assert res.calibration_path == str(cal)
    rec = json.loads(cal.read_text())
    assert rec["t_A"] == 0.199 and rec["n_neg"] == 200 and rec["written"] is True
    assert rec["preds_sha16"] == _util.sha_file(preds) and rec["written_utc"].endswith("Z")
    ct.CalibrationResult(**rec)                                     # round-trips (extra=forbid)
    # a second --write archives to _2 and replaces the key in place (202 negatives: the count
    # blocker must be told)
    _preds(preds, _frame(NEG200 + [0.3, 0.31], pos_scores=[0.5]))
    with pytest.raises(SystemExit, match="n-neg-expected 200"):
        ct.run(preds, "opus5_api", thr, write=True, archive_dir=arch)
    res2 = ct.run(preds, "opus5_api", thr, write=True, archive_dir=arch, n_neg_expected=202)
    assert res2.archive_path == str(arch / "thresholds_v2.pre_opus5_2.json")
    assert Path(res2.archive_path).read_text() == text
    assert res2.key_present and res2.existing_entry == {"tau0": 0.15, "t_A": 0.199, "t_B": 0.191}
    new2 = json.loads(thr.read_text())
    assert list(new2) == list(new) and new2["opus5_api"]["t_A"] == 0.3        # 0.31, 0.3 are the top two of 202
    assert {k: v for k, v in new2.items() if k != "opus5_api"} == old


def test_run_refuses_below_min_neg(tmp_path):
    preds = _preds(tmp_path / "preds.parquet", _frame([i / 100 for i in range(1, 41)], pos_scores=[0.5]))
    thr = _tmp_thresholds(tmp_path)
    before = thr.read_text()
    res = ct.run(preds, "opus5_api", thr, min_neg=50, archive_dir=tmp_path / "arch")
    assert any("n_neg=40 < --min-neg 50" in b for b in res.write_blockers)
    assert any("n_neg=40 != --n-neg-expected 200" in b for b in res.write_blockers)
    assert any("+inf" in b for b in res.write_blockers)             # 40 negatives at 1 % allow 0
    with pytest.raises(SystemExit, match="REFUSED --write"):
        ct.run(preds, "opus5_api", thr, min_neg=50, write=True, archive_dir=tmp_path / "arch")
    assert thr.read_text() == before and not (tmp_path / "arch").exists()
    # lowering --min-neg alone is not enough here (t_A is +inf, the count is 40); a dev fit at
    # fpr_a = 0.05 with the count stated writes
    with pytest.raises(SystemExit):
        ct.run(preds, "opus5_api", thr, min_neg=10, write=True, archive_dir=tmp_path / "arch")
    with pytest.raises(SystemExit, match="n-neg-expected"):
        ct.run(preds, "opus5_api", thr, fpr_a=0.05, fpr_b=0.10, min_neg=10, write=True, archive_dir=tmp_path / "arch")
    res = ct.run(preds, "opus5_api", thr, fpr_a=0.05, fpr_b=0.10, min_neg=10, write=True,
                 archive_dir=tmp_path / "arch", n_neg_expected=40)
    assert res.written and res.t_A == 0.39 and res.t_B == 0.37


def test_run_refuses_incomplete_replicate_nan_negatives_and_wrong_arm(tmp_path):
    """The completeness blockers (reviews: a premature --write on a parquet still being
    flushed, or on a subset of the 200 negatives, would fit on a partial file): no
    .meta.json, a negative row with S NaN, a count other than 200, an arm other than a2."""
    thr = _tmp_thresholds(tmp_path)
    before = thr.read_text()
    arch = tmp_path / "arch"
    # (1) 160 finite + 40 NaN negatives: the fit would silently use 160 (t_A 0.16 instead of 0.199)
    df = _frame(NEG200[40:] + [float("nan")] * 40, pos_scores=[0.5])
    preds = _preds(tmp_path / "preds.parquet", df)
    res = ct.run(preds, "opus5_api", thr, archive_dir=arch)
    assert (res.n_neg, res.n_neg_nan, res.t_A, res.t_B) == (160, 40, 0.2, 0.193)     # fit on the 160, silently, before
    assert any("40 negative non-anchor row(s) have S NaN" in b for b in res.write_blockers)
    assert any("n_neg=160 != --n-neg-expected 200" in b for b in res.write_blockers)
    with pytest.raises(SystemExit, match="REFUSED --write"):
        ct.run(preds, "opus5_api", thr, write=True, archive_dir=arch)
    # --allow-nan-neg lifts only the NaN blocker; the count blocker stands unless stated
    res = ct.run(preds, "opus5_api", thr, archive_dir=arch, allow_nan_neg=True)
    assert not any("S NaN" in b for b in res.write_blockers) and any("n-neg-expected" in b for b in res.write_blockers)
    assert ct.run(preds, "opus5_api", thr, archive_dir=arch, allow_nan_neg=True, n_neg_expected=160).write_blockers == []
    assert ct.run(preds, "opus5_api", thr, archive_dir=arch, allow_nan_neg=True, n_neg_expected=0).write_blockers == []
    # (2) no meta json: an interrupted / still-running replicate
    preds = _preds(tmp_path / "preds.parquet", _frame(NEG200, pos_scores=[0.5]), meta=False)
    res = ct.run(preds, "opus5_api", thr, archive_dir=arch)
    assert not res.meta_present and res.write_blockers and all("meta.json is absent" in b for b in res.write_blockers)
    with pytest.raises(SystemExit, match="meta.json is absent"):
        ct.run(preds, "opus5_api", thr, write=True, archive_dir=arch)
    # (3) the a1 design parquet is not the calibration run
    preds = _preds(tmp_path / "preds.parquet", _frame(NEG200, arm="a1"))
    res = ct.run(preds, "opus5_api", thr, archive_dir=arch)
    assert res.write_blockers == ["preds arm='a1' is not the registered calibration arm 'a2'"]
    assert thr.read_text() == before and not arch.exists()


def test_run_refuses_non_design_model_mismatch_tau0_and_noncanonical(tmp_path):
    thr = _tmp_thresholds(tmp_path)
    preds = _preds(tmp_path / "preds.parquet", _frame(NEG200, half="holdout"))
    res = ct.run(preds, "opus5_api", thr, archive_dir=tmp_path)
    assert any("half='holdout'" in b for b in res.write_blockers)
    _preds(preds, _frame(NEG200, model="sonnet"))
    res = ct.run(preds, "opus5_api", thr, archive_dir=tmp_path)
    assert any("disagrees with the parquet's model='sonnet'" in b for b in res.write_blockers)
    _preds(preds, _frame(NEG200, tau0=0.2))
    res = ct.run(preds, "opus5_api", thr, archive_dir=tmp_path)
    assert any("tau0" in b for b in res.write_blockers)
    _preds(preds, _frame(NEG200))
    thr.write_text(json.dumps(json.loads(thr.read_text()), indent=1))      # not canonical
    res = ct.run(preds, "opus5_api", thr, archive_dir=tmp_path)
    assert any("canonical" in b for b in res.write_blockers)
    with pytest.raises(SystemExit):
        ct.run(preds, "opus5_api", thr, write=True, archive_dir=tmp_path)
    # a frame without the meta columns (no half/model/tau0) raises no blocker
    df = _frame(NEG200).drop(columns=["half", "model", "tau0", "arm", "run_tag", "thinking", "effort",
                                      "thresholds_sha16"])
    _preds(preds, df)
    thr = _tmp_thresholds(tmp_path)
    res = ct.run(preds, "opus5_api", thr, archive_dir=tmp_path)
    assert res.write_blockers == [] and res.half is None and res.run_tag is None


def test_cli_subprocess(tmp_path):
    preds = _preds(tmp_path / "preds.parquet", _frame(NEG200, pos_scores=[0.5, 0.05]))
    thr = _tmp_thresholds(tmp_path)
    arch = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPT), "--preds", str(preds), "--model-key", "opus5_api",
           "--thresholds", str(thr), "--fpr-a", "0.01", "--fpr-b", "0.05", "--min-neg", "50",
           "--archive-dir", str(arch)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_util.REPRO))
    assert r.returncode == 0, r.stderr
    assert "[calib] t_A = 0.199" in r.stdout and "[calib] t_B = 0.191" in r.stdout
    rec = json.loads(r.stdout[r.stdout.index("\n{") + 1:])
    assert rec["written"] is False and rec["t_A"] == 0.199 and not arch.exists()
    r = subprocess.run(cmd + ["--write", "--out", str(tmp_path / "cal.json")],
                       capture_output=True, text=True, cwd=str(_util.REPRO))
    assert r.returncode == 0, r.stderr
    assert (arch / "thresholds_v2.pre_opus5.json").read_text() == REAL_THRESHOLDS.read_text()
    assert json.loads((tmp_path / "cal.json").read_text())["t_B"] == 0.191
    assert json.loads(thr.read_text())["opus5_api"] == {"tau0": 0.15, "t_A": 0.199, "t_B": 0.191}
    # refusal exits non-zero and leaves the file alone
    _preds(preds, _frame(NEG200[:30]))
    text = thr.read_text()
    r = subprocess.run(cmd + ["--write"], capture_output=True, text=True, cwd=str(_util.REPRO))
    assert r.returncode != 0 and "REFUSED --write" in r.stderr and thr.read_text() == text
    assert "n-neg-expected 200" in r.stderr
    # the CLI flags reach run(): --n-neg-expected / --allow-nan-neg
    _preds(preds, _frame(NEG200[:30] + [float("nan")], pos_scores=[0.5]))
    r = subprocess.run(cmd + ["--n-neg-expected", "30", "--allow-nan-neg", "--min-neg", "10", "--fpr-a", "0.1",
                              "--fpr-b", "0.2"], capture_output=True, text=True, cwd=str(_util.REPRO))
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout[r.stdout.index("\n{") + 1:])
    assert rec["write_blockers"] == [] and rec["n_neg_expected"] == 30 and rec["allow_nan_neg"] is True


@pytest.mark.skipif(not HOLDOUT_A2.exists(), reason="holdout a2 parquet not on this machine")
def test_smoke_real_holdout_a2_read_only(tmp_path):
    """Schema discovery on the real a2/opus5 HOLDOUT parquet — report mode only, on a tmp
    copy of the thresholds file; the holdout half is a write blocker by construction."""
    thr = _tmp_thresholds(tmp_path)
    before = thr.read_text()
    res = ct.run(HOLDOUT_A2, "opus5_api", thr, archive_dir=tmp_path / "arch")
    assert thr.read_text() == before and not (tmp_path / "arch").exists() and not res.written
    assert (res.n_neg, res.n_pos, res.n_anchor_excluded, res.half, res.model, res.arm) == \
        (200, 42, 0, "holdout", "opus5", "a2")
    assert res.n_neg_ge_tA <= 2 and res.n_neg_ge_tB <= 10
    assert res.t_A is not None and res.t_B is not None and res.t_A >= res.t_B
    assert res.fpr_A <= 0.01 + 1e-12 and res.fpr_B <= 0.05 + 1e-12
    assert any("half='holdout'" in b for b in res.write_blockers)
    assert res.run_thresholds_sha16 == "a40ae6e201a03e65"           # the provisional tuple sha
    with pytest.raises(SystemExit, match="REFUSED"):
        ct.run(HOLDOUT_A2, "opus5_api", thr, write=True, archive_dir=tmp_path / "arch")
    assert thr.read_text() == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
