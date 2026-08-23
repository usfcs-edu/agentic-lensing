#!/usr/bin/env python3
"""No-API, no-network tests for the golden SFT corpus builder (WP-F) and the DESI
golden-by-agreement arm.

Everything runs on synthetic labels/splits/frame/key fixtures + tiny PIL JPEGs in a temp dir.
Runs under the lensjudge venv via pytest, or directly:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_sft.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.golden import _util  # noqa: E402
from lensjudge.golden import build_corpus_golden as bcg  # noqa: E402
from lensjudge.golden import build_desi_agreement_arm as arm  # noqa: E402

KIT = "kit_test"


# ----------------------------------------------------------------- fixtures
def _jpeg(path: Path, seed: int) -> str:
    """Tiny unique JPEG; returns its sha16 (what the key/labels carry as render_sha)."""
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (16, 24, 3), dtype=np.uint8)).save(path, "JPEG")
    return _util.sha_file(path)


def _fixture(tmp: Path, n: int = 24, plant_leak: bool = False, split_of=None):
    """n units: grades cycle A/B/C/D, confidences cycle H/M/L, coords on a 1-degree grid
    (never within 2"), alternating align/validate. One pass-2 repeat in the key. Returns the
    four CSV paths + kits dir."""
    kits = tmp / "kits"
    items = kits / KIT / "items"
    items.mkdir(parents=True)
    frame, labels, splits, key = [], [], [], []
    for i in range(n):
        uid, cid = f"u{i + 1:04d}", f"jw{i + 1:03d}"
        ra, dec = 150.0 + i * 1.0, 2.0 + (i % 5) * 1.0
        split = split_of(i) if split_of else ("align" if i % 2 == 0 else "validate")
        if plant_leak and i == 1:        # 1" east of unit 0 (align); validate unless split_of says
            ra, dec = 150.0 + 1.0 / 3600.0, 2.0
        score = 4 - (i % 4)
        lmh = ("H", "M", "L")[i % 3]
        item = f"{i + 1:03d}"
        sha = _jpeg(items / f"{item}.jpg", seed=i)
        frame.append(dict(unit_id=uid, candidate_id=cid, ra_deg=ra, dec_deg=dec,
                          system_id=i, stratum="T_U", sw_filter="F150W", lw_filter="F444W",
                          layout="color", pipe_grade_passcount="U"))
        labels.append(dict(unit_id=uid, candidate_id=cid, ra_deg=ra, dec_deg=dec, stratum="T_U",
                           score_1_4=score, grade_letter=_util.score_to_letter(score),
                           confidence_lmh=lmh, confidence01=_util.CONF_TO_01[lmh],
                           pass2_score_1_4="", pass2_confidence_lmh="", n_passes=1,
                           label_stable="", render_sha=sha, grade_scale=_util.GRADE_SCALE,
                           grader_id="XH"))
        splits.append(dict(unit_id=uid, system_id=i, split=split, forced=False, stratum="T_U",
                           grade_letter=_util.score_to_letter(score)))
        key.append(dict(kit_id=KIT, item_id=item, presentation_index=i + 1, unit_id=uid,
                        candidate_id=cid, **{"pass": 1}, repeat_of_item="", render_sha=sha,
                        layout="color", stratum="T_U"))
    # a pass-2 repeat of unit 1 (byte-identical image under a new item id) must be ignored
    rep = f"{n + 1:03d}"
    (items / f"{rep}.jpg").write_bytes((items / "001.jpg").read_bytes())
    key.append(dict(kit_id=KIT, item_id=rep, presentation_index=n + 1, unit_id="u0001",
                    candidate_id="jw001", **{"pass": 2}, repeat_of_item="001",
                    render_sha=key[0]["render_sha"], layout="color", stratum="T_U"))
    paths = {}
    for stem, rows in (("frame", frame), ("golden_labels", labels), ("splits", splits),
                       ("kit_key", key)):
        paths[stem] = tmp / f"{stem}.csv"
        pd.DataFrame(rows).to_csv(paths[stem], index=False)
    return paths, kits


def _build(tmp: Path, **kw):
    paths, kits = _fixture(tmp, **kw)
    df = bcg.load_rows(paths["golden_labels"], paths["splits"], paths["frame"],
                       [paths["kit_key"]], kits)
    return df, paths, kits


# ------------------------------------------------------------ corpus builder
def test_build_corpus_writes_valid_jsonl():
    tmp = Path(tempfile.mkdtemp())
    df, paths, kits = _build(tmp, n=80)       # 40 align -> int(40 * 0.03) = 1 val row
    assert len(df) == 80 and set(df.split) == {"align", "validate"}
    bcg.firewall(df)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sysp = bcg.system_prompt(tmp / "missing_note.md")   # placeholder path -> warning
        assert any("placeholder" in str(x.message) for x in w)
    assert sysp.startswith(bcg.RUBRIC_V2) and "placeholder" in sysp
    note = tmp / "jwst_note.md"
    note.write_text("# JWST NOTE\nreal note text\n")
    assert bcg.system_prompt(note).endswith("real note text\n")

    train, valsel, vs_labels = bcg.build_records(df, sysp)
    out = tmp / "corpus_golden"
    man = bcg.write_corpus(df, train, valsel, vs_labels, out)

    tr = [json.loads(x) for x in open(out / "sft_golden_train.jsonl")]
    va = [json.loads(x) for x in open(out / "sft_golden_val.jsonl")]
    vs = [json.loads(x) for x in open(out / "valsel_golden.jsonl")]
    # 40 align rows (20 A + 20 C): valsel = ceil(0.2 * 20) = 4 per letter = 8, the rest train
    # (+ 3% val); the 40 validate rows are written NOWHERE in the corpus dir
    assert len(vs) == 8 and len(tr) + len(va) == 32
    assert len(va) == int(32 * bcg.VAL_FRAC) == 0
    align_names = set(df[df.split == "align"].name)
    validate_names = set(df[df.split == "validate"].name)
    for rec in tr + va + vs:
        assert Path(rec["images"][0]).name in {f"{i + 1:03d}.jpg" for i in range(0, 80, 2)}
    assert not any(n in json.dumps(rec) for rec in tr + va + vs for n in validate_names)
    for rec in tr + va:
        assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
        assert rec["messages"][1]["content"] == bcg.USER_MSG_JWST
        assert rec["messages"][1]["content"].count("<image>") == 1 == len(rec["images"])
        assert Path(rec["images"][0]).is_absolute() and Path(rec["images"][0]).exists()
        assert rec["images"][0].endswith(".jpg") and f"/{KIT}/items/" in rec["images"][0]
        g = ImageGrade.model_validate_json(rec["messages"][2]["content"])
        assert rec["label"] == bcg._LABEL[g.grade] and g.contaminant is None
        assert rec["name"].startswith("jw")
    for rec in vs:
        assert [m["role"] for m in rec["messages"]] == ["system", "user"]
        assert "label" not in rec and Path(rec["images"][0]).exists()
    lab = pd.read_csv(out / "valsel_golden_labels.csv")
    assert list(lab.columns) == ["idx", "label", "name", "grade", "label_source", "split"]
    assert lab.idx.tolist() == list(range(8)) and (lab.label_source == "golden_huang").all()
    assert set(lab.name) <= align_names and not set(lab.name) & validate_names
    assert lab.grade.value_counts().to_dict() == {"A": 4, "C": 4}      # align rows are A/C only here
    assert (lab.split == "align_valsel").all()
    assert set(lab.name) == bcg.valsel_names(df) and not set(lab.name) & {r["name"] for r in tr + va}
    # the carve is deterministic and shrinks with the fraction
    assert bcg.valsel_names(df) == bcg.valsel_names(df) and len(bcg.valsel_names(df, 0.1)) == 4
    assert bcg.valsel_names(df, 0.0) == set()

    man2 = _util.read_pinned(out / "corpus_manifest.csv")     # .sha sidecar verifies
    want = ["name", "unit_id", "ra", "dec", "survey_key", "grade", "score_1_4", "confidence_lmh",
            "label_source", "soft_target", "split", "label", "render_sha", "criteria_source"]
    assert list(man2.columns)[:len(want)] == want
    assert (man2.survey_key == "jwst").all() and (man2.criteria_source == "synthetic_hash").all()
    assert man2.split.value_counts().to_dict() == {"train": 32, "valsel": 8}
    assert len(man) == 40 and set(man.name) == align_names and man2.p_lens_target.between(0.01, 0.99).all()
    # the pass-2 repeat item never became the served image
    assert not any(r["images"][0].endswith("/081.jpg") for r in tr + va + vs)
    # the gate refuses a labels file that names validate units
    bad = lab.copy(); bad.loc[0, "split"] = "validate"
    try:
        bcg.assert_not_validate(bad, None, None); raise AssertionError("no raise")
    except SystemExit as e:
        assert "validate" in str(e)
    bad2 = pd.DataFrame({"idx": [0], "label": ["lens"], "name": [sorted(validate_names)[0]],
                         "grade": ["A"], "label_source": ["golden_huang"]})
    try:
        bcg.assert_not_validate(bad2, paths["golden_labels"], paths["splits"]); raise AssertionError("no raise")
    except SystemExit as e:
        assert "validate units" in str(e)
    bcg.assert_not_validate(lab, paths["golden_labels"], paths["splits"])        # clean


def test_targets_unique_with_identical_grade_and_confidence():
    tmp = Path(tempfile.mkdtemp())
    paths, kits = _fixture(tmp, n=30)
    lab = pd.read_csv(paths["golden_labels"])
    lab["score_1_4"], lab["grade_letter"], lab["confidence_lmh"] = 4, "A", "H"
    lab.to_csv(paths["golden_labels"], index=False)
    spl = pd.read_csv(paths["splits"]); spl["grade_letter"] = "A"; spl.to_csv(paths["splits"], index=False)
    df = bcg.load_rows(paths["golden_labels"], paths["splits"], paths["frame"],
                       [paths["kit_key"]], kits)
    assert (df.grade == "A").all() and (df.confidence_lmh == "H").all()
    train, valsel, _ = bcg.build_records(df, "SYS")       # asserts uniqueness internally
    targets = [r["messages"][2]["content"] for r in train]
    assert len(set(targets)) == len(targets) == 12 and len(valsel) == 3   # 15 align, 20% carve
    assert len(bcg.build_records(df, "SYS", valsel_frac=0)[0]) == 15
    # even with coordinates stripped from the rationale the hash channels separate them
    ps = {json.loads(t)["p_lens"] for t in targets}
    assert len(ps) > 1


def test_rationale_truthful_and_no_banned_words():
    for score in (1, 2, 3, 4):
        for lmh in ("L", "M", "H"):
            for i in range(20):
                r = bcg.golden_rationale(f"jw{i}", 150.123456, -2.5, "F150W", "F444W", score, lmh)
                low = r.lower()
                assert not any(w.lower() in low for w in bcg.BANNED_WORDS), r
                assert f"score {score}/4" in r and f"confidence {lmh}" in r
                assert "(150.1235,-2.5000)" in r and "F150W/F444W" in r
    # gray layouts: one filter missing -> the other alone; none -> NIRCam
    assert "JWST F150W:" in bcg.golden_rationale("x", 1, 2, "F150W", "", 4, "H") or \
        "JWST F150W " in bcg.golden_rationale("x", 1, 2, "F150W", "", 4, "H")
    assert "NIRCam" in bcg.golden_rationale("y", 1, 2, "", float("nan"), 4, "H")


def test_p_monotone_in_confidence_and_ordered_by_grade():
    names = [f"jw{i}" for i in range(50)]
    for name in names:
        for g in "ABCD":
            # sureness pulls p away from 0.5 toward SOFT[g]; same name -> same jitter
            # (strict on the raw value; C's whole L->H spread is 0.013, so only non-decreasing
            # survives the 2-dp rounding)
            d = [abs(bcg.golden_p(g, lmh, name, ndigits=None) - 0.5) for lmh in ("L", "M", "H")]
            assert d[0] < d[1] < d[2], (g, name, d)
            d2 = [abs(bcg.golden_p(g, lmh, name) - 0.5) for lmh in ("L", "M", "H")]
            assert d2[0] <= d2[1] <= d2[2], (g, name, d2)
            sign = 1 if bcg.SOFT[g] > 0.5 else -1
            assert all(sign * (bcg.golden_p(g, lmh, name) - 0.5) > 0 for lmh in "LMH")
        for lmh in "LMH":
            ps = [bcg.golden_p(g, lmh, name) for g in "ABCD"]
            assert ps[0] > ps[1] > ps[2] > ps[3], (lmh, name, ps)
            assert all(0.01 <= p <= 0.99 for p in ps)
    # across DIFFERENT names the +-0.02 jitter never flips the grade order
    lo = {g: min(bcg.golden_p(g, "L", n) for n in names) for g in "ABCD"}
    hi = {g: max(bcg.golden_p(g, "L", n) for n in names) for g in "ABCD"}
    assert lo["A"] > hi["B"] > hi["C"] and lo["B"] > hi["C"] and lo["C"] > hi["D"]
    # the no-jitter centre is exactly the contract formula
    c = bcg.golden_p("D", "L", "n") - (_util.hash01("n", "p") - 0.5) * 0.04
    assert abs(c - (0.5 + (0.05 - 0.5) * (0.6 + 0.4 * 0.33))) < 0.006   # 2-dp rounding
    # confidence mapping + jitter stays in its L/M/H band
    for lmh, base in _util.CONF_TO_01.items():
        for n in names:
            assert abs(bcg.golden_confidence(lmh, n) - base) <= 0.021
    # criteria: per-example varying, bounded, correlated with SOFT
    ca = [bcg.golden_criteria("A", n) for n in names]
    cd = [bcg.golden_criteria("D", n) for n in names]
    assert all(0 <= v <= 10 for c in ca + cd for v in c.values())
    assert np.mean([np.mean(list(c.values())) for c in ca]) > \
        np.mean([np.mean(list(c.values())) for c in cd]) + 4
    assert len({json.dumps(c) for c in ca}) > 1


def test_firewall_raises_on_planted_cross_split_pair():
    tmp = Path(tempfile.mkdtemp())
    df, _, _ = _build(tmp, plant_leak=True)
    try:
        bcg.firewall(df)
    except AssertionError as e:
        assert "POSITION leakage" in str(e)
    else:
        raise AssertionError("firewall did not raise on a 1\" align/validate pair")
    # the same geometry inside ONE half is fine (dup pairs share a half)
    df2, _, _ = _build(Path(tempfile.mkdtemp()), plant_leak=True,
                       split_of=lambda i: "align" if i < 12 else "validate")
    bcg.firewall(df2)


def test_load_rows_rejects_bad_inputs():
    tmp = Path(tempfile.mkdtemp())
    paths, kits = _fixture(tmp, n=8)
    # a corrupted served JPEG (sha mismatch vs key/labels) must be refused
    item = kits / KIT / "items" / "003.jpg"
    item.write_bytes(item.read_bytes() + b"\x00")
    try:
        bcg.load_rows(paths["golden_labels"], paths["splits"], paths["frame"], [paths["kit_key"]], kits)
    except AssertionError as e:
        assert "sha" in str(e)
    else:
        raise AssertionError("sha mismatch not detected")
    # a score outside 1-4 must raise (strict score_to_letter, never ImageGrade coercion)
    paths, kits = _fixture(Path(tempfile.mkdtemp()), n=4)
    lab = pd.read_csv(paths["golden_labels"]); lab.loc[0, "score_1_4"] = 5
    lab.to_csv(paths["golden_labels"], index=False)
    try:
        bcg.load_rows(paths["golden_labels"], paths["splits"], paths["frame"], [paths["kit_key"]], kits)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("score 5 accepted")


def test_gate_aucs():
    labels = pd.DataFrame({"idx": range(6), "label": ["lens", "lens", "softmid", "nonlens",
                                                      "lens", "nonlens"],
                           "name": [f"n{i}" for i in range(6)], "grade": list("ABCDAD"),
                           "label_source": "golden_huang"})
    preds = pd.DataFrame({"name": [f"n{i}" for i in range(6)],
                          "p_lens": [0.9, 0.8, 0.3, 0.1, 0.7, 0.2],
                          "p_lens_logprob": [0.9, 0.2, 0.3, 0.1, 0.7, 0.8],
                          "gp_A": [0.8, 0.7, 0.1, 0.0, 0.6, 0.1], "gp_B": [0.2, 0.2, 0.2, 0.1, 0.3, 0.1],
                          "gp_C": [0.0, 0.1, 0.5, 0.2, 0.1, 0.3], "gp_D": [0.0, 0.0, 0.2, 0.7, 0.0, 0.5]})
    res = bcg.gate_aucs(preds, labels)
    assert res["n"] == 6 and res["n_lens"] == 3
    assert res["p_lens"] == 1.0 and res["s_exp"] == 1.0 and res["p_lens_logprob"] < 1.0
    res2 = bcg.gate_aucs(preds.drop(columns="name"), labels)      # idx-order join
    assert res2["p_lens"] == 1.0
    # s_exp recomputed from gp_* uses llm_client.ORDINAL_W (A=1,B=2/3,C=1/3,D=0), the same
    # weights run_batch writes into the parquet — one definition, not two
    from lensjudge.common.llm_client import logprob_ordinal
    preds2 = preds.copy(); preds2["s_exp"] = [logprob_ordinal({"A": .8, "B": .2})] + [0.0] * 5
    preds3 = preds.copy(); preds3.loc[1:, ["gp_A", "gp_B", "gp_C", "gp_D"]] = 0.0
    j = labels.merge(preds3, on="name"); assert "s_exp" not in j.columns
    res3 = bcg.gate_aucs(preds3, labels)        # rows with no letter mass -> NaN s_exp, dropped
    assert np.isnan(res3["s_exp"])
    preds4 = preds.copy().drop(columns=["p_lens", "p_lens_logprob"])
    res4 = bcg.gate_aucs(preds4, labels)
    assert abs(res4["s_exp"] - 1.0) < 1e-9 and "p_lens" not in res4


# --------------------------------------------------------- agreement arm
def _synthetic_vizier(n_a=4, n_b=3, n_c=5, n_dis=6) -> pd.DataFrame:
    rows = []
    k = 0
    for q, s, n in (("A", 4.0, n_a), ("B", 3.0, n_b), ("C", 2.0, n_c)):
        for _ in range(n):
            rows.append(dict(Name=f"DESI-{k:03d}.0000+00.0000", Q=q, Score=s, delSc=0, pair_ok=True,
                             _RA=float(k), _DE=0.0)); k += 1
    for i in range(n_dis):                                  # disagreeing pairs
        rows.append(dict(Name=f"DESI-{k:03d}.0000+00.0000", Q="B", Score=3.5, delSc=1,
                         pair_ok=True, _RA=float(k), _DE=0.0)); k += 1
    rows.append(dict(Name="DESI-glitch", Q="A", Score=3.5, delSc=0, pair_ok=False, _RA=1.0, _DE=1.0))
    return pd.DataFrame(rows)


def test_select_agreement_counts_and_join():
    viz = _synthetic_vizier()
    sel = arm.select_agreement(viz, expected={"A": 4, "B": 3, "C": 5})
    assert len(sel) == 12 and "DESI-glitch" not in set(sel.name)
    assert sel.score_1_4.map(_util.score_to_letter).equals(sel.grade_letter)
    try:
        arm.select_agreement(viz, expected={"A": 100, "B": 165, "C": 461})
    except AssertionError:
        pass
    else:
        raise AssertionError("count assertion did not fire")
    local = pd.DataFrame({"name": viz.Name, "RA": viz._RA, "DEC": viz._DE, "grade": viz.Q})
    j = arm.join_local_catalog(sel, local)
    assert list(j.columns) == ["name", "score_1_4", "grade_letter", "ra", "dec"] and len(j) == 12
    # a real-table sanity check on the pinned CSV when it is present (no network)
    if arm.VIZIER.exists():
        real = arm.select_agreement(pd.read_csv(arm.VIZIER))
        assert len(real) == 726


def test_manifest_flags_and_fewshot_pick_deterministic():
    viz = _synthetic_vizier(n_a=6, n_b=6, n_c=6)
    local = pd.DataFrame({"name": viz.Name, "RA": viz._RA, "DEC": viz._DE})
    gd = pd.DataFrame({"name": [f"DESI-D{i}" for i in range(12)], "ra": 200.0 + np.arange(12),
                       "dec": -10.0, "survey_key": "storfer", "score_1_4": 1, "grade_letter": "D",
                       "label_source": arm.LABEL_D})
    # bench: one agreement row by NAME, one grade-D row by POSITION (1.5" off), one unrelated
    bench = pd.DataFrame({"name": ["DESI-000.0000+00.0000", "other", "x"],
                          "ra": [999.0, 201.0 + 1.5 / 3600.0, 50.0], "dec": [0.0, -10.0, 0.0]})
    pool = pd.DataFrame({"name": ["DESI-D2", "DESI-D3", "DESI-001.0000+00.0000"],
                         "split": ["gate", "train", "valsel"]})
    man = arm.build_manifest(viz, local, gd, bench, pool, expected={"A": 6, "B": 6, "C": 6})
    assert list(man.columns) == ["name", "ra", "dec", "survey_key", "score_1_4", "grade_letter",
                                 "label_source", "bench_overlap", "pool_split"]
    assert len(man) == 18 + 12
    assert set(man.loc[man.bench_overlap, "name"]) == {"DESI-000.0000+00.0000", "DESI-D1"}
    assert dict(zip(man.name, man.pool_split))["DESI-D2"] == "gate"
    assert (man.loc[man.label_source == arm.LABEL_AGREE, "survey_key"] == "ls-dr9").all()

    fs = arm.pick_fewshot(man)
    assert fs.grade.value_counts().to_dict() == {"D": 6, "A": 3, "B": 3, "C": 3}
    assert list(fs.columns) == ["name", "ra", "dec", "survey_key", "label", "grade", "note",
                                "grade_source"]
    assert not set(fs.name) & {"DESI-000.0000+00.0000", "DESI-D1", "DESI-D2"}   # excluded
    assert set(fs.label) == {"LENS", "POSSIBLE LENS", "NON-LENS"}
    assert (fs.loc[fs.grade == "D", "grade_source"] == arm.GRADE_SOURCE_D).all()
    assert (fs.loc[fs.grade != "D", "grade_source"] == arm.GRADE_SOURCE_AGREE).all()
    assert (fs.note.fillna("") == "").all()
    # determinism: same pick from a shuffled manifest
    fs2 = arm.pick_fewshot(man.sample(frac=1, random_state=7).reset_index(drop=True))
    assert fs.name.tolist() == fs2.name.tolist()


def test_agreement_sft_rows_only_with_images():
    tmp = Path(tempfile.mkdtemp())
    viz = _synthetic_vizier(n_a=2, n_b=2, n_c=2, n_dis=0)
    local = pd.DataFrame({"name": viz.Name, "RA": viz._RA, "DEC": viz._DE})
    gd = pd.DataFrame({"name": ["DESI-D0", "DESI-D1"], "ra": [200.0, 201.0], "dec": -10.0,
                       "survey_key": "storfer", "score_1_4": 1, "grade_letter": "D",
                       "label_source": arm.LABEL_D})
    man = arm.build_manifest(viz, local, gd, pd.DataFrame(columns=["name", "ra", "dec"]),
                             pd.DataFrame(columns=["name", "split"]), expected=None)
    img = tmp / "images"; img.mkdir()
    with_img = ["DESI-000.0000+00.0000", "DESI-003.0000+00.0000", "DESI-D1"]
    for n in with_img:
        for v in arm.desi.VIEWS:
            Image.new("RGB", (8, 8)).save(img / f"{n}_{v}.png")
    Image.new("RGB", (8, 8)).save(img / "DESI-D0_full.png")          # incomplete -> skipped
    recs, stats = arm.build_sft(man, img)
    assert stats["rows"] == 3 and stats["with_images"] == {arm.LABEL_AGREE: 2, arm.LABEL_D: 1}
    assert {r["name"] for r in recs} == set(with_img)
    for r in recs:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert r["messages"][0]["content"] == arm.desi.DIRECT_SYS
        assert r["messages"][1]["content"] == arm.desi.USER_MSG
        assert len(r["images"]) == 4 and all(Path(p).exists() for p in r["images"])
        g = ImageGrade.model_validate_json(r["messages"][2]["content"])
        assert r["label"] == bcg._LABEL[g.grade]
        low = g.rationale.lower()
        assert not any(w.lower() in low for w in bcg.BANNED_WORDS), g.rationale
        if g.grade == "D":
            assert g.contaminant == "contaminant" and g.p_lens <= 0.08
        else:
            assert g.contaminant is None and abs(g.confidence - 0.90) <= 0.021
    assert len({r["messages"][2]["content"] for r in recs}) == 3
    # DESI-pool gate/valsel rows never enter the mix-in (corpus_desi's frozen gate + selection set)
    man.loc[man.name == "DESI-D1", "pool_split"] = "valsel"
    man.loc[man.name == "DESI-003.0000+00.0000", "pool_split"] = "gate"
    recs2, stats2 = arm.build_sft(man, img)
    assert {r["name"] for r in recs2} == {"DESI-000.0000+00.0000"} and stats2["rows"] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
