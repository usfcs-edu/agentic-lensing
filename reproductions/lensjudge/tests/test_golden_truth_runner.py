#!/usr/bin/env python3
"""No-API tests for WP-6: golden/run_truth_eval.py (tuple, holdout gate, outputs, exposure
registry), the run_golden_eval `section=`/`cols=` seams, golden/REGISTRY.md's truth-eval
sections and anchors, and golden/make_verifier_patch.py (Nate's drop-in: embedded-brief sha
parity, the patch applying to a temp copy of J, 09_rank_report_v2.py on a synthetic verdict
dir with a ctl file and malformed lines).

The runner is exercised with golden/panel.py swapped for a stub (`rte._PANEL`) and a fake
exposure registry (`rge._REGISTRY`); one smoke test drives the REAL panel with
grader_direct.grade_candidate stubbed. NO network, NO API spend. J-dependent checks (the
patch apply) skip when the jwst-strong-lens-search checkout is absent; the node syntax
check skips without `node`.

    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_truth_runner.py
(also pytest-compatible)
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from lensjudge.golden import _util, aggregate_v2  # noqa: E402
from lensjudge.golden import make_verifier_patch as mvp  # noqa: E402
from lensjudge.golden import run_golden_eval as rge  # noqa: E402
from lensjudge.golden import run_truth_eval as rte  # noqa: E402
from lensjudge import config  # noqa: E402

SYNTH_PI = ["needs a much closer look at the ring", "ask the tool to hyperlink every catalogue entry"]

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"
PY = sys.executable
J = _util.JWST_REPO
ANCHOR_IDS = {"J20954380-1094330": "15", "J18805344+1121596": "13", "J18030075+2309921": "7",
              "J18030108+2309932": "14", "J5186648-1343587": "16"}


# ------------------------------------------------------------------ fixtures
def _jpeg(path: Path, seed: int, size=(752, 540)) -> None:
    from PIL import Image
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 60, (size[1], size[0], 3), dtype=np.uint8)).save(path, format="JPEG")


def make_world(n_design: int = 3, n_holdout: int = 3):
    """A temp truth world: manifest (half column, unit_id on every other row), pinned splits,
    composites, a REGISTRY.md copy, a frame.csv, a lexicon."""
    d = Path(tempfile.mkdtemp(prefix="truth_runner_"))
    (d / "kits_truth").mkdir()
    rows, sp = [], []
    i = 0
    for half, n in (("design", n_design), ("holdout", n_holdout)):
        for j in range(n):
            i += 1
            cid = f"J{1000000 + i * 7919}-{100000 + i * 31}"
            p = d / "kits_truth" / f"{cid}.jpg"
            _jpeg(p, i)
            unit = f"u{i:04d}" if i % 2 else ""
            rows.append({"name": cid, "ra": 150.0 + i, "dec": 2.0, "survey_key": "jwst", "grade_truth": "",
                         "binary_label": "lens" if i % 3 == 0 else "nonlens", "source": "truth_jwst",
                         "region": 1727, "tractor_type": "", "p_meta": np.nan, "leak": "no", "unit_id": unit,
                         "image_path": str(p), "image_path_v2r": "", "render_sha": _util.sha_file(p),
                         "truth_class": "cowls" if i % 3 == 0 else "negative", "is_positive": i % 3 == 0,
                         "is_stress": False, "is_anchor": False, "cowls_band": "", "cowls_ranking": "",
                         "cowls_theta_E": np.nan, "known_lens_name": "", "known_type": "", "known_sep_arcsec": np.nan,
                         "centre_is_deflector": i % 3 == 0, "layout": "color" if i != 2 else "gray_sw_only",
                         "field_class": "cosmos", "proposal": 1727, "mag_r": 20.0, "prior_exposure": 0,
                         "pipe_grade_passcount": "", "pipe_inspector_conf": np.nan, "pipe_score": np.nan,
                         "in_frame": bool(unit), "half": half})
            sp.append({"candidate_id": cid, "system_id": i, "half": half, "forced": False, "forced_reason": "",
                       "truth_class": rows[-1]["truth_class"], "cell": "x"})
    man = pd.DataFrame(rows)
    _util.pin(man, d / "truth_manifest.csv")
    _util.pin(pd.DataFrame(sp), d / "truth_splits.csv")
    fr = man[man["unit_id"] != ""][["unit_id", "name"]].rename(columns={"name": "candidate_id"})
    fr["desi_pool_overlap"] = False
    _util.pin(fr, d / "frame.csv")
    shutil.copy(GOLDEN / "REGISTRY.md", d / "REGISTRY.md")
    hold = man.loc[man["half"] == "holdout", "name"].tolist()
    (d / "banned.txt").write_text("\n".join(["J9999999-9999999", "this exact sentence must never appear"] + hold + SYNTH_PI) + "\n")
    (d / "banned_stale.txt").write_text("J9999999-9999999\nthis exact sentence must never appear\n")   # pre-Part-2 lexicon
    (d / "out").mkdir()
    return d, man


class _FakeRegistry:
    def __init__(self):
        self.exposed, self.seeded = [], []

    def seed_from_frame(self, frame, path=None):
        self.seeded.append(sorted(frame["unit_id"].astype(str)))

    def mark_exposed(self, unit_ids, run_tag, kind, path=None):
        self.exposed.append((tuple(unit_ids), run_tag, kind))

    def assert_unexposed(self, unit_ids, kinds=("fewshot", "sft"), path=None):
        pass


def make_stub_panel(p_evidence: float = 0.6, letter: str = "B", wrong_sha_role: str | None = None,
                    cost: float = 0.02):
    """A stand-in for golden/panel.py with the five names the runner uses. Prompts are fixed
    texts; the reported per-role sha16s are computed from (persona_set, note) exactly as the
    runner does, unless `wrong_sha_role` is set (the drift the runner must refuse)."""
    stub = types.SimpleNamespace()
    stub.calls = []
    texts = {"advocate": "ADVOCATE BRIEF\n", "artifact": "COMMON\nARTIFACT\n", "geometry": "COMMON\nGEOMETRY\n",
             "morphology": "COMMON\nMORPHOLOGY\n", "arbitrator": "ARBITRATOR\n"}
    inc = {"artifact": "WRAPPER art {claim}\n", "morphology": "WRAPPER mor\n", "geometry": "WRAPPER geo\n"}

    def load_persona_set(persona_dir, note_text=None):
        return dict(texts)

    def load_incumbent_set(persona_dir, claim_in_user=False):
        return {r: t.replace("{claim}", "user" if claim_in_user else "absent") for r, t in inc.items()}

    def persona_set_sha16(persona_dir):
        return "5tub5tub5tub5tub"

    async def grade_panel(cand, *, model, persona_set=None, note_text=None, thresholds=None, mode="full",
                          claim=None, trace_dir=None, stamp_dir=None, render="v1", persona_set_noclaim=None, **kw):
        stub.calls.append({"name": cand["name"], "mode": mode, "claim": claim, "render": render,
                           "trace_dir": trace_dir, "stamp_dir": stamp_dir, "tau0": thresholds["tau0"],
                           "noclaim_set": persona_set_noclaim is not None})
        roles = ("artifact", "morphology", "geometry") if mode == "incumbent" else \
            (("advocate",) if mode == "advocate_only" else ("advocate", "artifact", "geometry", "morphology"))
        note = "" if mode == "incumbent" else (note_text or "")
        ps = persona_set_noclaim if (mode == "incumbent" and claim is None and persona_set_noclaim) else persona_set
        shas = {r: _util.sha_text(rte.with_note(ps[r], note)) for r in roles}
        if wrong_sha_role in shas:
            shas[wrong_sha_role] = "0000000000000000"
        if trace_dir:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            (Path(trace_dir) / f"{cand['name']}_panel.jsonl").write_text("{}\n")
        return types.SimpleNamespace(
            S=p_evidence, S_arb=float("nan"), letter=letter, letter_source=thresholds["letter_source"],
            cost_usd=cost, calls=len(roles), parse_failures=[], system_sha16s=shas,
            raw={r: '{"stub": true}' for r in roles}, meta={"cost_by_role": {r: cost / len(roles) for r in roles}},
            p_evidence=p_evidence, parse_ok=True)

    def to_row(res, cand):
        return {"name": cand["name"], "grade_truth": cand.get("grade"), "parse_ok": res.parse_ok,
                "grade_pred": res.letter, "p_lens": res.S, "confidence": res.p_evidence, "S": res.S,
                "S_arb": res.S_arb, "p_evidence": res.p_evidence, "letter_source": res.letter_source,
                "parse_fail_roles": "", "calls": res.calls, "cost_usd": res.cost_usd}

    stub.load_persona_set, stub.load_incumbent_set = load_persona_set, load_incumbent_set
    stub.persona_set_sha16, stub.grade_panel, stub.to_row = persona_set_sha16, grade_panel, to_row
    return stub


def _register(md: Path, row: str, section: str = rte.SECTION) -> None:
    """Insert a row right under the section's separator line (the test_golden_model idiom)."""
    lines = md.read_text().splitlines()
    start = lines.index(f"## {section}")
    sep = next(i for i in range(start, len(lines)) if lines[i].strip() == rte.TRUTH_TABLE_SEP)
    lines.insert(sep + 1, row)
    md.write_text("\n".join(lines) + "\n")


def _live_topup_row(md: Path) -> dict:
    """A world's REGISTRY.md is a copy of the live file, whose "Truth-eval rescores" holds
    exactly ONE row: the registered 2026-08-24 top-up of the 51 empty a1-opus5 holdout rows
    (v2-deploy item 9). Every ledger count in the tests below sits on top of it; a second row,
    or one without the top-up prefix (a genuine rescore), fails here."""
    rows = rge._tables(md.read_text())[rte.RESCORE_SECTION]
    assert len(rows) == 1 and rows[0]["reason"].startswith(rte.TOPUP_PREFIX), rows
    assert len(rows[0]) == 15
    return rows[0]


def _capture(fn, *a, **kw) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **kw)
    return buf.getvalue()


@contextlib.contextmanager
def _stubbed(stub, fake, out_dir=None):
    """Swap in the stub panel, the fake registry, a synthetic PI-comment loader and (when
    given) the outputs root the holdout --out rule and the score-once scan look under."""
    saved = {k: os.environ.pop(k, None) for k in ("LENSJUDGE_BACKEND", "LENSJUDGE_THINKING", "LENSJUDGE_EFFORT")}
    rte._PANEL, rge._REGISTRY, rge._REGISTRY_PATH = stub, fake, None
    rte._PI_LOADER = lambda path: list(SYNTH_PI)
    old_out = config.OUT
    if out_dir is not None:
        config.OUT = Path(out_dir)
    try:
        yield
    finally:
        rte._PANEL, rge._REGISTRY, rge._REGISTRY_PATH, rte._PI_LOADER = None, None, None, None
        config.OUT = old_out
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------ REGISTRY.md + seams
def test_registry_md_truth_sections_and_anchors():
    txt = (GOLDEN / "REGISTRY.md").read_text()
    t = rge._tables(txt)
    for s in ("Truth-eval endpoints", "Truth-eval registered arms", "Truth-eval rescores",
              "Design anchors (PI-derived, design-only, never truth)"):
        assert s in t, s
    # 2026-08-23: the six holdout tuples are now registered (holdout scored once);
    # the table must parse with the TRUTH_COLS header and every row must carry an arm+model.
    arms = t["Truth-eval registered arms"]
    assert len(arms) >= 6, "registered-arms table lost rows"
    assert all(r.get("arm") and r.get("model") for r in arms)
    # score-once: no GENUINE rescore was ever needed. The only rows this table may hold are the
    # registered top-ups (REGISTRY › Deployment rule v2-deploy item 9: NaN rows re-scored, no
    # scored row touched); a real rescore row must fail here loudly
    assert all(r["reason"].startswith(rte.TOPUP_PREFIX) for r in t["Truth-eval rescores"]), t["Truth-eval rescores"]
    # the golden separator is still unique (test_golden_model counts it once); ours appears twice
    assert txt.count(rge.TABLE_SEP + "\n") == 1
    assert txt.count(rte.TRUTH_TABLE_SEP + "\n") == 2
    # the endpoints are the plan's, verbatim pieces
    for s in ("P1 recall of holdout positives at 5%", "P2 letters frozen on design", "P3 fraction of holdout positives at A/B",
              "forbidden-ground rate", "no truth at θ_E>2″"):
        assert s in txt, s
    anchors = t["Design anchors (PI-derived, design-only, never truth)"]
    assert {a["candidate_id"]: a["rank"] for a in anchors} == ANCHOR_IDS
    preds = {a["rank"]: a["written prediction"] for a in anchors}
    assert "A or B" in preds["15"] and "spiral_arm" in preds["13"] and preds["13"].startswith("letter D")
    assert preds["7"] == preds["14"] and "≤ 1" in preds["7"]
    assert "cluster" in preds["16"] and "not D" in preds["16"]
    # the tuple header matches TRUTH_COLS
    hdr = next(l for l in txt.splitlines() if l.startswith("| registered | arm | model | persona_set_sha16"))
    cells = [c.strip() for c in hdr.strip("|").split("|")]
    assert cells == ["registered", *rte.TRUTH_COLS, "note"]
    # anchors are real frame units / ids when the frame exists
    fr = GOLDEN / "frame.csv"
    if fr.exists():
        f = _util.read_pinned(fr, dtype={"candidate_id": str, "alias_ids": str})
        ids = set(f["candidate_id"]) | {a for s in f["alias_ids"].fillna("") for a in s.split("|") if a}
        assert set(ANCHOR_IDS) <= ids


def test_run_golden_eval_seams_default_and_section():
    """The factored helpers keep their golden defaults and work on the truth table; rescore
    rows land in the NAMED section (no longer at EOF, which is now the anchors table)."""
    d = Path(tempfile.mkdtemp(prefix="truth_seams_"))
    try:
        md = d / "REGISTRY.md"
        shutil.copy(GOLDEN / "REGISTRY.md", md)
        pre = _live_topup_row(md)
        t = rte.TruthTuple("a1", "sonnet", "p" * 16, "n" * 16, "advocate:" + "a" * 16, "jwst_v1", "d" * 16,
                           "s" * 16, "none", "off", "default", 1)
        assert not rge.is_registered(md, t, rte.SECTION, rte.TRUTH_COLS)
        _register(md, t.row())
        assert rge.is_registered(md, t, rte.SECTION, rte.TRUTH_COLS)
        assert not rge.is_registered(md, t)                      # not in the golden table
        t2 = rte.TruthTuple(**{**t.__dict__, "k": 3})
        assert not rge.is_registered(md, t2, rte.SECTION, rte.TRUTH_COLS)   # k is part of the tuple
        # the int-vs-str cell rule: '1' == 1, shas compare as strings
        assert rge._cell_same("1", 1) and rge._cell_same("a" * 16, "a" * 16) and not rge._cell_same("b" * 16, "a" * 16)
        outs = rge.out_paths(str(d / "preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet"), "a1", "holdout", 1, "sonnet")
        assert outs[0].name == "preds_truth_a1_sonnet_holdout_k1_r1.parquet"
        # k in the name: the k=1 arm and its k=3 replicate study never share a file
        k3 = rge.out_paths(str(d / "preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet"), "a1", "holdout", 3, "sonnet")
        assert [x.name for x in k3] == [f"preds_truth_a1_sonnet_holdout_k3_r{i}.parquet" for i in (1, 2, 3)]
        assert not set(k3) & set(outs)
        assert rte.TruthTuple(**{**t.__dict__, "k": 3}).run_tag("holdout", 2) == "truth_a1_sonnet_holdout_k3_r2"
        assert rge.check_validate_gate(md, t, outs, section=rte.SECTION, cols=rte.TRUTH_COLS, split="holdout",
                                       prefix=rte.PREFIX, rescore_section=rte.RESCORE_SECTION) is False
        pd.DataFrame({"name": ["x"]}).to_parquet(outs[0], index=False)
        rge.write_meta(outs[0], {"tuple": t.__dict__})
        try:
            rge.check_validate_gate(md, t, outs, section=rte.SECTION, cols=rte.TRUTH_COLS, split="holdout",
                                    prefix=rte.PREFIX, rescore_section=rte.RESCORE_SECTION)
            raise AssertionError("no raise")
        except SystemExit as e:
            assert "already scored on holdout" in str(e)
        # the scan is name-agnostic and recursive: a replicate of the SAME tuple written under
        # another name or into a sub-directory is found (--out peek.parquet, outputs/smoke/);
        # a parquet whose meta records another split, or a golden RunTuple, is not
        alt = d / "peek.parquet"
        pd.DataFrame({"name": ["x"]}).to_parquet(alt, index=False)
        rge.write_meta(alt, {"tuple": t.__dict__, "split": "holdout"})
        (d / "smoke").mkdir()
        sub = d / "smoke" / "preds_truth_a1_sonnet_holdout_k1_r1.parquet"
        pd.DataFrame({"name": ["x"]}).to_parquet(sub, index=False)
        rge.write_meta(sub, {"tuple": t.__dict__, "split": "holdout"})
        des = d / "preds_truth_a1_sonnet_design_k1_r1.parquet"
        pd.DataFrame({"name": ["x"]}).to_parquet(des, index=False)
        rge.write_meta(des, {"tuple": t.__dict__, "split": "design"})
        gold = d / "preds_golden_e1_sonnet_validate_r1.parquet"
        pd.DataFrame({"name": ["x"]}).to_parquet(gold, index=False)
        rge.write_meta(gold, {"tuple": rge.RunTuple("e1", "sonnet", "r" * 16, 0, "s" * 16, "y" * 16, "off", "default").__dict__,
                              "split": "validate"})
        found = set(rge.scored_outputs([d / "nothing.parquet"], t, "holdout", rte.TRUTH_COLS, rte.PREFIX))
        assert found == {outs[0], alt, sub}, found
        assert rge.check_validate_gate(md, t, outs, force=True, reason="trace dir lost", section=rte.SECTION,
                                       cols=rte.TRUTH_COLS, split="holdout", prefix=rte.PREFIX,
                                       rescore_section=rte.RESCORE_SECTION) is True
        tabs = rge._tables(md.read_text())
        rows = tabs["Truth-eval rescores"]                       # the live top-up row, then ours
        assert len(rows) == 2 and rows[0] == pre and rows[1]["reason"] == "trace dir lost"
        assert tabs["Rescores"] == [] and len(tabs["Design anchors (PI-derived, design-only, never truth)"]) == 5
        # the golden default still appends to the golden Rescores table, not to EOF
        g = rge.RunTuple("e1", "sonnet", "r" * 16, 0, "s" * 16, "y" * 16, "off", "default")
        rge.append_rescore(md, g, "golden reason")
        tabs = rge._tables(md.read_text())
        assert len(tabs["Rescores"]) == 1 and tabs["Rescores"][0]["reason"] == "golden reason"
        assert len(tabs["Design anchors (PI-derived, design-only, never truth)"]) == 5
        # check_system_prompt: gated split configurable
        try:
            rge.check_system_prompt("x", d / "absent.txt", "holdout", gated_splits=("holdout",)); raise AssertionError("no raise")
        except SystemExit as e:
            assert "build it first" in str(e)
        rge.check_system_prompt("x", d / "absent.txt", "design", gated_splits=("holdout",))
        rge.check_system_prompt("x", d / "absent.txt", "holdout")        # golden default gates validate only
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------ runner helpers
def test_runner_helpers():
    d = Path(tempfile.mkdtemp(prefix="truth_helpers_"))
    try:
        (d / "ids.csv").write_text("# a comment\nid,role\nA1,x\nA2,y\n")
        assert rte.read_ids_file(d / "ids.csv") == ["A1", "A2"]
        (d / "ids.txt").write_text("A1\n# c\n\nA3\n")
        assert rte.read_ids_file(d / "ids.txt") == ["A1", "A3"]
        assert rte.with_note("abc", "") == "abc" and rte.with_note("abc", "N") == "abcN" and rte.with_note("abcN", "N") == "abcN"
        assert rte.render_version("v1", False) == "jwst_v1" and rte.render_version("v1", True) == "jwst_v1+ctx20"
        assert rte.render_version("v2r", False) == "jwst_v2r"
        df = pd.DataFrame({"name": ["a", "b", "c"], "unit_id": ["u1", "", "u3"], "in_frame": [True, False, False]})
        assert rte.frame_units(df) == ["u1"]
        df = pd.DataFrame({"name": ["a", "b"], "unit_id": ["u1", ""]})
        assert rte.frame_units(df) == ["u1"]
        # thresholds: provisional fallback + tau0 override + model key recorded
        (d / "thr.json").write_text(json.dumps(mvp.THRESHOLDS_DEFAULT))
        thr = rte.load_thresholds(d / "thr.json", "sonnet_api")
        assert thr["letter_source"] == "provisional" and thr["t_A"] == 0.8 and thr["model_key"] == "sonnet_api"
        assert rte.load_thresholds(d / "thr.json", "sonnet_api", 0.2)["tau0"] == 0.2
        # inspector claims: flagged rows get the 400-char evidence, others nothing
        pd.DataFrame({"id": ["a", "b"], "lens_at_center": ["yes", "no"], "quadrant_lens": ["none", "NE"],
                      "evidence": ["e" * 500, "x"], "flagged": [True, False]}).to_csv(d / "ins.csv", index=False)
        out = rte.attach_claims(pd.DataFrame({"name": ["a", "b", "zz"]}), d / "ins.csv")
        assert out["has_claim"].tolist() == [True, False, False] and len(out.at[0, "claimed_evidence"]) == 400
        assert rte.claim_for(out.iloc[0].to_dict(), "inspector") == {"claim_center": "yes", "claim_quadrant": "none",
                                                                      "claimed_evidence": "e" * 400}
        assert rte.claim_for(out.iloc[1].to_dict(), "inspector") is None and rte.claim_for(out.iloc[0].to_dict(), "none") is None
        # select_items
        df = pd.DataFrame({"name": ["a", "b", "c"]})
        assert rte.select_items(df, ["c", "a", "nope"], 0)["name"].tolist() == ["a", "c"]
        assert len(rte.select_items(df, None, 2)) == 2
        try:
            rte.select_items(df, ["nope"], 0); raise AssertionError("no raise")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(d)


def test_truth_runner_claude5_models():
    """The two Claude-5 advocate-only holdout arms are runnable: --model choices carry the
    aliases, model_key routes them to their own thresholds keys, every alias has a budget
    estimate, and the tuple records thinking/effort (adaptive/xhigh for these runs).
    Thresholds: opus5_api is calibrated (2026-08-24, v2-deploy item 1), so a fresh opus5
    holdout run no longer needs --allow-provisional-thresholds -- but its tuple sha is then
    424a8aa9875bacd2, not the a40ae6e201a03e65 the registered opus5 rows carry (they were
    scored under the provisional numbers); sonnet5_api is still null and resolves provisional."""
    assert rte.MODELS == ("sonnet", "opus", "opus5", "sonnet5")
    assert rte.model_key("opus5") == "opus5_api" and rte.model_key("sonnet5") == "sonnet5_api"
    assert set(rte.MODELS) <= set(rte.COST_PER_CALL)
    thr = rte.load_thresholds(rte.THRESHOLDS, rte.model_key("opus5"))
    assert thr["letter_source"] == "opus5_api_calibrated" and thr["model_key"] == "opus5_api"
    assert (thr["t_A"], thr["t_B"], thr["tau0"]) == (0.2, 0.17, 0.15)
    assert rte.thresholds_sha(thr) == "424a8aa9875bacd2"
    thr5 = rte.load_thresholds(rte.THRESHOLDS, rte.model_key("sonnet5"))
    assert thr5["letter_source"] == "provisional" and thr5["model_key"] == "sonnet5_api"
    assert (thr5["t_A"], thr5["t_B"], thr5["tau0"]) == (0.80, 0.50, 0.15)
    assert rte.thresholds_sha(thr5) == "a40ae6e201a03e65"
    reg = rge._tables((GOLDEN / "REGISTRY.md").read_text())["Truth-eval registered arms"]
    assert {r["thresholds_sha16"] for r in reg if r["model"] == "opus5"} == {"a40ae6e201a03e65"}
    t = rte.TruthTuple("a2", "opus5", "p" * 16, "n" * 16, "advocate:" + "a" * 16, "jwst_v1",
                       "d" * 16, "s" * 16, "none", "adaptive", "xhigh", 1, "t" * 16)
    assert t.run_tag("holdout", 1) == "truth_a2_opus5_holdout_k1_r1"
    assert "| adaptive | xhigh |" in t.row()


# ------------------------------------------------------------------ the runner with a stub panel
def test_truth_runner_gate_outputs_registry():
    d, man = make_world()
    pre = _live_topup_row(d / "REGISTRY.md")
    stub, fake = make_stub_panel(p_evidence=0.6, letter="B"), _FakeRegistry()
    OUT_T = str(d / "out" / "preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet")
    common = ["--manifest", str(d / "truth_manifest.csv"), "--splits", str(d / "truth_splits.csv"),
              "--frame", str(d / "frame.csv"), "--registry-md", str(d / "REGISTRY.md"),
              "--out", OUT_T,
              "--banned", str(d / "banned.txt"), "--model", "sonnet", "--concurrency", "2",
              "--thresholds", str(d / "nothing.json"), "--persona-set", str(d)]
    hold = ["--allow-provisional-thresholds"]          # the world has no frozen t_A/t_B
    with _stubbed(stub, fake, out_dir=d / "out"):
        try:
            # ---- holdout, unregistered -> refused before any panel call
            try:
                rte.main(["--arm", "a1", "--split", "holdout"] + common + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "not registered" in str(e) and "Truth-eval registered arms" in str(e) and stub.calls == []
            # ---- design is ungated: writes parquet + votes + meta, marks only frame units
            rte.main(["--arm", "a1", "--split", "design"] + common)
            out = d / "out" / "preds_truth_a1_sonnet_design_k1_r1.parquet"
            df = pd.read_parquet(out)
            assert len(df) == 3 and set(df["name"]) == set(man[man.half == "design"]["name"])
            for c in ("p_lens", "grade_pred", "S", "S_arb", "p_evidence", "letter_source", "truth_class", "is_positive",
                      "half", "unit_id", "layout", "run_tag", "k", "arm", "model", "persona_set_sha16", "note_sha16",
                      "system_sha16s", "render_version", "render_desc_sha16", "splits_sha16", "claim_mode",
                      "thinking", "effort", "rescored", "tau0", "t_A", "t_B", "cost_usd", "parse_fail_roles",
                      "thresholds_sha16"):
                assert c in df.columns, c
            assert (df["p_lens"] == 0.6).all() and (df["grade_pred"] == "B").all() and (df["split"] == "design").all()
            assert (df["run_tag"] == "truth_a1_sonnet_design_k1_r1").all() and (df["letter_source"] == "provisional").all()
            assert (df["render_version"] == "jwst_v1").all() and (df["thinking"] == "off").all()
            votes = pd.read_parquet(d / "out" / "preds_truth_a1_sonnet_design_k1_r1_votes.parquet")
            assert len(votes) == 12 and set(votes["role"]) == {"advocate", "artifact", "geometry", "morphology"}
            assert set(votes.columns) >= {"name", "role", "parse_ok", "raw", "cost_usd", "system_sha16", "k"}
            assert (votes["raw"] == '{"stub": true}').all()
            meta = json.loads(rge.meta_path(out).read_text())
            assert meta["tuple"]["arm"] == "a1" and meta["n"] == 3 and meta["n_frame_units"] == 2
            assert meta["frame_units"] == ["u0001", "u0003"] and meta["n_over_cost_cap_this_run"] == 0
            assert (d / "out" / "traces_truth_a1_sonnet_design_k1_r1").is_dir()
            assert all(c["mode"] == "full" and c["claim"] is None and c["tau0"] == 0.15 for c in stub.calls)
            # registry: seeded from the whole frame, then ONLY this half's frame units marked, kind eval
            assert fake.seeded == [["u0001", "u0003", "u0005"]]
            assert fake.exposed == [(("u0001", "u0003"), "truth_a1_sonnet_design_k1_r1", "eval")]
            # the system sha the stub reports equals the tuple's (computed from the same texts)
            tup = rte.TruthTuple(**meta["tuple"])
            assert tup.system_sha16s.startswith("advocate:") and tup.note_sha16 == _util.sha_text(rte.NOTE_V2.read_text())
            assert tup.thresholds_sha16 == rte.thresholds_sha(meta["thresholds_resolved"]) and meta["out"] == str(out)
            # ---- resume: a second design call grades nothing new
            n = len(stub.calls)
            rte.main(["--arm", "a1", "--split", "design"] + common)
            assert len(stub.calls) == n
            # ---- resume under a DIFFERENT tuple is refused (E3): an interrupted parquet whose
            # rows carry another prompt sha must not be completed as this tuple's record
            rge.meta_path(out).unlink()
            pq = pd.read_parquet(out)
            pq.loc[pq.index[:2], "system_sha16s"] = "advocate:" + "0" * 16
            pq.to_parquet(out, index=False)
            try:
                rte.main(["--arm", "a1", "--split", "design"] + common); raise AssertionError("no raise")
            except SystemExit as e:
                assert "DIFFERENT tuple" in str(e) and "system_sha16s" in str(e)
            out.unlink()
            rte.main(["--arm", "a1", "--split", "design"] + common)        # fresh, complete again
            assert rge.meta_path(out).exists()
            # ---- per-item cost cap: warned and counted (design subsets are allowed)
            rte._PANEL = make_stub_panel(cost=0.5)
            rte.main(["--arm", "a2", "--split", "design", "--limit", "2"] + common)
            m2 = json.loads(rge.meta_path(d / "out" / "preds_truth_a2_sonnet_design_k1_r1.parquet").read_text())
            assert m2["n_over_cost_cap_this_run"] == 2 and m2["n"] == 2
            rte._PANEL = stub
            # ---- --ctx20 only on a1 (the attribution arm attaches nothing)
            try:
                rte.main(["--arm", "attr", "--split", "design", "--ctx20"] + common); raise AssertionError("no raise")
            except SystemExit:
                pass
            # ---- holdout hygiene, each refused before any call: --limit / --ids-file, a non-default
            # --out, a stale lexicon (no holdout ids), provisional thresholds without the flag
            row = _capture(rte.main, ["--arm", "a1", "--split", "holdout", "--print-tuple"] + common).strip()
            assert row.startswith("| ") and row.count("|") == len(rte.TRUTH_COLS) + 3
            _register(d / "REGISTRY.md", row)
            n = len(stub.calls)
            for extra, why in ((["--limit", "2"], "refused on the holdout"), (["--ids-file", str(d / "frame.csv")], "refused on the holdout"),
                               (["--out", str(d / "out" / "peek.parquet")], "default"),
                               (["--out", str(d / "elsewhere" / "preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet")], "not under")):
                try:
                    rte.main(["--arm", "a1", "--split", "holdout"] + common + hold + extra); raise AssertionError("no raise " + why)
                except SystemExit as e:
                    assert why in str(e), (why, str(e))
            try:
                rte.main(["--arm", "a1", "--split", "holdout"] + common[:-6] + ["--banned", str(d / "banned_stale.txt")]
                         + common[-4:] + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "lacks" in str(e) and "holdout ids" in str(e)
            try:
                rte.main(["--arm", "a1", "--split", "holdout"] + common); raise AssertionError("no raise")
            except SystemExit as e:
                assert "no frozen t_A/t_B" in str(e)
            assert len(stub.calls) == n
            # ---- run ONCE on the registered tuple
            rte.main(["--arm", "a1", "--split", "holdout"] + common + hold)
            dfh = pd.read_parquet(d / "out" / "preds_truth_a1_sonnet_holdout_k1_r1.parquet")
            assert len(dfh) == 3 and (dfh["half"] == "holdout").all() and len(stub.calls) == n + 3
            assert fake.exposed[-1] == (("u0005",), "truth_a1_sonnet_holdout_k1_r1", "eval")
            mh = json.loads(rge.meta_path(d / "out" / "preds_truth_a1_sonnet_holdout_k1_r1.parquet").read_text())
            assert mh["allow_provisional_thresholds"] is True and mh["split"] == "holdout"
            # ---- completed -> refused; k=3 is a different tuple -> unregistered; registered k=3
            # writes its own files beside the k=1 record (F4: R2 = a2 --k 3 is runnable)
            try:
                rte.main(["--arm", "a1", "--split", "holdout"] + common + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "already scored on holdout" in str(e)
            try:
                rte.main(["--arm", "a1", "--split", "holdout", "--k", "3"] + common + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "not registered" in str(e)
            row3 = _capture(rte.main, ["--arm", "a1", "--split", "holdout", "--k", "3", "--print-tuple"] + common).strip()
            _register(d / "REGISTRY.md", row3)
            rte.main(["--arm", "a1", "--split", "holdout", "--k", "3"] + common + hold)
            for i in (1, 2, 3):
                assert rge.meta_path(d / "out" / f"preds_truth_a1_sonnet_holdout_k3_r{i}.parquet").exists()
            assert pd.read_parquet(d / "out" / "preds_truth_a1_sonnet_holdout_k1_r1.parquet")["k"].eq(1).all()
            # ---- holdout without a lexicon -> refused before the registry is consulted
            try:
                rte.main(["--arm", "a1", "--split", "holdout"] + common[:-6] + ["--banned", str(d / "absent.txt")]
                         + common[-4:] + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "build it first" in str(e)
            # ---- force-rescore logs into the truth rescores table (not the anchors), archives the
            # previous replicate and re-grades every item
            n = len(stub.calls)
            rte.main(["--arm", "a1", "--split", "holdout", "--force-rescore", "--rescore-reason", "trace dir lost"] + common + hold)
            tabs = rge._tables((d / "REGISTRY.md").read_text())
            rows = tabs["Truth-eval rescores"]                   # the live top-up row, then ours
            assert len(rows) == 2 and rows[0] == pre and rows[1]["reason"] == "trace dir lost"
            assert len(tabs["Design anchors (PI-derived, design-only, never truth)"]) == 5
            assert pd.read_parquet(d / "out" / "preds_truth_a1_sonnet_holdout_k1_r1.parquet")["rescored"].all()
            assert len(stub.calls) == n + 3
            assert len(list((d / "out").glob("preds_truth_a1_sonnet_holdout_k1_r1.parquet.pre_rescore_*"))) == 1
            assert len(list((d / "out").glob("preds_truth_a1_sonnet_holdout_k1_r1.meta.json.pre_rescore_*"))) == 1
            # ---- a0 with inspector claims: incumbent mode, claims only where flagged, no note; the
            # wrapper variant follows the claim (C2) and BOTH variants are in the tuple; a claim
            # that hits the lexicon is blanked before the call (E10)
            ev = ["arc"] * 6
            ev[2] = "arc like J9999999-9999999"                       # names a banned id
            pd.DataFrame({"id": man["name"].tolist(), "lens_at_center": "yes", "quadrant_lens": "none",
                          "evidence": ev, "flagged": [True, False] * 3}).to_csv(d / "ins.csv", index=False)
            stub.calls.clear()
            rte.main(["--arm", "a0", "--split", "design", "--claim-mode", "inspector", "--inspections", str(d / "ins.csv")] + common)
            assert all(c["mode"] == "incumbent" and c["noclaim_set"] for c in stub.calls)
            assert sum(c["claim"] is not None for c in stub.calls) == 1          # one of the two flagged was blanked
            m0 = json.loads(rge.meta_path(d / "out" / "preds_truth_a0_sonnet_design_k1_r1.parquet").read_text())
            assert m0["tuple"]["note_sha16"] == _util.sha_text("") and m0["tuple"]["claim_mode"] == "inspector"
            parts = m0["tuple"]["system_sha16s"].split("+")
            assert [x.split(":")[0] for x in parts] == ["artifact", "morphology", "geometry",
                                                        "artifact@noclaim", "morphology@noclaim", "geometry@noclaim"]
            shas0 = {x.split(":")[0]: x.split(":")[1] for x in parts}
            assert shas0["artifact"] != shas0["artifact@noclaim"]        # the stub's only claim-bearing wrapper
            assert m0["n_claims"] == 1 and m0["n_claims_blanked"] == 1
            v0 = pd.read_parquet(d / "out" / "preds_truth_a0_sonnet_design_k1_r1_votes.parquet")
            art = v0[v0["role"] == "artifact"]["system_sha16"]
            assert set(art) == {shas0["artifact"], shas0["artifact@noclaim"]}
            assert (art == shas0["artifact"]).sum() == 1                  # the one item with a (surviving) claim
            # without claims the tuple carries the three wrappers only
            m0n = _capture(rte.main, ["--arm", "a0", "--split", "design", "--print-tuple"] + common)
            assert "@noclaim" not in m0n
            # claims on a non-incumbent arm are refused at argparse
            try:
                rte.main(["--arm", "a1", "--split", "design", "--claim-mode", "inspector"] + common); raise AssertionError("no raise")
            except SystemExit:
                pass
            # ---- a manifest whose `half` disagrees with the pinned splits is refused (E11)
            sp = _util.read_pinned(d / "truth_splits.csv", dtype=str)
            sp.loc[0, "half"] = "holdout" if sp.loc[0, "half"] == "design" else "design"
            _util.pin(sp, d / "bad_splits.csv")
            try:
                rte.main(["--arm", "a2", "--split", "design"] + common[:2] + ["--splits", str(d / "bad_splits.csv")] + common[4:])
                raise AssertionError("no raise")
            except SystemExit as e:
                assert "disagrees" in str(e)
            # ---- the sha assertion: a panel that sends a different prompt is refused, nothing written
            rte._PANEL = make_stub_panel(wrong_sha_role="geometry")
            try:
                rte.main(["--arm", "attr", "--split", "design"] + common); raise AssertionError("no raise")
            except SystemExit as e:
                assert "REFUSED" in str(e) and "geometry" in str(e)
            assert not (d / "out" / "preds_truth_attr_sonnet_design_k1_r1.parquet").exists()
        finally:
            shutil.rmtree(d)


def _with_failures(stub, fail: set) -> None:
    """Wrap the stub's grade_panel: an item whose name is in `fail` (checked at call time)
    comes back as an advocate parse failure — S NaN, no letter, one call, parse_ok False —
    the shape of the 49 transport failures in the real a1-opus5 holdout parquet."""
    inner = stub.grade_panel

    async def grade_panel(cand, **kw):
        res = await inner(cand, **kw)
        if cand["name"] in fail:
            res.S, res.p_evidence, res.letter, res.parse_ok = float("nan"), float("nan"), None, False
            res.parse_failures, res.calls = ["advocate"], 1
            res.system_sha16s = {"advocate": res.system_sha16s["advocate"]}
            res.raw, res.meta = {"advocate": "garbled"}, {"cost_by_role": {"advocate": 0.01}}
        return res

    stub.grade_panel = grade_panel


def test_flush_replace_and_votes_replace():
    """The --only-nan merges on a synthetic parquet: a re-scored row replaces its NaN
    predecessor in place (row order, column order, dtypes kept; untouched rows byte-equal),
    run_batch._flush would have dropped it; the votes replace drops the failed attempt's
    roles for the re-scored names only."""
    d = Path(tempfile.mkdtemp(prefix="flush_replace_"))
    try:
        out = d / "p.parquet"
        prev = pd.DataFrame({"name": ["a", "b", "c"], "p_lens": [0.3, np.nan, 0.7], "grade_pred": ["C", None, "A"],
                             "parse_ok": [True, False, True], "calls": [4, 1, 5], "rescored": [False] * 3,
                             "rescore_reason": [None] * 3, "k": [1, 1, 1]})
        prev.to_parquet(out, index=False)
        prev = pd.read_parquet(out)
        new = [{"name": "b", "p_lens": 0.9, "grade_pred": "A", "parse_ok": True, "calls": 5, "rescored": False,
                "rescore_reason": "top-up(only-nan): x", "k": 1}]
        prev.to_parquet(d / "plain.parquet", index=False)
        rte.run_batch._flush(list(new), d / "plain.parquet", set())
        assert pd.read_parquet(d / "plain.parquet")["p_lens"].isna().sum() == 1     # the plain flush drops the new row
        rte._flush_replace(list(new), out)
        after = pd.read_parquet(out)
        assert after["name"].tolist() == ["a", "b", "c"] and list(after.columns) == list(prev.columns)
        # every column that held a value keeps its dtype and its untouched rows byte-equal; a
        # column that was ALL null (rescore_reason) necessarily takes the string type once the
        # top-up rows carry a reason — its untouched rows stay null
        cols = [c for c in prev.columns if prev[c].notna().any()]
        assert "rescore_reason" not in cols and all(after[c].dtype == prev[c].dtype for c in cols), after.dtypes
        keep = prev["name"] != "b"
        pd.testing.assert_frame_equal(prev.loc[keep, cols].reset_index(drop=True), after.loc[keep, cols].reset_index(drop=True))
        assert prev.loc[keep, cols].to_parquet(index=False) == after.loc[keep, cols].to_parquet(index=False)
        b = after[after["name"] == "b"].iloc[0]
        assert b["p_lens"] == 0.9 and b["grade_pred"] == "A" and bool(b["parse_ok"]) and b["calls"] == 5
        assert b["rescore_reason"] == "top-up(only-nan): x" and after["rescore_reason"][keep].isna().all()
        # a second flush of the same accumulated rows is idempotent
        rte._flush_replace(list(new), out)
        pd.testing.assert_frame_equal(after, pd.read_parquet(out))
        # votes: replace drops every stored role of the flushed names, then dedupes on name+role
        vout = d / "v.parquet"
        pd.DataFrame({"name": ["a", "b", "b", "c"], "role": ["advocate", "advocate", "artifact", "advocate"],
                      "parse_ok": [True, False, True, True]}).to_parquet(vout, index=False)
        rte._flush_votes([{"name": "b", "role": "advocate", "parse_ok": True}], vout, replace=True)
        v = pd.read_parquet(vout)
        assert v[["name", "role"]].values.tolist() == [["a", "advocate"], ["c", "advocate"], ["b", "advocate"]]
        assert v.loc[v["name"] == "b", "parse_ok"].all()
        rte._flush_votes([{"name": "b", "role": "geometry", "parse_ok": True}], vout)      # plain: append + dedupe
        assert len(pd.read_parquet(vout)) == 4
    finally:
        shutil.rmtree(d)


def test_todo_rows_restricts_the_topup_to_the_stored_nan_names():
    """Item 9: --only-nan scores ONLY rows whose stored S is NaN. check_resume(only_nan=True)
    returns the finite-S names as done, so a manifest name ABSENT from the parquet (a design
    replicate scored with --limit / --ids-file) would otherwise be scored and appended; the
    to-do list is restricted to the NaN target names read before any write."""
    df = pd.DataFrame({"name": ["a", "b", "c"], "x": [1, 2, 3]})
    done = {"a"}                                           # a finite, b NaN, c never in the parquet
    assert [r["name"] for r in rte.todo_rows(df, done)] == ["b", "c"]            # the plain resume
    assert [r["name"] for r in rte.todo_rows(df, done, {"b"})] == ["b"]          # the top-up
    assert rte.todo_rows(df, done, set()) == []
    assert rte.todo_rows(df, {"a", "b", "c"}, {"b"}) == []
    # score() itself refuses only_nan without the target set (never a silent superset)
    import asyncio
    with pytest.raises(ValueError, match="only_names"):
        asyncio.run(rte.score(df, Path("/nonexistent/p.parquet"), Path("/nonexistent/v.parquet"),
                              Path("/nonexistent/t"), arm="a1", model="sonnet", persona_set={}, note="",
                              render="v1", claim_mode="none", thresholds={}, concurrency=1, extra_cols={},
                              expected_shas={}, only_nan=True))


def test_topup_prefix_is_refused_outside_only_nan():
    """A --force-rescore may not borrow the 'top-up(only-nan): ' prefix: the ledger (and
    test_registry_md_truth_sections_and_anchors) tell a top-up from a genuine rescore by it."""
    for args in (["--arm", "a1", "--split", "holdout", "--force-rescore", "--rescore-reason", "top-up(only-nan): x"],
                 ["--arm", "a2", "--split", "design", "--rescore-reason", "  top-up(only-nan):y"]):
        with pytest.raises(SystemExit, match="may not start with"):
            rte.main(args)


def test_truth_runner_only_nan_topup():
    """REGISTRY › Deployment rule v2-deploy item 9: --only-nan re-scores ONLY the NaN rows of a
    complete replicate, merges them in place (scored rows byte-equal), copies parquet / votes /
    meta to *.pre_topup_<UTC> before the first write, re-writes the meta with a topup block,
    logs ONE 'top-up(only-nan): …' row in Truth-eval rescores after the live file's registered
    top-up row (holdout; 15 cells; separators intact), needs --rescore-reason, refuses --force-rescore / --ids-file / --limit and an
    unregistered or incomplete target, and with zero NaN rows exits 0 writing nothing."""
    d, man = make_world(n_design=3, n_holdout=4)
    pre = _live_topup_row(d / "REGISTRY.md")
    stub, fake = make_stub_panel(p_evidence=0.6, letter="B"), _FakeRegistry()
    fail: set = set()
    _with_failures(stub, fail)
    OUT_T = str(d / "out" / "preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet")
    common = ["--manifest", str(d / "truth_manifest.csv"), "--splits", str(d / "truth_splits.csv"),
              "--frame", str(d / "frame.csv"), "--registry-md", str(d / "REGISTRY.md"), "--out", OUT_T,
              "--banned", str(d / "banned.txt"), "--model", "sonnet", "--concurrency", "2",
              "--thresholds", str(d / "nothing.json"), "--persona-set", str(d)]
    hold = ["--allow-provisional-thresholds"]
    H = ["--arm", "a1", "--split", "holdout"]
    TOP = ["--only-nan", "--rescore-reason", " transport  failures "]
    out = d / "out" / "preds_truth_a1_sonnet_holdout_k1_r1.parquet"
    votes_out = out.with_name(out.stem + "_votes.parquet")
    hnames = man.loc[man["half"] == "holdout", "name"].tolist()
    with _stubbed(stub, fake, out_dir=d / "out"):
        try:
            _register(d / "REGISTRY.md", _capture(rte.main, H + ["--print-tuple"] + common).strip())
            # nothing to top up yet: refused before any call
            try:
                rte.main(H + TOP + common + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "does not exist" in str(e) and stub.calls == []
            # ---- the registered run, two of four items fail (S NaN)
            fail.update(hnames[1:3])
            rte.main(H + common + hold)
            prev, prev_bytes, prev_votes = pd.read_parquet(out), out.read_bytes(), pd.read_parquet(votes_out)
            assert len(prev) == 4 and sorted(prev.loc[prev["p_lens"].isna(), "name"]) == sorted(hnames[1:3])
            assert rge.meta_path(out).exists() and json.loads(rge.meta_path(out).read_text())["n_parse_fail_this_run"] == 2
            # the resume sets: plain = every stored name; only_nan = the finite-S names
            extra = {c: prev[c].iloc[0] for c in rte.RESUME_COLS}
            assert rte.check_resume(out, extra) == set(hnames)
            assert rte.check_resume(out, extra, only_nan=True) == set(hnames) - set(hnames[1:3])
            assert rte.nan_names(out) == sorted(hnames[1:3])
            assert rte.topup_reason(" transport  failures ") == "top-up(only-nan): transport failures"
            # ---- refusals, nothing written, no call: no reason; + --force-rescore / --ids-file / --limit;
            # an unregistered tuple (k=3); the plain command (complete record) is still refused
            n = len(stub.calls)
            (d / "ids.txt").write_text(hnames[1] + "\n")
            for extra_args, why in ((["--only-nan"], "rescore-reason"),
                                    (TOP + ["--force-rescore"], "--force-rescore"),
                                    (TOP + ["--ids-file", str(d / "ids.txt")], "--ids-file"),
                                    (TOP + ["--limit", "1"], "--limit")):
                try:
                    rte.main(H + extra_args + common + hold); raise AssertionError("no raise " + why)
                except SystemExit as e:
                    assert "--only-nan" in str(e) and why in str(e), (why, str(e))
            try:
                rte.main(H + ["--k", "3"] + TOP + common + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "not registered" in str(e)
            try:
                rte.main(H + common + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "already scored on holdout" in str(e)
            assert len(stub.calls) == n and out.read_bytes() == prev_bytes
            assert not list((d / "out").glob("*.pre_topup_*"))
            assert rge._tables((d / "REGISTRY.md").read_text())["Truth-eval rescores"] == [pre]
            # ---- the top-up: exactly the two NaN rows are re-scored and merged in place
            fail.clear()
            rte.main(H + TOP + common + hold)
            assert len(stub.calls) == n + 2 and {c["name"] for c in stub.calls[n:]} == set(hnames[1:3])
            after = pd.read_parquet(out)
            assert after["name"].tolist() == prev["name"].tolist() and after["p_lens"].notna().all()
            assert list(after.columns) == list(prev.columns)
            # scored rows byte-equal on every column that held a value (rescore_reason was all-null
            # and takes the string type once the top-up rows carry a reason; its scored rows stay null)
            cols = [c for c in prev.columns if prev[c].notna().any()]
            assert "rescore_reason" not in cols and all(after[c].dtype == prev[c].dtype for c in cols), after.dtypes
            ok = (prev["p_lens"].notna()).values
            pd.testing.assert_frame_equal(prev.loc[ok, cols].reset_index(drop=True), after.loc[ok, cols].reset_index(drop=True))
            assert prev.loc[ok, cols].to_parquet(index=False) == after.loc[ok, cols].to_parquet(index=False)
            new = after[~ok]
            assert (new["p_lens"] == 0.6).all() and (new["grade_pred"] == "B").all() and new["parse_ok"].all()
            assert (new["rescore_reason"] == "top-up(only-nan): transport failures").all() and not after["rescored"].any()
            assert after.loc[ok, "rescore_reason"].isna().all()
            assert (after["run_tag"] == "truth_a1_sonnet_holdout_k1_r1").all()
            # the pre_topup copies: one each, the parquet byte-equal to the pre-run file
            cps = {p.name.split(".pre_topup_")[0]: p for p in (d / "out").glob("*.pre_topup_*")}
            assert set(cps) == {out.name, votes_out.name, rge.meta_path(out).name}, cps
            assert cps[out.name].read_bytes() == prev_bytes
            assert re.fullmatch(r"\d{8}T\d{6}Z", cps[out.name].name.split(".pre_topup_")[1])
            assert not list((d / "out").glob("*.pre_rescore_*"))                            # never archived
            # votes: the failed attempt's rows of the two names are replaced, the others untouched
            va = pd.read_parquet(votes_out)
            for nme in hnames[1:3]:
                assert set(va.loc[va["name"] == nme, "role"]) == {"advocate", "artifact", "geometry", "morphology"}
                assert va.loc[va["name"] == nme, "parse_ok"].all() and (va.loc[va["name"] == nme, "raw"] != "garbled").all()
            keep = ~prev_votes["name"].isin(hnames[1:3])
            pd.testing.assert_frame_equal(prev_votes[keep].reset_index(drop=True),
                                          va[~va["name"].isin(hnames[1:3])].reset_index(drop=True))
            assert not va.duplicated(["name", "role"]).any()
            # meta re-written on completion with the topup block
            m = json.loads(rge.meta_path(out).read_text())
            tp = m["topup"]
            assert tp["n_topup"] == 2 and tp["topup_names_count"] == 2 and tp["n_nan_before"] == 2 and tp["n_nan_after"] == 0
            assert tp["topup_names"] == sorted(hnames[1:3]) and tp["still_nan_names"] == []
            assert tp["reason"] == "top-up(only-nan): transport failures"
            assert set(tp["pre_topup"]) == {"parquet", "votes", "meta"}
            assert {Path(p) for p in tp["pre_topup"].values()} == set(cps.values())
            assert tp["previous"]["n_scored_this_run"] == 4 and tp["previous"]["n_parse_fail_this_run"] == 2
            assert m["topup_history"] == [] and m["rescored"] is False and m["rescore_reason"] is None
            assert m["n_scored_this_run"] == 2 and m["n_parse_fail_this_run"] == 0 and m["n"] == 4
            assert m["tuple"] == json.loads(cps[rge.meta_path(out).name].read_text())["tuple"]
            # the ledger: ONE new row after the live top-up row, 15 cells, the prefix; both table
            # separators intact; anchors untouched
            txt = (d / "REGISTRY.md").read_text()
            tabs = rge._tables(txt)
            rows = tabs["Truth-eval rescores"]
            assert len(rows) == 2 and rows[0] == pre and rows[1]["reason"] == "top-up(only-nan): transport failures"
            assert len(rows[1]) == 15 and rows[1]["arm"] == "a1" and rows[1]["model"] == "sonnet" and rows[1]["k"] == "1"
            line = next(l for l in txt.splitlines() if "top-up(only-nan): transport failures" in l)
            assert line.count("|") == 16                                                   # 15 cells
            assert txt.count(rte.TRUTH_TABLE_SEP + "\n") == 2 and txt.count(rge.TABLE_SEP + "\n") == 1
            assert tabs["Rescores"] == [] and len(tabs["Design anchors (PI-derived, design-only, never truth)"]) == 5
            assert fake.exposed[-1] == (("u0005", "u0007"), "truth_a1_sonnet_holdout_k1_r1", "eval")
            # ---- zero NaN rows: exit 0, a message, nothing written, no call, no ledger row
            n, snap = len(stub.calls), out.read_bytes()
            msg = _capture(rte.main, H + ["--only-nan", "--rescore-reason", "again"] + common + hold)
            assert "nothing to do" in msg and len(stub.calls) == n and out.read_bytes() == snap
            assert len(list((d / "out").glob("*.pre_topup_*"))) == 3
            assert rge._tables((d / "REGISTRY.md").read_text())["Truth-eval rescores"] == rows
            assert json.loads(rge.meta_path(out).read_text()) == m
            # the score-once rule still holds after the top-up
            try:
                rte.main(H + common + hold); raise AssertionError("no raise")
            except SystemExit as e:
                assert "already scored on holdout" in str(e)
            # ---- design: ungated (no registry row), no ledger row; an interrupted replicate is refused
            D = ["--arm", "a2", "--split", "design"]
            dnames = man.loc[man["half"] == "design", "name"].tolist()
            fail.add(dnames[0])
            rte.main(D + common)
            fail.clear()
            outd = d / "out" / "preds_truth_a2_sonnet_design_k1_r1.parquet"
            assert rte.nan_names(outd) == [dnames[0]]
            rte.main(D + ["--only-nan", "--rescore-reason", "design top-up"] + common)
            assert pd.read_parquet(outd)["p_lens"].notna().all()
            assert rge._tables((d / "REGISTRY.md").read_text())["Truth-eval rescores"] == rows
            assert len(list((d / "out").glob("preds_truth_a2_sonnet_design_k1_r1*.pre_topup_*"))) == 3
            assert json.loads(rge.meta_path(outd).read_text())["topup"]["n_topup"] == 1
            fail.add(dnames[1])
            rge.meta_path(outd).unlink()
            try:
                rte.main(D + ["--only-nan", "--rescore-reason", "x"] + common); raise AssertionError("no raise")
            except SystemExit as e:
                assert "interrupted" in str(e)
        finally:
            shutil.rmtree(d)


def test_truth_runner_real_panel_smoke():
    """The runner through the REAL golden/panel.py (persona files, views, schemas) with
    grader_direct.grade_candidate stubbed to return a below-tau0 AdvocateRecord: one call per
    item, S = p_evidence, letter D, the reported sha16s equal the tuple's."""
    from lensjudge.golden import grader_jwst, panel, schemas_panel  # noqa: F401
    from lensjudge.imaging import grader_direct
    from lensjudge.imaging.grader_lean import GradeResult
    if not (rte.PERSONA_SET_DEFAULT / "advocate.md").exists():
        print("  (skipped: persona set absent)")
        return
    d, man = make_world(n_design=2, n_holdout=1)
    fake = _FakeRegistry()
    calls = []

    async def stub(cand, *, model=None, system_prompt=None, content=None, trace_path=None, schema=None, **kw):
        calls.append({"schema": getattr(schema, "__name__", None), "n_images": sum(1 for b in content if b.get("type") == "image"),
                      "system_prompt": system_prompt})
        rec = schema(id="item", persona="advocate",
                     criteria={"source_contrast": 0, "low_surface_brightness": 0, "curvature": 0, "counter_image": 0, "arc_morphology": 0},
                     items=[], scale_class="none", n_red_neighbours_10as=0, bcg_like_halo=False, deflector_is_centre=True,
                     p_evidence=0.05, nothing_because="isolated elliptical", notes="stub")
        return GradeResult(rec, rec.model_dump_json(), cost_usd=0.002, num_turns=1, parse_ok=True,
                           meta={"name": cand.get("name"), "mode": "direct", "n_images": 1, "wall_s": 0.0,
                                 "n_thinking_blocks": 0, "thinking_chars": 0})
    orig = grader_direct.grade_candidate
    grader_direct.grade_candidate = stub
    saved = {k: os.environ.pop(k, None) for k in ("LENSJUDGE_BACKEND", "LENSJUDGE_THINKING", "LENSJUDGE_EFFORT")}
    rge._REGISTRY, rte._PANEL = fake, None
    try:
        common = ["--manifest", str(d / "truth_manifest.csv"), "--splits", str(d / "truth_splits.csv"),
                  "--frame", str(d / "frame.csv"), "--registry-md", str(d / "REGISTRY.md"),
                  "--out", str(d / "out" / "preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet"),
                  "--banned", str(d / "banned.txt"), "--model", "sonnet", "--thresholds", str(GOLDEN / "thresholds_v2.json")]
        rte.main(["--arm", "a2", "--split", "design"] + common)
        df = pd.read_parquet(d / "out" / "preds_truth_a2_sonnet_design_k1_r1.parquet")
        assert len(df) == 2 and len(calls) == 2 and all(c["schema"] == "AdvocateRecord" for c in calls)
        assert np.allclose(df["p_lens"], 0.05) and (df["grade_pred"] == "D").all() and (df["calls"] == 1).all()
        assert (df["system_sha16_advocate"] == _util.sha_text(calls[0]["system_prompt"])).all()
        meta = json.loads(rge.meta_path(d / "out" / "preds_truth_a2_sonnet_design_k1_r1.parquet").read_text())
        assert meta["tuple"]["system_sha16s"] == "advocate:" + df["system_sha16_advocate"].iloc[0]
        assert calls[0]["system_prompt"].endswith(rte.NOTE_V2.read_text())          # note last
        assert meta["tuple"]["persona_set_sha16"] == panel.persona_set_sha16(rte.PERSONA_SET_DEFAULT)
        votes = pd.read_parquet(d / "out" / "preds_truth_a2_sonnet_design_k1_r1_votes.parquet")
        assert len(votes) == 2 and (votes["role"] == "advocate").all() and votes["parse_ok"].all()
        assert fake.exposed == [(("u0001",), "truth_a2_sonnet_design_k1_r1", "eval")]
        traces = sorted(p.name for p in (d / "out" / "traces_truth_a2_sonnet_design_k1_r1").iterdir())
        assert any(n.endswith("_advocate.jsonl") for n in traces) and any(n.endswith("_panel.jsonl") for n in traces)
    finally:
        grader_direct.grade_candidate = orig
        rge._REGISTRY, rte._PANEL = None, None
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d)


# ------------------------------------------------------------------ Nate's drop-in generator
_FILES_RE = re.compile(r'^\s*(\w+): ("(?:[^"\\]|\\.)*"),\s*// sha16 ([0-9a-f]{16})$', re.M)


def extract_briefs(js: str) -> dict:
    """{name: (text, sha16)} re-extracted from the FILES block of verify_workflow_v2.js."""
    return {m.group(1): (json.loads(m.group(2)), m.group(3)) for m in _FILES_RE.finditer(js)}


def test_generator_sha_parity_and_contents():
    d = Path(tempfile.mkdtemp(prefix="vpatch_"))
    try:
        info = mvp.build(d, patch=False)
        js = (d / "scripts" / "verify_workflow_v2.js").read_text()
        briefs = extract_briefs(js)
        assert set(briefs) == set(mvp.PROMPT_FILES)
        for name in mvp.PROMPT_FILES:
            text, sha = briefs[name]
            assert _util.sha_text(text) == sha == info["prompt_sha16s"][name], name
            p = mvp.PERSONA_DIR / f"{name}.md"
            if p.exists():
                assert text == p.read_text(), f"{name}: embedded text differs from the .md"
                assert sha == _util.sha_text(p.read_text())
        # composition = panel.CRITIC_JOIN, so BRIEFS.artifact == load_persona_set's artifact prompt
        from lensjudge.golden import panel
        assert mvp.PART_JOIN == panel.CRITIC_JOIN
        assert f"FILES.critic_common + {json.dumps(mvp.PART_JOIN)} + FILES.critic_artifact" in js
        if (mvp.PERSONA_DIR / "advocate.md").exists():
            sysp = panel.load_persona_set(mvp.PERSONA_DIR)
            assert briefs["critic_common"][0] + mvp.PART_JOIN + briefs["critic_artifact"][0] == sysp["artifact"]
        # no veto sentences; dispatch on the job prefix; his conventions
        for s in mvp.FORBIDDEN_SENTENCES:
            assert s not in js.lower(), s
        for s in ("job.split('_')[0]", "${BASE}/data/verify/${job}.json", "${BASE}/results/verdicts/verify_${job}.jsonl",
                  "phase('Verify')", "await pipeline(", "n_no_opinion", "n_named"):
            assert s in js, s
        # aggregate_v2 byte copy + thresholds
        if mvp.AGGREGATE.exists():
            assert (d / "scripts" / "aggregate_v2.py").read_bytes() == mvp.AGGREGATE.read_bytes()
            assert info["aggregate_sha16"] == _util.sha_file(mvp.AGGREGATE)
        thr = json.loads((d / "scripts" / "thresholds_v2.json").read_text())
        assert "provisional" in thr and "opus_claude_code" in thr
        # calibration ids
        ids = pd.read_csv(d / "scripts" / "calibration_ids.csv", comment="#", dtype=str)
        assert list(ids.columns) == ["id", "role", "source"]
        if mvp.TRUTH_NEGATIVES.exists():
            assert (ids["role"] == "negative").sum() == 200
        if mvp.CONTROL_RECOVERY.exists():
            # design-half COWLS only: the holdout-half COWLS ids are withheld (E9)
            ctl = set(pd.read_csv(mvp.CONTROL_RECOVERY, dtype={"id": str})["id"].astype(str))
            cow = set(ids.loc[ids["role"] == "cowls", "id"])
            assert cow <= ctl and 0 < len(cow) < len(ctl), (len(cow), len(ctl))
            if mvp.TRUTH_SPLITS.exists():
                sp = pd.read_csv(mvp.TRUTH_SPLITS, dtype=str).set_index("candidate_id")["half"]
                assert all(sp.get(i) == "design" for i in cow)
                assert cow == {i for i in ctl if sp.get(i) == "design"}
                assert not (ids["id"].isin({i for i in ctl if sp.get(i) == "holdout"})).any()
                assert f"{len(ctl) - len(cow)} holdout-half COWLS ids withheld" in (d / "scripts" / "calibration_ids.csv").read_text()
        assert ids.groupby("role")["id"].nunique().to_dict() == ids["role"].value_counts().to_dict()   # no dupes
        # README: the five failures with numbers, letter_source, the 12 ctl verdicts, inspector, next step, validation-only
        rd = (d / "README.md").read_text()
        for s in ("1/24 (4%)", "0/31 COWLS", "4.9 / 3.4 / 2.3", "κ 0.46–0.63", "19.6 %", "62 %", "88 %", "0.537", "0.505", "0.463",
                  "sonnet_thresholds_uncalibrated", "## The 12 ctl verdicts", "7632b39", "## Inspector recommendation",
                  "p_evidence", "bluer", "0.5–2.5", "1,674", "validation-only", "## Run order", "1,520",
                  "advocate_e0..e30", "08e_make_refuter_batches.py", "09_rank_report_v2.py --model-key opus_claude_code"):
            assert s in rd, s
        if not info["todo"]:
            assert "TODO at generation time" not in rd
        # every generated python file compiles; the shipped test passes against the shipped aggregator
        for p in list((d / "scripts").glob("*.py")) + [d / "tests" / "test_aggregate_v2.py"]:
            subprocess.run([PY, "-m", "py_compile", str(p)], check=True)
        r = subprocess.run([PY, str(d / "tests" / "test_aggregate_v2.py")], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        # 08d/08e/08f share the measured composite geometry
        for f in ("08d_make_evidence_batches.py", "08e_make_refuter_batches.py", "08f_make_arbitrator_batches.py"):
            t = (d / "scripts" / f).read_text()
            assert "PX, GAP, TH, FOOT_H = 240, 8, 18, 22" in t and "FOOTER_Y = COMPOSITE_H - FOOT_H" in t, f
            assert '"claimed_evidence"' not in t and '"claim_center"' not in t, f   # no inspector fields in any job
        e = (d / "scripts" / "08e_make_refuter_batches.py").read_text()
        # the panel sets come from panel_gloss.json (gray morphology = (a),(d),(e), as in lensjudge)
        assert "GEOMETRY_PANELS = ('b', 'd', 'e')" in e and "'gray': ('a', 'd', 'e')" in e and "'color': ('c', 'd', 'e')" in e
        assert "p_evidence" not in e.split("payload = {", 1)[1].split("}", 1)[0]   # critics never get p_evidence
        assert 'extra={"tau0"' not in e                                           # tau0 stays out of the jobs
        # views per role directory; the finite gate decides the layout like the render did
        for f in ("08d_make_evidence_batches.py", "08e_make_refuter_batches.py", "08f_make_arbitrator_batches.py"):
            t = (d / "scripts" / f).read_text()
            assert 'view_path("full", ' in t and "MIN_FINITE = 0.55" in t and "finite_{ch}" in t, f
        assert 'view_path("geometry", cid)' in e and 'view_path("morphology", cid)' in e
        # blinding text and no "crop with Bash" invitation; gray morphology text says (a)
        assert "BLINDING: read NOTHING but the job file" in js and "You may crop/zoom" not in js
        assert "in the gray layout (a) normal 10\", (d) and (e)" in js
        # README: letter_source table, the divergence list, the withheld holdout COWLS
        for s in ("`provisional`", "sonnet_thresholds_uncalibrated", "opus_claude_code_calibrated",
                  "## What differs from the lensjudge truth evaluation", "Blinding is advisory", "withheld"):
            assert s in rd, s
    finally:
        shutil.rmtree(d)


def test_js_syntax_when_node_available():
    node = shutil.which("node")
    if node is None:
        print("  (skipped: node not installed)")
        return
    d = Path(tempfile.mkdtemp(prefix="vpatch_js_"))
    try:
        mvp.build(d, patch=False)
        src = (d / "scripts" / "verify_workflow_v2.js").read_text()
        # the workflow DSL runs the body inside an async function (top-level await + return),
        # exactly like his verify_workflow.js; wrap it the same way for the parser
        body = re.sub(r"^export const meta = \{.*?\n\}\n", "", src, count=1, flags=re.S)
        (d / "wrapped.js").write_text("async function __wf(args, phase, pipeline, agent) {\n" + body + "\n}\n")
        r = subprocess.run([node, "--check", str(d / "wrapped.js")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        shutil.rmtree(d)


def test_patch_applies_to_temp_copy_of_J():
    need = [J / "scripts" / "verify_workflow.js", J / "scripts" / "09_rank_report.py", J / "scripts" / "util.py"]
    missing = [p for p in need if not p.exists()]
    if missing or shutil.which("git") is None:
        print(f"  (skipped: missing {missing[0] if missing else 'git'})")
        return
    d = Path(tempfile.mkdtemp(prefix="vpatch_apply_"))
    try:
        out = d / "patch"
        info = mvp.build(out, patch=True)
        patch = Path(info["patch"])
        assert patch.exists() and patch.stat().st_size > 1000
        txt = patch.read_text()
        assert txt.count("new file mode") == len(mvp.PATCH_FILES) and "deleted file" not in txt
        assert all(f"+++ b/{dst}" in txt for dst in mvp.PATCH_FILES.values())
        # a temp copy of J (no .git, no bulk data): scripts/, tests/ (absent), top-level files
        jc = d / "jcopy"
        shutil.copytree(J / "scripts", jc / "scripts")
        for p in J.iterdir():
            if p.is_file():
                shutil.copy(p, jc / p.name)
        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        git = lambda *a: subprocess.run(["git", *a], cwd=jc, env=env, capture_output=True, text=True)  # noqa: E731
        assert git("init", "-q").returncode == 0
        assert git("add", "-A").returncode == 0 and git("commit", "-q", "-m", "base").returncode == 0
        r = git("apply", "--check", str(patch))
        assert r.returncode == 0, r.stderr
        r = git("apply", str(patch))
        assert r.returncode == 0, r.stderr
        for dst in mvp.PATCH_FILES.values():
            assert (jc / dst).exists(), dst
        assert (jc / "scripts" / "aggregate_v2.py").read_bytes() == (out / "scripts" / "aggregate_v2.py").read_bytes()
        # nothing of his changed
        assert git("status", "--porcelain", "--untracked-files=no").stdout.strip() == ""
        assert (jc / "scripts" / "verify_workflow.js").read_bytes() == (J / "scripts" / "verify_workflow.js").read_bytes()
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------ 09_rank_report_v2 on synthetic verdicts
def _adv(i, p, items, **c):
    crit = {"source_contrast": 5, "low_surface_brightness": 5, "curvature": 5, "counter_image": 5, "arc_morphology": 5}
    crit.update(c)
    return {"id": i, "persona": "advocate", "criteria": crit, "items": items, "arc_radius_arcsec": None,
            "arc_pa_span_deg": None, "counter_image_pos": None, "centre_of_curvature_offset_arcsec": None,
            "scale_class": "galaxy", "n_red_neighbours_10as": 0, "bcg_like_halo": False, "deflector_is_centre": True,
            "p_evidence": p, "nothing_because": "" if items else "isolated elliptical", "notes": "n"}


def _item(k, r, a, b):
    return {"k": k, "what": "arc", "panel": "d", "r_arcsec": r, "pa_deg_from": a, "pa_deg_to": b,
            "visible_in_direct": True, "criteria": [3]}


def _crit(i, p, alt, r, cov, loc, noop=False):
    return {"id": i, "persona": p, "no_opinion": noop, "no_opinion_reason": "outside_competence" if noop else None,
            "alternative": None if noop else alt, "alternative_desc": alt or "", "location": loc, "accounts_for": cov,
            "leaves_standing": [], "refutation_strength": r, "measured": None, "scale_class": None, "notes": "n"}


def _box(r0, r1, a, b):
    return {"r_arcsec_from": r0, "r_arcsec_to": r1, "pa_deg_from": a, "pa_deg_to": b}


def _leg(i, p, v):
    return {"id": i, "persona": p, "verdict": v, "alternative": "", "notes": "x"}


def make_synthetic_repo(d: Path) -> None:
    """A fake J layout: scripts/ (generated 09 + aggregate + thresholds + a stub util.py with
    BASE), results/inspections.csv, results/results.csv, data/controls.csv and verdict files:
    v2 records (incl. an arbitrator), legacy pass/fail rows in a *_ctl* file, a malformed
    line, a record with unknown keys, a scale_tension above 0.4, and an id absent from
    inspections.csv."""
    mvp.build(d / "gen", patch=False)
    (d / "scripts").mkdir()
    for f in ("09_rank_report_v2.py", "aggregate_v2.py", "thresholds_v2.json", "calibrate_thresholds_v2.py"):
        shutil.copy(d / "gen" / "scripts" / f, d / "scripts" / f)
    # the synthetic repo starts UNCALIBRATED regardless of whether the shipped file has been
    # frozen: the first half of test_rank_report_v2_synthetic pins the provisional behaviour,
    # the second half writes its own sonnet_api numbers to pin the calibrated fallback (C5)
    thr_p = d / "scripts" / "thresholds_v2.json"
    thr0 = json.loads(thr_p.read_text())
    thr0["sonnet_api"] = {"tau0": 0.15, "t_A": None, "t_B": None}
    thr_p.write_text(json.dumps(thr0))
    (d / "scripts" / "util.py").write_text('import os\nBASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))\n')
    (d / "results" / "verdicts").mkdir(parents=True)
    (d / "data").mkdir()
    ids = ["R15", "R13", "NEG1", "NEG2", "CTL1", "LEG1", "UU1"]
    pd.DataFrame({"id": ids, "ra": 1.0, "dec": 2.0, "mag_r": 19.0, "type": "DEV", "proposal": 1727, "field_id": "f",
                  "sw_obs": "x", "lw_obs": "y", "sw_filter": "F150W", "lw_filter": "F277W", "status": "ok",
                  "lens_at_center": "yes", "quadrant_lens": "none", "evidence": "e", "center_galaxy_type": "E",
                  "confidence": [45, 50, 10, 20, 30, 35, 60], "flagged": True, "png": "cutouts/x.jpg",
                  "arc_score": 1}).to_csv(d / "results" / "inspections.csv", index=False)
    pd.DataFrame({"id": ["R15", "R13", "LEG1", "CTL1"], "grade": ["C", "C", "D", "U"]}).to_csv(d / "results" / "results.csv", index=False)
    pd.DataFrame({"id": ["CTL1"], "cowls_code": ["COSJ"]}).to_csv(d / "data" / "controls.csv", index=False)
    V = d / "results" / "verdicts"
    with open(V / "verify_advocate_e0.jsonl", "w") as f:
        for rec in (_adv("R15", 0.8, [_item(1, 3.6, 40, 170), _item(2, 3.2, 220, 250)], curvature=8, counter_image=7, arc_morphology=8),
                    _adv("R13", 0.6, [_item(1, 0.7, 10, 120)]), _adv("NEG1", 0.03, []), _adv("NEG2", 0.2, [_item(1, 1.0, 0, 90)])):
            f.write(json.dumps(rec) + "\n")
        f.write("this is not json\n")
        f.write(json.dumps({"id": "R13", "persona": "advocate", "bogus": 1}) + "\n")
    with open(V / "verify_geometry_v0.jsonl", "w") as f:
        f.write(json.dumps(_crit("R15", "geometry", "scale_tension", 0.4, [1], _box(3.0, 4.2, 30, 180))) + "\n")
        f.write(json.dumps(_crit("R13", "geometry", None, 0.0, [], None, noop=True)) + "\n")
        f.write(json.dumps(_crit("NEG2", "geometry", "scale_tension", 0.9, [1], _box(0.5, 1.5, 0, 90))) + "\n")
    with open(V / "verify_morphology_v0.jsonl", "w") as f:
        f.write(json.dumps(_crit("R13", "morphology", "spiral_arm", 0.9, [1], _box(0.4, 1.2, 0, 140))) + "\n")
        f.write(json.dumps(_crit("R15", "morphology", None, 0.0, [], None, noop=True)) + "\n")
    with open(V / "verify_artifact_v0.jsonl", "w") as f:
        f.write(json.dumps(_crit("R15", "artifact", None, 0.0, [], None)) + "\n")
        f.write(json.dumps(_crit("R13", "artifact", None, 0.0, [], None)) + "\n")
        f.write(json.dumps(_crit("NEG2", "artifact", "psf_wing", 0.7, [1], None)) + "\n")   # no location -> rejected
    with open(V / "verify_arbitrator_a0.jsonl", "w") as f:
        f.write(json.dumps({"id": "R13", "persona": "arbitrator",
                            "rulings": [{"persona": "morphology", "ruling": "upheld", "covers": [1], "why": "bridge"}],
                            "surviving_items": [], "letter_llm": "D", "scale_class_final": "galaxy",
                            "needs_human": False, "rationale": "spiral"}) + "\n")
    with open(V / "verify_artifact_ctl0.jsonl", "w") as f:
        f.write(json.dumps(_leg("CTL1", "artifact", "fail")) + "\n")
        f.write(json.dumps(_leg("ORPHAN", "artifact", "pass")) + "\n")
    with open(V / "verify_geometry_ctl0.jsonl", "w") as f:
        f.write(json.dumps(_leg("CTL1", "geometry", "fail")) + "\n")
    with open(V / "verify_morphology_ctl0.jsonl", "w") as f:
        f.write(json.dumps(_leg("CTL1", "morphology", "fail")) + "\n")
    with open(V / "verify_artifact_0.jsonl", "w") as f:
        f.write(json.dumps(_leg("LEG1", "artifact", "pass")) + "\n")
        f.write(json.dumps(_leg("LEG1", "geometry", "uncertain")) + "\n")


def test_rank_report_v2_synthetic():
    d = Path(tempfile.mkdtemp(prefix="rank_v2_"))
    try:
        make_synthetic_repo(d)
        r = subprocess.run([PY, str(d / "scripts" / "09_rank_report_v2.py"), "--model-key", "opus_claude_code"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "REPORT_V2_DONE" in r.stdout
        res = pd.read_csv(d / "results" / "results_v2.csv", dtype={"id": str}).set_index("id")
        # every verdict id lands in the output, the orphan (not in inspections.csv) included
        assert set(res.index) == {"R15", "R13", "NEG1", "NEG2", "CTL1", "LEG1", "UU1", "ORPHAN"}
        assert not bool(res.at["ORPHAN", "in_inspections"]) and bool(res.at["R15", "in_inspections"])
        # rejected: the malformed line, the unknown-key record, the scale_tension > 0.4, the
        # named alternative without a location box (C9: the same record is a parse failure in lensjudge)
        rej = [json.loads(l) for l in (d / "results" / "verdicts_rejected.jsonl").read_text().splitlines()]
        assert len(rej) == 4 and sorted(x["reason"].split(":")[0] for x in rej) == [
            "alternative 'psf_wing' needs a location box", "json", "scale_tension strength > 0.4", "unknown keys ['bogus']"]
        # scores: rank-15-like above rank-13-like; D rule on 13 and on the nothing-located negative
        assert abs(res.at["R15", "S"] - 0.64) < 1e-9 and abs(res.at["R13", "S"] - 0.06) < 1e-9
        assert res.at["R15", "letter"] == "B" and res.at["R13", "letter"] == "D" and res.at["NEG1", "letter"] == "D"
        assert res.at["NEG2", "letter"] == "C"                          # its only critic was rejected
        assert res.at["R13", "letter_llm"] == "D" and res.at["R13", "alternative_final"] == "spiral_arm"
        # S_arb == S when no arbitrator ran (lensjudge's definition); letter_arb carries the arbitrated guards
        assert abs(res.at["R13", "S_arb"] - 0.06) < 1e-9 and abs(res.at["R15", "S_arb"] - 0.64) < 1e-9
        assert res.at["R13", "letter_arb"] == "D" and res.at["R15", "letter_arb"] == "B"
        # both sonnet_api and opus_claude_code are null in the shipped file -> provisional, labelled so
        assert (res.loc[res["examined"], "letter_source"] == "provisional").all()
        # ranking: examined strictly above U; U ordered by inspector confidence; score_v2 = -1 + conf/100
        order = res.sort_values("rank_v2").index.tolist()
        assert order[:4] == ["R15", "NEG2", "R13", "NEG1"] and order[4:] == ["UU1", "LEG1", "CTL1", "ORPHAN"]
        assert res.at["UU1", "letter"] == "U" and abs(res.at["UU1", "score_v2"] - (-0.4)) < 1e-9
        assert (res.loc[res["examined"], "score_v2"] > res.loc[~res["examined"], "score_v2"].max()).all()
        # the ctl recovery: the v1 export said U, the jsonl say 0/3 = D
        assert res.at["CTL1", "legacy_grade_v1"] == "U" and res.at["CTL1", "legacy_grade_from_jsonl"] == "D"
        assert res.at["LEG1", "legacy_n_pass"] == 1 and res.at["LEG1", "legacy_grade_from_jsonl"] == "C"
        # the other outputs
        top = pd.read_csv(d / "results" / "top100_v2.csv", dtype={"id": str})
        assert top["id"].tolist()[:4] == ["R15", "NEG2", "R13", "NEG1"]
        diff = pd.read_csv(d / "results" / "regrade_diff.csv", dtype={"id": str}).set_index("id")
        assert set(diff.index) == {"R15", "R13", "NEG1", "NEG2"} and diff.at["R15", "delta_ordinal"] == 1 and diff.at["R13", "delta_ordinal"] == -1
        ver = [json.loads(l) for l in (d / "results" / "verifications_v2.jsonl").read_text().splitlines()]
        assert len(ver) == 17 and all({"file", "line", "kind", "record"} <= set(v) for v in ver)
        assert res.at["NEG2", "n_critics"] == 0                       # both of NEG2's critics were rejected
        assert sum(v["kind"] == "legacy" for v in ver) == 6 and any(v["file"].endswith("_ctl0.jsonl") for v in ver)
        rep = (d / "results" / "report_v2.md").read_text()
        assert "letter_source=provisional" in rep and "## Positive controls" in rep and "| 1 | R15 |" in rep
        # after the lensjudge design freeze (sonnet_api written) the same verdicts are lettered with
        # the Sonnet numbers and stamped sonnet_thresholds_uncalibrated (C5: sonnet_api is in the chain)
        thr_path = d / "scripts" / "thresholds_v2.json"
        thr0 = json.loads(thr_path.read_text())
        thr0["sonnet_api"] = {"tau0": 0.15, "t_A": 0.63, "t_B": 0.41}
        thr_path.write_text(json.dumps(thr0))
        r = subprocess.run([PY, str(d / "scripts" / "09_rank_report_v2.py"), "--model-key", "opus_claude_code"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        res2 = pd.read_csv(d / "results" / "results_v2.csv", dtype={"id": str}).set_index("id")
        assert (res2.loc[res2["examined"], "letter_source"] == "sonnet_thresholds_uncalibrated").all()
        assert res2.at["R15", "letter"] == "A"                           # 0.64 >= t_A 0.63 with the A guards met
        thr_path.write_text(json.dumps({**thr0, "sonnet_api": {"tau0": 0.15, "t_A": None, "t_B": None}}))
        # the calibration script on the same results (COWLS id + 2 negatives as the calibration set)
        (d / "scripts" / "calibration_ids.csv").write_text("id,role,source\nNEG1,negative,t\nNEG2,negative,t\nR15,cowls,t\n")
        r = subprocess.run([PY, str(d / "scripts" / "calibrate_thresholds_v2.py"), "--model-key", "opus_claude_code", "--write"],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "fewer than 50" in r.stdout, r.stdout + r.stderr
        thr = json.loads((d / "scripts" / "thresholds_v2.json").read_text())
        assert thr["opus_claude_code"] is None                         # refused to write on 2 negatives
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
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
