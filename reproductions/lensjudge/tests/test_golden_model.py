#!/usr/bin/env python3
"""No-API tests for the golden model-facing path (WP-E): fewshot, grader_jwst, eval manifest,
run_golden_eval's registry gate, audit_traces, logprob_ordinal, and the run_batch /
grader_direct seams.

Pure logic on a synthetic kit (tiny PIL JPEGs in a temp dir) with grader_direct.grade_candidate
monkeypatched to a stub — NO network, NO API spend. Run as a script (no pytest installed):
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_model.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.common import llm_client  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.golden import _util, audit_traces, build_eval_manifest, fewshot, grader_jwst  # noqa: E402
from lensjudge.golden import run_golden_eval as rge  # noqa: E402
from lensjudge.imaging import grader_direct, run_batch  # noqa: E402
from lensjudge.imaging.grader_lean import GradeResult  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"


# ------------------------------------------------------------------ synthetic kit fixture
def _jpeg(path: Path, seed: int) -> None:
    from PIL import Image
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path, format="JPEG")


def make_fixture(n: int = 10):
    """A 10-unit golden world: kit JPEGs, key, labels, frame, splits. Units u0001..u0005
    align, u0006..u0010 validate. Letters cycle A,B,C,D; confidences mostly H."""
    d = Path(tempfile.mkdtemp(prefix="golden_model_"))
    kit = "kit01"
    items = d / "kits" / kit / "items"
    items.mkdir(parents=True)
    key_rows, lab_rows, fr_rows, sp_rows = [], [], [], []
    letters = ["A", "B", "C", "D"]
    for i in range(1, n + 1):
        u, cid = f"u{i:04d}", f"J{1000000 + i * 7919}-{100000 + i * 31}"
        item = f"{i:03d}"
        p = items / f"{item}.jpg"
        _jpeg(p, seed=i)
        sha = _util.sha_file(p)
        layout = "gray_sw_only" if i == 4 else "color"      # u0004 is a gray layout
        letter = letters[(i - 1) % 4]
        score = _util.LETTER_TO_SCORE[letter]
        conf = "M" if i == 5 else "H"                        # u0005 is confidence M
        n_passes = 2 if i in (1, 9) else 1
        stable = (i != 9)                                     # u0009 repeated and unstable
        split = "align" if i <= 5 else "validate"
        key_rows.append(dict(kit_id=kit, item_id=item, presentation_index=i, unit_id=u, candidate_id=cid,
                             **{"pass": 1}, repeat_of_item="", render_sha=sha, layout=layout, stratum="T_U"))
        lab_rows.append(dict(unit_id=u, candidate_id=cid, ra_deg=10.0 + i, dec_deg=-5.0 - i, stratum="T_U",
                             score_1_4=score, grade_letter=letter, confidence_lmh=conf,
                             confidence01=_util.CONF_TO_01[conf],
                             pass2_score_1_4=(score if n_passes == 2 and stable else (score - 1 if n_passes == 2 else np.nan)),
                             pass2_confidence_lmh=("H" if n_passes == 2 else ""), n_passes=n_passes,
                             label_stable=(stable if n_passes == 2 else np.nan), render_sha=sha,
                             grade_scale=_util.GRADE_SCALE, grader_id="XH"))
        fr_rows.append(dict(unit_id=u, candidate_id=cid, alias_ids=("Jalias%d" % i if i == 7 else ""),
                            system_id=i, ra_deg=10.0 + i, dec_deg=-5.0 - i, stratum="T_U", substratum="",
                            # u0002 was never flagged by the pipeline (no pass-count, no confidence);
                            # u0006 is flagged but carries no confidence
                            pipe_grade_passcount=("" if i == 2 else "U"),
                            pipe_inspector_conf=(np.nan if i in (2, 6) else 10.0 * i),
                            pipe_score=0.5, center_galaxy_type="", rank_top100=i, prior_exposure=0,
                            lit_known=False, layout=layout, desi_pool_overlap=(i == 3), proposal=f"prog{i % 2}"))
        sp_rows.append(dict(unit_id=u, system_id=i, split=split, forced=False, stratum="T_U", grade_letter=letter))
    key = pd.DataFrame(key_rows)
    labels = pd.DataFrame(lab_rows)
    frame = pd.DataFrame(fr_rows)
    splits = pd.DataFrame(sp_rows)
    (d / "keys").mkdir()
    _util.pin(key, d / "keys" / f"{kit}_key.csv")
    _util.pin(labels, d / "golden_labels.csv")
    _util.pin(frame, d / "frame.csv")
    _util.pin(splits, d / "splits.csv")
    return d, key, labels, frame, splits


# ------------------------------------------------------------------ stub grader
class _Stub:
    """Replaces grader_direct.grade_candidate; records every call's content/system prompt."""
    def __init__(self, grade="B"):
        self.calls = []
        self.grade = grade

    async def __call__(self, cand, *, model=None, system_prompt=None, content=None, trace_path=None, **kw):
        self.calls.append({"cand": cand, "model": model, "system_prompt": system_prompt,
                           "content": content, "trace_path": trace_path})
        g = ImageGrade(grade=self.grade, p_lens=0.7, confidence=0.8, rationale="stub")
        return GradeResult(g, g.model_dump_json(), cost_usd=0.001, num_turns=1, parse_ok=True,
                           meta={"name": cand.get("name"), "mode": "direct", "n_images": 1,
                                 "wall_s": 0.0, "n_thinking_blocks": 0, "thinking_chars": 0})


def _patch_stub(grade="B"):
    orig = grader_direct.grade_candidate
    stub = _Stub(grade)
    grader_direct.grade_candidate = stub
    return stub, orig


# ------------------------------------------------------------------ llm_client / seams
def test_logprob_ordinal_arithmetic():
    assert abs(llm_client.logprob_ordinal({"A": .5, "B": .5}) - 5 / 6) < 1e-4
    # missing letters renormalise: {A:.2, C:.2} -> (1*.2 + 1/3*.2) / .4 = 2/3
    assert abs(llm_client.logprob_ordinal({"A": .2, "C": .2}) - 2 / 3) < 1e-4
    assert llm_client.logprob_ordinal({"D": 0.9}) == 0.0
    assert llm_client.logprob_ordinal({"A": 0.05}) == 1.0
    assert llm_client.logprob_ordinal(None) is None and llm_client.logprob_ordinal({}) is None
    assert llm_client.logprob_ordinal({"X": 0.5}) is None          # no known letter mass
    assert llm_client.logprob_ordinal({"A": .5, "B": .5}, w={"A": 1.0, "B": 0.0}) == 0.5
    assert llm_client.gp_coverage({"A": .5, "B": .3}) == 0.8
    assert llm_client.gp_coverage({"A": .9, "B": .3}) == 1.0
    assert llm_client.gp_coverage(None) is None


def test_grader_direct_constants_and_header():
    assert grader_direct.FEWSHOT_LEAD.startswith("Below are REFERENCE EXAMPLES")
    assert grader_direct.FEWSHOT_TRAIL.startswith("END OF REFERENCE EXAMPLES")
    from lensjudge.common import fetch, render
    orig = fetch.get_cube, render.render_views, render.png_b64
    fetch.get_cube = lambda **k: object()
    render.render_views = lambda cube, views: {v: object() for v in views}
    render.png_b64 = lambda img: "B64"
    try:
        hdr = lambda ex: grader_direct._example_blocks(ex, ("full",))[0]["text"]  # noqa: E731
        assert hdr({"name": "x", "label": "LENS", "grade": "A"}) == "REFERENCE EXAMPLE -- LENS (consensus grade A)"
        assert hdr({"name": "x", "label": "LENS", "grade": "A", "grade_source": "expert"}) == \
            "REFERENCE EXAMPLE -- LENS (expert grade A)"
        assert "(consensus grade A)" in hdr({"name": "x", "label": "LENS", "grade": "A", "grade_source": float("nan")})
    finally:
        fetch.get_cube, render.render_views, render.png_b64 = orig


def test_grader_direct_claude5_wiring():
    """Claude 5 advocate arms: alias map, list-price cost accounting, larger thinking cap."""
    assert grader_direct._MODEL_IDS["opus5"] == "claude-opus-5"
    assert grader_direct._MODEL_IDS["sonnet5"] == "claude-sonnet-5"
    # cost accounting at LIST price (Sonnet 5's $2/$10 intro through 2026-08-31 is not assumed)
    assert grader_direct._PRICE["claude-opus-5"] == (5.0, 25.0, 0.50, 6.25)
    assert grader_direct._PRICE["claude-sonnet-5"] == (3.0, 15.0, 0.30, 3.75)
    # both ids are in the set the thinking branch reads for the 16384 max_tokens cap
    assert grader_direct._CLAUDE5_IDS == {"claude-opus-5", "claude-sonnet-5"}
    # every alias is priced: a Claude 5 run can never fall back to the Sonnet 4.6 default
    assert {grader_direct._MODEL_IDS[a] for a in ("opus5", "sonnet5")} <= set(grader_direct._PRICE)

    class _U:
        input_tokens, output_tokens = 1_000_000, 100_000
        cache_read_input_tokens = cache_creation_input_tokens = 0
    assert abs(grader_direct._cost("claude-opus-5", _U()) - 7.5) < 1e-9      # 5.0 + 2.5
    assert abs(grader_direct._cost("claude-sonnet-5", _U()) - 4.5) < 1e-9    # 3.0 + 1.5


def test_run_batch_seams():
    assert run_batch._grader("jwst") is grader_jwst
    g = GradeResult(None, "", meta={"s_exp": 0.61, "grade_probs": {"A": .5, "B": .4}, "p_lens_logprob": 0.9})
    row = run_batch._row_dict(g, {"name": "n"})
    assert row["s_exp"] == 0.61 and row["gp_A"] == 0.5 and row["p_lens_logprob"] == 0.9
    assert "s_exp" in run_batch._row_dict(GradeResult(None, "", meta={}), {"name": "n"})
    # golden rows (survey_key=jwst) are refused by every non-jwst mode before any call
    import asyncio
    d = Path(tempfile.mkdtemp(prefix="golden_rb_"))
    try:
        df = pd.DataFrame({"name": ["J1-1"], "survey_key": ["jwst"], "catalog": ["jwst"], "grade": ["A"]})
        for mode in ("direct", "lean", "escalate", "matched"):
            try:
                asyncio.run(run_batch.run(df, d / "p.parquet", 1, None, mode))
                raise AssertionError(f"mode {mode} accepted jwst rows")
            except SystemExit as e:
                assert "--mode jwst" in str(e)
        assert not (d / "p.parquet").exists()
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------ grader_jwst
def test_jwst_content_order_and_media_type():
    d, key, labels, frame, splits = make_fixture()
    try:
        p = d / "kits" / "kit01" / "items" / "001.jpg"
        cand = {"name": labels.candidate_id[0], "image_path": str(p), "render_sha": labels.render_sha[0]}
        c = grader_jwst.jwst_content(cand)
        assert [b["type"] for b in c] == ["text", "image"]
        assert c[0]["text"].endswith("Respond with ONLY the JSON object for the required schema.")
        assert c[1]["source"]["media_type"] == "image/jpeg" and c[1]["source"]["type"] == "base64"
        assert base64.b64decode(c[1]["source"]["data"]) == p.read_bytes()
        # nothing identifying in the text: no id, no coordinates
        assert labels.candidate_id[0] not in c[0]["text"] and "RA=" not in c[0]["text"]
        # sha mismatch refuses; missing composite -> None
        try:
            grader_jwst.jwst_content({**cand, "render_sha": "0" * 16}); raise AssertionError("no raise")
        except ValueError:
            pass
        assert grader_jwst.jwst_content({"name": "x", "image_path": str(d / "nope.jpg")}) is None
        # key-based resolution (no image_path) via the fixture keys dir
        grader_jwst._KEY_CACHE = None
        orig_keys, orig_kits = grader_jwst.KEYS_DIR, grader_jwst.KITS_DIR
        grader_jwst.KITS_DIR = d / "kits"
        try:
            grader_jwst._KEY_CACHE = grader_jwst._keys(d / "keys")
            assert grader_jwst.resolve_image_path({"name": labels.candidate_id[1]}) == d / "kits/kit01/items/002.jpg"
            assert grader_jwst.resolve_image_path({"unit_id": "u0003"}) == d / "kits/kit01/items/003.jpg"
        finally:
            grader_jwst.KEYS_DIR, grader_jwst.KITS_DIR, grader_jwst._KEY_CACHE = orig_keys, orig_kits, None
    finally:
        shutil.rmtree(d)


def test_grade_candidate_audit_event_and_meta():
    d, key, labels, frame, splits = make_fixture()
    stub, orig = _patch_stub("A")
    try:
        blocks, ex_ids = fewshot.build_exemplar_blocks(labels, key, n_per_grade=1, eligible_units=["u0001", "u0002", "u0003"],
                                                       kits_dir=d / "kits")
        assert ex_ids == ["u0001", "u0002", "u0003"]
        cand = {"name": labels.candidate_id[7], "unit_id": "u0008",
                "image_path": str(d / "kits/kit01/items/008.jpg"), "render_sha": labels.render_sha[7]}
        tp = d / "traces" / "t.jsonl"
        res = asyncio.run(grader_jwst.grade_candidate(cand, model="sonnet", trace_path=str(tp),
                                                      fewshot_blocks=blocks, exemplar_unit_ids=ex_ids))
        assert res.parse_ok and res.grade.grade == "A"
        assert res.meta["mode"] == "jwst" and res.meta["n_exemplars"] == 3
        assert res.meta["render_sha"] == labels.render_sha[7]
        assert res.meta["exemplar_unit_ids"] == "u0001|u0002|u0003"
        # content order: exemplar blocks then [gloss text, candidate image]
        call = stub.calls[-1]
        assert call["content"][:len(blocks)] == blocks
        assert [b["type"] for b in call["content"][len(blocks):]] == ["text", "image"]
        assert call["system_prompt"] == grader_jwst.DIRECT_SYS_JWST
        assert call["system_prompt"].startswith("You are an expert astronomer") and "NIRCam" in call["system_prompt"]
        # a caller rubric (run_batch --rubric, E3) always gets the note appended, once
        for given in ("E3 RUBRIC", "E3 RUBRIC" + grader_jwst.JWST_NOTE):
            asyncio.run(grader_jwst.grade_candidate(cand, model="sonnet", system_prompt=given))
            assert stub.calls[-1]["system_prompt"] == "E3 RUBRIC" + grader_jwst.JWST_NOTE
        # audit event is the FIRST record in the trace, before any request event
        recs = [json.loads(l) for l in tp.read_text().splitlines()]
        ev = recs[0]
        assert ev["event"] == "golden_content_audit"
        assert ev["n_images"] == 4 and ev["n_exemplars"] == 3
        assert ev["exemplar_image_shas"] == [labels.render_sha[0], labels.render_sha[1], labels.render_sha[2]]
        assert ev["candidate_image_sha"] == labels.render_sha[7]
        assert ev["exemplar_unit_ids"] == ex_ids and ev["unit_id"] == "u0008"
        assert ev["system_sha16"] == _util.sha_text(grader_jwst.DIRECT_SYS_JWST)
        tb = ev["text_blocks"]
        assert all(set(t) >= {"sha16", "head", "n_chars"} and len(t["head"]) <= 200 for t in tb)
        assert tb[0]["sha16"] == _util.sha_text(fewshot.FEWSHOT_LEAD)
        assert tb[-1]["sha16"] == _util.sha_text(grader_jwst.PANEL_GLOSS)
        assert all(i["media_type"] == "image/jpeg" for i in ev["images"])
        # missing composite -> clean error result, no call
        n0 = len(stub.calls)
        r2 = asyncio.run(grader_jwst.grade_candidate({"name": "x", "image_path": str(d / "nope.jpg")}))
        assert r2.error == "no composite" and len(stub.calls) == n0
        # env guard
        os.environ["LENSJUDGE_FEWSHOT_MANIFEST"] = "/tmp/x.csv"
        try:
            asyncio.run(grader_jwst.grade_candidate(cand)); raise AssertionError("no raise")
        except RuntimeError:
            pass
        finally:
            os.environ.pop("LENSJUDGE_FEWSHOT_MANIFEST", None)
    finally:
        grader_direct.grade_candidate = orig
        shutil.rmtree(d)


# ------------------------------------------------------------------ fewshot
def test_fewshot_eligibility_and_determinism():
    d, key, labels, frame, splits = make_fixture()
    try:
        align = splits.loc[splits.split == "align", "unit_id"].tolist()      # u0001..u0005
        b1, ids1 = fewshot.build_exemplar_blocks(labels, key, n_per_grade=3, eligible_units=align, kits_dir=d / "kits")
        b2, ids2 = fewshot.build_exemplar_blocks(labels, key, n_per_grade=3, eligible_units=align, kits_dir=d / "kits")
        assert ids1 == ids2 and b1 == b2
        # u0004 gray layout, u0005 confidence M -> excluded; u0001 (repeated, stable) stays
        assert ids1 == ["u0001", "u0002", "u0003"]
        txt = [b["text"] for b in b1 if b["type"] == "text"]
        assert txt[0] == grader_direct.FEWSHOT_LEAD and txt[-1] == grader_direct.FEWSHOT_TRAIL
        assert txt[1] == "REFERENCE EXAMPLE -- LENS (expert score 4/4 = A, confidence H)"
        assert txt[3] == "REFERENCE EXAMPLE -- LENS (expert score 3/4 = B, confidence H)"
        assert txt[5] == "REFERENCE EXAMPLE -- POSSIBLE LENS (expert score 2/4 = C, confidence H)"
        assert txt[2] == txt[4] == "[composite]"
        imgs = [b for b in b1 if b["type"] == "image"]
        assert len(imgs) == 3 and all(i["source"]["media_type"] == "image/jpeg" for i in imgs)
        assert base64.b64decode(imgs[0]["source"]["data"]) == (d / "kits/kit01/items/001.jpg").read_bytes()
        # block order per exemplar: header, [composite], image
        assert [b["type"] for b in b1] == ["text"] + ["text", "text", "image"] * 3 + ["text"]
        # no split restriction: D available from validate (u0008 D H; u0009 unstable excluded; u0010 B)
        _, ids_all = fewshot.build_exemplar_blocks(labels, key, n_per_grade=3, kits_dir=d / "kits")
        assert "u0008" in ids_all and "u0009" not in ids_all and "u0004" not in ids_all and "u0005" not in ids_all
        # n = min(n_per_grade, available): n_per_grade=1 -> one per letter present
        _, ids_1 = fewshot.build_exemplar_blocks(labels, key, n_per_grade=1, kits_dir=d / "kits")
        assert len(ids_1) == 4
        # validate-only eligible set with only u0006.. -> letters B,C,D present
        _, ids_v = fewshot.build_exemplar_blocks(labels, key, n_per_grade=3, eligible_units=["u0006", "u0007", "u0008"],
                                                 kits_dir=d / "kits")
        assert ids_v == ["u0006", "u0007", "u0008"]
        # nothing eligible -> empty
        assert fewshot.build_exemplar_blocks(labels, key, eligible_units=["u0004", "u0005"], kits_dir=d / "kits") == ([], [])
        # dict lookup works too (layout then from labels' absence -> color)
        lk = {r.unit_id: str(d / "kits/kit01/items" / f"{r.item_id}.jpg") for r in key.itertuples()}
        _, ids_d = fewshot.build_exemplar_blocks(labels, lk, n_per_grade=1, eligible_units=["u0002"], kits_dir=d / "kits")
        assert ids_d == ["u0002"]
        # a different seed only changes the tie-break hash salt, never eligibility
        _, ids_s = fewshot.build_exemplar_blocks(labels, key, n_per_grade=3, seed=7, eligible_units=align, kits_dir=d / "kits")
        assert sorted(ids_s) == sorted(ids1)
    finally:
        shutil.rmtree(d)


def test_fewshot_embargo_env_guard_and_sha():
    d, key, labels, frame, splits = make_fixture()
    try:
        # free-text column -> embargo raise
        bad = labels.copy(); bad["note"] = ""; bad.loc[0, "note"] = "free text that must not ride along"
        try:
            fewshot.build_exemplar_blocks(bad, key, kits_dir=d / "kits"); raise AssertionError("no raise")
        except RuntimeError as e:
            assert "embargo" in str(e)
        # embargo=False tolerates the column but still emits only the template
        blocks, _ = fewshot.build_exemplar_blocks(bad, key, n_per_grade=1, embargo=False, kits_dir=d / "kits")
        fewshot.check_embargo(blocks)
        # a tampered text block is caught by check_embargo
        blocks[1] = {"type": "text", "text": blocks[1]["text"] + ": some appended free text"}
        try:
            fewshot.check_embargo(blocks); raise AssertionError("no raise")
        except RuntimeError:
            pass
        # env-var guard
        os.environ["LENSJUDGE_FEWSHOT_MANIFEST"] = "/tmp/fs.csv"
        try:
            fewshot.build_exemplar_blocks(labels, key, kits_dir=d / "kits"); raise AssertionError("no raise")
        except RuntimeError as e:
            assert "LENSJUDGE_FEWSHOT_MANIFEST" in str(e)
        finally:
            os.environ.pop("LENSJUDGE_FEWSHOT_MANIFEST", None)
        # served JPEG differing from the graded bytes -> refuse
        _jpeg(d / "kits/kit01/items/002.jpg", seed=999)
        try:
            fewshot.build_exemplar_blocks(labels, key, n_per_grade=1, eligible_units=["u0002"], kits_dir=d / "kits")
            raise AssertionError("no raise")
        except ValueError as e:
            assert "render_sha" in str(e)
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------ eval manifest
def test_eval_manifest_columns_and_rules():
    d, key, labels, frame, splits = make_fixture()
    try:
        man = build_eval_manifest.build(labels, frame, splits, key, d / "kits", "all")
        assert list(man.columns)[:11] == build_eval_manifest.EVAL_COLS
        assert list(man.columns)[11:] == build_eval_manifest.EXTRA_COLS
        assert len(man) == 10 and (man.survey_key == "jwst").all() and (man.source == "golden_huang").all()
        m = man.set_index("unit_id")
        assert m.loc["u0001", "grade_truth"] == "A" and m.loc["u0003", "grade_truth"] == "C" and m.loc["u0004", "grade_truth"] == "D"
        # binary_label is the registered primary endpoint (score >= 3); the >= 2 view is extra
        assert m.loc["u0002", "binary_label"] == "lens" and m.loc["u0003", "binary_label"] == "nonlens"
        assert (man.binary_label == np.where(man.grade_truth.isin(["A", "B"]), "lens", "nonlens")).all()
        assert (man.binary_label_ge2 == np.where(man.grade_truth.isin(["A", "B", "C"]), "lens", "nonlens")).all()
        assert m.loc["u0003", "leak"] == "desi_train" and m.loc["u0001", "leak"] == "no"
        assert man.p_meta.isna().all() and (man.tractor_type == "").all()
        assert m.loc["u0001", "region"] == "prog1" and m.loc["u0002", "region"] == "prog0"
        # incumbent: flagged -> conf/100; never flagged -> 0 (ranked below every flagged row,
        # NOT dropped); flagged without a confidence -> NaN
        assert abs(m.loc["u0001", "p_pipeline"] - 0.1) < 1e-9 and m.loc["u0002", "p_pipeline"] == 0.0
        assert np.isnan(m.loc["u0006", "p_pipeline"])
        assert not m.loc["u0002", "pipe_flagged"] and m.loc["u0001", "pipe_flagged"] and m.loc["u0006", "pipe_flagged"]
        assert m.loc["u0006", "split"] == "validate" and m.loc["u0001", "split"] == "align"
        assert m.loc["u0001", "image_path"].endswith("kits/kit01/items/001.jpg")
        assert m.loc["u0001", "render_sha"] == labels.render_sha[0]
        assert m.loc["u0001", "name"] == labels.candidate_id[0] and abs(m.loc["u0001", "ra"] - 11.0) < 1e-9
        # split filter
        mv = build_eval_manifest.build(labels, frame, splits, key, d / "kits", "validate")
        assert set(mv.unit_id) == {f"u{i:04d}" for i in range(6, 11)}
        # the pinned file round-trips through run_batch's manifest convention
        out = d / "man.csv"
        _util.pin(man, out)
        df = rge.load_manifest(out, "align")
        assert "grade" in df.columns and len(df) == 5 and (df.catalog == "jwst").all()
        # lensbench_gate.merge_manifest / score.py read these columns unchanged
        from lensjudge.eval.lensbench_gate import merge_manifest
        preds = pd.DataFrame({"name": man.name, "p_lens": 0.5, "parse_ok": True})
        mm = merge_manifest(preds, pd.read_csv(out))
        assert len(mm) == 10 and mm.is_lens.sum() == 6            # A/B = score >= 3
    finally:
        shutil.rmtree(d)


# ------------------------------------------------------------------ registry gate
def _register(md: Path, *rows: str) -> None:
    """Insert registered-arm rows right under the table's separator line."""
    txt = md.read_text()
    sep = rge.TABLE_SEP + "\n"
    assert txt.count(sep) == 1, "REGISTRY.md separator drifted from run_golden_eval.TABLE_SEP"
    md.write_text(txt.replace(sep, sep + "".join(r + "\n" for r in rows), 1))


def test_registry_md_parse_and_gate():
    d = Path(tempfile.mkdtemp(prefix="golden_reg_"))
    try:
        md = d / "REGISTRY.md"
        shutil.copy(GOLDEN / "REGISTRY.md", md)
        assert rge.registered_rows(md) == []                 # shipped file has an empty table
        t = rge.RunTuple("e2", "sonnet", "abcd1234abcd1234", 3, "ffff0000ffff0000", "5y5t3m5ha5y5t3m5", "off", "default")
        outs = rge.out_paths(str(d / "preds_golden_{arm}_{model}_{split}_r{k}.parquet"), "e2", "validate", 2, "sonnet")
        assert [p.name for p in outs] == ["preds_golden_e2_sonnet_validate_r1.parquet", "preds_golden_e2_sonnet_validate_r2.parquet"]
        # a served open-model id is made path-safe; the run tag carries the model too
        assert rge.out_paths("x/{arm}_{model}_{k}.parquet", "e1", "validate", 1, "org/Qwen-27B")[0].name == "e1_org_Qwen-27B_1.parquet"
        assert t.run_tag("validate", 2) == "golden_e2_sonnet_validate_r2"
        # unregistered -> refused
        try:
            rge.check_validate_gate(md, t, outs); raise AssertionError("no raise")
        except SystemExit as e:
            assert "not registered" in str(e)
        # register (sibling rows differing in n_exemplars / thinking / system sha must not match)
        other = rge.RunTuple("e2", "sonnet", "abcd1234abcd1234", 2, "ffff0000ffff0000", "5y5t3m5ha5y5t3m5", "off", "default")
        _register(md, other.row("rubric_imaging_v2.md"), t.row("rubric_imaging_v2.md"))
        assert rge.is_registered(md, t) and rge.is_registered(md, other)
        assert not rge.is_registered(md, rge.RunTuple("e1", "sonnet", "abcd1234abcd1234", 0, "ffff0000ffff0000", "5y5t3m5ha5y5t3m5", "off", "default"))
        assert not rge.is_registered(md, rge.RunTuple("e2", "sonnet", "abcd1234abcd1234", 3, "ffff0000ffff0000", "5y5t3m5ha5y5t3m5", "adaptive", "default"))
        assert not rge.is_registered(md, rge.RunTuple("e2", "sonnet", "abcd1234abcd1234", 3, "ffff0000ffff0000", "0000000000000000", "off", "default"))
        assert not rge.is_registered(md, rge.RunTuple("e2", "sonnet", "abcd1234abcd1234", 3, "ffff0000ffff0000", "5y5t3m5ha5y5t3m5", "off", "high"))
        assert rge.check_validate_gate(md, t, outs) is False
        # an interrupted replicate (parquet WITHOUT .meta.json) is resumable, not "scored"
        pd.DataFrame({"name": ["x"]}).to_parquet(outs[0], index=False)
        assert rge.check_validate_gate(md, t, outs) is False
        # a completed one (parquet + meta recording this tuple) -> refused
        rge.write_meta(outs[0], {"tuple": {c: getattr(t, c) for c in rge.REGISTERED_COLS}})
        try:
            rge.check_validate_gate(md, t, outs); raise AssertionError("no raise")
        except SystemExit as e:
            assert "already scored" in str(e)
        # the target path holding ANOTHER tuple is a collision, not a rescore
        rge.write_meta(outs[0], {"tuple": {**{c: getattr(other, c) for c in rge.REGISTERED_COLS}}})
        try:
            rge.check_validate_gate(md, t, outs); raise AssertionError("no raise")
        except SystemExit as e:
            assert "DIFFERENT" in str(e)
        outs[0].unlink(); rge.meta_path(outs[0]).unlink()
        # already scored under a DIFFERENT file name but the same tuple (meta sidecar) -> refused
        alt = d / "preds_golden_e2x_validate_r1.parquet"
        pd.DataFrame({"name": ["x"]}).to_parquet(alt, index=False)
        rge.write_meta(alt, {"tuple": {c: getattr(t, c) for c in rge.REGISTERED_COLS}})
        try:
            rge.check_validate_gate(md, t, outs); raise AssertionError("no raise")
        except SystemExit as e:
            assert "already scored" in str(e)
        # force without reason -> refused; with reason -> rescore row appended
        try:
            rge.check_validate_gate(md, t, outs, force=True); raise AssertionError("no raise")
        except SystemExit as e:
            assert "rescore-reason" in str(e)
        assert rge.check_validate_gate(md, t, outs, force=True, reason="trace dir lost") is True
        resc = rge._tables(md.read_text())["Rescores"]
        assert len(resc) == 1 and resc[0]["reason"] == "trace dir lost" and resc[0]["arm"] == "e2"
        assert resc[0]["system_sha16"] == t.system_sha16 and resc[0]["thinking"] == "off"
        assert rge.is_registered(md, t)                     # still registered after the append
        # the system-prompt embargo check: a validate run needs a lexicon; a hit refuses
        try:
            rge.check_system_prompt("rubric text", d / "absent.txt", "validate"); raise AssertionError("no raise")
        except SystemExit as e:
            assert "build it first" in str(e)
        rge.check_system_prompt("rubric text", d / "absent.txt", "align")         # only a note
        (d / "lex.txt").write_text("J1234567-1234567\nthis exact sentence must never appear\n")
        rge.check_system_prompt("a rubric mentioning nothing banned", d / "lex.txt", "validate")
        for bad in ("see J1234567-1234567 for an example", "This EXACT sentence must never appear."):
            try:
                rge.check_system_prompt(bad, d / "lex.txt", "align"); raise AssertionError("no raise")
            except SystemExit as e:
                assert "banned string" in str(e)
    finally:
        shutil.rmtree(d)


class _FakeRegistry:
    def __init__(self):
        self.exposed, self.asserted = [], []

    def mark_exposed(self, unit_ids, run_tag, kind):
        self.exposed.append((tuple(unit_ids), run_tag, kind))

    def assert_unexposed(self, unit_ids, kinds=("fewshot", "sft")):
        self.asserted.append((tuple(unit_ids), tuple(kinds)))


def test_run_golden_eval_end_to_end_with_stub():
    """main() on the synthetic kit with grader_direct stubbed: e2/align leaves exemplars out,
    writes r1..rk parquets + meta; validate refuses unregistered, runs once registered, refuses
    twice; audit_traces passes on the traces and catches planted violations."""
    d, key, labels, frame, splits = make_fixture()
    stub, orig = _patch_stub("B")
    saved_env = {k: os.environ.pop(k, None) for k in ("LENSJUDGE_BACKEND", "LENSJUDGE_FEWSHOT_MANIFEST")}
    fake = _FakeRegistry()
    rge._REGISTRY = fake
    try:
        man = build_eval_manifest.build(labels, frame, splits, key, d / "kits", "all")
        _util.pin(man, d / "man.csv")
        md = d / "REGISTRY.md"
        shutil.copy(GOLDEN / "REGISTRY.md", md)
        lex = audit_traces.build_lexicon(splits, frame, labels, pi_comments=SYNTH_PI)
        (d / "banned.txt").write_text("\n".join(lex) + "\n")
        common = ["--manifest", str(d / "man.csv"), "--labels", str(d / "golden_labels.csv"),
                  "--splits", str(d / "splits.csv"), "--keys-dir", str(d / "keys"), "--kits-dir", str(d / "kits"),
                  "--registry-md", str(md), "--out", str(d / "out" / "preds_golden_{arm}_{model}_{split}_r{k}.parquet"),
                  "--banned", str(d / "banned.txt"), "--model", "sonnet", "--concurrency", "2"]
        (d / "out").mkdir()
        # ---- e2 on align: 3 exemplars (u0001..u0003) left out -> 2 scored units, k=2 parquets
        rge.main(["--arm", "e2", "--split", "align", "--n-exemplars", "3", "--k", "2"] + common)
        for k in (1, 2):
            pq = d / "out" / f"preds_golden_e2_sonnet_align_r{k}.parquet"
            df = pd.read_parquet(pq)
            assert set(df.unit_id) == {"u0004", "u0005"} and (df.k == k).all() and (df.n_exemplars_seen == 3).all()
            assert (df.grade_pred == "B").all() and (df.rescored == False).all()  # noqa: E712
            assert "s_exp" in df.columns and (df.grade_truth.isin(["D", "A"])).all()
            assert (df.thinking == "off").all() and (df.effort == "default").all()
            meta = json.loads((d / "out" / f"preds_golden_e2_sonnet_align_r{k}.meta.json").read_text())
            assert meta["exemplar_unit_ids"] == ["u0001", "u0002", "u0003"] and meta["tuple"]["n_exemplars"] == 3
            assert meta["system_sha16"] == meta["tuple"]["system_sha16"] == _util.sha_text(grader_jwst.DIRECT_SYS_JWST)
            assert (d / "out" / f"traces_golden_e2_sonnet_align_r{k}").is_dir()
        kinds = [(k, tag) for _, tag, k in fake.exposed]
        assert ("fewshot", "golden_e2_sonnet_align_r1") in kinds and ("eval", "golden_e2_sonnet_align_r2") in kinds
        assert fake.asserted == []                              # align runs do not need the proof
        assert os.environ.get("LENSJUDGE_BACKEND") == "anthropic"   # alias selected the Claude engine
        # ---- e2 on validate: unregistered -> refused before any call
        n_calls = len(stub.calls)
        try:
            rge.main(["--arm", "e2", "--split", "validate", "--n-exemplars", "3", "--k", "1"] + common)
            raise AssertionError("no raise")
        except SystemExit as e:
            assert "not registered" in str(e) and len(stub.calls) == n_calls
        # a validate run without a lexicon is refused before the registry is even consulted
        try:
            rge.main(["--arm", "e2", "--split", "validate", "--n-exemplars", "3", "--k", "1"]
                     + common[:-6] + ["--banned", str(d / "absent.txt"), "--model", "sonnet", "--concurrency", "2"])
            raise AssertionError("no raise")
        except SystemExit as e:
            assert "build it first" in str(e) and len(stub.calls) == n_calls
        # register the exact tuple (taken from --print-tuple's row format) and run
        t = rge.RunTuple("e2", "sonnet", _util.sha_text(grader_jwst.RUBRIC_V2_PATH.read_text()), 3,
                         rge.splits_sha(d / "splits.csv"), _util.sha_text(grader_jwst.DIRECT_SYS_JWST), "off", "default")
        _register(md, t.row("rubric_imaging_v2.md", 1))
        rge.main(["--arm", "e2", "--split", "validate", "--n-exemplars", "3", "--k", "1"] + common)
        dfv = pd.read_parquet(d / "out" / "preds_golden_e2_sonnet_validate_r1.parquet")
        assert set(dfv.unit_id) == {f"u{i:04d}" for i in range(6, 11)} and (dfv.split == "validate").all()
        assert fake.asserted and set(fake.asserted[-1][0]) == set(dfv.unit_id)   # validate units proven unexposed first
        # second validate call on the same tuple -> refused
        try:
            rge.main(["--arm", "e2", "--split", "validate", "--n-exemplars", "3", "--k", "1"] + common)
            raise AssertionError("no raise")
        except SystemExit as e:
            assert "already scored" in str(e)
        # the same tuple under --thinking adaptive is a different tuple: unregistered
        try:
            rge.main(["--arm", "e2", "--split", "validate", "--n-exemplars", "3", "--k", "1", "--thinking", "adaptive"] + common)
            raise AssertionError("no raise")
        except SystemExit as e:
            assert "not registered" in str(e)
        finally:
            os.environ.pop("LENSJUDGE_THINKING", None)
        # an interrupted replicate resumes instead of being refused: drop the meta, the
        # parquet stays -> the run finishes it (0 new calls, units already done) and re-writes meta
        rge.meta_path(d / "out" / "preds_golden_e2_sonnet_validate_r1.parquet").unlink()
        n_calls = len(stub.calls)
        rge.main(["--arm", "e2", "--split", "validate", "--n-exemplars", "3", "--k", "1"] + common)
        assert len(stub.calls) == n_calls and rge.meta_path(d / "out" / "preds_golden_e2_sonnet_validate_r1.parquet").exists()
        # e1 on validate (zero-shot) also proves the validate units unexposed first
        t1 = rge.RunTuple("e1", "sonnet", t.rubric_sha16, 0, t.splits_sha16, t.system_sha16, "off", "default")
        _register(md, t1.row("rubric_imaging_v2.md", 1))
        n_asserted = len(fake.asserted)
        rge.main(["--arm", "e1", "--split", "validate", "--k", "1"] + common)
        assert len(fake.asserted) == n_asserted + 1 and set(fake.asserted[-1][0]) == set(dfv.unit_id)
        # e1 needs no exemplars; e2 insists on them
        try:
            rge.main(["--arm", "e1", "--split", "align", "--n-exemplars", "2", "--k", "1"] + common)
            raise AssertionError("no raise")
        except SystemExit:
            pass
        # ---- audit the validate traces: clean
        rc = audit_traces.main(["--traces-dir", str(d / "out" / "traces_golden_e2_sonnet_validate_r1"),
                                "--banned", str(d / "banned.txt"), "--splits", str(d / "splits.csv"),
                                "--key", str(d / "keys" / "kit01_key.csv"), "--out", str(d / "audit.json"),
                                "--check-text", str(grader_jwst.RUBRIC_V2_PATH), "--check-text", str(grader_jwst.JWST_NOTE_PATH)])
        rep = json.loads((d / "audit.json").read_text())
        assert rc == 0 and rep["passed"] and rep["n_events"] == 5 and rep["n_violations"] == 0
        assert rep["exemplar_unit_ids"] == ["u0001", "u0002", "u0003"]
        assert all(v is None for v in rep["checked_files"].values())
    finally:
        grader_direct.grade_candidate = orig
        rge._REGISTRY = None
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d)


def test_run_golden_eval_with_real_registry():
    """Same flow against WP-D's registry.py on a temp ledger (--registry-csv): sync, a validate
    run records eval + fewshot exposure, and a validate unit later marked as a fewshot
    exemplar makes the next validate run die with ExposureError before any model call."""
    from lensjudge.golden import registry
    d, key, labels, frame, splits = make_fixture()
    stub, orig = _patch_stub("C")
    saved_env = {k: os.environ.pop(k, None) for k in ("LENSJUDGE_BACKEND", "LENSJUDGE_FEWSHOT_MANIFEST")}
    rge._REGISTRY = None
    try:
        ledger = d / "golden_registry.csv"
        registry.sync_from(labels, splits, frame, path=ledger)
        assert len(registry.load(ledger)) == 10
        man = build_eval_manifest.build(labels, frame, splits, key, d / "kits", "all")
        _util.pin(man, d / "man.csv")
        md = d / "REGISTRY.md"
        shutil.copy(GOLDEN / "REGISTRY.md", md)
        t = rge.RunTuple("e2", "sonnet", _util.sha_text(grader_jwst.RUBRIC_V2_PATH.read_text()), 1,
                         rge.splits_sha(d / "splits.csv"), _util.sha_text(grader_jwst.DIRECT_SYS_JWST), "off", "default")
        _register(md, t.row("rubric_imaging_v2.md", 1))
        lex = audit_traces.build_lexicon(splits, frame, labels, pi_comments=SYNTH_PI)
        (d / "banned.txt").write_text("\n".join(lex) + "\n")
        (d / "out").mkdir()
        common = ["--manifest", str(d / "man.csv"), "--labels", str(d / "golden_labels.csv"),
                  "--splits", str(d / "splits.csv"), "--keys-dir", str(d / "keys"), "--kits-dir", str(d / "kits"),
                  "--registry-md", str(md), "--registry-csv", str(ledger), "--banned", str(d / "banned.txt"),
                  "--out", str(d / "out" / "preds_golden_{arm}_{split}_r{k}.parquet"), "--model", "sonnet"]
        rge.main(["--arm", "e2", "--split", "validate", "--n-exemplars", "1", "--k", "1"] + common)
        reg = registry.load(ledger).set_index("unit_id")
        assert reg.loc["u0006", "exposed_runs"] == "golden_e2_sonnet_validate_r1" and reg.loc["u0006", "in_fewshot"] == ""
        assert reg.loc["u0001", "in_fewshot"] == "golden_e2_sonnet_validate_r1"      # the A exemplar
        assert set(u for u in reg.index if reg.loc[u, "in_fewshot"]) == {"u0001", "u0002", "u0003"}
        # poison: a validate unit becomes a fewshot exemplar somewhere -> the proof must fail,
        # for the few-shot arm AND for a zero-shot arm (an SFT student scored as e1)
        registry.mark_exposed(["u0007"], "rogue_run", "fewshot", path=ledger)
        (d / "out" / "preds_golden_e2_validate_r1.parquet").unlink()
        (d / "out" / "preds_golden_e2_validate_r1.meta.json").unlink()
        t1 = rge.RunTuple("e1", "served/student-9b", t.rubric_sha16, 0, t.splits_sha16, t.system_sha16, "off", "default")
        _register(md, t1.row("rubric_imaging_v2.md", 1))
        n_calls = len(stub.calls)
        for argv in (["--arm", "e2", "--split", "validate", "--n-exemplars", "1", "--k", "1"] + common,
                     ["--arm", "e1", "--split", "validate", "--k", "1"] + common[:-2] + ["--model", "served/student-9b"]):
            saved_backend = os.environ.pop("LENSJUDGE_BACKEND", None)
            try:
                rge.main(argv)
                raise AssertionError("no raise")
            except registry.ExposureError as e:
                assert "u0007" in str(e) and len(stub.calls) == n_calls
            finally:
                if saved_backend is not None:
                    os.environ["LENSJUDGE_BACKEND"] = saved_backend
        # smoke on the synthetic frame: zero-shot, marks exposure with run_tag smoke; once a
        # splits file is given, validate units are never picked
        rge.main(["--smoke", "2", "--frame", str(d / "frame.csv"), "--keys-dir", str(d / "keys"),
                  "--kits-dir", str(d / "kits"), "--registry-csv", str(ledger), "--model", "sonnet",
                  "--out", str(d / "out" / "preds_golden_{arm}_{split}_r{k}.parquet")])
        sm = pd.read_parquet(d / "out" / "preds_golden_smoke_frame_r1.parquet")
        assert len(sm) == 2 and (sm.run_tag == "smoke").all() and sm.grade_truth.isna().all()
        reg = registry.load(ledger).set_index("unit_id")
        assert "smoke" in reg.loc["u0001", "exposed_runs"]
        assert (d / "out" / "traces_golden_smoke").is_dir()
        big = rge.smoke_frame(d / "frame.csv", d / "keys", d / "kits", 10, splits_csv=d / "splits.csv")
        assert set(big.unit_id) == {f"u{i:04d}" for i in range(1, 6)}      # the align half only
        assert len(rge.smoke_frame(d / "frame.csv", d / "keys", d / "kits", 10)) == 10   # no splits given
    finally:
        grader_direct.grade_candidate = orig
        rge._REGISTRY, rge._REGISTRY_PATH = None, None
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d)


# ------------------------------------------------------------------ audit_traces
# synthetic stand-ins for the PI's comments (the real strings live only in the gitignored
# golden/pi_comments.txt and never in a test)
SYNTH_PI = ("needs a much closer look at the ring", "ask the tool to hyperlink every catalogue entry",
            "the dissenting persona is right here")


def test_pi_comments_file_contract():
    """The comment strings are not in any tracked module; the loader pins count + sha16 and
    refuses an edited file; the lexicon builder takes them as an explicit argument."""
    import re
    src = (GOLDEN / "audit_traces.py").read_text()
    assert "PI_COMMENTS = (" not in src and not re.search(r"closer look", src, re.I)
    assert audit_traces.PI_COMMENTS_N == 16 and re.fullmatch(r"[0-9a-f]{16}", audit_traces.PI_COMMENTS_SHA16)
    assert "golden/pi_comments.txt" in (HERE.parent / ".gitignore").read_text()
    d = Path(tempfile.mkdtemp(prefix="golden_pi_"))
    try:
        assert audit_traces.load_pi_comments(d / "absent.txt", required=False) == []
        try:
            audit_traces.load_pi_comments(d / "absent.txt"); raise AssertionError("no raise")
        except FileNotFoundError as e:
            assert "allow-missing-pi-comments" in str(e)
        (d / "bad.txt").write_text("\n".join(SYNTH_PI) + "\n")
        try:
            audit_traces.load_pi_comments(d / "bad.txt"); raise AssertionError("no raise")
        except ValueError as e:
            assert "expected 16 lines" in str(e)
        if audit_traces.PI_COMMENTS_PATH.exists():       # this machine holds the real file
            real = audit_traces.load_pi_comments()
            assert len(real) == 16 and all(r.strip() for r in real)
        else:
            print("  (golden/pi_comments.txt absent here; real-file check skipped)")
    finally:
        shutil.rmtree(d)


def test_audit_lexicon_and_planted_violations():
    d, key, labels, frame, splits = make_fixture()
    try:
        lex = audit_traces.build_lexicon(splits, frame, labels, pi_comments=SYNTH_PI)
        val_ids = set(labels[labels.unit_id.isin([f"u{i:04d}" for i in range(6, 11)])].candidate_id)
        assert val_ids <= set(lex) and "Jalias7" in lex
        assert not (set(labels[labels.unit_id.isin(["u0001", "u0002"])].candidate_id) & set(lex))
        assert all(c in lex for c in SYNTH_PI)
        assert "u0008 score_1_4=1 confidence_lmh=H" in lex
        # word-window matching: rubric phrases do not trip it, verbatim copies do
        assert audit_traces.banned_hit("the lens light removed; over-subtraction residuals", lex) is None
        assert audit_traces.banned_hit("rank 7 NEEDS a  much closer look at the ring", lex) is not None
        assert audit_traces.banned_hit(f"see {sorted(val_ids)[0]} here", lex) is not None
        assert audit_traces.banned_hit("the dissenting persona is right here", lex)[0] == SYNTH_PI[2]
        align_shas, val_shas = audit_traces.split_shas(key, splits)
        assert len(align_shas) == 5 and len(val_shas) == 5 and not (align_shas & val_shas)
        a1, a2 = sorted(align_shas)[:2]
        v1 = sorted(val_shas)[0]
        known = audit_traces.known_template_shas()

        def ev(texts, ex, cand, n_ex, **kw):
            return {"event": "golden_content_audit", "name": "x",
                    "text_blocks": [{"sha16": _util.sha_text(t), "head": t[:200], "n_chars": len(t)} for t in texts],
                    "exemplar_image_shas": ex, "candidate_image_sha": cand, "image_shas": ex + [cand],
                    "n_images": len(ex) + 1, "n_exemplars": n_ex, "exemplar_unit_ids": [], "system_sha16": "s", **kw}

        gloss = grader_jwst.PANEL_GLOSS
        clean = ev([fewshot.FEWSHOT_LEAD, fewshot.header_text(4, "A", "H"), "[composite]", fewshot.FEWSHOT_TRAIL, gloss],
                   [a1], v1, 1)
        rep = audit_traces.audit([clean], lex, align_shas, val_shas, known)
        assert rep["passed"] and rep["n_violations"] == 0
        planted = ev(["PS: " + sorted(val_ids)[1] + "\n" + gloss], [a1], v1, 1)               # banned id in the head
        bad_ex = ev([gloss], [v1], a2, 1)                                                   # validate sha as exemplar
        n_mis = ev([gloss], [a1, a2], v1, 2, n_images=2)                                    # n_images != 1 + n_ex
        long_unknown = ev([gloss.replace("JWST", "JWST!")], [a1], v1, 1)                    # >200 chars, not a template
        rep = audit_traces.audit([clean, planted, bad_ex, n_mis, long_unknown], lex, align_shas, val_shas, known)
        checks = sorted(v["check"] for v in rep["violations"])
        assert not rep["passed"]
        assert "banned_text" in checks and "validate_as_exemplar" in checks and "exemplar_not_align" in checks
        assert "n_images" in checks and "unverifiable_text_block" in checks
        # the planted block also trips the unverifiable check (PS+gloss is long and not a template) -> fine
        assert [v for v in rep["violations"] if v["check"] == "banned_text"][0]["ngram"] == sorted(val_ids)[1].casefold()
        # CLI path on a planted trace dir -> exit 1 and the JSON names the check
        td = d / "traces_bad"; td.mkdir()
        (td / "x.jsonl").write_text(json.dumps(planted) + "\n" + json.dumps({"event": "direct_request"}) + "\n")
        (d / "banned.txt").write_text("\n".join(lex) + "\n")
        rc = audit_traces.main(["--traces-dir", str(td), "--banned", str(d / "banned.txt"), "--splits", str(d / "splits.csv"),
                                "--key", str(d / "keys" / "kit01_key.csv"), "--out", str(d / "audit.json")])
        assert rc == 1 and not json.loads((d / "audit.json").read_text())["passed"]
        # --build-lexicon CLI writes the file; it refuses to run without the PI comments file
        # unless told so explicitly, and then says what it cannot catch
        try:
            audit_traces.main(["--build-lexicon", "--splits", str(d / "splits.csv"), "--frame", str(d / "frame.csv"),
                               "--labels", str(d / "golden_labels.csv"), "--banned", str(d / "lex2.txt"),
                               "--pi-comments", str(d / "absent.txt")])
            raise AssertionError("no raise")
        except FileNotFoundError:
            pass
        rc = audit_traces.main(["--build-lexicon", "--splits", str(d / "splits.csv"), "--frame", str(d / "frame.csv"),
                                "--labels", str(d / "golden_labels.csv"), "--banned", str(d / "lex2.txt"),
                                "--pi-comments", str(d / "absent.txt"), "--allow-missing-pi-comments"])
        assert rc == 0
        lex_no_pi = audit_traces.load_lexicon(d / "lex2.txt")
        assert [e for e in lex if e not in SYNTH_PI] == lex_no_pi
    finally:
        shutil.rmtree(d)


def test_prompts_are_item_agnostic_and_lexicon_clean():
    """No candidate id / coordinate / PI-comment text in the note, the gloss or the rubric;
    the note says what it must (resolution, layout, no CNN, Huang scale, JWST FP families)."""
    note = grader_jwst.JWST_NOTE
    for must in ("0.031", "40", "32 px", "6 panels", "N up", "yellow", "NO CNN", "Huang", "diffraction",
                 "Spiral", "ring", "Tidal", "over-subtraction"):
        assert must.lower() in note.lower(), must
    import re
    assert not re.search(r"J\d{6,}[+-]\d{5,}", note + grader_jwst.PANEL_GLOSS + grader_jwst.DIRECT_SYS_JWST)
    pi = audit_traces.load_pi_comments(required=False) or list(SYNTH_PI)
    assert audit_traces.banned_hit(grader_jwst.DIRECT_SYS_JWST, pi) is None
    assert audit_traces.banned_hit(grader_jwst.PANEL_GLOSS, pi) is None
    assert grader_jwst.DIRECT_SYS_JWST == grader_jwst.RUBRIC_V2_PATH.read_text() + note
    assert grader_jwst.with_note("X") == "X" + note
    assert (GOLDEN / "REGISTRY.md").read_text().count("## Registered arms") == 1
    assert "golden/banned_lexicon.txt" in (HERE.parent / ".gitignore").read_text()


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
