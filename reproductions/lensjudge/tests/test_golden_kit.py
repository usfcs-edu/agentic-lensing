#!/usr/bin/env python3
"""No-network tests for the golden kit (WP-C): schema, build_kit, add-repeats, collect, serve.

Everything runs on a synthetic 12-unit frame with generated 752x562 JPEGs in a temp dir —
NO network, NO API calls, no real candidate ids. Runs under pytest or directly:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_kit.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from lensjudge.golden import _util, build_kit, collect, schema  # noqa: E402

KIT = "testkit"
GRADER = "XH"


# ------------------------------------------------------------------ fixtures
def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="golden_kit_"))


def _make_frame(root: Path, n: int = 12) -> Path:
    """Synthetic frame + source JPEGs. Exclusion cases: u0001 prior_exposure=2, u0002
    lit_known, u0003 gray layout, u0004/u0005 share a system_id. u0006 comes from the primary
    stamps/ path, all others from the fallback dir (exercises both lookups)."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(1, n + 1):
        cid = f"J{10000000 + i * 1234567:08d}{'+' if i % 2 else '-'}{i * 7654321 % 10000000:07d}"
        rows.append({
            "unit_id": f"u{i:04d}", "candidate_id": cid, "alias_ids": "",
            "system_id": 4 if i in (4, 5) else i, "ra_deg": 100.0 + i * 1.2345,
            "dec_deg": -10.0 - i * 0.54321,
            "stratum": ["T_verified", "T_U", "K_cowls", "D_refuted"][i % 4],
            "substratum": "", "pipe_grade_passcount": "ABCDU"[i % 5],
            "pipe_inspector_conf": 0.5, "pipe_score": 1.0, "center_galaxy_type": "",
            "rank_top100": i, "prior_exposure": 2 if i == 1 else 0,
            "lit_known": i == 2, "layout": "gray_sw_only" if i == 3 else "color",
            "cowls_ranking": "", "cowls_theta_E": np.nan,
            "known_lens_name": "FAKELENS-1" if i == 2 else "", "known_type": "",
            "known_sep_arcsec": np.nan, "desi_pool_overlap": False, "desi_pool_split": "",
            "sw_obs": "x", "lw_obs": "y", "sw_filter": "F115W", "lw_filter": "F444W",
            "proposal": 1234, "field_id": "f",
        })
    frame = pd.DataFrame(rows)
    _util.pin(frame, root / "frame.csv")
    fb, st = root / "fallback", root / "stamps"
    fb.mkdir(); st.mkdir()
    for r in frame.itertuples():
        a = rng.integers(0, 255, size=(562, 752, 3), dtype=np.uint8)
        a[540:, :, :] = 255                       # a white "footer" strip to crop away
        im = Image.fromarray(a)
        if r.unit_id == "u0006":
            (st / r.candidate_id).mkdir()
            im.save(st / r.candidate_id / f"{r.candidate_id}_v1.jpg", quality=93)
        else:
            im.save(fb / f"{r.candidate_id}.jpg", quality=93)
    return root / "frame.csv"


def _build(root: Path, collision_dirs=()) -> tuple[Path, Path]:
    frame = _make_frame(root)
    return build_kit.build(frame, KIT, GRADER, 20260822, root / "stamps", root / "fallback",
                           kits_root=root / "kits", keys_dir=root / "keys",
                           collision_dirs=collision_dirs)


def _event(manifest_sha: str, item: str, pos: int, session: str, score: int, conf: str,
           rev: int = 1, ts: str = "2026-08-22T10:00:00.000Z", flag: bool = False,
           **over) -> dict:
    d = dict(kit_id=KIT, manifest_sha=manifest_sha, item_id=item, presentation_index=pos,
             session_id=session, grader_id=GRADER, score_1_4=score, confidence_lmh=conf,
             seconds=12.5, revision=rev, flag=flag, timestamp=ts, ua="test", tool_version="t")
    d.update(over)
    return d


def _write_events(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def _manifest(root: Path) -> dict:
    return json.loads((root / "kits" / KIT / "kit_manifest.json").read_text())


# -------------------------------------------------------------------- schema
def test_schema_forbid_and_validators():
    good = _event("abc", "001", 1, "s1", 4, "H")
    ev = schema.GradeEvent(**good)
    assert ev.score_1_4 == 4 and ev.dt.tzinfo is not None
    assert set(good) == set(schema.EVENT_KEYS)
    for bad in ({"note": "free text"}, {"score_1_4": 5}, {"score_1_4": "4"},
                {"score_1_4": True}, {"confidence_lmh": "l"}, {"confidence_lmh": "X"},
                {"revision": 0}, {"timestamp": "yesterday"}, {"timestamp": "2026-08-22T10:00:00"},
                {"flag": 1}, {"presentation_index": 0}, {"seconds": -1}):
        d = dict(good); d.update(bad)
        try:
            schema.GradeEvent(**d)
        except Exception as e:  # noqa: BLE001
            assert "ValidationError" in type(e).__name__, (bad, e)
        else:
            raise AssertionError(f"accepted {bad}")
    # records/labels: extra keys and a bad letter are refused; 'pass' alias round-trips
    rec = dict(unit_id="u1", candidate_id="J1", kit_id=KIT, item_id="001", session_id="s",
               presentation_index=1, score_1_4=3, grade_letter="B", confidence_lmh="M",
               confidence01=0.7, seconds=1.0, revision_count=1, flag=False, render_sha="x",
               manifest_sha="y", render_version="jwst_v1", grade_scale=_util.GRADE_SCALE,
               timestamp="2026-08-22T10:00:00Z", grader_id=GRADER)
    rec["pass"] = 1
    r = schema.GradeRecord(**rec).check_consistent()
    assert r.model_dump(by_alias=True)["pass"] == 1
    try:
        schema.GradeRecord(**dict(rec, grade_letter="A")).check_consistent()
        raise AssertionError("letter/score mismatch accepted")
    except ValueError:
        pass
    try:
        schema.GradeRecord(**dict(rec, rationale="smuggled"))
        raise AssertionError("extra key accepted")
    except Exception as e:  # noqa: BLE001
        assert "ValidationError" in type(e).__name__
    assert schema.RECORD_COLS[4] == "pass" and "pass_" not in schema.RECORD_COLS
    assert schema.LABEL_COLS == (
        "unit_id", "candidate_id", "ra_deg", "dec_deg", "stratum", "score_1_4", "grade_letter",
        "confidence_lmh", "confidence01", "pass2_score_1_4", "pass2_confidence_lmh",
        "n_passes", "label_stable", "render_sha", "grade_scale", "grader_id")


def test_events_from_jsonl_refuses_smuggled():
    root = _tmp()
    try:
        good = _event("abc", "001", 1, "s1", 4, "H")
        p = _write_events(root / "e.jsonl", [good, dict(good, comment="a note")])
        try:
            schema.events_from_jsonl(p)
            raise AssertionError("smuggled key accepted")
        except ValueError as e:
            assert "smuggled" in str(e) and "comment" in str(e) and ":2:" in str(e)
        p2 = _write_events(root / "ok.jsonl", [good, dict(good, revision=2)])
        assert len(schema.events_from_jsonl(p2)) == 2
    finally:
        shutil.rmtree(root)


# ----------------------------------------------------------------- build_kit
def test_build_kit_blind():
    root = _tmp()
    try:
        kit_dir, key_path = _build(root)
        frame = _util.read_pinned(root / "frame.csv", dtype=str, keep_default_na=False)
        key = schema.read_key(key_path)
        assert list(key.columns) == list(schema.KEY_COLS)
        assert len(key) == 12 and (key["pass"] == 1).all() and (key["repeat_of_item"] == "").all()
        assert set(key["candidate_id"]) == set(frame["candidate_id"])
        # item ids: opaque 3-digit draws without replacement, NOT a running count in
        # presentation order (a running count would mark later-added repeats)
        ids = key["item_id"].tolist()
        assert len(set(ids)) == 12 and all(re.fullmatch(r"\d{3}", i) for i in ids)
        assert ids != [f"{i:03d}" for i in range(1, 13)] and ids != sorted(ids)
        assert key["presentation_index"].tolist() == list(range(1, 13))
        # seeded permutation: reproducible and not the frame order; ids drawn from the same
        # generator right after it
        assert key["unit_id"].tolist() != sorted(key["unit_id"])
        rng2 = np.random.default_rng(20260822)
        order2 = rng2.permutation(12)
        assert key["unit_id"].tolist() == [f"u{int(i) + 1:04d}" for i in order2]
        assert ids == build_kit.draw_item_ids(rng2, 12)
        # served JPEGs are 752x540, quality 92 (not the scrambled set's 93), one frozen mtime,
        # and the key sha is the sha of the served bytes
        for r in key.itertuples():
            p = kit_dir / "items" / f"{r.item_id}.jpg"
            assert Image.open(p).size == (752, 540)
            assert _util.sha_file(p) == r.render_sha
            assert int(p.stat().st_mtime) == build_kit.ITEM_MTIME
        assert len(set(key["render_sha"])) == 12
        assert build_kit.JPEG_KW["quality"] == 92
        # the primary stamps/ path was used for u0006, fallback for the rest (both cropped)
        html = (kit_dir / "grade.html").read_text()
        low = html.lower()
        for v in list(frame["candidate_id"]) + list(frame["unit_id"]) + list(key["render_sha"]) + \
                ["FAKELENS-1", (key_path.with_suffix(".csv.sha")).read_text().strip()]:
            assert v.lower() not in low, v
        for ra, dec in zip(frame["ra_deg"], frame["dec_deg"]):
            for v in (ra, dec):
                assert f"{float(v):.4f}" not in html and f"{float(v):.2f}" not in html
        for w in build_kit.FORBIDDEN_WORDS:
            assert w not in low, w
        assert "<!--" not in html and '<script type="application/json" id="kit">' in html
        assert "<input" in html and 'type="text"' not in html and "<textarea" not in html
        m = _manifest(root)
        assert m["kit_id"] == KIT and m["grader_id"] == GRADER and m["n_items"] == 12
        assert m["render_version"] == "jwst_v1" and m["grade_scale"] == _util.GRADE_SCALE
        assert [i["item_id"] for i in m["items"]] == key["item_id"].tolist()
        assert build_kit.manifest_sha_of(m) == m["manifest_sha"]
        assert set(m) == {"kit_id", "kit_version", "grader_id", "grade_scale", "render_version",
                          "confidence", "legend", "items", "n_items", "manifest_sha", "built_at"}
        assert "pass-count" in m["legend"]["banner"]
        assert m["confidence"]["keys"] == ["L", "M", "H"]
        emb = json.loads(re.search(r'id="kit">(.*?)</script>', html, re.S).group(1))
        assert emb == m
        sheet = pd.read_csv(kit_dir / "grading_sheet.csv", dtype=str, keep_default_na=False)
        assert list(sheet.columns) == ["item_id", "score_1_4", "confidence_lmh"]
        assert sheet["item_id"].tolist() == sorted(key["item_id"]) and (sheet["score_1_4"] == "").all()
        readme = (kit_dir / "README.txt").read_text()
        assert "top100_clean_scrambled" in readme and "SUPERSEDES" in readme
        assert "Einstein" in readme and "4 = A" in readme and "pass-count" in readme
        assert (kit_dir / "serve.py").read_text() == build_kit.SERVE_PY.read_text()
        hist = build_kit.load_manifest_history(root / "keys", KIT)
        assert len(hist) == 1 and hist[0]["manifest_sha"] == m["manifest_sha"]
        assert hist[0]["key_sha"] == build_kit.key_pin_sha(root / "keys", KIT)
        assert "key_sha" not in m                       # history-only, never shipped
        # a second build must refuse to overwrite the key
        try:
            build_kit.build(root / "frame.csv", KIT, GRADER, 1, root / "stamps", root / "fallback",
                            kits_root=root / "kits", keys_dir=root / "keys")
            raise AssertionError("overwrote an existing kit")
        except FileExistsError:
            pass
        # a served JPEG byte-identical to a file the grader already holds is refused
        held = root / "held"; held.mkdir()
        shutil.copyfile(kit_dir / "items" / f"{ids[0]}.jpg", held / "007.jpg")
        try:
            build_kit.assert_no_collision(kit_dir / "items", (held, root / "nonexistent"))
            raise AssertionError("byte collision not detected")
        except AssertionError as e:
            if "byte-identical" not in str(e):
                raise
        (held / "007.jpg").write_bytes(b"\xff\xd8" + b"x" * 10)
        rep = build_kit.assert_no_collision(kit_dir / "items", (held,))
        assert rep[str(held)] == {"n_theirs": 1, "byte_hits": 0, "size_hits": 0}
        assert build_kit.assert_no_collision(kit_dir / "items", (root / "nonexistent",)) == {}
        # --force wipes the key AND the manifest history (no old event can join a new key)
        build_kit.build(root / "frame.csv", KIT, GRADER, 1, root / "stamps", root / "fallback",
                        kits_root=root / "kits", keys_dir=root / "keys", force=True, collision_dirs=())
        hist2 = build_kit.load_manifest_history(root / "keys", KIT)
        assert len(hist2) == 1 and hist2[0]["manifest_sha"] != m["manifest_sha"]
        assert schema.read_key(key_path)["item_id"].tolist() != ids      # new seed, new draw
        # a missing composite is reported by id before anything is written
        try:
            build_kit.build(root / "frame.csv", "other", GRADER, 1, root / "stamps", None,
                            kits_root=root / "kits", keys_dir=root / "keys")
            raise AssertionError("missing sources not detected")
        except FileNotFoundError as e:
            assert "11 composite" in str(e)
            assert not (root / "keys" / "other_key.csv").exists()
    finally:
        shutil.rmtree(root)


def _grade_prefix(root: Path, n_graded: int, session: str = "s1") -> list[dict]:
    """Pass-1 events for presentation positions 1..n_graded, score = 1 + unit_no % 4."""
    key = schema.read_key(root / "keys" / f"{KIT}_key.csv").sort_values("presentation_index")
    m = _manifest(root)
    evs = []
    for r in key.head(n_graded).itertuples():
        score = 1 + int(r.unit_id[1:]) % 4
        ts = f"2026-08-22T10:{r.presentation_index:02d}:00.000Z"
        evs.append(_event(m["manifest_sha"], r.item_id, r.presentation_index, session, score,
                          "LMH"[score % 3], ts=ts))
    return evs


def test_add_repeats_gap_rule():
    root = _tmp()
    try:
        kit_dir, key_path = _build(root)
        key1 = schema.read_key(key_path)
        m1 = _manifest(root)
        evs = _grade_prefix(root, 10)
        _write_events(root / "records" / "e1.jsonl", evs)
        collect.collect(KIT, [root / "records" / "e1.jsonl"], frame_path=root / "frame.csv",
                        kits_root=root / "kits", keys_dir=root / "keys",
                        grades_path=root / "golden_grades.csv", labels_path=root / "golden_labels.csv")
        build_kit.add_repeats(KIT, root / "frame.csv", root / "golden_grades.csv", n=4, min_gap=3,
                              seed=20260822, kits_root=root / "kits", keys_dir=root / "keys",
                              min_per_level=1, collision_dirs=())
        key2 = schema.read_key(key_path)
        m2 = _manifest(root)
        rep = key2[key2["pass"] == 2]
        assert len(rep) == 4 and len(key2) == 16
        # repeat ids come from the unused remainder and are indistinguishable from the rest:
        # no continuation of a sequence, no id above every pass-1 id
        assert key2["item_id"].is_unique and not set(rep["item_id"]) & set(key1["item_id"])
        assert max(rep["item_id"]) != max(key2["item_id"]) or min(rep["item_id"]) < max(key1["item_id"])
        assert m2["kit_version"] == 2 and m2["n_items"] == 16
        assert m2["manifest_sha"] != m1["manifest_sha"]
        assert [i["item_id"] for i in m2["items"]] == key2.sort_values("presentation_index")["item_id"].tolist()
        pos = dict(zip(key2["item_id"], key2["presentation_index"]))
        grades = collect.load_grades(root / "golden_grades.csv")
        graded_items = set(grades["item_id"])
        frame = _util.read_pinned(root / "frame.csv", dtype={"unit_id": str}, keep_default_na=False)
        excluded = {"u0001", "u0002", "u0003", "u0004", "u0005"}
        for r in rep.itertuples():
            orig = key1[key1["item_id"] == r.repeat_of_item].iloc[0]
            assert orig["unit_id"] == r.unit_id and orig["render_sha"] == r.render_sha
            assert r.repeat_of_item in graded_items            # drawn from graded pass-1 units
            assert r.unit_id not in excluded
            assert (kit_dir / "items" / f"{r.item_id}.jpg").read_bytes() == \
                (kit_dir / "items" / f"{r.repeat_of_item}.jpg").read_bytes()
            assert r.presentation_index - pos[r.repeat_of_item] >= 3, (r.item_id, r.presentation_index)
            assert r.presentation_index > 10                   # only in the ungraded tail
        # graded prefix untouched, ungraded originals only pushed later
        p1 = dict(zip(key1["item_id"], key1["presentation_index"]))
        for it in graded_items:
            assert pos[it] == p1[it]
        for it in set(p1) - graded_items:
            assert pos[it] >= p1[it]
        # stratified on observed pass-1 score: no level takes more than its fair share
        sc = dict(zip(grades["unit_id"], grades["score_1_4"]))
        counts = pd.Series([sc[u] for u in rep["unit_id"]]).value_counts()
        assert counts.max() <= 2, counts.to_dict()
        # shipped files updated, one frozen mtime for originals and repeats, still blind
        html = (kit_dir / "grade.html").read_text().lower()
        for it in rep["item_id"]:
            assert f'"{it}"' in html and f"{it}.jpg" in html
        assert {int(q.stat().st_mtime) for q in (kit_dir / "items").glob("*.jpg")} == {build_kit.ITEM_MTIME}
        for v in list(frame["candidate_id"]) + list(frame["unit_id"]) + list(key2["render_sha"]):
            assert v.lower() not in html
        for w in build_kit.FORBIDDEN_WORDS:
            assert w not in html
        sheet = pd.read_csv(kit_dir / "grading_sheet.csv", dtype=str, keep_default_na=False)
        assert len(sheet) == 16
        hist = build_kit.load_manifest_history(root / "keys", KIT)
        assert [h["kit_version"] for h in hist] == [1, 2]
        # second round: only 1 eligible graded unit is left (5 eligible among the 10 graded,
        # 4 already repeated) -> a short draw with a warning, never an error; a tail too
        # short for min_gap -> appended at the end; never the same unit twice
        build_kit.add_repeats(KIT, root / "frame.csv", root / "golden_grades.csv", n=2, min_gap=50,
                              seed=20260822, kits_root=root / "kits", keys_dir=root / "keys",
                              min_per_level=1, collision_dirs=())
        key3 = schema.read_key(key_path)
        assert len(key3) == 17 and (key3["pass"] == 2).sum() == 5
        last = key3.sort_values("presentation_index").iloc[-1]
        assert last["pass"] == 2 and last["item_id"] not in set(key2["item_id"])
        assert len(set(key3.loc[key3["pass"] == 2, "unit_id"])) == 5
        assert [h["kit_version"] for h in build_kit.load_manifest_history(root / "keys", KIT)] == [1, 2, 3]
    finally:
        shutil.rmtree(root)


def test_drop_units_pilot_fallback():
    """--drop-units retires ungraded items of the named strata as a new manifest version:
    graded items and a deterministic hash core are kept, the key of every remaining item is
    unchanged, dropped rows go to keys/<kit>_dropped.csv, old events still collect."""
    root = _tmp()
    try:
        kit_dir, key_path = _build(root)
        key1 = schema.read_key(key_path)
        m1 = _manifest(root)
        evs = _grade_prefix(root, 4)
        _write_events(root / "records" / "e1.jsonl", evs)
        kw = dict(frame_path=root / "frame.csv", kits_root=root / "kits", keys_dir=root / "keys",
                  grades_path=root / "golden_grades.csv", labels_path=root / "golden_labels.csv")
        collect.collect(KIT, [root / "records" / "e1.jsonl"], **kw)
        graded = set(collect.load_grades(root / "golden_grades.csv")["unit_id"])
        # strata in the synthetic frame cycle T_verified/T_U/K_cowls/D_refuted (3 each)
        keep = {"T_U": 1, "D_refuted": 2}
        want_drop = build_kit.pick_drop_units(key1, graded, keep)
        build_kit.drop_units(KIT, root / "frame.csv", root / "golden_grades.csv", keep,
                             kits_root=root / "kits", keys_dir=root / "keys", collision_dirs=())
        key2 = schema.read_key(key_path)
        m2 = _manifest(root)
        assert set(key1["unit_id"]) - set(key2["unit_id"]) == set(want_drop) and want_drop
        for st, n in keep.items():
            left = key2[key2["stratum"] == st]
            assert len(left) == max(n, int((key1["stratum"] == st).sum() - len(
                [u for u in want_drop if u in set(key1.loc[key1["stratum"] == st, "unit_id"])])))
            assert len(left) >= min(n, int((key1["stratum"] == st).sum()))
        assert graded <= set(key2["unit_id"])                 # never a graded unit
        # key rows of the survivors are unchanged apart from the re-packed position
        j = key1.set_index("unit_id").loc[key2["unit_id"]]
        for c in ("item_id", "candidate_id", "render_sha", "layout", "stratum"):
            assert j[c].tolist() == key2[c].tolist()
        assert key2["presentation_index"].tolist() == list(range(1, len(key2) + 1))
        assert m2["kit_version"] == 2 and m2["n_items"] == len(key2) == 12 - len(want_drop)
        assert [i["item_id"] for i in m2["items"]] == key2["item_id"].tolist()
        for u in want_drop:
            it = key1.loc[key1["unit_id"] == u, "item_id"].iloc[0]
            assert not (kit_dir / "items" / f"{it}.jpg").exists()
        dropped = _util.read_pinned(root / "keys" / f"{KIT}_dropped.csv", dtype=str, keep_default_na=False)
        assert set(dropped["unit_id"]) == set(want_drop) and (dropped["dropped_in_version"] == "2").all()
        # the rule is deterministic: same inputs, same core
        assert build_kit.pick_drop_units(key1, graded, keep) == want_drop
        # session-1 events (manifest v1) still collect under v2; a dropped id is never reused
        collect.collect(KIT, [root / "records" / "e1.jsonl"], **kw)
        build_kit.add_repeats(KIT, root / "frame.csv", root / "golden_grades.csv", n=2, min_gap=1,
                              seed=20260822, kits_root=root / "kits", keys_dir=root / "keys",
                              min_per_level=1, collision_dirs=())
        key3 = schema.read_key(key_path)
        assert not set(key3["item_id"]) & set(dropped["item_id"])
        assert [h["kit_version"] for h in build_kit.load_manifest_history(root / "keys", KIT)] == [1, 2, 3]
        # a key regenerated underneath the campaign is refused by collect (key_sha pin check)
        _util.pin(key3.assign(candidate_id=key3["candidate_id"].str[::-1]), key_path)
        try:
            collect.collect(KIT, [root / "records" / "e1.jsonl"], **kw)
            raise AssertionError("regenerated key accepted")
        except ValueError as e:
            assert "regenerated key" in str(e)
        assert m1["manifest_sha"] != m2["manifest_sha"]
    finally:
        shutil.rmtree(root)


# ------------------------------------------------------------------- collect
def test_collect_last_revision_wins_and_pass_assignment():
    root = _tmp()
    try:
        kit_dir, key_path = _build(root)
        m1 = _manifest(root)
        key1 = schema.read_key(key_path).sort_values("presentation_index")
        evs = _grade_prefix(root, 8, "s1")
        it3 = key1.iloc[2]
        # a Backspace re-grade of position 3: revision 2, different score, later timestamp
        evs.append(_event(m1["manifest_sha"], it3.item_id, 3, "s1", 1, "L", rev=2,
                          ts="2026-08-22T10:30:00.000Z", flag=True))
        # a stale revision-1 duplicate (auto-download overlap) must collapse, not conflict
        evs.append(dict(evs[2]))
        f1 = _write_events(root / "records" / f"events_{KIT}_s1.jsonl", evs)
        kw = dict(frame_path=root / "frame.csv", kits_root=root / "kits", keys_dir=root / "keys",
                  grades_path=root / "golden_grades.csv", labels_path=root / "golden_labels.csv")
        out = collect.collect(KIT, [f1], **kw)
        g = collect.load_grades(root / "golden_grades.csv")
        assert list(g.columns) == list(schema.RECORD_COLS)
        assert len(g) == 8 and (g["pass"] == 1).all() and out["pass1_n"] == 8
        row = g[g["item_id"] == it3.item_id].iloc[0]
        assert row["score_1_4"] == 1 and row["grade_letter"] == "D" and row["confidence_lmh"] == "L"
        assert row["revision_count"] == 2 and bool(row["flag"]) is True
        assert abs(row["confidence01"] - _util.CONF_TO_01["L"]) < 1e-9
        assert row["timestamp"] == "2026-08-22T10:30:00.000Z"
        assert (g["revision_count"] == 1).sum() == 7
        assert (g["grade_scale"] == _util.GRADE_SCALE).all() and (g["render_version"] == "jwst_v1").all()
        labels = _util.read_pinned(root / "golden_labels.csv", dtype=str, keep_default_na=False)
        assert list(labels.columns) == list(schema.LABEL_COLS) and len(labels) == 8
        assert (labels["n_passes"] == "1").all() and (labels["pass2_score_1_4"] == "").all()
        assert (labels["label_stable"] == "").all()
        lab3 = labels[labels["unit_id"] == it3.unit_id].iloc[0]
        assert lab3["score_1_4"] == "1" and lab3["grade_letter"] == "D"
        assert abs(float(lab3["ra_deg"]) - (100.0 + int(it3.unit_id[1:]) * 1.2345)) < 1e-9
        # now hidden repeats, then a later session grades the rest (incl. the repeats)
        build_kit.add_repeats(KIT, root / "frame.csv", root / "golden_grades.csv", n=3, min_gap=2,
                              seed=1, kits_root=root / "kits", keys_dir=root / "keys",
                              min_per_level=1, collision_dirs=())
        m2 = _manifest(root)
        key2 = schema.read_key(key_path).sort_values("presentation_index")
        evs2 = []
        for r in key2[key2["presentation_index"] > 8].itertuples():
            orig = key2[key2["item_id"] == r.repeat_of_item] if r.repeat_of_item else None
            if orig is not None:           # repeat: same score for 2 of 3, different for one
                u = int(r.unit_id[1:]); s1 = 1 + u % 4
                score = s1 if r.item_id != key2.loc[key2["pass"] == 2, "item_id"].iloc[0] else (s1 % 4) + 1
            else:
                score = 2
            evs2.append(_event(m2["manifest_sha"], r.item_id, r.presentation_index, "s2", score, "M",
                               ts=f"2026-08-23T09:{r.presentation_index:02d}:00.000Z"))
        f2 = _write_events(root / "records" / f"events_{KIT}_s2.jsonl", evs2)
        # an event under a foreign manifest sha is refused
        bad = _write_events(root / "records" / "bad.jsonl",
                            [_event("deadbeefdeadbeef", "001", 1, "x", 4, "H")])
        try:
            collect.collect(KIT, [f1, bad], **kw)
            raise AssertionError("foreign manifest_sha accepted")
        except ValueError as e:
            assert "manifest_sha" in str(e)
        out2 = collect.collect(KIT, [f1, f2], **kw)
        g2 = collect.load_grades(root / "golden_grades.csv")
        assert len(g2) == 15 and (g2["pass"] == 2).sum() == 3 and out2["pass2_n"] == 3
        assert out2["n_repeat_pairs"] == 3 and abs(out2["repeat_exact"] - 2 / 3) < 1e-9
        p2 = g2[g2["pass"] == 2]
        assert (p2["session_id"] == "s2").all()
        assert set(p2["unit_id"]) <= set(g2.loc[g2["pass"] == 1, "unit_id"])
        labels2 = _util.read_pinned(root / "golden_labels.csv", dtype=str, keep_default_na=False)
        assert len(labels2) == 12
        two = labels2[labels2["n_passes"] == "2"]
        assert len(two) == 3 and (two["pass2_confidence_lmh"] == "M").all()
        assert sorted(two["label_stable"]) == ["False", "True", "True"]
        for r in two.itertuples():
            assert (r.label_stable == "True") == (r.score_1_4 == r.pass2_score_1_4)
        one = labels2[labels2["n_passes"] == "1"]
        assert (one["pass2_score_1_4"] == "").all() and (one["label_stable"] == "").all()
        # canonical label is pass 1 even when pass 2 disagreed
        for r in two.itertuples():
            assert r.score_1_4 == str(g2[(g2["unit_id"] == r.unit_id) & (g2["pass"] == 1)].iloc[0]["score_1_4"])
        # sheet fallback fills only items with no tool event, without timing
        sheet = pd.DataFrame({"item_id": [key2.iloc[0].item_id, "999"], "score_1_4": ["4", ""],
                              "confidence_lmh": ["H", ""]})
        sheet.to_csv(root / "sheet.csv", index=False)
        collect.collect(KIT, [f1, f2], sheet=root / "sheet.csv", **kw)
        g3 = collect.load_grades(root / "golden_grades.csv")
        assert len(g3) == 15 and g3[g3["item_id"] == key2.iloc[0].item_id].iloc[0]["session_id"] == "s1"
    finally:
        shutil.rmtree(root)


def test_collect_idempotent_merge():
    root = _tmp()
    try:
        _build(root)
        m = _manifest(root)
        key = schema.read_key(root / "keys" / f"{KIT}_key.csv").sort_values("presentation_index")
        e1 = _write_events(root / "records" / "a.jsonl", _grade_prefix(root, 5, "s1"))
        evs_b = [_event(m["manifest_sha"], r.item_id, r.presentation_index, "s2", 2, "M",
                        ts=f"2026-08-23T08:{r.presentation_index:02d}:00.000Z")
                 for r in key.iloc[5:9].itertuples()]
        e2 = _write_events(root / "records" / "b.jsonl", evs_b)
        kw = dict(frame_path=root / "frame.csv", kits_root=root / "kits", keys_dir=root / "keys",
                  grades_path=root / "golden_grades.csv", labels_path=root / "golden_labels.csv")
        collect.collect(KIT, [e1], **kw)
        t1 = (root / "golden_grades.csv").read_text()
        out = collect.collect(KIT, [e1], **kw)
        assert (root / "golden_grades.csv").read_text() == t1 and out["grades_changed"] is False
        # new session only: earlier rows are kept (merge), not dropped
        collect.collect(KIT, [e2], **kw)
        g = collect.load_grades(root / "golden_grades.csv")
        assert len(g) == 9 and set(g["session_id"]) == {"s1", "s2"}
        t2 = (root / "golden_grades.csv").read_text()
        collect.collect(KIT, [e1, e2], **kw)
        assert (root / "golden_grades.csv").read_text() == t2
        # a later revision arriving in a new file supersedes the merged row
        it = key.iloc[0]
        e3 = _write_events(root / "records" / "c.jsonl",
                           [_event(m["manifest_sha"], it.item_id, 1, "s1", 4, "H", rev=2,
                                   ts="2026-08-24T08:00:00.000Z")])
        collect.collect(KIT, [e3], **kw)
        g = collect.load_grades(root / "golden_grades.csv")
        row = g[g["item_id"] == it.item_id].iloc[0]
        assert row["score_1_4"] == 4 and row["grade_letter"] == "A" and len(g) == 9
        # labels .sha sidecar verifies
        _util.read_pinned(root / "golden_labels.csv")
    finally:
        shutil.rmtree(root)


def test_collect_refuses_extra_key():
    root = _tmp()
    try:
        _build(root)
        m = _manifest(root)
        p = _write_events(root / "records" / "x.jsonl",
                          [_event(m["manifest_sha"], "001", 1, "s", 3, "M", rationale="arc at 2 o'clock")])
        try:
            collect.collect(KIT, [p], frame_path=root / "frame.csv", kits_root=root / "kits",
                            keys_dir=root / "keys", grades_path=root / "golden_grades.csv",
                            labels_path=root / "golden_labels.csv")
            raise AssertionError("extra key accepted")
        except ValueError as e:
            assert "smuggled" in str(e) and "rationale" in str(e)
        assert not (root / "golden_grades.csv").exists()
    finally:
        shutil.rmtree(root)


# --------------------------------------------------------------------- tool
def test_template_js_syntax():
    """node --check on the inline script (skipped when node is absent)."""
    html = build_kit.TEMPLATE.read_text()
    assert "__KIT_MANIFEST_JSON__" in html
    assert 'type="text"' not in html and "<textarea" not in html and "contenteditable" not in html
    assert "https://" not in html and "http://" not in html      # self-contained, file:// safe
    for k in schema.EVENT_KEYS:                                   # the tool emits every key
        assert re.search(rf"\b{k}\s*:", html), k
    js = re.search(r"<script>\n(.*?)</script>", html, re.S).group(1)
    node = shutil.which("node")
    if not node:
        print("  (node not found; JS syntax check skipped)")
        return
    p = Path(tempfile.mkdtemp()) / "tool.js"
    p.write_text(js)
    try:
        r = subprocess.run([node, "--check", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        shutil.rmtree(p.parent)


def test_serve_post_and_list():
    """serve.py (stdlib, loopback only): POST appends a JSONL line; extra keys are refused."""
    root = _tmp()
    try:
        _build(root)
        kit_dir = root / "kits" / KIT
        sys.path.insert(0, str(kit_dir))
        import importlib
        serve = importlib.import_module("serve")
        sys.path.pop(0)
        serve.Handler.kit_dir = str(kit_dir)
        cwd = os.getcwd(); os.chdir(kit_dir)
        srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            m = _manifest(root)
            ev = _event(m["manifest_sha"], "001", 1, "sessA", 3, "M")
            req = urllib.request.Request(f"{base}/api/event", data=json.dumps(ev).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                assert json.loads(r.read())["ok"] is True
            bad = urllib.request.Request(f"{base}/api/event", data=json.dumps(dict(ev, note="x")).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(bad)
                raise AssertionError("extra key accepted by serve.py")
            except urllib.error.HTTPError as e:
                assert e.code == 400
            with urllib.request.urlopen(f"{base}/api/events?grader={GRADER}") as r:
                got = json.loads(r.read())
            assert got == [ev]
            with urllib.request.urlopen(f"{base}/grade.html") as r:
                assert b'id="kit"' in r.read()
            f = kit_dir / "records" / f"events_{KIT}_sessA.jsonl"
            assert schema.events_from_jsonl(f)[0].model_dump() == ev
        finally:
            srv.shutdown(); srv.server_close(); os.chdir(cwd)
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
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
