#!/usr/bin/env python3
"""No-network test for golden/regrade_scrambled.py (the one-off blind regrade of the
scrambled top-100).

Covers, on synthetic keys / images / prediction rows only (no API, no real key file):
  1. the blind-cand builder emits EXACTLY {name, image_path, layout} — no candidate id,
     rank, coordinate, filter or any other key column can reach a model-facing row;
  2. filename -> layout from blank sw_filter / lw_filter alone (color; lw blank ->
     gray_sw_only; sw blank -> gray_lw_only, the build_frame.derive_layout naming; both
     blank refuses), NaN treated as blank;
  3. the de-scramble join (filename -> key -> candidate_id), the comparison-CSV shape,
     the our_S-descending order and the agree_letter logic, on canned rows.

Runs under pytest or directly:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_scrambled.py
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import regrade_scrambled as rs  # noqa: E402


def _synthetic_key() -> pd.DataFrame:
    """Three scrambled rows with deliberately leak-shaped extra columns (synthetic ids)."""
    return pd.DataFrame({
        "index": ["1", "2", "3"],
        "filename": ["001.jpg", "002.jpg", "003.jpg"],
        "rank": ["42", "3", "77"],
        "candidate_id": ["JTEST0001+0000001", "JTEST0002+0000002", "JTEST0003+0000003"],
        "ra_deg": ["150.1", "150.2", "150.3"],
        "dec_deg": ["2.1", "2.2", "2.3"],
        "verifier_grade": ["U", "A", "C"],
        "verifiers_pass": ["0", "3", "1"],
        "inspector_confidence": ["26.0", "78.0", ""],
        "discovery_status": ["new", "new", "known"],
        "blind_theta_E_arcsec": ["1.4", "1.1", ""],
        "sw_filter": ["F150W", "F150W2", np.nan],
        "lw_filter": ["F277W", "", "F444W"],
        "evidence": ["secret free text", "more secret text", "even more"],
    })


def test_blind_build_layout_and_descramble():
    # ---- 2: layout from the filter blanks alone (the build_frame.derive_layout naming)
    assert rs.derive_layout("F150W", "F277W") == "color"
    assert rs.derive_layout("F150W2", "") == "gray_sw_only"          # lw blank -> SW present
    assert rs.derive_layout("", "F444W") == "gray_lw_only"           # sw blank -> LW present
    assert rs.derive_layout(float("nan"), "F277W") == "gray_lw_only" # NaN == blank
    for sw, lw in (("", np.nan), (" ", None), (None, "")):
        try:
            rs.derive_layout(sw, lw)
            raise AssertionError("both-blank must refuse")
        except ValueError:
            pass
    assert rs.scr_name("007.jpg") == "scr_007"
    assert rs.scr_to_filename("scr_007") == "007.jpg"
    for bad in ("7.jpg", "0007.jpg", "001.png"):
        try:
            rs.scr_name(bad)
            raise AssertionError(bad)
        except ValueError:
            pass

    key = _synthetic_key()
    d = Path(tempfile.mkdtemp(prefix="golden_scrambled_"))
    try:
        # ---- 1: builder output carries nothing but name / image_path / layout
        from PIL import Image
        for fn in key["filename"]:
            Image.new("RGB", rs.IMG_SIZE, (10, 10, 12)).save(d / fn, quality=90)
        cands = rs.build_blind_cands(key, d)
        assert list(cands.columns) == list(rs.BLIND_COLS)
        assert cands["name"].tolist() == ["scr_001", "scr_002", "scr_003"]
        assert cands["layout"].tolist() == ["color", "gray_sw_only", "gray_lw_only"]
        leak_values = set(key["candidate_id"]) | set(key["rank"]) | set(key["ra_deg"]) | \
            set(key["evidence"]) | set(key["verifier_grade"]) | {"F150W", "F277W", "F444W", "F150W2"}
        for _, row in cands.iterrows():
            for v in row.tolist():
                assert str(v) not in leak_values, (v, "key column leaked into a blind cand")
            assert "JTEST" not in str(row.tolist())
        # a wrong-size image refuses (the scrambled set is footer-stripped 752x540)
        Image.new("RGB", (752, 562), (0, 0, 0)).save(d / "001.jpg", quality=90)
        try:
            rs.build_blind_cands(key, d)
            raise AssertionError("wrong-size image must refuse")
        except ValueError:
            pass

        # ---- 3: de-scramble join + comparison shape + agree_letter, canned prediction rows
        preds = pd.DataFrame({
            "name": ["scr_001", "scr_002", "scr_003"],
            "S": [0.30, 0.05, float("nan")],
            "S_arb": [0.30, 0.05, float("nan")],
            "grade_pred": ["A", "C", None],                # scr_003 = parse failure
            "letter_llm": ["A", "C", None],
            "p_evidence": [0.85, 0.20, 0.10],
            "scale_class_final": ["galaxy", "galaxy", "none"],
            "alternative_final": [None, "merger", None],
            "needs_human": [False, False, False],
            "parse_ok": [True, True, False],
            "letter_source": ["sonnet_api_calibrated"] * 3,
            "cost_usd": [0.08, 0.06, 0.02],
        })
        comp = rs.descramble(preds, key)
        assert list(comp.columns) == list(rs.COMPARISON_COLS)
        # sorted by our_S descending, NaN (parse failure) last
        assert comp["scrambled_item"].tolist() == ["scr_001", "scr_002", "scr_003"]
        assert comp["candidate_id"].tolist() == ["JTEST0001+0000001", "JTEST0002+0000002",
                                                 "JTEST0003+0000003"]
        assert comp["rank"].tolist() == [42, 3, 77] and comp["rank"].dtype.kind == "i"
        assert comp["nate_grade"].tolist() == ["U", "A", "C"]
        assert comp["nate_n_pass"].tolist() == [0, 3, 1]
        assert math.isnan(comp["nate_inspector_conf"].iloc[2])       # blank conf -> NaN
        assert math.isnan(comp["blind_theta_E_arcsec"].iloc[2])
        # agree only on exact letter equality: U != A; A != C is checked the other way round
        assert comp["agree_letter"].tolist() == [False, False, False]
        preds2 = preds.assign(grade_pred=["A", "A", "C"])            # scr_002 nate=A ours=A
        comp2 = rs.descramble(preds2, key).set_index("scrambled_item")
        assert bool(comp2.loc["scr_002", "agree_letter"]) is True
        assert bool(comp2.loc["scr_001", "agree_letter"]) is False   # U never agrees
        assert bool(comp2.loc["scr_003", "agree_letter"]) is True    # C == C
        # a preds row without a key row refuses
        try:
            rs.descramble(preds.assign(name=["scr_001", "scr_002", "scr_099"]), key)
            raise AssertionError("missing key row must refuse")
        except ValueError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
