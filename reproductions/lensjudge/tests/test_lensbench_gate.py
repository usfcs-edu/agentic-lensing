#!/usr/bin/env python3
"""No-API unit tests for the v4 LensBench-VI gate + arcsinh render option.

Pure-logic only — NO network, NO API spend. Run under the lensjudge venv via pytest, or:
    PYTHONPATH=reproductions python reproductions/lensjudge/tests/test_lensbench_gate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import render  # noqa: E402
from lensjudge.eval import lensbench_gate as G  # noqa: E402


def _manifest():
    return pd.DataFrame({
        "name": ["L1", "L2", "R1", "R2", "M1", "M2"],
        "source": ["graded", "graded", "random_neg", "random_neg", "mimic_neg", "mimic_neg"],
        "binary_label": ["lens", "lens", "nonlens", "nonlens", "nonlens", "nonlens"],
        "grade_truth": ["A", "B", "D", "D", "D", "D"],
    })


def _preds(p_lens):
    names = ["L1", "L2", "R1", "R2", "M1", "M2"]
    return pd.DataFrame({
        "name": names,
        "grade_truth": ["A", "B", "D", "D", "D", "D"],
        "p_lens": p_lens,
        "parse_ok": [True] * 6,
        "grade_pred": ["A", "B", "D", "D", "C", "D"],
        "cost_usd": [0.0] * 6,
        "wall_s": [1.0] * 6,
    })


# --------------------------------------------------------------- gate metrics
def test_gate_metrics_auc_splits():
    man = _manifest()
    # lens high; random low (perfect detection); one mimic overlaps a lens (mimic AUC = 0.75)
    preds = _preds([0.90, 0.80, 0.10, 0.20, 0.85, 0.50])
    m = G.gate_metrics(G.merge_manifest(preds, man))
    assert m["n"] == 6 and m["parse_rate"] == 1.0
    assert m["n_pos"] == 2 and m["n_random"] == 2 and m["n_mimic"] == 2
    assert m["detection_auc"] == 1.0          # lens {0.9,0.8} vs random {0.1,0.2}
    assert m["mimic_auc"] == 0.75             # lens {0.9,0.8} vs mimic {0.85,0.5}
    assert m["mean_p_lens_A"] == 0.9


def test_merge_drops_unknown_names():
    man = _manifest()
    preds = _preds([0.9, 0.8, 0.1, 0.2, 0.85, 0.5])
    preds = pd.concat([preds, pd.DataFrame([{
        "name": "GHOST", "grade_truth": "A", "p_lens": 0.5, "parse_ok": True,
        "grade_pred": "A", "cost_usd": 0.0, "wall_s": 1.0}])], ignore_index=True)
    merged = G.merge_manifest(preds, man)
    assert "GHOST" not in set(merged["name"])   # not in manifest -> dropped
    assert len(merged) == 6


# ----------------------------------------------------------------- verdict
def test_verdict_pass_and_fail():
    oracle = {"detection_auc": 1.0, "parse_rate": 1.0}
    v_pass = G.verdict({"detection_auc": 0.99, "parse_rate": 0.98}, oracle, 0.03, 0.95)
    assert v_pass["passed"] is True
    v_fail_auc = G.verdict({"detection_auc": 0.80, "parse_rate": 0.99}, oracle, 0.03, 0.95)
    assert v_fail_auc["passed"] is False
    v_fail_parse = G.verdict({"detection_auc": 0.99, "parse_rate": 0.50}, oracle, 0.03, 0.95)
    assert v_fail_parse["passed"] is False


# --------------------------------------------------------------- flip rate
def test_grade_flip_rate():
    a = pd.DataFrame({"name": ["x", "y", "z"], "grade_pred": ["A", "B", "C"],
                      "p_lens": [0.9, 0.6, 0.3]})
    b = pd.DataFrame({"name": ["x", "y", "z"], "grade_pred": ["A", "C", "C"],
                      "p_lens": [0.9, 0.4, 0.3]})
    fr = G.grade_flip_rate(a, b)
    assert fr["n_shared"] == 3
    assert abs(fr["grade_flip_rate"] - (1 / 3)) < 1e-3   # only y flips B->C (rounded to 4dp)
    assert fr["lens_call_flip_rate"] == 0.0              # all A/B/C stay lens-calls
    assert abs(fr["mean_abs_p_lens_delta"] - (0.2 / 3)) < 1e-3


# ----------------------------------------------------------- arcsinh render
def test_arcsinh_rgb_shape():
    rng = np.random.RandomState(0)
    cube = (rng.rand(3, config.SIZE_PIX, config.SIZE_PIX).astype(np.float32) * 100.0)
    img = render.arcsinh_rgb(cube)
    assert img.size == (config.RENDER_PX, config.RENDER_PX)
    assert img.mode == "RGB"


def test_render_stretch_toggle():
    saved = os.environ.get("LENSJUDGE_RENDER_STRETCH")
    rng = np.random.RandomState(1)
    cube = (rng.rand(3, config.SIZE_PIX, config.SIZE_PIX).astype(np.float32) * 100.0)
    try:
        os.environ["LENSJUDGE_RENDER_STRETCH"] = "arcsinh"
        assert render.render_stretch() == "arcsinh"
        views = render.render_views(cube, views=("full", "zoom"))
        assert set(views) == {"full", "zoom"}
        assert views["full"].size == (config.RENDER_PX, config.RENDER_PX)
        os.environ["LENSJUDGE_RENDER_STRETCH"] = "lupton"
        assert render.render_stretch() == "lupton"
    finally:
        if saved is None:
            os.environ.pop("LENSJUDGE_RENDER_STRETCH", None)
        else:
            os.environ["LENSJUDGE_RENDER_STRETCH"] = saved


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
