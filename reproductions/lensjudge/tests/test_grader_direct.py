#!/usr/bin/env python3
"""No-API unit tests for grader_direct v4 additions (few-shot in-context prefix).

Pure-logic, mocked render/fetch — NO network, NO API spend. Run under the lensjudge venv via pytest, or:
    PYTHONPATH=reproductions python reproductions/lensjudge/tests/test_grader_direct.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from lensjudge.imaging import grader_direct  # noqa: E402
from lensjudge.common import fetch, render  # noqa: E402


def test_fewshot_prefix_unset_is_empty():
    saved = os.environ.pop("LENSJUDGE_FEWSHOT_MANIFEST", None)
    try:
        assert grader_direct._fewshot_prefix(("full",)) == []
    finally:
        if saved is not None:
            os.environ["LENSJUDGE_FEWSHOT_MANIFEST"] = saved


def test_fewshot_prefix_builds_blocks(monkeypatch=None):
    # mock cutout fetch + render so no FITS/network is needed
    orig_cube, orig_views, orig_b64 = fetch.get_cube, render.render_views, render.png_b64
    fetch.get_cube = lambda **k: object()                      # non-None sentinel
    render.render_views = lambda cube, views: {v: object() for v in views}
    render.png_b64 = lambda img: "B64DATA"
    d = Path(tempfile.mkdtemp())
    csv = d / "fs.csv"
    pd.DataFrame([
        {"name": "L1", "ra": 1.0, "dec": 2.0, "survey_key": "storfer",
         "label": "LENS", "grade": "A", "note": "clear lens"},
        {"name": "N1", "ra": 3.0, "dec": 4.0, "survey_key": "claudenet",
         "label": "NON-LENS", "grade": "D", "note": "mimic"},
    ]).to_csv(csv, index=False)
    saved = os.environ.get("LENSJUDGE_FEWSHOT_MANIFEST")
    os.environ["LENSJUDGE_FEWSHOT_MANIFEST"] = str(csv)
    try:
        blks = grader_direct._fewshot_prefix(("full", "zoom"))
        imgs = [b for b in blks if b.get("type") == "image"]
        txts = [b["text"] for b in blks if b.get("type") == "text"]
        assert len(imgs) == 4                       # 2 examples x 2 views
        assert imgs[0]["source"]["data"] == "B64DATA"
        assert any(t.startswith("REFERENCE EXAMPLE -- LENS") for t in txts)
        assert any(t.startswith("REFERENCE EXAMPLE -- NON-LENS") for t in txts)
        assert txts[-1].startswith("END OF REFERENCE EXAMPLES")
    finally:
        fetch.get_cube, render.render_views, render.png_b64 = orig_cube, orig_views, orig_b64
        if saved is None:
            os.environ.pop("LENSJUDGE_FEWSHOT_MANIFEST", None)
        else:
            os.environ["LENSJUDGE_FEWSHOT_MANIFEST"] = saved


def test_fewshot_prefix_skips_when_no_cutout():
    orig_cube = fetch.get_cube
    fetch.get_cube = lambda **k: None                          # nothing renders
    d = Path(tempfile.mkdtemp()); csv = d / "fs.csv"
    pd.DataFrame([{"name": "X", "ra": 1.0, "dec": 2.0, "survey_key": "storfer",
                   "label": "LENS", "grade": "A", "note": "x"}]).to_csv(csv, index=False)
    saved = os.environ.get("LENSJUDGE_FEWSHOT_MANIFEST")
    os.environ["LENSJUDGE_FEWSHOT_MANIFEST"] = str(csv)
    try:
        assert grader_direct._fewshot_prefix(("full",)) == []   # no example rendered -> no prefix
    finally:
        fetch.get_cube = orig_cube
        if saved is None:
            os.environ.pop("LENSJUDGE_FEWSHOT_MANIFEST", None)
        else:
            os.environ["LENSJUDGE_FEWSHOT_MANIFEST"] = saved


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
