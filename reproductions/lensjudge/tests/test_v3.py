#!/usr/bin/env python3
"""No-API unit/regression tests for the lensjudge-v3 additions.

Pure-logic + graceful-fail paths only — NO network, NO API spend. Runs under the lensjudge venv
via pytest, or directly:
    PYTHONPATH=reproductions python reproductions/lensjudge/tests/test_v3.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from lensjudge.common import hsc_fetch, highres  # noqa: E402
from lensjudge.eval.export_hard_negatives import (  # noqa: E402
    build_hard_negatives, BANK_COLS, BANK_TYPES)
from lensjudge.eval.run_cascade import top_frac_indices  # noqa: E402
from lensjudge.eval.crossmatch_external import crossmatch  # noqa: E402


class _Resp:
    def __init__(self, status, content):
        self.status_code, self.content = status, content


def _pop_creds():
    return {k: os.environ.pop(k, None) for k in ("HSC_USER", "HSC_PASSWORD")}


def _restore(saved):
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# --------------------------------------------------------------- HSC fetch
def test_hsc_fetch_no_credentials():
    saved = _pop_creds()
    try:
        assert hsc_fetch.have_credentials() is False
        assert hsc_fetch.fetch_hsc_cutout(10.0, 2.0) is None
    finally:
        _restore(saved)


def test_hsc_fetch_out_of_footprint():
    saved = _pop_creds()
    os.environ["HSC_USER"], os.environ["HSC_PASSWORD"] = "x", "y"
    orig = hsc_fetch.requests.get
    hsc_fetch.requests.get = lambda *a, **k: _Resp(404, b"<html>404 Not Found</html>")
    try:
        # absurd coords -> not cached -> hits the (mocked) service -> 404 on primary band -> None
        assert hsc_fetch.fetch_hsc_cutout(123.456789, -89.987654) is None
    finally:
        hsc_fetch.requests.get = orig
        shutil.rmtree(hsc_fetch.HSC_CACHE / "pdr3_wide" / "123.456789_-89.987654",
                      ignore_errors=True)
        _restore(saved)


def test_hsc_fetch_success_mock():
    import glob
    fits = glob.glob(str(hsc_fetch.HSC_CACHE / "**" / "i.fits"), recursive=True)
    if not fits:
        print("  [skip] no cached HSC FITS to mock with"); return
    body = Path(fits[0]).read_bytes()
    saved = _pop_creds()
    os.environ["HSC_USER"], os.environ["HSC_PASSWORD"] = "x", "y"
    orig = hsc_fetch.requests.get
    hsc_fetch.requests.get = lambda *a, **k: _Resp(200, body)
    try:
        bands = hsc_fetch.fetch_hsc_cutout(200.111, 33.222)   # fresh coords
        assert bands is not None and "i" in bands and bands["i"].ndim == 2
    finally:
        hsc_fetch.requests.get = orig
        shutil.rmtree(hsc_fetch.HSC_CACHE / "pdr3_wide" / "200.111000_+33.222000",
                      ignore_errors=True)
        _restore(saved)


# ---------------------------------------------------------- resolve_highres
def test_resolve_highres_no_coverage():
    saved = _pop_creds()                       # no creds + no staged Euclid catalog -> None
    try:
        assert highres.resolve_highres("nope", 200.0, 50.0) is None
    finally:
        _restore(saved)


def test_resolve_highres_priority():
    oe, oh = highres._resolve_euclid, highres._resolve_hsc
    # (a) euclid hit takes precedence; hsc must NOT be consulted
    highres._resolve_euclid = lambda n, r, d, rad: {"survey": "euclid", "id_str": "E1", "sep_arcsec": 0.0}
    highres._resolve_hsc = lambda n, r, d: (_ for _ in ()).throw(AssertionError("hsc consulted!"))
    try:
        assert highres.resolve_highres("x", 1.0, 2.0)["survey"] == "euclid"
    finally:
        highres._resolve_euclid, highres._resolve_hsc = oe, oh
    # (b) euclid miss -> hsc hit (carries ra/dec for the position-based grader)
    highres._resolve_euclid = lambda *a: None
    highres._resolve_hsc = lambda n, r, d: {"survey": "hsc", "id_str": "H", "ra": r, "dec": d,
                                            "sep_arcsec": 0.0}
    try:
        hit = highres.resolve_highres("x", 1.0, 2.0)
        assert hit["survey"] == "hsc" and hit["ra"] == 1.0 and hit["dec"] == 2.0
    finally:
        highres._resolve_euclid, highres._resolve_hsc = oe, oh


# ------------------------------------------------------ B6 export mapping
def test_export_build_hard_negatives():
    preds = pd.DataFrame({
        "name": ["m1", "m2_headlogit", "lensA", "mD_nocoord", "m_lowp", "m_weirdtype"],
        "grade_pred": ["D", "D", "A", "D", "C", "D"],
        "contaminant": ["lrg_companion", "merger", None, "blend", "ring_galaxy", "weird_type"],
        "p_lens": [0.03, 0.04, 0.90, 0.02, 0.10, 0.05],
        "p_meta": [0.95, 14.4, 0.99, 0.80, 0.97, 0.70],     # m2 carries a v3-head logit (>1)
        "parse_ok": [True, True, True, True, True, True],
    })
    coords = pd.DataFrame({"name": ["m1", "m2_headlogit", "lensA", "m_lowp", "m_weirdtype"],
                           "ra": [10., 20., 30., 40., 50.], "dec": [1., 2., 3., 4., 5.]})
    bank = build_hard_negatives(preds, coords, p_max=0.3)
    # kept: m1 (D+cont), m2 (D+cont), m_lowp (C but p_lens<=0.3 + cont), m_weirdtype (D+cont)
    # dropped: lensA (grade A, no contaminant), mD_nocoord (no manifest coords)
    assert set(bank["row_id"]) == {"m1", "m2_headlogit", "m_lowp", "m_weirdtype"}
    assert list(bank.columns) == BANK_COLS
    assert ((bank["p_final"] >= 0) & (bank["p_final"] <= 1)).all()          # clamp
    assert bank.loc[bank.row_id == "m2_headlogit", "p_final"].iloc[0] == 1.0  # head-logit -> 1.0
    assert bank.loc[bank.row_id == "m_weirdtype", "mimic_type"].iloc[0] == "other"  # enum clamp
    assert set(bank["mimic_type"]) <= BANK_TYPES
    assert not bank[["RA", "DEC"]].isna().any().any()
    assert (bank["source"] == "lensjudge_v2").all()


# ---------------------------------------------------- cascade rank routing
def test_top_frac_indices():
    s = [0.1, 0.9, 0.5, 0.3]
    assert top_frac_indices(s, 0.5) == {1, 2}        # top-2: 0.9, 0.5
    assert top_frac_indices(s, 0.25) == {1}          # top-1
    assert top_frac_indices(s, 1.0) == {0, 1, 2, 3}
    assert top_frac_indices([], 0.5) == set()
    assert top_frac_indices([0.2, 0.1], 0.01) == {0}  # >=1 always escalates when there is data


# ------------------------------------------------------- crossmatch matcher
def test_crossmatch_matcher():
    master = pd.DataFrame({"ra": [10.0, 200.0], "dec": [5.0, -30.0], "name": ["c0", "c1"]})
    cat = pd.DataFrame({"RA": [10.00005], "DEC": [5.00005]})   # ~0.25" from c0, far from c1
    m, n = crossmatch(master, cat, "RA", "DEC", radius_arcsec=2.0, tag="t")
    assert int(n) == 1
    assert set(m["name"]) == {"c0"}
    assert float(m["t_sep_arcsec"].iloc[0]) < 2.0


# -------------------------------------------------- DESI name -> RA/Dec fallback
def test_parse_radec_from_name():
    from lensjudge.common.fetch import parse_radec_from_name
    ra, dec = parse_radec_from_name("DESI-029.0755-01.1297")
    assert abs(ra - 29.0755) < 1e-4 and abs(dec + 1.1297) < 1e-4
    ra2, dec2 = parse_radec_from_name("DESI-091.7214-58.9787")
    assert abs(ra2 - 91.7214) < 1e-4 and abs(dec2 + 58.9787) < 1e-4
    ra3, dec3 = parse_radec_from_name("DESI-150.5180+04.7380")
    assert abs(ra3 - 150.5180) < 1e-4 and abs(dec3 - 4.7380) < 1e-4
    assert parse_radec_from_name("s_358314_2329") == (None, None)   # non-coord name
    assert parse_radec_from_name(None) == (None, None)


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
