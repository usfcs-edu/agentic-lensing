#!/usr/bin/env python3
"""No-network tests for the vendored JWST fetch/render module and the stamp builder (WP-A).

Synthetic arrays only — NO network, NO API spend. Runs as a plain script or under pytest:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_jwst_fetch.py

What is pinned here: the composite geometry (752x562, footer crop at y=540), the FITS
stamp round-trip (float32, contract header keys, WCS orientation), the run's layout rule
(a missing/under-filled channel flips the composite to the gray layout exactly as
util.render_cutout does), and — when the JWST run repo is present — that the vendored
block in common/jwst_fetch.py is still byte-identical to scripts/util.py @ 4f81493.
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from lensjudge.common import jwst_fetch as jf  # noqa: E402
from lensjudge.golden import _util, build_stamps  # noqa: E402

_META = dict(id="JTEST+000", ra=150.12345, dec=2.54321, mag_r=19.87, type="DEV",
             proposal="1234", sw_filter="F150W", lw_filter="F277W")


def _synth(seed=0, n=320, amp=60.0, sigma_px=6.0, ring_r=None):
    """Sky noise (sigma=1) + a central Gaussian, optionally with a faint ring (an 'arc')."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2.0
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    im = rng.normal(0.0, 1.0, (n, n)) + amp * np.exp(-0.5 * r2 / sigma_px ** 2)
    if ring_r:
        im += 8.0 * np.exp(-0.5 * (np.sqrt(r2) - ring_r) ** 2 / 1.5 ** 2)
    return im


def _panel(img, row, col, px=240, gap=8, th=18):
    """Pixel block of one panel (the geometry inside render_cutout)."""
    x = gap + col * (px + gap)
    y = gap + row * (px + th + gap) + th
    return np.asarray(img)[y:y + px, x:x + px].astype(int)


def _gray_fraction(block):
    return float(((block[..., 0] == block[..., 1]) & (block[..., 1] == block[..., 2])).mean())


# -------------------------------------------------- geometry

def test_render_v1_composite_size_and_mode():
    img = jf.render_v1_composite(_synth(1), _synth(2, sigma_px=9.0, ring_r=40), _META)
    assert img.size == (752, 562) == jf.COMPOSITE_SIZE
    assert img.mode == "RGB"
    # gray-only input renders the same size (layout changes, canvas does not)
    assert jf.render_v1_composite(_synth(1), None, _META).size == (752, 562)
    assert jf.render_v1_composite(None, None, _META) is None


def test_crop_footer():
    img = jf.render_v1_composite(_synth(1), _synth(2), _META)
    c = jf.crop_footer(img)
    assert c.size == (752, 540) and jf.FOOTER_Y == 540
    # the crop is a pure crop: pixels above the cut are untouched
    assert np.array_equal(np.asarray(c), np.asarray(img)[:540])
    # the footer text (id/coords) lives in the dropped strip: it is not all background there
    strip = np.asarray(img)[540:]
    assert (strip != np.array([12, 12, 16])).any()


def test_run_constants():
    assert (jf.CUT_ARCSEC, jf.OUT_PX, jf.PIX, jf.ZOOM_ARCSEC) == (10.0, 320, 0.03125, 3.5)
    assert (jf.MIN_FINITE, jf.JPEG_QUALITY, jf.FOOTER_Y) == (0.55, 95, 540)
    assert jf.RENDER_VERSION == "jwst_v1"
    assert jf.SW_PRIORITY[0] == "F150W" and jf.LW_PRIORITY[0] == "F277W"


def test_product_urls():
    s3, mast = jf.product_urls("jw01837-c1002_t000_nircam_clear-f150w")
    assert s3 == ("https://stpubdata.s3.amazonaws.com/jwst/public/jw01837/L3/t/c1002/"
                  "jw01837-c1002_t000_nircam_clear-f150w_i2d.fits")
    assert mast.endswith("mast:JWST/product/jw01837-c1002_t000_nircam_clear-f150w_i2d.fits")


# -------------------------------------------------- FITS stamps

def test_write_stamp_fits_roundtrip():
    arr = _synth(3).astype(np.float64)
    hdr = jf.stamp_header("JTEST+000", 150.12345, 2.54321, "SW", "F150W",
                          "jw01837-c1002_t000_nircam_clear-f150w",
                          "https://stpubdata.s3.amazonaws.com/x/y.fits", 10.0, 320, 0.987,
                          fetched_at="2026-08-22T00:00:00+00:00")
    want = {"OBJECT", "RA_DEG", "DEC_DEG", "FILTER", "CHANNEL", "OBS_ID", "SRC_URL",
            "PIXSCALE", "CUTARC", "FINITE", "ORIENT", "CREATOR", "FETCHED"}
    assert set(hdr) == want, set(hdr) ^ want
    assert all(len(k) <= 8 for k in hdr)
    d = Path(tempfile.mkdtemp(prefix="golden_fits_"))
    try:
        p = jf.write_stamp_fits(d / "t_SW_10as.fits", arr, hdr)
        back, h = jf.read_stamp_fits(p)
        assert back.dtype == np.float32 and back.shape == (320, 320)
        assert np.allclose(back, arr.astype(np.float32))
        for k, v in hdr.items():
            got = h[k]
            assert (abs(got - v) < 1e-9) if isinstance(v, float) else (got == v), (k, got, v)
        assert h["PIXSCALE"] == 0.03125 and h["CUTARC"] == 10.0 and h["FINITE"] == 0.987
        # minimal TAN WCS: column 0 is East (larger RA), row 0 is North (larger Dec)
        from astropy.wcs import WCS
        w = WCS(h)
        c = w.pixel_to_world(159.5, 159.5)
        assert abs(c.ra.deg - 150.12345) < 1e-6 and abs(c.dec.deg - 2.54321) < 1e-6
        assert w.pixel_to_world(0, 159.5).ra.deg > w.pixel_to_world(319, 159.5).ra.deg
        assert w.pixel_to_world(159.5, 0).dec.deg > w.pixel_to_world(159.5, 319).dec.deg
        assert abs((w.pixel_to_world(159.5, 0).dec.deg - w.pixel_to_world(159.5, 319).dec.deg)
                   * 3600 - 319 * 0.03125) < 1e-3
        # a non-float array and a 20" context stamp keep the scale honest
        hdr20 = jf.stamp_header("JTEST+000", 150.12345, 2.54321, "LW", "F277W", "o", "u",
                                20.0, 640, 1.0, fetched_at="x")
        assert hdr20["PIXSCALE"] == 0.03125
        # missing filter/obs (NaN from pandas) become '' not 'nan'
        h2 = jf.stamp_header("J", 0.0, 0.0, "LW", float("nan"), float("nan"), None, 10.0, 320, 0.0,
                             fetched_at="x")
        assert h2["FILTER"] == "" and h2["OBS_ID"] == "" and h2["SRC_URL"] == ""
        # over-long keys are refused (FITS cards are 8 chars)
        try:
            jf.write_stamp_fits(d / "bad.fits", arr, {"TOO_LONG_KEY": 1})
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


# -------------------------------------------------- layout rule

def test_layout_rule_matches_render_cutout():
    sw, lw = _synth(1), _synth(2, sigma_px=9.0, ring_r=40)
    both = jf.render_v1_composite(sw, lw, _META)
    gray = jf.render_v1_composite(sw, None, _META)
    # the wrapper is render_cutout, nothing more
    assert np.array_equal(np.asarray(both), np.asarray(jf.render_cutout(sw, lw, _META)))
    assert np.array_equal(np.asarray(gray), np.asarray(jf.render_cutout(sw, None, _META)))
    # colour layout: row-1 col-3 and row-2 col-2 are RGB (R != B on most pixels)
    assert _gray_fraction(_panel(both, 0, 2)) < 0.5
    assert _gray_fraction(_panel(both, 1, 1)) < 0.5
    # gray layout: the same slots become grayscale panels (only overlays are coloured)
    assert _gray_fraction(_panel(gray, 0, 2)) > 0.95
    assert _gray_fraction(_panel(gray, 1, 1)) > 0.95
    # panels that do not depend on LW are identical across layouts
    for rc in ((0, 0), (0, 1), (1, 0), (1, 2)):
        assert np.array_equal(_panel(both, *rc), _panel(gray, *rc)), rc
    # the run's gate: an LW below MIN_FINITE is dropped => gray layout, pixel-identical
    ch = {"SW": (sw, 1.0), "LW": (lw, 0.4)}
    g = jf.gate_min_finite(ch)
    assert g["SW"] is sw and g["LW"] is None
    assert np.array_equal(np.asarray(jf.render_v1_composite(g["SW"], g["LW"], _META)),
                          np.asarray(gray))
    g2 = jf.gate_min_finite({"SW": (sw, 0.55), "LW": (lw, 0.551)})
    assert g2["SW"] is sw and g2["LW"] is lw
    assert jf.gate_min_finite({"SW": (None, 0.0), "LW": (lw, 1.0)})["SW"] is None
    # LW-only systems render with the LW filter as the base name
    lw_only = jf.render_v1_composite(None, lw, _META)
    assert lw_only.size == (752, 562)
    assert not np.array_equal(np.asarray(lw_only), np.asarray(gray))


def test_composite_jpeg_encoding_is_the_runs():
    """save_composite must use the run's call (quality 95, PIL defaults): standard
    tables + 4:2:0 subsampling, no progressive/optimize flags."""
    from PIL import JpegImagePlugin
    img = jf.render_v1_composite(_synth(1), _synth(2), _META)
    d = Path(tempfile.mkdtemp(prefix="golden_jpg_"))
    try:
        p = jf.save_composite(img, d / "x_v1.jpg")
        j = Image.open(p)
        assert j.format == "JPEG" and j.size == (752, 562)
        assert JpegImagePlugin.get_sampling(j) == 2          # 4:2:0
        assert "progressive" not in j.info and "progression" not in j.info
        assert list(j.quantization[0])[:8] == [2, 1, 1, 2, 2, 4, 5, 6]   # libjpeg q95 luma
    finally:
        shutil.rmtree(d, ignore_errors=True)


# -------------------------------------------------- vendoring guard

def _trim_util(src: str) -> str:
    """util.py minus the sections jwst_fetch deliberately drops."""
    s = src.replace('BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))\n', "")
    s = re.sub(r"def parse_s_region.*?(?=\ndef product_urls)", "", s, flags=re.S)
    s = re.sub(r"\ndef disk_free_gib.*?(?=\n# -+ panels)", "\n", s, flags=re.S)
    return s


def test_vendored_block_is_verbatim():
    util = _util.JWST_REPO / "scripts" / "util.py"
    if not util.exists():
        print("  (skip: JWST run repo not present)")
        return
    mine = Path(jf.__file__).read_text()
    block = mine.split("# ============================================================================ VENDORED\n")[1]
    block = block.split("# ======================================================================== END VENDORED")[0]
    block = block.split("\n", 1)[1]
    a = [ln for ln in _trim_util(util.read_text()).splitlines() if ln.strip()]
    b = [ln for ln in block.splitlines() if ln.strip()]
    assert a == b, "vendored block drifted from util.py@4f81493 — re-vendor, do not patch"


# -------------------------------------------------- build_stamps bookkeeping

def test_stamp_paths_follow_contract():
    p = build_stamps.stamp_paths("J3440482-522486", Path("/x/stamps"))
    assert p[("SW", 10.0)] == Path("/x/stamps/J3440482-522486/J3440482-522486_SW_10as.fits")
    assert p[("LW", 10.0)] == Path("/x/stamps/J3440482-522486/J3440482-522486_LW_10as.fits")
    assert p[("SW", 20.0)] == Path("/x/stamps/J3440482-522486/J3440482-522486_SW_20as.fits")
    assert p["COMPOSITE"] == Path("/x/stamps/J3440482-522486/J3440482-522486_v1.jpg")
    assert build_stamps.MANIFEST_COLS == ["candidate_id", "channel", "filter", "obs_id", "url",
                                          "arcsec", "out_px", "finite_fraction", "path",
                                          "sha256", "fetched_at"]


def test_manifest_is_done_and_pin_roundtrip():
    """is_done needs a COMPOSITE row and every non-empty path present with its sha;
    the pinned CSV reloads into the same rows."""
    root = Path(tempfile.mkdtemp(prefix="golden_manifest_"))
    try:
        # files must live under LENSJUDGE for the relative-path convention; use a temp
        # subdir inside the (gitignored) stamps tree
        sd = build_stamps.STAMPS / f"_test_{root.name}"
        sd.mkdir(parents=True, exist_ok=True)
        cid = "JTEST+000"
        paths = build_stamps.stamp_paths(cid, sd)
        jf.write_stamp_fits(paths[("SW", 10.0)], _synth(1),
                            jf.stamp_header(cid, 1.0, 2.0, "SW", "F150W", "o", "u", 10.0, 320, 1.0,
                                            fetched_at="x"))
        jf.save_composite(jf.render_v1_composite(_synth(1), None, _META), paths["COMPOSITE"])
        rows = [build_stamps._row(cid, "SW", "F150W", "o", "u", 10.0, 320, 1.0,
                                  paths[("SW", 10.0)], "x"),
                build_stamps._row(cid, "LW", "F277W", "o2", "u2", 10.0, 320, 0.0, None, "x"),
                build_stamps._row(cid, "COMPOSITE", "F150W", "", "", 10.0, 240, 1.0,
                                  paths["COMPOSITE"], "x")]
        assert rows[1]["path"] == "" and rows[1]["sha256"] == ""
        assert len(rows[0]["sha256"]) == 64
        m = build_stamps.Manifest(root / "m.csv")
        assert not m.is_done(cid)
        m.replace(cid, rows)
        assert m.is_done(cid)
        sha = m.pin([cid])
        m2 = build_stamps.Manifest(root / "m.csv")
        assert m2.is_done(cid) and m2.pin([cid]) == sha
        assert list(m2.to_frame().columns) == build_stamps.MANIFEST_COLS
        assert len(m2.to_frame()) == 3
        # without the composite row it is not done; with a corrupted file it is not done
        m.replace(cid, rows[:2])
        assert not m.is_done(cid)
        m.replace(cid, rows)
        paths["COMPOSITE"].write_bytes(b"corrupt")
        assert not m.is_done(cid)
        m.drop(cid)
        assert not m.is_done(cid)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(build_stamps.STAMPS / f"_test_{root.name}", ignore_errors=True)


def test_compare_jpeg():
    d = Path(tempfile.mkdtemp(prefix="golden_cmp_"))
    try:
        img = jf.render_v1_composite(_synth(1), _synth(2), _META)
        a = jf.save_composite(img, d / "a.jpg")
        b = jf.save_composite(img, d / "b.jpg")
        r = build_stamps.compare_jpeg(a, b)
        assert r["bytes_identical"] and r["max_abs_diff"] == 0 and r["frac_pixels_differ"] == 0.0
        # same pixels, different encoder settings: not byte-identical, small pixel diffs
        img.save(str(d / "c.jpg"), quality=93, optimize=True)
        r = build_stamps.compare_jpeg(a, d / "c.jpg")
        assert not r["bytes_identical"] and r["max_abs_diff"] > 0 and 0 < r["frac_pixels_differ"] < 1
        # different image: large diff
        jf.save_composite(jf.render_v1_composite(_synth(7, amp=5.0), None, _META), d / "d.jpg")
        r = build_stamps.compare_jpeg(a, d / "d.jpg")
        assert r["max_abs_diff"] > 50
        # shape mismatch is reported, not raised
        jf.crop_footer(img).save(str(d / "e.jpg"), quality=95)
        r = build_stamps.compare_jpeg(a, d / "e.jpg")
        assert r["frac_pixels_differ"] == 1.0 and "shape" in r["note"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_load_frame_joins_targets():
    """--frame mode: only candidate_id (+ra_deg/dec_deg cross-check) is taken from the frame;
    the fetch/footer fields come from J/data/targets.parquet. Local file read, no network."""
    if not build_stamps.TARGETS.exists():
        print("  (skip: JWST run repo not present)")
        return
    import pandas as pd
    d = Path(tempfile.mkdtemp(prefix="golden_frame_"))
    try:
        fr = pd.DataFrame([
            dict(unit_id="u0001", candidate_id="J3440482-522486", ra_deg=34.404817, dec_deg=-5.224858,
                 stratum="T_verified", layout="color"),
            dict(unit_id="u0002", candidate_id="J18030075+2309921", ra_deg=180.300749, dec_deg=23.099210,
                 stratum="T_verified", layout="gray_sw_only"),
            dict(unit_id="u0003", candidate_id="JNOTREAL+0000000", ra_deg=1.0, dec_deg=1.0,
                 stratum="N_unflagged", layout="color")])
        _util.pin(fr, d / "frame.csv")
        j = build_stamps.load_frame(d / "frame.csv")
        assert list(j["candidate_id"]) == list(fr["candidate_id"])
        r0 = j.iloc[0]
        assert r0.sw_obs == "jw01837-c1002_t000_nircam_clear-f150w" and r0.sw_filter == "F150W"
        assert r0.lw_obs == "jw01837-c1002_t000_nircam_clear-f277w" and r0.lw_filter == "F277W"
        assert r0.type in ("SER", "DEV") and abs(float(r0.mag_r) - 20.41) < 0.01
        assert str(r0.proposal) == "1837" and abs(float(r0.ra) - 34.404817) < 1e-5
        r1 = j.iloc[1]
        assert pd.isna(r1.lw_obs) and r1.sw_filter == "F150W2"       # the gray_sw_only case
        assert pd.isna(j.iloc[2]["ra"])                                 # unknown id -> will fail, not crash
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_rendered_layout_matches_frame_rule():
    """rendered_layouts reads the layout off the manifest (colour iff '+' in the COMPOSITE
    filter; gray picks the channel that passed the gate) and check_layout flags a frame whose
    `layout` disagrees — the maskbar case: LW observation present, finite 0.0, rendered gray."""
    root = Path(tempfile.mkdtemp(prefix="golden_layout_"))
    try:
        def R(cid, ch, filt, ff, path, arcsec=10.0):   # manifest row without hashing a real file
            return {"candidate_id": cid, "channel": ch, "filter": filt, "obs_id": "o", "url": "u",
                    "arcsec": arcsec, "out_px": 240 if ch == "COMPOSITE" else (320 if arcsec == 10.0 else 640),
                    "finite_fraction": ff, "path": path, "sha256": "0" * 64 if path else "",
                    "fetched_at": "x"}
        m = build_stamps.Manifest(root / "m.csv")
        m.replace("A", [R("A", "SW", "F150W", 1.0, "golden/stamps/A/a_sw.fits"),
                        R("A", "LW", "F277W", 0.97, "golden/stamps/A/a_lw.fits"),
                        R("A", "COMPOSITE", "F150W+F277W", 1.0, "golden/stamps/A/a.jpg")])
        m.replace("B", [R("B", "SW", "F150W2", 1.0, "golden/stamps/B/b_sw.fits"),
                        R("B", "LW", "F444W", 0.0, "golden/stamps/B/b_lw.fits"),
                        R("B", "COMPOSITE", "F150W2", 1.0, "golden/stamps/B/b.jpg")])
        m.replace("C", [R("C", "LW", "F444W", 0.9, "golden/stamps/C/c_lw.fits"),
                        R("C", "COMPOSITE", "F444W", 0.9, "golden/stamps/C/c.jpg")])
        # D: SW fails the gate at 10" (so the composite is LW-only) but passes at 20" —
        # only the 10" rows may decide the rendered side
        m.replace("D", [R("D", "SW", "F150W", 0.2, "golden/stamps/D/d_sw.fits"),
                        R("D", "SW", "F150W", 0.8, "golden/stamps/D/d_sw20.fits", arcsec=20.0),
                        R("D", "LW", "F277W", 1.0, "golden/stamps/D/d_lw.fits"),
                        R("D", "LW", "F277W", 1.0, "golden/stamps/D/d_lw20.fits", arcsec=20.0),
                        R("D", "COMPOSITE", "F277W", 1.0, "golden/stamps/D/d.jpg")])
        got = build_stamps.rendered_layouts(m)
        assert got.to_dict() == {"A": "color", "B": "gray_sw_only", "C": "gray_lw_only", "D": "gray_lw_only"}, got.to_dict()
        frame = pd.DataFrame({"candidate_id": ["A", "B", "C", "D", "E"],
                              "layout": ["color", "color", "gray_lw_only", "gray_lw_only", "color"]})
        bad = build_stamps.check_layout(frame, m)
        assert bad["candidate_id"].tolist() == ["B"] and bad["rendered"].tolist() == ["gray_sw_only"]
        frame.loc[1, "layout"] = "gray_sw_only"
        assert build_stamps.check_layout(frame, m).empty
        assert build_stamps.check_layout(frame.drop(columns=["layout"]), m).empty
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
