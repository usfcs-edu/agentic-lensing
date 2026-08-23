#!/usr/bin/env python3
"""No-network tests for golden/{incumbent_replay,reason_audit,render_v2,analyze_truth}.py (WP-5).

Synthetic stamps / preds only; the two checks that need the JWST run repo (the 350-id replay
and the 12 recovered ctl verdicts; the real-stamp compose) skip when LENSJUDGE_JWST_REPO is
absent or the files are missing. Runs under pytest or directly:
    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_analysis.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from lensjudge.common import jwst_fetch, render  # noqa: E402
from lensjudge.golden import _util, analyze_truth as at, incumbent_replay as ir, reason_audit as ra, render_v2  # noqa: E402

# Verbatim incumbent notes from diag_forensics.md (the canned cases the audit must hit):
# rank 15 (J20954380-1094330) GEOMETRY fail — the theta_E prior written back verbatim
RANK15_GEOMETRY_NOTE = (
    'The orange features are broad clumpy patches at ~3-4" arranged with 180-deg rotational '
    '(spiral/ring) symmetry rather than lensing mirror symmetry, and a 3.5" Einstein radius would '
    'demand a group/cluster halo, not this single galaxy sitting next to an obviously tidally '
    'disturbed edge-on companion.')
RANK15_GEOMETRY_ALT = "dusty spiral arms / tidal debris in a low-z interacting system"
# COWLS AAAAAB (COSJ100052+014856 = J15021825+181565, theta_E 1.00") ARTIFACT fail — the
# circular-subtraction bowtie blamed on the sky
COWLS_AAAAAB_ARTIFACT_NOTE = (
    "A bright star with the unmistakable JWST 6-spike diffraction pattern sits in the NW corner and "
    "its spikes and wings sweep across the field along exactly the direction of the claimed 'bluish "
    "tangential structure ~1.2\" NW', and the residual is a perfectly 180 deg-symmetric two-lobed "
    "pattern along the galaxy's N-S major axis - a Sersic/ellipticity mismatch, not an offset arc.")
COWLS_AAAAAB_ARTIFACT_ALT = ("PSF wings/diffraction spikes of the bright star in the NW corner, plus a "
                             "symmetric bipolar over-subtraction residual")
COWLS_AAAAAB_GEOMETRY_NOTE = (
    "The residual is the textbook bipolar bowtie aligned exactly with the galaxy's NNW-SSE major axis "
    "and negative along the minor axis; the lobes are radially elongated and follow the galaxy's own "
    "isophotes, the exact opposite of tangential lensed images.")
A68_MORPHOLOGY_NOTE = ("a broad clumpy orange arm containing dusty star-forming knots that is redder than "
                       "the blue central galaxy (wrong sign for a lensed source)")
RANK1_GEOMETRY_PASS_NOTE = (
    'Blue arc traces r=1.2-1.4" over ~150 deg E of the red elliptical with its centre of curvature on '
    'the deflector, and a compact blue counter-image sits 1.0" W almost diametrically opposite - a '
    'textbook fold/partial-ring configuration.')


def _j_present(*rel) -> bool:
    J = _util.JWST_REPO
    return J.exists() and all((J / r).exists() for r in rel)


# ================================================================== incumbent_replay
def test_passcount_table_identical_to_grade():
    """All 27 verdict triples (+ empty) through passcount_incumbent vs 09_rank_report.grade()."""
    import itertools
    for trip in itertools.product(ir.VERDICTS, repeat=3):
        n_pass, n_fail, n_unc, letter = ir.passcount_incumbent(list(trip))
        want = {3: "A", 2: "B", 1: "C"}.get(trip.count("pass"), "D")
        assert (n_pass, n_fail, n_unc) == (trip.count("pass"), trip.count("fail"), trip.count("uncertain"))
        assert letter == want, (trip, letter, want)
    assert ir.passcount_incumbent([]) == (0, 0, 0, "U")
    # unknown strings are "uncertain" (the loader's coercion), dict records are accepted
    assert ir.passcount_incumbent(["PASS ", "maybe", {"verdict": "fail"}]) == (1, 1, 1, "C")


def test_loader_rule_keep_first_and_coercion():
    """Synthetic verdict files: sorted-glob keep-first per (id, persona), rstrip(','), malformed
    lines skipped, unknown verdict -> uncertain, verbatim text (no 300-char truncation)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        long_note = "x" * 450
        (d / "verify_artifact_0.jsonl").write_text(
            json.dumps({"id": "J1", "persona": "Artifact", "verdict": "Pass", "alternative": "", "notes": long_note}) + ",\n"
            + "not json\n" + json.dumps(["no", "dict"]) + "\n" + json.dumps({"persona": "artifact"}) + "\n"
            + json.dumps({"id": "J2", "persona": "artifact", "verdict": "weird", "alternative": "a", "notes": "n"}) + "\n")
        (d / "verify_artifact_t1_1.jsonl").write_text(
            json.dumps({"id": "J1", "persona": "artifact", "verdict": "fail", "alternative": "dup", "notes": "second"}) + "\n")
        (d / "verify_artifact_ctl0.jsonl").write_text(
            json.dumps({"id": "J3", "persona": "artifact", "verdict": "fail", "alternative": "c", "notes": "ctl"}) + "\n")
        ver = ir.load_verdicts(d)
        assert ver.attrs["malformed"] == 3
        assert len(ver) == 3 and set(ver["id"]) == {"J1", "J2", "J3"}
        j1 = ver[ver["id"] == "J1"].iloc[0]
        assert j1["verdict"] == "pass" and j1["notes"] == long_note and j1["file"] == "verify_artifact_0.jsonl"
        assert ver[ver["id"] == "J2"].iloc[0]["verdict"] == "uncertain"
        allrows = ir.load_verdicts(d, keep="all")
        assert len(allrows) == 4 and (allrows["dup_rank"] == 1).sum() == 1
        trunc = ir.load_verdicts(d, truncate=True)
        assert len(trunc[trunc["id"] == "J1"].iloc[0]["notes"]) == ir.NOTES_TRUNC
        agg = ir.aggregate(ver)
        assert agg.set_index("id").loc["J3", "any_ctl_file"] and not agg.set_index("id").loc["J1", "any_ctl_file"]
        assert agg.set_index("id").loc["J1", "letter_incumbent"] == "C"       # 1 pass / 0 / 0 over one persona


def test_counterfactual_rules():
    agg = pd.DataFrame({
        "id": ["a", "b", "c"], "n_pass": [2, 1, 0], "n_fail": [0, 1, 3], "n_uncertain": [1, 1, 0],
        "letter_incumbent": ["B", "C", "D"],
        "artifact_verdict": ["pass", "pass", "fail"], "artifact_alternative": ["", "", "spiral"],
        "geometry_verdict": ["pass", "fail", "fail"], "geometry_alternative": ["", "", ""],
        "morphology_verdict": ["uncertain", "uncertain", "fail"], "morphology_alternative": ["", "", "shell"]})
    cf = ir.counterfactual_letters(agg).set_index("id")
    assert list(cf.loc["a"]) == ["B", "B", "A"]        # 2+0.5 floors to B; no named fail -> 3 -> A
    assert list(cf.loc["b"]) == ["C", "C", "A"]        # geometry fail without an alternative is not a veto
    assert list(cf.loc["c"]) == ["D", "D", "C"]        # two named fails veto, the unnamed one does not
    t = ir.counterfactual_table(agg, cowls_ids=["c"])
    assert t.set_index("rule").loc["named_alt_veto_only", "cowls_ABC"] == 1


def test_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    from sklearn.metrics import cohen_kappa_score
    assert abs(ir._kappa(x, y) - cohen_kappa_score(x, y)) < 1e-12


def test_real_replay_reproduces_350_and_recovers_12():
    """J-dependent: the replay reproduces results.csv for every CSV-verified id, recovers the
    12 *_ctl* ids, and reproduces diag_forensics §a (17/12/8 passes; kappa 0.63/0.49/0.46)."""
    if not _j_present("results/verdicts", "results/results.csv", "results/verifications.csv"):
        print("  (skipped: JWST repo verdicts not present)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = ir.replay(out_csv=Path(tmp) / "incumbent_replay.csv", quiet=True)
        pinned = _util.read_pinned(Path(tmp) / "incumbent_replay.csv", dtype={"id": str})
    assert len(out["hidden"]) == 12 and all(out["table"].set_index("id").loc[out["hidden"], "recovered_ctl"])
    assert len(pinned) == 362 and pinned["id"].is_unique
    pt = out["persona"]
    assert {p: int(pt.loc[p, "pass"]) for p in ir.PERSONAS} == {"artifact": 17, "geometry": 8, "morphology": 12}
    k = out["kappa"].set_index("pair")["kappa"]
    assert abs(k["artifact-geometry"] - 0.628) < 0.002 and abs(k["geometry-morphology"] - 0.486) < 0.002 \
        and abs(k["artifact-morphology"] - 0.461) < 0.002
    rec = pinned[pinned["recovered_ctl"].astype(str).str.lower() == "true"]
    assert len(rec) == 12 and (rec["grade_published"] == "U").all() and (rec["letter_incumbent"] == "D").all()
    assert int(pinned["any_ctl_file"].astype(str).str.lower().eq("true").sum()) == 15   # 12 hidden + 3 D in the CSV
    assert "COSJ100052+014856" in set(rec["cowls_code"])          # the AAAAAB lens is among the 12
    # the verbatim text is longer than the CSV's truncation for at least one recovered note
    assert pinned[[f"{p}_notes" for p in ir.PERSONAS]].map(len).max().max() > ir.NOTES_TRUNC


# ================================================================== reason_audit
def test_canned_notes_hit_the_right_categories():
    g15 = ra.grounds(RANK15_GEOMETRY_ALT + " | " + RANK15_GEOMETRY_NOTE)
    assert "theta_e_prior" in g15 and "spiral_ring_disk" in g15, g15
    assert ra.locates_feature(RANK15_GEOMETRY_NOTE)                 # ~3-4"
    ga = ra.grounds(COWLS_AAAAAB_ARTIFACT_ALT + " | " + COWLS_AAAAAB_ARTIFACT_NOTE)
    assert "over_subtraction" in ga and "diffraction_detector" in ga, ga
    assert ra.locates_feature(COWLS_AAAAAB_ARTIFACT_NOTE)           # ~1.2" NW, 180 deg
    gg = ra.grounds(COWLS_AAAAAB_GEOMETRY_NOTE)
    assert "over_subtraction" in gg and "not_tangential" in gg, gg
    assert "colour_only" in ra.grounds(A68_MORPHOLOGY_NOTE)
    # a PASS note with pixels located but no refutation ground of the forbidden kind
    gp = ra.grounds(RANK1_GEOMETRY_PASS_NOTE)
    assert not any(c in gp for c in ra.FORBIDDEN), gp
    assert ra.locates_feature(RANK1_GEOMETRY_PASS_NOTE)


def test_forbidden_only_and_locates_feature_edges():
    assert ra.forbidden_only('a 3" Einstein radius for this single galaxy is implausible')
    assert not ra.forbidden_only(RANK15_GEOMETRY_NOTE)             # theta_E prior + spiral + companion
    assert not ra.forbidden_only("nothing tagged here at all")
    assert ra.forbidden_only("the residual is a four-lobed butterfly from the subtraction")
    assert not ra.locates_feature("low S/N feature")               # S/N is not a compass token
    assert not ra.locates_feature("")
    assert ra.locates_feature("arc at 1.3 arcsec") and ra.locates_feature("PA 93-164 deg") \
        and ra.locates_feature('knot 1.0" W') and ra.locates_feature("to the north-east")
    assert not ra.locates_feature(np.nan)


def test_audit_table_incumbent_and_new_scheme():
    rep = pd.DataFrame({
        "id": ["x", "y"],
        "artifact_verdict": ["fail", "pass"], "artifact_alternative": [COWLS_AAAAAB_ARTIFACT_ALT, ""],
        "artifact_notes": [COWLS_AAAAAB_ARTIFACT_NOTE, RANK1_GEOMETRY_PASS_NOTE],
        "geometry_verdict": ["fail", "fail"], "geometry_alternative": [RANK15_GEOMETRY_ALT, ""],
        "geometry_notes": [RANK15_GEOMETRY_NOTE, 'a 3" Einstein radius for this single galaxy is implausible'],
        "morphology_verdict": ["uncertain", "fail"], "morphology_alternative": ["", ""],
        "morphology_notes": ["", A68_MORPHOLOGY_NOTE]})
    long = ra.incumbent_long(rep)
    assert len(long) == 6 and ra.refutation_mask(long).sum() == 4
    t = ra.audit_table(long)
    assert t.loc["n_refutations", "all"] == 4 and t.loc["n_refutations", "geometry"] == 2
    assert t.loc["theta_e_prior", "geometry"] == 1.0
    # only the bare theta_E sentence rests on a forbidden ground ALONE (the A68 colour note
    # also names "star-forming knots" -> spiral_ring_disk, so a mention is not a sole ground)
    assert t.loc["forbidden_only", "all"] == 0.25
    assert abs(ra.forbidden_rate(long) - 0.25) < 1e-12
    # new scheme: votes parquet-like frame with raw JSON; no_opinion and alternative None excluded.
    # The artifact critic names the SANCTIONED channel subtraction_residual for item 1 — but the
    # advocate marked item 1 visible_in_direct, so the bowtie note IS a forbidden ground here
    # (the channel is admissible only for features absent from every direct panel)
    votes = pd.DataFrame({
        "id": ["x", "x", "x", "x"], "role": ["advocate", "artifact", "geometry", "morphology"],
        "raw": [json.dumps({"p_evidence": 0.7, "items": [{"k": 1, "visible_in_direct": True, "r_arcsec": 0.4},
                                                          {"k": 2, "visible_in_direct": False, "r_arcsec": 0.5}]}),
                json.dumps({"no_opinion": False, "alternative": "subtraction_residual", "accounts_for": [1],
                            "alternative_desc": "bowtie over-subtraction lobes", "notes": 'symmetric lobes at 0.4"'}),
                json.dumps({"no_opinion": True, "no_opinion_reason": "outside_competence", "alternative": None,
                            "alternative_desc": "", "notes": ""}),
                json.dumps({"no_opinion": False, "alternative": "spiral_arm", "accounts_for": [1, 2],
                            "alternative_desc": "arm joined to the nucleus", "notes": 'radius grows with angle from 1.1" E to 1.6" S'})]})
    crit = ra.votes_to_critics(votes)
    assert len(crit) == 3 and ra.refutation_mask(crit).tolist() == [True, False, True]
    assert crit["covers_direct"].tolist() == [True, False, True] and crit["accounts_for"].tolist() == [[1], [], [1, 2]]
    t2 = ra.audit_table(crit)
    assert t2.loc["n_refutations", "all"] == 2 and t2.loc["over_subtraction", "artifact"] == 1.0
    assert t2.loc["over_subtraction_forbidden", "artifact"] == 1.0 and t2.loc["uses_subtraction_residual", "all"] == 0.5
    assert t2.loc["forbidden_only", "all"] == 0.5 and t2.loc["locates_feature", "all"] == 1.0
    # ... the same note on an item the advocate saw ONLY in the subtracted panel is the channel
    # working as designed: not forbidden (F2: the monitor must not count sanctioned channels)
    votes_ok = votes.copy()
    votes_ok.loc[1, "raw"] = json.dumps({"no_opinion": False, "alternative": "subtraction_residual", "accounts_for": [2],
                                         "alternative_desc": "bowtie over-subtraction lobes", "notes": 'symmetric lobes at 0.4"'})
    t_ok = ra.audit_table(ra.votes_to_critics(votes_ok))
    assert t_ok.loc["over_subtraction", "artifact"] == 1.0 and t_ok.loc["over_subtraction_forbidden", "artifact"] == 0.0
    assert t_ok.loc["forbidden_only", "all"] == 0.0
    # the other two sanctioned channels: a scale argument routed into scale_tension is not a
    # theta_E-prior refutation; a colour remark beside a structural alternative is not colour-only;
    # the same words under "other" / another alternative ARE forbidden
    st = pd.DataFrame([
        {"id": "a", "persona": "geometry", "no_opinion": False, "alternative": "scale_tension", "alternative_desc": "",
         "notes": 'the 3.5" radius implies a group-scale halo but I see no second red member and no envelope',
         "accounts_for": [1], "covers_direct": True},
        {"id": "b", "persona": "geometry", "no_opinion": False, "alternative": "companion_projection", "alternative_desc": "",
         "notes": 'a 3.5" Einstein radius would demand a group halo; none present', "accounts_for": [1], "covers_direct": True},
        {"id": "c", "persona": "morphology", "no_opinion": False, "alternative": "other", "alternative_desc": "host-coloured knot",
         "notes": "redder than the deflector, consistent with a dusty source", "accounts_for": [1], "covers_direct": True},
        {"id": "d", "persona": "morphology", "no_opinion": False, "alternative": "spiral_arm", "alternative_desc": "arm",
         "notes": 'same colour as the host disk and a stellar bridge; spiral arm at r=1.2" PA 40', "accounts_for": [1], "covers_direct": True}])
    tg = ra.tag_frame(st).set_index("id")
    assert tg.loc["a", "theta_e_prior"] and not tg.loc["a", "theta_e_prior_forbidden"] and not tg.loc["a", "forbidden_only"]
    assert tg.loc["b", "theta_e_prior_forbidden"] and tg.loc["b", "forbidden_only"]
    assert tg.loc["c", "colour_only"] and tg.loc["c", "colour_only_forbidden"] and tg.loc["c", "forbidden_only"]
    assert tg.loc["d", "colour_only"] and not tg.loc["d", "colour_only_forbidden"] and not tg.loc["d", "forbidden_only"]
    assert tg["uses_scale_tension"].tolist() == [True, False, False, False]
    assert ra.forbidden_only(st.loc[0, "notes"], alternative="scale_tension") is False
    assert ra.forbidden_only(st.loc[0, "notes"]) is True                 # prose alone (the incumbent rule)
    assert ra.audit_table(st).loc["forbidden_only", "all"] == 0.5
    # the runner's votes carry the model's RAW text (prose + JSON, arcsecond quotes in the
    # prose), not bare JSON: the reader must still recover the record (first API smoke: every
    # critic row was dropped and audit_table crashed on the empty frame)
    votes2 = votes.copy()
    votes2["raw"] = ['Looking at the 3.5" zoom first.\n' + r for r in votes["raw"]]
    crit2 = ra.votes_to_critics(votes2)
    assert len(crit2) == 3 and crit2["alternative"].fillna("").tolist() == ["subtraction_residual", "", "spiral_arm"]
    pd.testing.assert_frame_equal(ra.audit_table(crit2), t2)
    # without an advocate record in the votes covers_direct is unknown -> the channel counts as sanctioned
    crit3 = ra.votes_to_critics(votes[votes["role"] != "advocate"])
    assert crit3["covers_direct"].isna().all() and ra.audit_table(crit3).loc["forbidden_only", "all"] == 0.0
    empty = ra.votes_to_critics(pd.DataFrame({"id": ["x"], "role": ["advocate"], "raw": ["{}"]}))
    assert len(empty) == 0
    t3 = ra.audit_table(empty)                      # empty frame: a table of NaNs, no crash
    assert t3.loc["n_refutations", "all"] == 0 and np.isnan(t3.loc["over_subtraction", "all"])


# ================================================================== render_v2
def sersic_arc(n: int = 320, eps: float = 0.4, theta: float = np.radians(30), re_px: float = 25,
               amp: float = 400.0, arc: bool = True, seed: int = 1, noise: float = 1.0,
               arc_amp: float = 12.0) -> np.ndarray:
    """A de Vaucouleurs deflector (eps, PA) + a tangential arc at 1.2" over PA 20-140 deg
    (dy > 0 = South in the N-up grid) + unit Gaussian noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n].astype(float)
    x0 = y0 = (n - 1) / 2
    dx, dy = xx - x0, yy - y0
    ct, st = np.cos(theta), np.sin(theta)
    xr, yr = dx * ct + dy * st, -dx * st + dy * ct
    r = np.sqrt(xr ** 2 + (yr / (1 - eps)) ** 2)
    im = amp * np.exp(-7.67 * ((r / re_px) ** 0.25 - 1))
    if arc:
        rr, pa = np.hypot(dx, dy), np.arctan2(dy, dx)
        r_arc = 1.2 / jwst_fetch.PIX
        im = im + arc_amp * noise * np.exp(-0.5 * ((rr - r_arc) / 2.0) ** 2) * ((pa > np.radians(20)) & (pa < np.radians(140)))
    return im + rng.normal(0, noise, (n, n))


def _annulus(n: int = 320, lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - 159.5, xx - 159.5) * jwst_fetch.PIX
    return (r >= lo) & (r < hi)


def test_render_desc_path_is_shared():
    """panel.py reads the description from the same file render_v2.py writes (no import
    between them: the panel stays numpy-free at import)."""
    from lensjudge.golden import panel
    assert panel.RENDER_V2_DESC_PATH == render_v2.DESC_PATH
    assert panel.read_render_desc("v1") is None
    if render_v2.DESC_PATH.exists():
        assert panel.read_render_desc("v2r") == render_v2.desc_text() and "do NOT treat blue" in render_v2.desc_text()


def test_slot_geometry_matches_render_cutout():
    """The hard-coded slot (f) box is where render_cutout pastes its sixth panel: render a
    synthetic composite with a flat white sixth panel stand-in and read the pixels back."""
    assert render_v2.SLOT_F == (504, 292) and render_v2.LABEL_XY == (505, 277)
    assert render_v2.CROP_PX == 112 and render_v2.CENTRE == (159.5, 159.5)
    sw = sersic_arc(seed=1)
    meta = dict(id="Jsynth", ra=150.0, dec=2.0, mag_r=20.0, type="SER", proposal="0", sw_filter="F150W", lw_filter="F277W")
    comp = jwst_fetch.render_v1_composite(sw, None, meta)
    arr = np.asarray(comp)
    x, y = render_v2.SLOT_F
    # the outline rectangle render_cutout draws around every panel sits exactly on the slot edge
    assert tuple(arr[y, x]) == render_v2.OUTLINE and tuple(arr[y + 239, x + 239]) == render_v2.OUTLINE
    assert tuple(arr[y - 1, x - 1]) == render_v2.BG                  # outside the slot: canvas


def test_elliptical_model_removes_quadrupole_and_keeps_arc():
    sw, lw = sersic_arc(seed=1), sersic_arc(seed=2, amp=600.0)
    sh = render_v2.estimate_shape_central(sw, lw)
    assert abs(sh["eps"] - 0.4) < 0.05 and abs(np.degrees(sh["theta"]) - 30) < 3
    assert abs(sh["x0"] - 159.5) < 0.5 and abs(sh["y0"] - 159.5) < 0.5    # offset back to stamp coords
    ann = _annulus()
    sw0 = sersic_arc(seed=1, arc=False)                                   # deflector only
    chi0 = render_v2.chi_band(sw0, sh["eps"], sh["theta"])
    med, _ = jwst_fetch.sky(sw0)
    circ = jwst_fetch.radial_profile_subtract(sw0 - med)
    circ_chi = circ / render._robust_sigma(circ)
    rms_ell, rms_circ = float(np.sqrt(np.mean(chi0[ann] ** 2))), float(np.sqrt(np.mean(circ_chi[ann] ** 2)))
    assert rms_circ > 10 and rms_ell < 0.25 * rms_circ, (rms_ell, rms_circ)   # the butterfly is gone
    assert rms_ell < 2.0
    chi = render_v2.chi_band(sw, sh["eps"], sh["theta"])
    yy, xx = np.mgrid[:320, :320]
    r = np.hypot(yy - 159.5, xx - 159.5) * jwst_fetch.PIX
    pa = np.degrees(np.arctan2(yy - 159.5, xx - 159.5))
    arcmask = (np.abs(r - 1.2) < 0.1) & (pa > 30) & (pa < 130)
    assert float(np.median(chi[arcmask])) > 3.0 and float(chi[arcmask].max()) > 3.0    # the arc survives
    tile = render_v2.chi_tile(sw, sh["eps"], sh["theta"])
    assert tile.shape == (112, 112, 3) and tile.dtype == np.uint8
    # single-band shape estimate: the same band stacked thrice gives the same eps/theta
    sh1 = render_v2.estimate_shape_central(sw, None)
    assert abs(sh1["eps"] - sh["eps"]) < 0.05


def test_chi_panel_layout_two_tiles_and_single():
    sw, lw = sersic_arc(seed=1), sersic_arc(seed=2, amp=600.0)
    panel, info = render_v2.chi_panel(sw, lw)
    assert panel.size == (240, 240) and info["n_tiles"] == 2 and info["bands"] == ["SW", "LW"]
    a = np.asarray(panel)
    assert tuple(a[120, 119]) == (255, 255, 255) and tuple(a[120, 120]) == (255, 255, 255)   # the divider
    assert tuple(a[30, 120]) == render_v2.BG                                                  # above the tiles
    single, info1 = render_v2.chi_panel(None, lw)
    assert single.size == (240, 240) and info1["n_tiles"] == 1 and info1["bands"] == ["LW"]
    try:
        render_v2.chi_panel(None, None)
        raise AssertionError("both bands absent must raise")
    except ValueError:
        pass


def test_compose_v2_on_synthetic_stamp_dir():
    sw, lw = sersic_arc(seed=1), sersic_arc(seed=2, amp=600.0)
    meta = dict(id="Jsynth", ra=150.0, dec=2.0, mag_r=20.0, type="SER", proposal="0", sw_filter="F150W", lw_filter="F277W")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "stamps" / "Jsynth"
        d.mkdir(parents=True)
        for ch, arr, filt in (("SW", sw, "F150W"), ("LW", lw, "F277W")):
            jwst_fetch.write_stamp_fits(d / f"Jsynth_{ch}_10as.fits", arr,
                                        jwst_fetch.stamp_header("Jsynth", 150.0, 2.0, ch, filt, "obs", "", 10.0, 320, 1.0))
        v1 = jwst_fetch.render_v1_composite(sw, lw, meta)
        jwst_fetch.save_composite(v1, d / "Jsynth_v1.jpg")
        out = Path(tmp) / "Jsynth_v2r.jpg"
        rsha, dsha = render_v2.compose_v2(d, out)
        im = Image.open(out)
        assert im.size == (752, 562) and im.format == "JPEG"
        assert rsha == _util.sha_file(out) and len(rsha) == 16
        assert dsha == _util.sha_text(render_v2.DESC_PATH.read_text()) == render_v2.desc_sha16()
        # the five other panels are byte-identical to the v1 composite (same JPEG generation
        # on both sides: decode both and compare outside the replaced slot)
        a, b = np.asarray(Image.open(d / "Jsynth_v1.jpg").convert("RGB")).astype(int), np.asarray(im.convert("RGB")).astype(int)
        x, y = render_v2.SLOT_F
        outside = np.ones(a.shape[:2], bool)
        outside[y - render_v2.TH:y + 240, x:x + 240] = False
        diff = np.abs(a - b)
        assert diff[outside].mean() < 1.0 and np.percentile(diff[outside], 99.9) <= 12   # re-encoding noise only
        assert diff[y:y + 240, x:x + 240].mean() > 20            # the slot did change
        # deterministic: a second compose writes the same bytes
        rsha2, _ = render_v2.compose_v2(d, Path(tmp) / "again.jpg")
        assert rsha2 == rsha
        # the kit crop: 752x540 q92, sha of the served bytes
        ksha = render_v2.crop_to_kit(out, Path(tmp) / "kit.jpg")
        assert Image.open(Path(tmp) / "kit.jpg").size == (752, 540) and len(ksha) == 16
        # the CLI path on a one-row manifest (unpinned CSV is accepted)
        man = Path(tmp) / "truth_manifest.csv"
        pd.DataFrame({"name": ["Jsynth", "Jmissing"]}).to_csv(man, index=False)
        df = render_v2.run(man, Path(tmp) / "kits_v2r", Path(tmp) / "stamps", quiet=True)
        assert list(df["status"]) [0] == "ok" and df["status"].iloc[1] != "ok"
        assert df.iloc[0]["render_sha_v2r"] == _util.sha_file(Path(tmp) / "kits_v2r" / "Jsynth.jpg")
        assert df.iloc[0]["render_version"] == "jwst_v2r" and df.iloc[0]["n_tiles"] == 2
        assert (Path(tmp) / "kits_v2r" / "render_v2r.csv.sha").exists()
    assert render_v2.render_tag() == "jwst_v2r" and render_v2.render_tag("isophote", 20) == "jwst_v2r-isophote-lim20"


def test_real_stamp_compose_smoke():
    """J-dependent-ish: one real colour stamp and one gray stamp from golden/stamps, if present."""
    stamps = render_v2.STAMPS_DIR
    if not stamps.exists():
        print("  (skipped: golden/stamps absent)")
        return
    dirs = [d for d in sorted(stamps.iterdir()) if d.is_dir() and (d / f"{d.name}_v1.jpg").exists()]
    if not dirs:
        print("  (skipped: no stamp dirs)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        rsha, dsha, info = render_v2.compose_v2_info(dirs[0], Path(tmp) / "x.jpg")
        assert Image.open(Path(tmp) / "x.jpg").size == (752, 562) and info["n_tiles"] in (1, 2)


# ================================================================== analyze_truth
def synth_truth(n_pos: int = 40, n_neg: int = 200, n_sd: int = 20, seed: int = 3):
    """Manifest + two arms built so every headline number is known exactly:
      a1 (S): the 10 top negatives sit at 0.70..0.79, 30 positives >= 0.80, 10 positives in
              0.30..0.40, the other negatives in 0.25..0.45 -> recall@5%FPR = 30/40 = 0.75
              exactly (threshold 0.70);
      a0 (pass-count): 5 negatives at n_pass=1 (FPR 2.5%), the first 25 positives at 1
              -> recall@5%FPR = 25/40; a1 promotes positives 25..29 (5) and demotes none
              -> McNemar one-sided p = 0.5^5 = 0.03125; dAUC > 0."""
    rng = np.random.default_rng(seed)
    pos = [f"P{i:03d}" for i in range(n_pos)]
    neg = [f"N{i:03d}" for i in range(n_neg)]
    sd = [f"D{i:03d}" for i in range(n_sd)]
    anc = list(at.ANCHORS)
    names = pos + neg + sd + anc
    tc = ["cowls"] * 20 + ["lit_galaxy"] * 10 + ["lit_cluster"] * 10 + ["negative"] * n_neg + ["stress_D"] * n_sd + ["anchor"] * 5
    man = pd.DataFrame({"name": names, "truth_class": tc})
    man["is_positive"] = man["truth_class"].isin(["cowls", "lit_galaxy", "lit_cluster"])
    man["is_anchor"] = man["truth_class"].eq("anchor")
    man["is_stress"] = man["truth_class"].eq("stress_D")
    man["cowls_band"] = np.where(man["truth_class"].eq("cowls"), rng.choice(["strong", "marginal", "weak"], len(man)), "")
    theta = np.linspace(0.4, 1.98, 20)
    man["cowls_theta_E"] = np.nan
    man.loc[:19, "cowls_theta_E"] = theta
    man["layout"] = rng.choice(["color", "gray_sw_only"], len(man), p=[.9, .1])
    man["field_class"] = rng.choice(["cosmos", "cluster", "blank"], len(man))
    man["known_type"] = np.where(man["truth_class"].eq("lit_cluster"), "LeG", np.where(man["truth_class"].eq("lit_galaxy"), "gLS", ""))
    man["centre_is_deflector"] = man["truth_class"].isin(["cowls", "lit_galaxy"])
    man["half"] = "holdout"
    # the 11th-highest negative (exactly 0.45) sits ABOVE every weak positive (0.30..0.40), so
    # the last ROC point with FPR <= 5 % is exactly the 10th-highest negative, 0.70; the weak
    # positives still beat about half of the other negatives (0.25..0.45) so AUC(a1) > AUC(a0)
    neg_s = np.concatenate([np.linspace(0.70, 0.79, 10), [0.45], rng.uniform(0.25, 0.449, n_neg - 11)])
    # positives: the 20 COWLS get S DEcreasing with theta_E (rho = -1: the registered failure)
    pos_s = np.concatenate([np.linspace(0.99, 0.80, 20), rng.uniform(0.80, 0.99, 10), np.linspace(0.30, 0.40, 10)])
    neg_p = np.array([1] * 5 + [0] * (n_neg - 5), float)
    pos_p = np.array([1] * 25 + [0] * 15, float)

    def mk(scores, letters):
        return pd.DataFrame({"name": names, "p_lens": scores, "grade_pred": letters, "cost_usd": 0.05,
                             "parse_fail_roles": "", "escalate": False})
    a1 = mk(np.concatenate([pos_s, neg_s, rng.uniform(0, 0.3, n_sd), [0.9, 0.5, 0.45, 0.7, 0.1]]),
            ["A"] * 20 + ["B"] * 10 + ["C"] * 10 + ["C"] * 10 + ["D"] * (n_neg - 10) + ["D"] * 18 + ["C"] * 2 + ["A", "B", "B", "C", "D"])
    a0 = mk(np.concatenate([pos_p, neg_p, [0] * n_sd, [1, 2, 1, 1, 1]]),
            ["C"] * 25 + ["D"] * 15 + ["C"] * 5 + ["D"] * (n_neg - 5) + ["D"] * n_sd + ["C", "B", "C", "C", "C"])
    return man, a0, a1


def _load_man(man: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.csv"
        man.to_csv(p, index=False)
        return at.load_manifest(p, None)


def _get(res, stat, arm):
    r = res[(res["statistic"] == stat) & (res["arm"] == arm)]
    assert len(r) == 1, (stat, arm, len(r))
    return r.iloc[0]


def test_small_statistics():
    assert at.mcnemar_exact(5, 0) == (0.03125, 0.0625)
    assert np.isnan(at.mcnemar_exact(0, 0)[0])
    lo, hi = at.clopper_pearson(0, 10)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 10))) < 1e-9
    lo, hi = at.clopper_pearson(10, 10)
    assert hi == 1.0 and abs(lo - 0.025 ** (1 / 10)) < 1e-9
    assert np.isnan(at.clopper_pearson(0, 0)[0])
    y = np.array([1] * 4 + [0] * 20)
    s = np.array([0.9, 0.8, 0.3, 0.2] + [0.85] + [0.5] * 19)           # one negative above 0.8, 19 above the weak positives
    rec, thr, fpr = at.threshold_at_fpr(y, s, 0.05)
    assert rec == 0.5 and thr == 0.8 and fpr == 0.05
    rho, p, n = at.spearman_perm(np.arange(10.0), -np.arange(10.0), n_perm=50)
    assert abs(rho + 1.0) < 1e-9 and p < 0.1 and n == 10


def test_analyze_exact_values_on_synthetic_preds():
    man, a0, a1 = synth_truth()
    M = _load_man(man)
    assert int(M["y"].sum()) == 40 and M["is_anchor"].sum() == 5 and M["is_negative"].sum() == 200
    a1b = a1.copy()
    a1b.loc[a1b["name"] == "P000", "grade_pred"] = "B"                # one letter flip in replicate 2
    res, anchors = at.analyze(M, {"a0": [a0], "a1": [a1, a1b]}, "holdout", "a0",
                              {"t_A": 0.8, "t_B": 0.5}, n_boot=150, n_perm=100)
    assert list(res.columns) == ["statistic", "arm", "split", "value", "ci_lo", "ci_hi", "n"]
    # P1 exact
    r = _get(res, "P1_recall_at_fpr5pct", "a1")
    assert r["value"] == 0.75 and r["n"] == 40 and r["ci_lo"] < 0.75 < r["ci_hi"]
    assert abs(_get(res, "P1_threshold_at_fpr5pct", "a1")["value"] - 0.70) < 1e-12
    assert _get(res, "P1_achieved_fpr5pct", "a1")["value"] == 0.05
    assert _get(res, "P1_recall_at_fpr5pct", "a0")["value"] == 25 / 40
    assert _get(res, "P1_achieved_fpr5pct", "a0")["value"] == 0.025
    assert _get(res, "P1_roc_point1_tpr", "a0")["value"] == 25 / 40          # the 4-point ROC is written out
    lo, hi = at.clopper_pearson(30, 40)
    assert abs(r["ci_lo"] - lo) < 1e-12 and abs(r["ci_hi"] - hi) < 1e-12
    # McNemar: 5 promotions, 0 demotions -> p = 0.03125
    assert _get(res, "P1_mcnemar_promotions", "a1_vs_a0")["value"] == 5
    assert _get(res, "P1_mcnemar_demotions", "a1_vs_a0")["value"] == 0
    assert _get(res, "P1_mcnemar_p_onesided", "a1_vs_a0")["value"] == 0.03125
    # dAUC sign and CI
    d = _get(res, "dAUC", "a1_minus_a0")
    assert d["value"] > 0 and d["ci_lo"] <= d["value"] <= d["ci_hi"] and d["n"] == 240
    assert _get(res, "AUC", "a1")["value"] > _get(res, "AUC", "a0")["value"]
    # P2 / P3
    assert _get(res, "P2_fpr_letter_A", "a1")["value"] == 0.0 and _get(res, "P2_fpr_letter_AB", "a1")["n"] == 200
    assert abs(_get(res, "P2_fpr_at_t_B", "a1")["value"] - 10 / 200) < 1e-12     # ten negatives >= 0.5 (0.70..0.79)
    assert _get(res, "P2_fpr_at_t_A", "a1")["value"] == 0.0
    assert res[(res["statistic"] == "P2_fpr_at_t_A") & (res["arm"] == "a0")].empty   # not for a pass-count arm
    assert _get(res, "P3_positives_at_AB", "a1")["value"] == 0.75 and _get(res, "P3_positives_at_AB", "a0")["value"] == 0.0
    # per-stratum rows present, counts right
    assert _get(res, "recall_at_fpr5pct[scale_group=galaxy]", "a1")["n"] == 30
    assert _get(res, "recall_at_fpr5pct[scale_group=cluster]", "a1")["n"] == 10
    assert _get(res, "n_detected[scale_group=galaxy]", "a0")["value"] == 25
    assert _get(res, "recall_at_fpr5pct[theta_bin=theta_E<=1]", "a1")["value"] == 1.0
    assert not res[res["statistic"].str.startswith("recall_at_fpr5pct[cowls_band=")].empty
    assert not res[res["statistic"].str.startswith("recall_at_fpr5pct[layout=")].empty
    assert not res[res["statistic"].str.startswith("recall_at_fpr5pct[field_class=")].empty
    # Spearman on the 20 COWLS: S decreasing in theta_E -> rho = -1 (the registered failure mode)
    assert abs(_get(res, "spearman_S_vs_theta_E", "a1")["value"] + 1.0) < 1e-9 and _get(res, "spearman_S_vs_theta_E", "a1")["n"] == 20
    # stress / quality
    assert _get(res, "D_rate_stress_D", "a1")["value"] == 0.9 and _get(res, "D_rate_stress_D", "a0")["value"] == 1.0
    assert _get(res, "parse_failure_rate", "a1")["value"] == 0.0
    assert abs(_get(res, "cost_usd_per_item", "a1")["value"] - 0.05) < 1e-12
    assert abs(_get(res, "letter_flip_rate[replicate pairs]", "a1")["value"] - 1 / 265) < 1e-4   # grade_flip_rate rounds to 4 dp
    assert res[(res["statistic"] == "letter_flip_rate[replicate pairs]") & (res["arm"] == "a0")].empty   # k = 1
    # anchors: 5 anchors x 2 arms, never in the positive count, predictions attached
    assert len(anchors) == 10 and set(anchors["arm"]) == {"a0", "a1"}
    assert anchors.set_index(["candidate_id", "arm"]).loc[("J20954380-1094330", "a1"), "letter"] == "A"
    assert all(anchors["prediction"].str.len() > 0)
    md = at.summary_md(res, anchors, "holdout", "a0")
    assert "| a1 | 40 | 200 | 75% [59%, 87%]" in md and "a1 vs a0 | 5 | 0 | 0.0312" in md
    # reseeding: the same endpoint twice gives identical bootstrap CIs
    res2, _ = at.analyze(M, {"a0": [a0], "a1": [a1, a1b]}, "holdout", "a0", {"t_A": 0.8, "t_B": 0.5}, n_boot=150, n_perm=100)
    pd.testing.assert_frame_equal(res, res2)


def test_parse_failures_excluded_and_votes_audit():
    man, a0, a1 = synth_truth()
    M = _load_man(man)
    a1 = a1.copy()
    a1.loc[a1["name"].isin(["P030", "N000"]), ["p_lens", "parse_fail_roles"]] = [np.nan, "advocate"]
    a1["S_arb"] = a1["p_lens"] * 0.9
    votes = pd.DataFrame({"id": ["P000", "P000", "P000", "P001"], "role": ["advocate", "artifact", "geometry", "morphology"],
                          "raw": [json.dumps({"p_evidence": 0.6, "items": [{"k": 1, "visible_in_direct": True}]}),
                                  json.dumps({"no_opinion": False, "alternative": "subtraction_residual", "accounts_for": [1],
                                              "alternative_desc": "bowtie lobes", "notes": 'symmetric lobes at 0.3"'}),
                                  json.dumps({"no_opinion": True, "no_opinion_reason": "outside_competence", "alternative": None,
                                              "alternative_desc": "", "notes": ""}),
                                  json.dumps({"no_opinion": False, "alternative": "spiral_arm",
                                              "alternative_desc": "arm from the nucleus", "notes": 'radius grows 1.1" to 1.6" E'})]})
    res, _ = at.analyze(M, {"a0": [a0], "a1": [a1]}, "holdout", "a0", None, {"a1": [votes]}, n_boot=100, n_perm=50)
    assert _get(res, "n_positives_scored", "a1")["value"] == 39 and _get(res, "n_negatives_scored", "a1")["value"] == 199
    assert abs(_get(res, "parse_failure_rate", "a1")["value"] - 2 / 265) < 1e-12
    assert "a1arb" in set(res["arm"]) and not res[res["arm"] == "a1arb_minus_a1"].empty
    assert _get(res, "refutation_forbidden_only_rate", "a1")["value"] == 0.5
    assert _get(res, "refutation_over_subtraction_forbidden_rate", "a1")["value"] == 0.5
    assert _get(res, "refutation_uses_subtraction_residual_rate", "a1")["value"] == 0.5
    assert _get(res, "refutation_locates_feature_rate", "a1")["value"] == 1.0
    assert _get(res, "no_opinion_rate_votes[geometry]", "a1")["value"] == 1.0
    assert _get(res, "n_refutations", "a1")["value"] == 2


def test_write_outputs_merges_by_split():
    man, a0, a1 = synth_truth()
    M = _load_man(man)
    res, anchors = at.analyze(M, {"a0": [a0], "a1": [a1]}, "holdout", "a0", None, n_boot=50, n_perm=20)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "truth"
        rp, ap, mp = at.write_outputs(res, anchors, "holdout", "a0", prefix)
        assert rp.name == "truth_results.csv" and ap.name == "truth_anchors.csv" and mp.name == "truth_summary_holdout.md"
        d = res.copy(); d["split"] = "design"
        at.write_outputs(d, anchors.assign(split="design"), "design", "a0", prefix)
        at.write_outputs(res, anchors, "holdout", "a0", prefix)            # re-run replaces, never duplicates
        merged = pd.read_csv(rp)
        assert set(merged["split"]) == {"design", "holdout"} and len(merged) == 2 * len(res)
    # thresholds file: null model key -> provisional; the runner's key map (opus -> opus_api, not
    # Nate's opus_claude_code); the parquet's own t_A/t_B take precedence in analyze()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "thresholds_v2.json"
        p.write_text(json.dumps({"sonnet_api": {"tau0": 0.15, "t_A": None, "t_B": None}, "opus_claude_code": {"tau0": 0.15, "t_A": 0.9, "t_B": 0.6},
                                 "opus_api": None, "provisional": {"tau0": 0.15, "t_A": 0.80, "t_B": 0.50}}))
        t = at.load_thresholds(p, "sonnet")
        assert t["t_A"] == 0.8 and t["source"] == "provisional"
        assert at.load_thresholds(p, "opus")["source"] == "provisional"          # opus_api null, not Nate's block
        assert at.load_thresholds(Path(tmp) / "missing.json", "sonnet") is None
    rows = [a1.assign(t_A=0.63, t_B=0.41, tau0=0.15, letter_source="sonnet_api_calibrated")]
    assert at.thresholds_from_preds(rows) == {"t_A": 0.63, "t_B": 0.41, "tau0": 0.15, "source": "sonnet_api_calibrated"}
    assert at.thresholds_from_preds([a1]) is None
    try:
        at.thresholds_from_preds(rows + [a1.assign(t_A=0.7, t_B=0.41, tau0=0.15)]); raise AssertionError("no raise")
    except SystemExit:
        pass
    res_p, _ = at.analyze(M, {"a1": rows}, "holdout", "a0", {"t_A": 0.8, "t_B": 0.5}, n_boot=20, n_perm=10)
    assert _get(res_p, "P2_t_A", "a1")["value"] == 0.63 and _get(res_p, "P2_t_B", "a1")["value"] == 0.41
    assert _get(res_p, "P2_holds_t_B", "a1")["value"] in (0.0, 1.0) and _get(res_p, "P2_upper_ci_ok_t_A", "a1") is not None
    # the registered P2 (CP lower bound <= target) vs the plan's upper-CI wording, on the
    # synthetic a1 (0/200 at t_A, 10/200 at t_B)
    res_t, _ = at.analyze(M, {"a1": [a1]}, "holdout", "a0", {"t_A": 0.8, "t_B": 0.5}, n_boot=20, n_perm=10)
    assert _get(res_t, "P2_holds_t_A", "a1")["value"] == 1.0 and _get(res_t, "P2_upper_ci_ok_t_A", "a1")["value"] == 1.0
    assert _get(res_t, "P2_holds_t_B", "a1")["value"] == 1.0 and _get(res_t, "P2_upper_ci_ok_t_B", "a1")["value"] == 0.0   # 10/200: upper 8.9 %
    assert abs(_get(res_t, "P2_fpr_at_t_B", "a1")["ci_lo"] - 0.0243) < 1e-3 and abs(_get(res_t, "P2_fpr_at_t_B", "a1")["ci_hi"] - 0.0897) < 1e-3
    # discovery: k in the name groups tuples (a1 = k1, a1k3 = the replicate study), the pre-k
    # name reads as k=1, votes parquets are ignored, replicates sort by r; on the holdout a
    # parquet without .meta.json, a row count != meta.n or a subset is REFUSED (design: warned)
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        for k in (2, 1):
            (tp / f"preds_truth_a1_sonnet_holdout_k3_r{k}.parquet").write_bytes(b"")
            (tp / f"preds_truth_a1_sonnet_holdout_k3_r{k}_votes.parquet").write_bytes(b"")
        (tp / "preds_truth_a1_sonnet_holdout_r1.parquet").write_bytes(b"")
        found = at.discover(tp, "sonnet", "holdout", strict=False)
        assert [p.name for p in found["a1k3"]] == ["preds_truth_a1_sonnet_holdout_k3_r1.parquet", "preds_truth_a1_sonnet_holdout_k3_r2.parquet"]
        assert [p.name for p in found["a1"]] == ["preds_truth_a1_sonnet_holdout_r1.parquet"] and "a0" not in found
        try:
            at.discover(tp, "sonnet", "holdout"); raise AssertionError("no raise")
        except SystemExit as e:
            assert "no .meta.json" in str(e)
        # complete replicates with matching metas pass; a tuple mismatch within a group and a
        # subset / short parquet are refused on the holdout
        a1.head(10).to_parquet(tp / "preds_truth_a2_sonnet_holdout_k1_r1.parquet", index=False)
        meta = {"tuple": {"arm": "a2", "k": 1, "persona_set_sha16": "p" * 16}, "n": 10, "split": "holdout", "limit": 0, "ids_file": None}
        (tp / "preds_truth_a2_sonnet_holdout_k1_r1.meta.json").write_text(json.dumps(meta))
        ok = at.discover(tp, "sonnet", "holdout", arms=("a2",), n_expected=10)
        assert [p.name for p in ok["a2"]] == ["preds_truth_a2_sonnet_holdout_k1_r1.parquet"]
        try:
            at.discover(tp, "sonnet", "holdout", arms=("a2",), n_expected=282); raise AssertionError("no raise")
        except SystemExit as e:
            assert "the holdout half has 282 rows" in str(e)
        a1.head(9).to_parquet(tp / "preds_truth_a2_sonnet_holdout_k1_r1.parquet", index=False)
        try:
            at.discover(tp, "sonnet", "holdout", arms=("a2",)); raise AssertionError("no raise")
        except SystemExit as e:
            assert "9 rows but meta.n = 10" in str(e)
        a1.head(10).to_parquet(tp / "preds_truth_a2_sonnet_holdout_k2_r1.parquet", index=False)
        a1.head(10).to_parquet(tp / "preds_truth_a2_sonnet_holdout_k2_r2.parquet", index=False)
        (tp / "preds_truth_a2_sonnet_holdout_k2_r1.meta.json").write_text(json.dumps({**meta, "tuple": {**meta["tuple"], "k": 2}}))
        (tp / "preds_truth_a2_sonnet_holdout_k2_r2.meta.json").write_text(json.dumps({**meta, "tuple": {**meta["tuple"], "k": 2, "persona_set_sha16": "q" * 16}}))
        try:
            at.discover(tp, "sonnet", "holdout", arms=("a2",)); raise AssertionError("no raise")
        except SystemExit as e:
            assert "2 different tuples" in str(e)
        found_d = at.discover(tp, "sonnet", "holdout", arms=("a2",), strict=False)      # warned, not refused
        assert set(found_d) == {"a2", "a2k2"}


# ================================================================== runner
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
