"""Repo-side test of scripts/aggregate_v2.py (runs as a script or under pytest; no data needed).

Two synthetic candidates modelled on the diagnosis: a rank-15-like object (a 3.6" arc with a
counter-image; the only critique is a soft scale_tension) and a rank-13-like object (a 0.7"
feature that the morphology critic identifies as a spiral arm with a located, covering box).
Pre-registered behaviour: S(15) > S(13); 13 is a D by the D rule; a critic whose location box
does not overlap the item it claims to cover contributes nothing (the geometric guard); the
legacy pass-count table reproduces 09_rank_report.py:grade(); U ranks below every examined item.

  python tests/test_aggregate_v2.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import aggregate_v2 as agg  # noqa: E402

THR = {"tau0": 0.15, "t_A": 0.80, "t_B": 0.50, "letter_source": "provisional"}


def advocate(p, items, **crit):
    c = {"source_contrast": 5, "low_surface_brightness": 5, "curvature": 5, "counter_image": 5, "arc_morphology": 5}
    c.update(crit)
    return {"id": "x", "persona": "advocate", "criteria": c, "items": items, "arc_radius_arcsec": None,
            "arc_pa_span_deg": None, "counter_image_pos": None, "centre_of_curvature_offset_arcsec": None,
            "scale_class": "galaxy", "n_red_neighbours_10as": 0, "bcg_like_halo": False,
            "deflector_is_centre": True, "p_evidence": p, "nothing_because": "", "notes": ""}


def item(k, r, pa0, pa1, panel="d"):
    return {"k": k, "what": "arc", "panel": panel, "r_arcsec": r, "pa_deg_from": pa0, "pa_deg_to": pa1,
            "visible_in_direct": True, "criteria": [3, 5]}


def critic(persona, alt, r, covers, loc, no_opinion=False):
    return {"id": "x", "persona": persona, "no_opinion": no_opinion,
            "no_opinion_reason": "outside_competence" if no_opinion else None,
            "alternative": None if no_opinion else alt, "alternative_desc": alt or "",
            "location": loc, "accounts_for": covers, "leaves_standing": [], "refutation_strength": r,
            "measured": None, "scale_class": None, "notes": ""}


def box(r0, r1, pa0, pa1):
    return {"r_arcsec_from": r0, "r_arcsec_to": r1, "pa_deg_from": pa0, "pa_deg_to": pa1}


RANK15 = advocate(0.80, [item(1, 3.6, 40, 170), item(2, 3.2, 220, 250)], curvature=8, counter_image=7, arc_morphology=8)
RANK15_CRITICS = [critic("artifact", None, 0.0, [], None),
                  critic("geometry", "scale_tension", 0.4, [1], box(3.0, 4.2, 30, 180)),
                  critic("morphology", None, 0.0, [], None, no_opinion=True)]
RANK13 = advocate(0.60, [item(1, 0.7, 10, 120)], curvature=4, counter_image=1, arc_morphology=3)
RANK13_CRITICS = [critic("artifact", None, 0.0, [], None),
                  critic("geometry", None, 0.0, [], None, no_opinion=True),
                  critic("morphology", "spiral_arm", 0.9, [1], box(0.4, 1.2, 0, 140))]


def test_coverage_and_scores():
    s15 = agg.score_S(RANK15, RANK15_CRITICS)
    s13 = agg.score_S(RANK13, RANK13_CRITICS)
    assert abs(s15 - 0.80 * (1 - 0.4 * 0.5)) < 1e-9, s15       # scale_tension covers 1 of 2 items
    assert abs(s13 - 0.60 * (1 - 0.9 * 1.0)) < 1e-9, s13       # spiral arm covers the only item
    assert s15 > s13


def test_geometric_guard():
    far = [critic("morphology", "spiral_arm", 0.9, [1], box(2.5, 3.5, 200, 300))]   # box misses the item
    assert abs(agg.score_S(RANK13, far) - 0.60) < 1e-9
    assert agg.coverage_fraction(RANK13["items"], far[0]) == 0.0
    assert agg.covers(RANK13["items"][0], box(0.4, 1.2, 350, 30))                    # PA wrap-around
    assert not agg.covers(RANK13["items"][0], None)
    # a near-ring written in wrap form (100 -> 80 = 340 deg) stays a ring after the +-20 deg
    # padding; a critic box on its far side covers it
    ring = item(1, 1.0, 100, 80)
    assert agg.covers(ring, box(0.5, 2.0, 180, 220)) and agg.covers(ring, box(0.5, 2.0, 180, 220), dpa=0.0)
    assert not agg.covers(item(1, 1.0, 100, 120), box(0.5, 2.0, 180, 220))


def test_letters():
    l15, src = agg.assign_letter(agg.score_S(RANK15, RANK15_CRITICS), RANK15, RANK15_CRITICS, THR)
    assert l15 in ("A", "B") and src == "provisional", (l15, src)
    l13, _ = agg.assign_letter(agg.score_S(RANK13, RANK13_CRITICS), RANK13, RANK13_CRITICS, THR)
    assert l13 == "D", l13
    nothing = advocate(0.02, [])
    nothing["nothing_because"] = "isolated elliptical"
    assert agg.assign_letter(0.02, nothing, [], THR)[0] == "D"
    weak = advocate(0.30, [item(1, 1.0, 0, 90)])
    assert agg.assign_letter(0.30, weak, [], THR)[0] == "C"        # located evidence below t_B, no D ground


def test_arbitrated():
    arb = {"id": "x", "persona": "arbitrator",
           "rulings": [{"persona": "morphology", "ruling": "overruled", "covers": [], "why": "no bridge"}],
           "surviving_items": [1], "letter_llm": "B", "scale_class_final": "galaxy", "needs_human": False,
           "rationale": ""}
    assert abs(agg.score_S_arb(RANK13, RANK13_CRITICS, arb) - 0.60) < 1e-9
    assert agg.assign_letter(0.60, RANK13, RANK13_CRITICS, THR, arbitrator=arb)[0] != "D"


def test_passcount_and_rank():
    v = lambda p, x: {"id": "x", "persona": p, "verdict": x, "alternative": "", "notes": ""}   # noqa: E731
    assert agg.passcount_incumbent([v("artifact", "pass"), v("geometry", "pass"), v("morphology", "pass")])[3] == "A"
    assert agg.passcount_incumbent([v("artifact", "pass"), v("geometry", "fail"), v("morphology", "pass")])[3] == "B"
    assert agg.passcount_incumbent([v("artifact", "pass"), v("geometry", "fail"), v("morphology", "uncertain")])[3] == "C"
    assert agg.passcount_incumbent([v("artifact", "fail"), v("geometry", "fail"), v("morphology", "fail")])[3] == "D"
    assert agg.passcount_incumbent([])[3] == "U"
    rows = [{"id": "u", "S": float("nan"), "p_evidence": float("nan"), "examined": False, "confidence": 90.0},
            {"id": "lo", "S": 0.05, "p_evidence": 0.1, "examined": True, "confidence": 10.0},
            {"id": "hi", "S": 0.70, "p_evidence": 0.8, "examined": True, "confidence": 20.0}]
    order = [r["id"] for r in sorted(rows, key=agg.rank_key)]      # ascending => best first
    assert order == ["hi", "lo", "u"], order
    assert agg.rank_score(rows[0]) < 0 < agg.rank_score(rows[1])    # U strictly below examined


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
