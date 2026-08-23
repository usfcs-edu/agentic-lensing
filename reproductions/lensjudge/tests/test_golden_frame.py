#!/usr/bin/env python3
"""No-network tests for golden/build_frame.py (WP-B, the LITE sampling frame).

Pure-logic tests run on synthetic DataFrames (no file dependencies): the 10" union-find, the
closest-match known-lens dedup, largest-remainder allocation, the D_refuted substratum fill
chain, the seeded unit_id shuffle, the exact contract schema, and an end-to-end build over a
small synthetic universe. One smoke test builds the REAL frame into a temp dir when the
jwst-strong-lens-search checkout and the parity outputs are present, and is skipped otherwise.

    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_frame.py
(also pytest-compatible)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402
from lensjudge.golden import build_frame as bf  # noqa: E402

ARCSEC = 1.0 / 3600.0

# the contract's frame.csv schema, copied verbatim from GOLDEN_CONTRACT.md (not imported, so a
# drift in build_frame.FRAME_COLUMNS is caught here)
CONTRACT_COLUMNS = [
    "unit_id", "candidate_id", "alias_ids", "system_id", "ra_deg", "dec_deg", "stratum", "substratum",
    "pipe_grade_passcount", "pipe_inspector_conf", "pipe_score", "center_galaxy_type", "rank_top100",
    "prior_exposure", "lit_known", "layout", "cowls_ranking", "cowls_theta_E", "known_lens_name",
    "known_type", "known_sep_arcsec", "desi_pool_overlap", "desi_pool_split", "sw_obs", "lw_obs",
    "sw_filter", "lw_filter", "proposal", "field_id",
]


# ------------------------------------------------------------------ union-find
def test_union_find_merges_close_pairs_not_12arcsec():
    ra = np.array([180.0, 180.0, 51.0, 51.0, 10.0, 10.0, 100.0])
    dec = np.array([23.0, 23.0 + 1.17 * ARCSEC,          # the rank 7/14 twin (1.17")
                    -13.0, -13.0 + 8.78 * ARCSEC,         # the rank 16/17 pair (8.78")
                    5.0, 5.0 + 12.0 * ARCSEC,             # 12" apart: NOT a system
                    0.0])
    sid = bf.union_find_systems(ra, dec, 10.0)
    assert sid[0] == sid[1], "1.17\" pair must share a system"
    assert sid[2] == sid[3], "8.78\" pair must share a system"
    assert sid[4] != sid[5], "12\" pair must not be merged"
    assert len({sid[0], sid[2], sid[4], sid[5], sid[6]}) == 5
    assert sid.min() == 1 and sid.max() == 5          # dense, 1-based, by first appearance
    assert list(sid) == [1, 1, 2, 2, 3, 4, 5]


def test_union_find_is_transitive():
    # a chain a-b-c with a-c at 16" (> radius) still forms one system through b
    ra = np.array([30.0, 30.0, 30.0])
    dec = np.array([1.0, 1.0 + 8 * ARCSEC, 1.0 + 16 * ARCSEC])
    sid = bf.union_find_systems(ra, dec, 10.0)
    assert sid[0] == sid[1] == sid[2]
    assert list(bf.union_find_systems([1.0], [1.0])) == [1]
    assert len(bf.union_find_systems([], [])) == 0


# ------------------------------------------------------------------ known-lens dedup
def test_dedup_keeps_closest_match():
    known = pd.DataFrame({
        "lens_name": ["far name", "closest name", "mid name", "other cutout", "cowls row", "too far"],
        "lens_src": ["SIMBAD:LeG", "SIMBAD:LeI", "SIMBAD:gLS", "SIMBAD:gLS", "COWLS", "SIMBAD:gLS"],
        "cutout_id": ["J1", "J1", "J1", "J2", "J3", "J4"],
        "sep_arcsec": [1.5, 0.3, 0.9, 1.99, 0.1, 2.0],
        "flagged": [True] * 6, "grade": ["U"] * 6,
    })
    d = bf.dedup_known_lenses(known, 2.0)
    assert list(d["cutout_id"]) == ["J1", "J2"]                 # COWLS and >= 2" dropped
    assert d.loc[d["cutout_id"] == "J1", "lens_name"].item() == "closest name"
    assert d.loc[d["cutout_id"] == "J1", "sep_arcsec"].item() == 0.3


# ------------------------------------------------------------------ allocation
def test_largest_remainder_allocation():
    alloc, left = bf.largest_remainder({"a": 342, "b": 342, "c": 211, "d": 73}, 20)
    assert sum(alloc.values()) == 20 and left == 0
    assert alloc == {"a": 7, "b": 7, "c": 4, "d": 2}
    # capacity caps and redistributes; a key missing from the capacity dict has ZERO rows
    alloc, left = bf.largest_remainder({"a": 10, "b": 10, "c": 10}, 9, capacity={"a": 1, "b": 100})
    assert alloc == {"a": 1, "b": 8} and left == 0
    alloc, left = bf.largest_remainder({"a": 1, "b": 1}, 5, capacity={"a": 1, "b": 1})
    assert alloc == {"a": 1, "b": 1} and left == 3                 # honest about the shortfall
    assert bf.largest_remainder({}, 5) == ({}, 5)


# ------------------------------------------------------------------ D_refuted fill chain
def _synthetic_D(n_merger=4, n_ring=6, n_spiral=4, n_ell=20, n_other=16):
    types = (["merger"] * n_merger + ["ring"] * n_ring + ["spiral"] * n_spiral
             + ["elliptical"] * n_ell + ["s0"] * (n_other // 2) + ["ambiguous"] * (n_other - n_other // 2))
    n = len(types)
    conf = np.linspace(90, 30, n)          # ellipticals get distinct, decreasing confidences
    return pd.DataFrame({"id": [f"D{i:03d}" for i in range(n)], "center_galaxy_type": types,
                         "confidence": conf, "score": conf / 100.0})


def test_refuted_substratum_fill():
    D = _synthetic_D()
    sel, notes = bf.select_refuted(D, np.random.default_rng(1), n_per=10)
    assert len(sel) == 40 and sel["id"].is_unique
    counts = sel["substratum"].value_counts().to_dict()
    # merger short by 6 -> ring_spiral (10 of 10 already used) -> elliptical supplies the 6
    assert counts == {"merger": 4, "ring_spiral": 10, "elliptical_nearmiss": 16, "other": 10}
    ell = sel[sel["substratum"] == "elliptical_nearmiss"]
    top = D[D["center_galaxy_type"] == "elliptical"].nlargest(16, "confidence")
    assert set(ell["id"]) == set(top["id"]), "near-miss ellipticals must be the highest-confidence ones"
    assert any("merger: only 4/10" in n for n in notes)
    assert any("filled 6 from elliptical_nearmiss" in n for n in notes)
    # a well-stocked pool fills every family from itself and reports nothing
    sel2, notes2 = bf.select_refuted(_synthetic_D(12, 8, 8, 20, 20), np.random.default_rng(1), 10)
    assert notes2 == [] and sel2["substratum"].value_counts().to_dict() == {
        "merger": 10, "ring_spiral": 10, "elliptical_nearmiss": 10, "other": 10}
    # determinism under the seed
    sel3, _ = bf.select_refuted(D, np.random.default_rng(1), n_per=10)
    assert list(sel3["id"]) == list(sel["id"])


# ------------------------------------------------------------------ unit ids
def test_unit_id_order_is_shuffled():
    n = 120
    strata = ["T_verified"] * 30 + ["K_cowls"] * 30 + ["D_refuted"] * 30 + ["N_unflagged"] * 30
    frame = pd.DataFrame({"candidate_id": [f"C{i}" for i in range(n)], "stratum": strata,
                          "ra_deg": 10.0 + 0.05 * np.arange(n), "dec_deg": np.zeros(n)})
    out = bf.assign_unit_ids(frame, np.random.default_rng(_util.SEED))
    assert list(out["unit_id"]) == [f"u{i + 1:04d}" for i in range(n)]
    assert set(out["candidate_id"]) == set(frame["candidate_id"])
    assert list(out["stratum"]) != strata, "unit_id order must not follow the stratum blocks"
    # strata interleave: the longest run of one stratum is far shorter than a block
    runs, best, cur = [], 1, 1
    for a, b in zip(out["stratum"], out["stratum"][1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    assert best < 10
    # and the rank correlation between unit number and original position is weak
    orig = out["candidate_id"].str[1:].astype(int).to_numpy()
    assert abs(np.corrcoef(orig, np.arange(n))[0, 1]) < 0.3
    assert out["system_id"].nunique() == n                  # 180" spacing -> every row its own system
    out2 = bf.assign_unit_ids(frame, np.random.default_rng(_util.SEED))
    assert list(out2["candidate_id"]) == list(out["candidate_id"])   # seeded -> reproducible


# ------------------------------------------------------------------ schema
def test_layout_uses_run_finite_fraction():
    """layout = gray when a channel is missing OR the run's cutout of it had < MIN_FINITE finite
    pixels (the F444W maskbar case: LW observation present, finite_lw 0.0). Presence-only
    when no render manifest is given; both channels unusable raises."""
    ids = pd.Series(["a", "b", "c", "d", "e"])
    sw = pd.Series(["s", None, "s", "s", "s"], dtype=object)
    lw = pd.Series(["l", "l", None, "l", "l"], dtype=object)
    render = pd.DataFrame({"id": ["a", "b", "c", "d", "zzz"],
                           "finite_sw": [1.0, 0.9, 1.0, 1.0, 0.0],
                           "finite_lw": [1.0, 1.0, 0.2, 0.0, 0.0]})   # 'e' absent -> assume 1.0
    got = list(bf.derive_layout(sw, lw, render, ids))
    assert got == ["color", "gray_lw_only", "gray_sw_only", "gray_sw_only", "color"], got
    # presence only
    got = list(bf.derive_layout(sw, lw, None, ids))
    assert got == ["color", "gray_lw_only", "gray_sw_only", "color", "color"], got
    # exactly at the gate is usable; both channels empty is an error, not a silent layout
    render2 = pd.DataFrame({"id": ["a"], "finite_sw": [0.55], "finite_lw": [0.54]})
    assert list(bf.derive_layout(sw[:1], lw[:1], render2, ids[:1])) == ["gray_sw_only"]
    try:
        bf.derive_layout(sw[:1], lw[:1],
                                  pd.DataFrame({"id": ["a"], "finite_sw": [0.1], "finite_lw": [0.1]}),
                                  ids[:1])
    except ValueError as e:
        assert "no renderable channel" in str(e)
    else:
        raise AssertionError("both-empty layout must raise")


def test_schema_columns_exact():
    assert list(bf.FRAME_COLUMNS) == CONTRACT_COLUMNS
    assert len(set(bf.FRAME_COLUMNS)) == len(bf.FRAME_COLUMNS)
    assert set(bf.STRATA) == {"T_verified", "T_U", "K_cowls", "L_known", "D_refuted", "U_tail", "N_unflagged"}


# ------------------------------------------------------------------ synthetic universe, end to end
def _universe():
    """400 targets on a 72" grid; top-20 'master' with a 1.17" twin (ranks 7/8) and an 8.78" pair
    (ranks 16/17); 200 flagged (U 1-150 except the verified 1-8, D 151-200); 6 COWLS (two inside the
    U_tail bands / D pool); a literature table with duplicate names, COWLS/top-N collisions and a
    > 2" miss; DESI positions on the top candidate and on ten unflagged targets. U_tail bands sit
    above the real TOP_N (100) because that constant is not parametrised."""
    n = 400
    ids = [f"S{i:04d}" for i in range(n)]
    ra = 150.0 + 0.02 * (np.arange(n) % 20)
    dec = 2.0 + 0.02 * (np.arange(n) // 20)
    dec[7] = dec[6] + 1.17 * ARCSEC; ra[7] = ra[6]          # rank 8 = twin of rank 7
    dec[16] = dec[15] + 8.78 * ARCSEC; ra[16] = ra[15]      # rank 17 = 8.78" from rank 16
    rank = np.arange(1, n + 1)
    flagged = rank <= 200
    grade = np.where(rank <= 8, np.array(list("AABBBCCC") + [""] * (n - 8)),
                     np.where(rank <= 150, "U", np.where(rank <= 200, "D", "")))
    ctype = np.array(["elliptical"] * n, dtype=object)
    for r0, r1, t in ((151, 154, "s0"), (155, 158, "merger"), (159, 164, "ring"), (165, 168, "spiral"),
                      (169, 188, "elliptical"), (189, 196, "ambiguous"), (197, 200, "s0")):
        ctype[r0 - 1:r1] = t
    conf = np.where(flagged, 100.0 - 0.2 * rank, 0.0)
    conf[168:188] = 90.0 - 2.0 * np.arange(20)               # ellipticals: ranks 169.. highest first
    score = np.where(flagged, 4.0 - 0.005 * rank, 0.0)
    proposal = np.where(rank <= 100, 1000, np.where(rank <= 160, 2000, 3000))   # flagged mix 100:60:40
    proposal[200:] = np.array([1000, 2000, 3000, 4000] * 50)                    # 4000 has no flagged rows
    status = np.array(["ok"] * n, dtype=object); status[-5:] = "no_coverage"
    results = pd.DataFrame({"id": ids, "ra": ra, "dec": dec, "grade": grade, "confidence": conf,
                            "score": score, "center_galaxy_type": ctype, "rank": rank})
    inspections = pd.DataFrame({"id": ids, "ra": ra, "dec": dec, "status": status, "flagged": flagged,
                                "proposal": proposal, "confidence": conf, "center_galaxy_type": ctype})
    master = pd.DataFrame({"rank": rank[:20], "candidate_id": ids[:20], "ra_deg": ra[:20], "dec_deg": dec[:20],
                           "verifier_grade": grade[:20],
                           "discovery_status": ["known", "new", "field_match"] + ["new"] * 17})
    k_ids = ["S0104", "S0030", "S0150", "S0300", "S0301", "S0302"]     # ranks 105, 31, 151, 301-303
    r = results.set_index("id")
    controls = pd.DataFrame({"id": k_ids, "ra": r.loc[k_ids, "ra"].to_numpy(), "dec": r.loc[k_ids, "dec"].to_numpy(),
                             "cowls_code": [f"COSJ{i}" for i in range(6)],
                             "einstein_radius": [0.5, 1.0, np.nan, 0.7, 0.9, 1.2], "sep_arcsec": [0.1] * 6,
                             "ranking": ["AAAAAB", None, "BBUXXX", "UUXXXX", "ABBBXX", "AABXXX"],
                             "flagged": [True, True, True, False, False, False],
                             "grade": ["U", "U", "D", None, None, None], "rank": [105, 31, 151, 301, 302, 303]})
    known = pd.DataFrame({
        "lens_name": ["Lens A", "Lens A2", "Lens A0", "Lens B", "Lens C", "Lens D", "Lens E", "Lens F",
                      "Lens G", "Lens H", "Lens I", "COSJ0"],
        "lens_src": ["SIMBAD:gLS", "SIMBAD:LeG", "SIMBAD:LeI", "SIMBAD:gLS", "SIMBAD:LeG", "SIMBAD:LeI",
                     "SIMBAD:gLS", "SIMBAD:gLS", "SIMBAD:gLS", "SIMBAD:gLS", "SIMBAD:gLe", "COWLS"],
        "cutout_id": ["S0110", "S0110", "S0110", "S0111", "S0152", "S0153", "S0310", "S0311", "S0312",
                      "S0000", "S0300", "S0104"],
        "sep_arcsec": [0.5, 1.5, 0.2, 1.9, 0.8, 1.2, 0.3, 1.0, 2.5, 0.1, 0.2, 0.1],
        "flagged": [True, True, True, True, True, True, False, False, False, True, False, True],
        "grade": ["U", "U", "U", "U", "D", "D", None, None, None, "A", None, "U"]})
    sw_obs = np.array(["jwXXXX_sw"] * n, dtype=object); sw_obs[1] = None
    lw_obs = np.array(["jwXXXX_lw"] * n, dtype=object); lw_obs[2] = None
    targets = pd.DataFrame({"id": ids, "ra": ra, "dec": dec, "sw_obs": sw_obs, "lw_obs": lw_obs,
                            "sw_filter": "F150W", "lw_filter": "F277W", "proposal": proposal.astype(str),
                            "field_id": np.arange(n) // 40})
    desi_idx = [0] + list(range(200, 210))
    desi = pd.DataFrame({"ra": ra[desi_idx], "dec": dec[desi_idx] + 0.5 * ARCSEC,
                         "split": ["valsel"] + ["train"] * 10, "label_source": ["graded"] + ["random_neg"] * 10})
    return dict(results=results, inspections=inspections, master=master, controls=controls, known=known,
                targets=targets, desi=desi)


_QUOTAS = {"L_known": 4, "D_refuted": 10, "U_tail": ((101, 130, 3), (131, None, 2)), "N_unflagged": 5}


def test_synthetic_universe_end_to_end():
    src = _universe()
    frame, S = bf.build_frame(src, np.random.default_rng(_util.SEED), _QUOTAS)
    assert list(frame.columns) == CONTRACT_COLUMNS
    assert len(frame) == 19 + 6 + 4 + 40 + 5 + 5 == 79
    assert frame["candidate_id"].is_unique
    assert list(frame["unit_id"]) == [f"u{i + 1:04d}" for i in range(len(frame))]
    counts = frame["stratum"].value_counts().to_dict()
    assert counts == {"T_verified": 7, "T_U": 12, "K_cowls": 6, "L_known": 4, "D_refuted": 40,
                      "U_tail": 5, "N_unflagged": 5}
    f = frame.set_index("candidate_id")
    # --- top-100 aliasing and systems
    assert "S0007" not in f.index and f.at["S0006", "alias_ids"] == "S0007"
    assert S["aliases"] == {"S0006": ["S0007"]}
    assert f.at["S0015", "system_id"] == f.at["S0016", "system_id"]
    assert (frame.groupby("system_id").size() > 1).sum() == 1       # the only multi-member system
    assert f.at["S0015", "stratum"] == "T_U" and f.at["S0016", "stratum"] == "T_U"
    # --- prior exposure / lit_known / layout / pipe columns on the top-N rows
    top = frame[frame["rank_top100"].notna()]
    assert sorted(top["rank_top100"].astype(int)) == [r for r in range(1, 21) if r != 8]   # rank 8 collapsed
    assert (top.loc[top["rank_top100"] <= 15, "prior_exposure"] == 2).all()
    assert (top.loc[top["rank_top100"] > 15, "prior_exposure"] == 1).all()
    assert (frame.loc[frame["rank_top100"].isna(), "prior_exposure"] == 0).all()
    assert f.at["S0000", "lit_known"] and f.at["S0002", "lit_known"] and not f.at["S0001", "lit_known"]
    assert f.at["S0001", "layout"] == "gray_lw_only" and f.at["S0002", "layout"] == "gray_sw_only"
    assert f.at["S0000", "layout"] == "color" and f.at["S0001", "sw_obs"] == ""
    assert f.at["S0000", "pipe_grade_passcount"] == "A" and f.at["S0000", "substratum"] == "A"
    assert f.at["S0000", "known_lens_name"] == "Lens H" and f.at["S0000", "known_type"] == "SIMBAD:gLS"
    assert f.at["S0000", "desi_pool_overlap"] and f.at["S0000", "desi_pool_split"] == "valsel"
    assert float(f.at["S0000", "pipe_inspector_conf"]) == 99.8
    # --- COWLS
    K = frame[frame["stratum"] == "K_cowls"].set_index("candidate_id")
    assert set(K.index) == {"S0104", "S0030", "S0150", "S0300", "S0301", "S0302"}
    assert K["lit_known"].all() and (K["known_type"] == "COWLS").all()
    assert K.at["S0300", "known_lens_name"] == "COSJ3"            # COWLS code wins over the SIMBAD name
    assert K.at["S0030", "cowls_ranking"] == "" and K.at["S0104", "cowls_ranking"] == "AAAAAB"
    assert np.isnan(K.at["S0150", "cowls_theta_E"]) and K.at["S0150", "substratum"] == "pipe_D"
    assert K.at["S0300", "substratum"] == "unflagged" and np.isnan(K.at["S0300", "pipe_inspector_conf"])
    assert K.at["S0300", "pipe_grade_passcount"] == "" and np.isnan(K.at["S0300", "pipe_score"])
    # --- L_known: both pipe-D fixed, one U + one unflagged fill; never a top-N or COWLS cutout
    Lk = frame[frame["stratum"] == "L_known"].set_index("candidate_id")
    assert {"S0152", "S0153"} <= set(Lk.index)
    assert Lk["substratum"].value_counts().to_dict() == {"pipe_D": 2, "pipe_U": 1, "unflagged": 1}
    assert not set(Lk.index) & {"S0000", "S0300", "S0312"}
    assert Lk["lit_known"].all()
    if "S0110" in Lk.index:
        assert Lk.at["S0110", "known_lens_name"] == "Lens A0" and Lk.at["S0110", "known_sep_arcsec"] == 0.2
    assert S["L_known_alloc"] == {"fixed": {"pipe_D": 2}, "fill": {"pipe_U": 1, "unflagged": 1}}
    # --- D_refuted: fill chain, exclusions, near-miss rule
    D = frame[frame["stratum"] == "D_refuted"]
    assert D["substratum"].value_counts().to_dict() == {"merger": 4, "ring_spiral": 10,
                                                        "elliptical_nearmiss": 16, "other": 10}
    assert (D["pipe_grade_passcount"] == "D").all()
    assert not set(D["candidate_id"]) & {"S0150", "S0152", "S0153"}
    ell = D[D["substratum"] == "elliptical_nearmiss"]
    assert set(ell["candidate_id"]) == {f"S{i:04d}" for i in range(168, 184)}
    assert any("merger: only 4/10" in n for n in S["fills"])
    # --- U_tail bands, none from K/L
    U = frame[frame["stratum"] == "U_tail"]
    assert U["substratum"].value_counts().to_dict() == {"rank_101_130": 3, "rank_131_200": 2}
    assert (U["pipe_grade_passcount"] == "U").all()
    assert not set(U["candidate_id"]) & {"S0104", "S0030", "S0110", "S0111"}      # K/L ids inside the bands
    # --- N_unflagged: proposal mix 100:60:40 -> {3,1,1}; DESI-overlap rows excluded; never proposal 4000
    N = frame[frame["stratum"] == "N_unflagged"]
    assert S["N_unflagged_alloc"] == {"1000": 3, "2000": 1, "3000": 1}
    assert N["substratum"].value_counts().to_dict() == {"proposal_1000": 3, "proposal_2000": 1, "proposal_3000": 1}
    assert not set(N["candidate_id"]) & {f"S{i:04d}" for i in range(200, 210)}
    assert (N["pipe_grade_passcount"] == "").all() and N["pipe_inspector_conf"].isna().all()
    assert not N["desi_pool_overlap"].any() and not N["lit_known"].any()
    assert S["pop"]["unflagged_desi_overlap_excluded"] == 10
    assert (N["candidate_id"].str[1:].astype(int) < 395).all()         # no_coverage rows never drawn
    # --- the same seed reproduces the same frame
    frame2, _ = bf.build_frame(_universe(), np.random.default_rng(_util.SEED), _QUOTAS)
    assert frame2.equals(frame)


# ------------------------------------------------------------------ real data smoke test (skipped if absent)
def test_real_frame_smoke():
    J = _util.JWST_REPO
    need = [J / "results" / "results.csv", J / "results" / "JWST_top100_master.csv",
            J / "results" / "control_recovery.csv", J / "results" / "known_lens_recovery.csv",
            J / "results" / "inspections.csv", J / "data" / "targets.parquet",
            bf.OUTPUTS / "parity_train_pool.csv", bf.OUTPUTS / "parity_bench_arm1.csv",
            bf.OUTPUTS / "parity_bench_arm2.csv"]
    missing = [p for p in need if not p.exists()]
    if missing:
        print(f"  (skipped: missing {missing[0]})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        frame, S = bf.build(Path(tmp))
        pinned = _util.read_pinned(Path(tmp) / "frame.csv", dtype={"candidate_id": str})
        assert (Path(tmp) / "frame_summary.md").exists()
    assert len(frame) == 250 and len(pinned) == 250
    assert list(frame.columns) == CONTRACT_COLUMNS
    assert frame["candidate_id"].is_unique and frame["unit_id"].is_unique
    assert int(frame["stratum"].isin(["T_verified", "T_U"]).sum()) == 99
    assert int((frame["stratum"] == "K_cowls").sum()) == 31
    assert frame["stratum"].value_counts().to_dict() == {
        "T_U": 78, "D_refuted": 40, "K_cowls": 31, "L_known": 30, "U_tail": 30, "T_verified": 21, "N_unflagged": 20}
    f = frame.set_index("candidate_id")
    assert f.at["J18030075+2309921", "alias_ids"] == "J18030108+2309932"    # rank 14 collapsed onto rank 7
    assert "J18030108+2309932" not in f.index
    assert f.at["J5186648-1343587", "system_id"] == f.at["J5186803-1343778", "system_id"]   # ranks 16/17
    assert int((frame["prior_exposure"] == 2).sum()) == 14 and int((frame["prior_exposure"] == 1).sum()) == 85
    assert set(frame["layout"]) <= {"color", "gray_sw_only", "gray_lw_only"}
    assert S["fills"] == [], S["fills"]
    # determinism regression against the tracked frame, when it has been built
    tracked = _util.HERE / "frame.csv"
    if tracked.exists():
        with tempfile.TemporaryDirectory() as tmp:
            bf.build(Path(tmp))
            assert (Path(tmp) / "frame.csv").read_text() == tracked.read_text(), \
                "rebuilt frame differs from golden/frame.csv — inputs or builder changed; rebuild and re-pin"


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
