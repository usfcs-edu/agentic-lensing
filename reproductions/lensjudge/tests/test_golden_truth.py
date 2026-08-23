#!/usr/bin/env python3
"""No-network tests for the Part-2 truth set: golden/sample_truth_negatives.py,
golden/build_truth_manifest.py, golden/split_truth.py, registry.seed_from_frame and the
`halves=` argument of split_halves.firewall.

Everything runs on one synthetic universe (frame + run tables + catalogue + AnomalyMatch +
DESI pool laid out on a 36" grid): the 5" catalogue purge, the frame / positive 10" buffer,
the DESI 2" exclusion, largest-remainder quotas to the positives' mix, the exact 200/200
half balance, the manifest's column order and class logic (U_tail row with a <= 2"
literature lens -> positive; AnomalyMatch precedence over stress_D; 2" positional dedup;
the rank-14 twin kept as an anchor), `cowls_band` on canned strings, `centre_is_deflector`,
the split's forced rules / whole-proposal literature / negative balance / firewall, and
determinism under the seed. J-dependent checks (the 195,818-position union, the tracked
truth_negatives.csv / truth_manifest.csv / truth_splits.csv regressions) run only when
LENSJUDGE_JWST_REPO / the default checkout exists, and are skipped otherwise.

    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/tests/test_golden_truth.py
(also pytest-compatible)
"""
from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util, registry, split_halves  # noqa: E402
from lensjudge.golden import build_frame as bf  # noqa: E402
from lensjudge.golden import build_truth_manifest as btm  # noqa: E402
from lensjudge.golden import sample_truth_negatives as stn  # noqa: E402
from lensjudge.golden import split_truth as st  # noqa: E402

ARCSEC = 1.0 / 3600.0
# the contract's truth_manifest.csv schema, copied verbatim (not imported) so drift is caught
CONTRACT_COLS = [
    "name", "ra", "dec", "survey_key", "grade_truth", "binary_label", "source", "region",
    "tractor_type", "p_meta", "leak",
    "unit_id", "image_path", "image_path_v2r", "render_sha", "render_sha_v2r", "truth_class", "is_positive", "is_stress",
    "is_anchor", "cowls_band", "cowls_ranking", "cowls_theta_E", "known_lens_name", "known_type",
    "known_sep_arcsec", "centre_is_deflector", "layout", "field_class", "proposal", "mag_r",
    "prior_exposure", "pipe_grade_passcount", "pipe_inspector_conf", "pipe_score", "in_frame", "half",
]
NEG_COLS = ["candidate_id", "ra_deg", "dec_deg", "proposal", "field_class", "layout", "mag_r",
            "sw_obs", "lw_obs", "sw_filter", "lw_filter", "nearest_cat_sep_arcsec", "half"]
SPLIT_COLS = ["candidate_id", "system_id", "half", "forced", "forced_reason", "truth_class", "cell"]


# ============================================================================ synthetic universe
class _U:
    """Builder for one synthetic universe. Objects sit on a 36" grid (RA step 0.01 deg at
    dec 0) so nothing is within 10" unless placed so on purpose."""

    def __init__(self):
        self.rows: list = []          # every target: id, ra, dec, flagged, grade, conf, score, ...
        self.frame: list = []
        self.known: list = []
        self.master: list = []
        self.controls: list = []
        self.cat: list = []
        self.anom: list = []
        self.desi: list = []
        self._i = 0

    def pos(self, d_ra_arcsec: float = 0.0, d_dec_arcsec: float = 0.0, base=None):
        if base is None:
            self._i += 1
            base = (100.0 + 0.01 * self._i, 0.0)
        return base[0] + d_ra_arcsec * ARCSEC, base[1] + d_dec_arcsec * ARCSEC

    def target(self, cid, ra, dec, *, flagged=False, grade="", conf=0.0, score=0.0, proposal="1727",
               sw="obsSW", lw="obsLW", mag_r=19.5, status="ok", ctype="elliptical"):
        self.rows.append(dict(id=cid, ra=ra, dec=dec, flagged=flagged, grade=grade or np.nan,
                              confidence=conf, score=score, proposal=proposal, sw_obs=sw or np.nan,
                              lw_obs=lw or np.nan, sw_filter="F150W" if sw else np.nan,
                              lw_filter="F277W" if lw else np.nan, mag_r=mag_r, status=status,
                              center_galaxy_type=ctype))
        return cid

    def unit(self, cid, stratum, substratum="", rank=np.nan, prior=0, known=("", "", np.nan),
             cowls=("", np.nan), desi=False):
        self.frame.append(dict(unit_id=f"u{len(self.frame) + 1:04d}", candidate_id=cid, stratum=stratum,
                               substratum=substratum, rank_top100=rank, prior_exposure=prior,
                               known_lens_name=known[0], known_type=known[1], known_sep_arcsec=known[2],
                               cowls_ranking=cowls[0], cowls_theta_E=cowls[1], desi_pool_overlap=desi))
        return self.frame[-1]["unit_id"]

    def lit(self, name, src, cid, sep):
        self.known.append(dict(lens_name=name, lens_src=src, cutout_id=cid, sep_arcsec=sep,
                               flagged=True, grade="U", confidence=20.0))

    def top(self, rank, cid, ra, dec, status="new", designation="", nearest_sep=np.nan):
        self.master.append(dict(rank=rank, candidate_id=cid, ra_deg=ra, dec_deg=dec, discovery_status=status,
                                designation=designation, nearest_sep_arcsec=nearest_sep, verifier_grade="C"))

    def src(self) -> dict:
        res = pd.DataFrame(self.rows)
        results = res[["id", "ra", "dec", "grade", "confidence", "score", "proposal", "center_galaxy_type"]].copy()
        inspections = res[["id", "ra", "dec", "status", "flagged", "proposal"]].copy()
        targets = res[["id", "ra", "dec", "mag_r", "sw_obs", "lw_obs", "sw_filter", "lw_filter", "proposal"]].copy()
        frame = pd.DataFrame(self.frame)
        rr = results.set_index("id")
        frame["ra_deg"] = rr.loc[frame["candidate_id"], "ra"].to_numpy()
        frame["dec_deg"] = rr.loc[frame["candidate_id"], "dec"].to_numpy()
        frame["proposal"] = rr.loc[frame["candidate_id"], "proposal"].to_numpy()
        frame["system_id"] = bf.union_find_systems(frame["ra_deg"], frame["dec_deg"], 10.0)
        desi = pd.DataFrame(self.desi, columns=["ra", "dec", "split", "label_source"])
        return dict(results=results, inspections=inspections, targets=targets, frame=frame,
                    master=pd.DataFrame(self.master),
                    controls=pd.DataFrame(self.controls, columns=["id", "ranking", "score_x"]),
                    known=pd.DataFrame(self.known, columns=["lens_name", "lens_src", "cutout_id", "sep_arcsec",
                                                            "flagged", "grade", "confidence"]),
                    desi=desi, render=None,
                    anomalymatch=pd.DataFrame(self.anom, columns=["RA", "DEC", "voted_class", "ID"]),
                    fields=None)

    def catalogue(self) -> pd.DataFrame:
        return pd.DataFrame(self.cat, columns=["ra", "dec", "catalog"])


def _universe(n_unflagged: int = 90) -> _U:
    u = _U()
    # ---- COWLS controls (frame K_cowls), all in COSMOS 1727
    for cid, rank_str, th in (("C1", "AAAAAB", 1.0), ("C2", "ABBUUX", 0.58), ("C3", "", np.nan)):
        ra, dec = u.pos()
        u.target(cid, ra, dec, flagged=True, grade="U", conf=20, score=1.1)
        u.unit(cid, "K_cowls", "pipe_U", known=(f"COS{cid}", "COWLS", 0.1), cowls=(rank_str, th))
        u.controls.append(dict(id=cid, ranking=rank_str or np.nan, score_x=f"S{btm.cowls_score(rank_str)[0]:02d}" if rank_str else "M25"))
        u.cat.append((ra + 0.1 * ARCSEC, dec, "cowls"))
    # ---- L_known (frame): galaxy-scale in 1727, two cluster knots in 6882
    for cid, src, sep, prop in (("L1", "SIMBAD:gLS", 0.5, "1727"), ("L2", "SIMBAD:LeG", 1.8, "6882"), ("L3", "SIMBAD:LeI", 1.5, "6882")):
        ra, dec = u.pos()
        u.target(cid, ra, dec, flagged=True, grade="D", conf=30, score=1.0, proposal=prop)
        u.unit(cid, "L_known", "pipe_D", known=(f"Lens {cid}", src, sep))
        u.lit(f"Lens {cid}", src, cid, sep)
        u.cat.append((ra + sep * ARCSEC, dec, "simbad"))
    # ---- top-100 (frame T_verified): knowns 1/4/16/17, anchors 7 (+ alias 14), 13, 15
    ra, dec = u.pos(); u.target("R01", ra, dec, flagged=True, grade="A", conf=95, score=3.8, proposal="1837")
    u.unit("R01", "T_verified", "A", rank=1, prior=2, known=("VIRTUAL PARENT SL2S X", "SIMBAD:gLS", 0.97))
    u.top(1, "R01", ra, dec, "known", "SL2S J02176-0513 (also written SL2S 0217)", 0.01)
    u.lit("VIRTUAL PARENT SL2S X", "SIMBAD:gLS", "R01", 0.97)
    ra, dec = u.pos(); u.target("R04", ra, dec, flagged=True, grade="A", conf=75, score=3.7, proposal="6882")
    u.unit("R04", "T_verified", "A", rank=4, prior=2)
    u.top(4, "R04", ra, dec, "known", "DESI-307.9137-40.5843", 0.0)
    ra, dec = u.pos(); u.target("R07", ra, dec, flagged=True, grade="B", conf=60, score=3.5, proposal="1635", lw="")
    u.unit("R07", "T_verified", "B", rank=7, prior=2)
    u.top(7, "R07", ra, dec)
    ra14, dec14 = u.pos(1.17, 0, base=(ra, dec))            # the rank-14 twin: 1.17" away, NOT a frame row
    u.target("R14", ra14, dec14, flagged=True, grade="B", conf=55, score=3.4, proposal="1635", lw="")
    u.top(14, "R14", ra14, dec14)
    ra, dec = u.pos(); u.target("R13", ra, dec, flagged=True, grade="C", conf=50, score=2.6, proposal="6434", ctype="spiral")
    u.unit("R13", "T_verified", "C", rank=13, prior=2)
    u.top(13, "R13", ra, dec)
    ra, dec = u.pos(); u.target("R15", ra, dec, flagged=True, grade="C", conf=45, score=2.5, proposal="2561")
    u.unit("R15", "T_verified", "C", rank=15, prior=2)
    u.top(15, "R15", ra, dec)
    ra, dec = u.pos(); u.target("R16", ra, dec, flagged=True, grade="C", conf=42, score=2.4, proposal="5594")
    u.unit("R16", "T_verified", "C", rank=16, prior=1)
    u.top(16, "R16", ra, dec, "known", "RCS J0327:[SBC2023] E2.2 - knot E, image 2.2 of the giant arc")
    ra17, dec17 = u.pos(0, 8.78, base=(ra, dec))           # rank 17: 8.78" from rank 16 -> same system
    u.target("R17", ra17, dec17, flagged=True, grade="C", conf=40, score=2.3, proposal="5594")
    u.unit("R17", "T_verified", "C", rank=17, prior=1)
    u.top(17, "R17", ra17, dec17, "known", "RCS J0327:[SBC2023] E1.1 (with B1 adjacent)")
    ra, dec = u.pos(); u.target("R48", ra, dec, flagged=True, grade="U", conf=27, score=1.2, proposal="5594")
    u.unit("R48", "T_U", "", rank=48, prior=1, known=("VIRTUAL PARENT eMACS Y", "SIMBAD:gLS", 1.47))
    u.top(48, "R48", ra, dec, "field_match", "nearest catalogued lens object: eMACS Y (1.5\")", 1.47)
    u.lit("VIRTUAL PARENT eMACS Y", "SIMBAD:gLS", "R48", 1.47)     # field_match: NOT a positive
    # ---- D_refuted (frame): four substrata; D3 is also an AnomalyMatch Class A target
    for cid, sub, ctype, prop in (("D1", "merger", "merger", "6882"), ("D2", "ring_spiral", "ring", "5594"),
                                  ("D3", "elliptical_nearmiss", "elliptical", "2561"), ("D4", "other", "s0", "1727")):
        ra, dec = u.pos()
        u.target(cid, ra, dec, flagged=True, grade="D", conf=40, score=1.3, proposal=prop, ctype=ctype)
        u.unit(cid, "D_refuted", sub)
        if cid == "D3":
            u.anom.append((ra + 0.04 * ARCSEC, dec, "Class A.", "ABELL2744_1"))
    # ---- U_tail (frame): U3 carries a SIMBAD lensed image at 1.31" -> positive, not stress
    for cid, sub, prop in (("U1", "rank_101_300", "6882"), ("U2", "rank_101_300", "1727"), ("U3", "rank_301_2024", "5890")):
        ra, dec = u.pos()
        u.target(cid, ra, dec, flagged=True, grade="U", conf=20, score=1.1, proposal=prop)
        known = ("[DSB2018] A370 2.4.1", "SIMBAD:LeG", 1.31) if cid == "U3" else ("", "", np.nan)
        u.unit(cid, "U_tail", sub, known=known)
        if cid == "U3":
            u.lit(known[0], known[1], cid, known[2]); u.cat.append((ra + 1.31 * ARCSEC, dec, "simbad"))
    # ---- N_unflagged (frame)
    for cid in ("N1", "N2"):
        ra, dec = u.pos(); u.target(cid, ra, dec, proposal="5594"); u.unit(cid, "N_unflagged", "proposal_5594")
    # ---- literature <= 2" NOT in the frame: 3 kept (+ one COWLS-src), one 2" twin dropped, one at 2.5" out
    ra, dec = u.pos(); u.target("F1", ra, dec, flagged=True, grade="U", conf=25, score=1.1, proposal="1727")
    u.lit("[FKC2008] COSMOS 1", "SIMBAD:gLS", "F1", 0.9); u.cat.append((ra, dec + 0.9 * ARCSEC, "simbad"))
    ra, dec = u.pos(); u.target("F2", ra, dec, proposal="2561", lw="")
    u.lit("[MRC2018] A2744 105.4", "SIMBAD:LeG", "F2", 1.42); u.cat.append((ra, dec + 1.42 * ARCSEC, "simbad"))
    u.anom.append((ra + 0.03 * ARCSEC, dec, "Class B.", "ABELL2744_2"))      # on a positive: excluded
    ra, dec = u.pos(); u.target("F3", ra, dec, flagged=True, grade="D", conf=30, score=1.0, proposal="1727")
    u.lit("COSJ100036+015220", "COWLS", "F3", 1.72); u.cat.append((ra, dec + 1.72 * ARCSEC, "cowls"))
    ra, dec = u.pos(); u.target("F4", ra, dec, proposal="4744")
    u.lit("[CNL2018] PLCK 8.3", "SIMBAD:LeG", "F4", 1.98); u.cat.append((ra, dec + 1.98 * ARCSEC, "simbad"))
    ra5, dec5 = u.pos(1.5, 0, base=(ra, dec)); u.target("F5", ra5, dec5, proposal="4744")       # 1.5" twin of F4
    u.lit("[CNL2018] PLCK 8.3b", "SIMBAD:LeI", "F5", 0.5)
    ra, dec = u.pos(); u.target("F6", ra, dec, proposal="6882")
    u.lit("far knot", "SIMBAD:LeG", "F6", 2.5); u.cat.append((ra, dec + 2.5 * ARCSEC, "simbad"))   # > 2": not a positive
    # ---- AnomalyMatch on a plain unflagged non-frame target + one far from any target
    ra, dec = u.pos(); u.target("A1", ra, dec, proposal="2561")
    u.anom.append((ra, dec + 0.02 * ARCSEC, "Class C.", "ABELL2744_3")); u.cat.append((ra, dec, "anomaly"))
    u.anom.append((50.0, 50.0, "Class A.", "nowhere"))
    # ---- the unflagged pool for the sampler: layouts / proposals cycle; some purged / DESI / buffered
    layouts = (("obsSW", "obsLW", "1727"), ("obsSW", "obsLW", "6882"), ("", "obsLW", "6882"),
               ("obsSW", "obsLW", "2662"), ("obsSW", "", "2662"))
    for k in range(n_unflagged):
        sw, lw, prop = layouts[k % len(layouts)]
        ra, dec = u.pos()
        u.target(f"P{k:03d}", ra, dec, proposal=prop, sw=sw, lw=lw, mag_r=18 + (k % 30) / 10)
        if k % 9 == 0:                                      # within 4" of a catalogue lens: purged
            u.cat.append((ra + 4.0 * ARCSEC, dec, "misc_recent"))
        if k % 9 == 1:                                      # within 1" of a DESI pool row: excluded
            u.desi.append((ra + 1.0 * ARCSEC, dec, "trainsel", "random_neg"))
    # an unflagged target 6" from frame unit C1 (inside the 10" buffer) and one with no coverage
    c1 = next(r for r in u.rows if r["id"] == "C1")
    ra, dec = u.pos(6.0, 0, base=(c1["ra"], c1["dec"])); u.target("PBUF", ra, dec, proposal="1727")
    ra, dec = u.pos(); u.target("PNOC", ra, dec, proposal="1727", status="no_coverage")
    # a flagged non-frame row must never be drawn as a negative
    ra, dec = u.pos(); u.target("PFLG", ra, dec, flagged=True, grade="U", conf=15, score=1.0, proposal="1727")
    return u


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# ============================================================================ cowls band / field class
def test_cowls_band_canned_strings():
    assert btm.cowls_score("AAAAAB") == (11, 5) and btm.cowls_band("AAAAAB") == "strong"
    assert btm.cowls_score("ABBUUX") == (4, 1) and btm.cowls_band("ABBUUX") == "marginal"
    assert btm.cowls_score("BBBXXX") == (3, 0) and btm.cowls_band("BBBXXX") == "weak"
    assert btm.cowls_score("UUUUUU") == (0, 0) and btm.cowls_band("UUUUUU") == "weak"
    for none in ("", None, np.nan, "nan"):
        assert btm.cowls_band(none) == "provenance", none
    assert btm.cowls_band("AAABBX") == "strong" and btm.cowls_score("AAABBX") == (8, 3)   # S >= 8 <=> nA >= 3 on controls
    assert btm.cowls_band("AASUXX") == "marginal"      # S = 5
    assert btm.cowls_band("SSUUXX") == "weak"          # S = 2
    assert btm.cowls_band("ABBBBS") == "marginal"      # S = 7, nA = 1


def test_field_class_and_cluster_review():
    assert btm.field_class_of("1727") == "cosmos" and btm.field_class_of(1727) == "cosmos"
    for p in ("6882", "5594", "5890", "2561", 6882, "2561.0"):
        assert btm.field_class_of(p) == "cluster", p
    assert btm.field_class_of("2662") == "blank" and btm.field_class_of("") == "blank" and btm.field_class_of(np.nan) == "blank"
    fields = pd.DataFrame({"proposal_id": ["6882", "6882", "9999", "9999", "1727", "2045"],
                           "target_name": ["A370", "MACSJ0416.1-2403", "SPT0202-61", "NGC-602", "CWEBTILE-0-0", "NAME-ARCHES-CLUSTER"]})
    d = btm.derive_cluster_proposals(fields)
    assert set(d) == {"6882", "9999", "2045"} and d["9999"] == ["SPT0202-61"]      # 9999 at exactly 1/2
    assert "1727" not in d
    assert "2045" not in btm.CLUSTER_PROPOSALS          # the star cluster the review deliberately leaves out


def test_short_designation():
    assert btm.short_designation("SL2S J02176-0513 (also written SL2S J021737-051329)") == "SL2S J02176-0513"
    assert btm.short_designation("RCS J0327:[SBC2023] E1.1 (with B1 adjacent) - knots") == "RCS J0327:[SBC2023] E1.1"
    assert btm.short_designation("MACS J0416.1-2403 ID14 -- main deflector = cluster member") == "MACS J0416.1-2403 ID14"
    assert btm.short_designation(np.nan) == "" and btm.short_designation("GDS J123730+621301") == "GDS J123730+621301"


def test_literature_le2_any_source_closest():
    known = pd.DataFrame({"lens_name": ["a", "b", "cowls", "far"], "lens_src": ["SIMBAD:LeG", "SIMBAD:gLS", "COWLS", "SIMBAD:gLS"],
                          "cutout_id": ["J1", "J1", "J2", "J3"], "sep_arcsec": [1.5, 0.4, 1.72, 2.01]})
    d = btm.literature_le2(known, 2.0)
    assert list(d["cutout_id"]) == ["J1", "J2"] and d.iloc[0]["lens_name"] == "b"     # COWLS kept, > 2" dropped


# ============================================================================ manifest (synthetic)
def _manifest(n_neg: int = 20):
    u = _universe()
    src = u.src()
    cat = u.catalogue()
    neg, S = _quiet(stn.sample, src, cat, n_neg, _util.SEED)[0]
    man, notes = btm.assemble(src, neg)
    return u, src, cat, neg, S, man, notes


def test_manifest_columns_and_classes():
    u, src, cat, neg, S, man, notes = _manifest()
    assert list(man.columns) == CONTRACT_COLS
    assert list(man.columns[:11]) == btm.EVAL_COLS
    assert man["name"].is_unique
    m = man.set_index("name")
    cls = man["truth_class"].value_counts().to_dict()
    # cowls 3; lit_galaxy: L1, R01, R04, F1, F3(COWLS src); lit_cluster: L2, L3, R16, R17, U3, F2, F4
    assert cls["cowls"] == 3 and cls["lit_galaxy"] == 5 and cls["lit_cluster"] == 7, cls
    assert cls["negative"] == 20 and cls["stress_D"] == 3 and cls["stress_U"] == 2 and cls["anomalymatch"] == 2 and cls["anchor"] == 4
    assert int(man["is_positive"].sum()) == 15
    assert set(man.loc[man["is_positive"], "truth_class"]) == {"cowls", "lit_galaxy", "lit_cluster"}
    assert set(man.loc[man["is_stress"], "truth_class"]) == {"stress_D", "stress_U", "anomalymatch"}
    # the U_tail unit with a SIMBAD lensed image at 1.31" is a positive, not stress_U
    assert m.at["U3", "truth_class"] == "lit_cluster" and m.at["U3", "in_frame"] and m.at["U3", "unit_id"].startswith("u")
    assert any("U3" in n for n in notes["frame_lit_outside_L_known"])
    # field_match top-100 row with a 1.47" VIRTUAL PARENT cluster record is NOT a positive and not in the manifest
    assert "R48" not in m.index
    assert "F6" not in m.index                              # 2.5": out
    # 2" positional dedup among positives: F5 dropped onto F4
    assert "F4" in m.index and "F5" not in m.index and any("F5" in n and "F4" in n for n in notes["dropped_dups"])
    # AnomalyMatch: D3 (frame D_refuted) takes anomalymatch precedence; F2's candidate is excluded (positive); A1 plain
    assert m.at["D3", "truth_class"] == "anomalymatch" and m.at["D3", "in_frame"] and m.at["D3", "known_type"] == "anomalymatch"
    assert m.at["A1", "truth_class"] == "anomalymatch" and not m.at["A1", "in_frame"] and m.at["A1", "unit_id"] == ""
    assert m.at["F2", "truth_class"] == "lit_cluster"
    assert notes["anomalymatch_classes"] == {"Class A": 1, "Class C": 1}
    assert any("D3" in n for n in notes["overlaps"])
    # anchors: ranks 15, 7, 14, 16, 13 flagged; 16 is a positive; 14 is a non-frame twin row with prior_exposure 2
    assert set(man.loc[man["is_anchor"], "name"]) == {"R15", "R07", "R14", "R16", "R13"}
    assert m.at["R16", "truth_class"] == "lit_cluster" and m.at["R16", "is_positive"] and m.at["R16", "is_anchor"]
    assert m.at["R14", "truth_class"] == "anchor" and not m.at["R14", "in_frame"] and m.at["R14", "unit_id"] == ""
    assert int(m.at["R14", "prior_exposure"]) == 2 and int(m.at["R16", "prior_exposure"]) == 1 and int(m.at["R07", "prior_exposure"]) == 2
    assert m.at["R14", "layout"] == "gray_sw_only" and m.at["R07", "layout"] == "gray_sw_only"
    # top-100 known rows: scale from the designation table, name/type/sep from the master when no SIMBAD match
    assert m.at["R04", "truth_class"] == "lit_galaxy" and m.at["R04", "known_type"] == btm.TOP100_KNOWN_TYPE
    assert m.at["R04", "known_lens_name"] == "DESI-307.9137-40.5843" and m.at["R04", "known_sep_arcsec"] == 0.0
    assert m.at["R01", "known_type"] == "SIMBAD:gLS" and m.at["R01", "known_sep_arcsec"] == 0.97
    assert m.at["R16", "known_lens_name"] == "RCS J0327:[SBC2023] E2.2" and np.isnan(m.at["R16", "known_sep_arcsec"])
    assert m.at["R17", "truth_class"] == "lit_cluster"
    # eval columns
    assert (man["survey_key"] == "jwst").all() and (man["source"] == "truth_jwst").all()
    assert (man["grade_truth"].astype(str) == "").all() and man["p_meta"].isna().all() and (man["tractor_type"] == "").all()
    assert man.loc[man["binary_label"] != "", "binary_label"].value_counts().to_dict() == {"nonlens": 20, "lens": 15}
    assert (man.loc[man["is_stress"] | (man["truth_class"] == "anchor"), "binary_label"] == "").all()
    assert (man["region"].astype(str) == man["proposal"].astype(str)).all()
    assert set(man["leak"]) <= {"no", "desi_train"}
    assert (man.loc[man["truth_class"] == "negative", "leak"] == "no").all()
    assert (man["image_path"] == "").all() and (man["half"] == "").all() and (man["image_path_v2r"] == "").all() and (man["render_sha_v2r"] == "").all()
    # pipeline columns: NaN conf/score for unflagged, '' grade
    assert m.at["C1", "pipe_grade_passcount"] == "U" and m.at["C1", "pipe_inspector_conf"] == 20
    assert m.at["F2", "pipe_grade_passcount"] == "" and np.isnan(m.at["F2", "pipe_inspector_conf"])
    # cowls band / theta
    assert m.at["C1", "cowls_band"] == "strong" and m.at["C2", "cowls_band"] == "marginal" and m.at["C3", "cowls_band"] == "provenance"
    assert (man.loc[man["truth_class"] != "cowls", "cowls_band"] == "").all()
    # centre_is_deflector: cowls | gLS/gLe/LeQ | sep <= 1", positives only
    assert m.at["C3", "centre_is_deflector"] and m.at["L1", "centre_is_deflector"] and m.at["R04", "centre_is_deflector"]
    assert m.at["F3", "centre_is_deflector"]                     # COWLS catalogue lens at 1.72": by type
    assert not m.at["L2", "centre_is_deflector"] and not m.at["U3", "centre_is_deflector"] and not m.at["R16", "centre_is_deflector"]
    assert not m.at["D3", "centre_is_deflector"] and not m.at["A1", "centre_is_deflector"]
    # row order: class order then id
    order = [btm.TRUTH_CLASSES.index(c) for c in man["truth_class"]]
    assert order == sorted(order)


def test_attach_v2r_merges_only_ok_rows_with_relative_paths():
    """render_v2.py's pinned CSV -> image_path_v2r / render_sha_v2r: ok rows with an existing
    JPEG only, lensjudge-relative paths, failed / stray / missing-file rows stay ''."""
    man = pd.DataFrame({"name": ["A", "B", "C", "D"]})
    with tempfile.TemporaryDirectory(dir=_util.LENSJUDGE / "outputs") as tmp:
        tmp = Path(tmp)
        for cid in ("A", "B"):
            (tmp / f"{cid}.jpg").write_bytes(b"\xff\xd8" + cid.encode())
        rv = pd.DataFrame({"id": ["A", "B", "C", "Z"],
                           "image_path_v2r": [str(tmp / "A.jpg"), str(tmp / "B.jpg"), str(tmp / "C.jpg"), str(tmp / "Z.jpg")],
                           "render_sha_v2r": ["aa", "bb", "cc", "zz"],
                           "status": ["ok", "ValueError: no stamp", "ok", "ok"]})
        csv = tmp / "render_v2r.csv"
        _util.pin(rv, csv)
        out, _ = _quiet(btm.attach_v2r, man, csv)
        assert out["image_path_v2r"].tolist()[0] == str((tmp / "A.jpg").relative_to(_util.LENSJUDGE))
        assert not Path(out["image_path_v2r"][0]).is_absolute()
        assert out["render_sha_v2r"].tolist() == ["aa", "", "", ""]      # B failed, C has no file, D absent
        assert out["image_path_v2r"].tolist()[1:] == ["", "", ""]
        # no CSV -> all ''
        out2 = btm.attach_v2r(man, tmp / "nope.csv")
        assert (out2["image_path_v2r"] == "").all() and (out2["render_sha_v2r"] == "").all()


def test_manifest_negatives_must_not_clash_and_frame_columns():
    u, src, cat, neg, S, man, notes = _manifest()
    bad = neg.copy(); bad.loc[0, "candidate_id"] = "C1"
    try:
        btm.assemble(src, bad)
        raise AssertionError("a negative that is also a positive must be refused")
    except AssertionError as e:
        assert "overlap" in str(e)
    # no negatives at all -> classes still build, no negative rows
    man0, _ = btm.assemble(src, None)
    assert (man0["truth_class"] != "negative").all() and len(man0) == len(man) - 20


def test_manifest_images_fetch_list_and_half(tmp_path=None):
    from PIL import Image
    u, src, cat, neg, S, man, notes = _manifest()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stamps, kits = tmp / "stamps", tmp / "kits_truth"
        for cid in ("C1", "R14"):
            (stamps / cid).mkdir(parents=True)
            Image.new("RGB", (752, 562), (40 + len(cid), 50, 60)).save(stamps / cid / f"{cid}_v1.jpg", quality=92)
        out = btm.write_images(man, stamps, kits)
        got = out[out["image_path"] != ""]
        assert set(got["name"]) == {"C1", "R14"}
        for _, r in got.iterrows():
            p = Path(r["image_path"])
            p = p if p.is_absolute() else _util.LENSJUDGE / p
            assert p.exists() and p.name == f"{r['name']}.jpg" and p.parent == kits
            assert Image.open(p).size == (752, 540), "footer must be cropped at y=540"
            assert r["render_sha"] == _util.sha_file(p) and len(r["render_sha"]) == 16
        # not byte-identical to what build_kit would serve (optimize=True) for the same source
        from lensjudge.golden import build_kit
        kit_item = tmp / "item.jpg"
        build_kit.crop_footer(stamps / "C1" / "C1_v1.jpg", kit_item)
        assert kit_item.read_bytes() != (kits / "C1.jpg").read_bytes()
        assert np.array_equal(np.asarray(Image.open(kit_item)), np.asarray(Image.open(kits / "C1.jpg"))), \
            "pixel-identical to the kit item (only the Huffman tables differ)"
        fl = btm.fetch_list(out)
        assert list(fl.columns) == ["candidate_id", "ra_deg", "dec_deg", "layout"]
        assert len(fl) == len(man) - 2 and "C1" not in set(fl["candidate_id"]) and "R07" in set(fl["candidate_id"])
    # half from a splits table that covers the row set exactly; anything else leaves it blank
    splits = pd.DataFrame({"candidate_id": man["name"], "half": ["design", "holdout"] * (len(man) // 2) + ["design"] * (len(man) % 2)})
    h = btm.attach_half(man, splits)
    assert (h["half"] != "").all() and h.loc[h["name"] == man["name"].iloc[0], "half"].item() == "design"
    h2, text = _quiet(btm.attach_half, man, splits.iloc[:-1])
    assert (h2["half"] == "").all() and "left blank" in text
    assert (btm.attach_half(man, None)["half"] == "").all()


# ============================================================================ sampler (synthetic)
def test_sampler_purge_exclusions_quotas_halves():
    u = _universe()
    src = u.src(); cat = u.catalogue()
    (neg, S), _ = _quiet(stn.sample, src, cat, 20, _util.SEED)
    stn.check(neg, 20)
    assert list(neg.columns) == NEG_COLS and len(neg) == 20
    ids = set(neg["candidate_id"])
    # only unflagged, ok, non-frame targets
    assert all(i.startswith("P") for i in ids) and "PNOC" not in ids and "PFLG" not in ids and "PBUF" not in ids
    assert not ids & set(src["frame"]["candidate_id"])
    # purge: nothing within 5" of the catalogue; the k % 9 == 0 rows (4" away) are gone
    assert (neg["nearest_cat_sep_arcsec"] >= 5.0).all()
    assert not any(int(i[1:]) % 9 == 0 for i in ids)
    # DESI: the k % 9 == 1 rows are gone
    assert not any(int(i[1:]) % 9 == 1 for i in ids)
    # purged: 10 P rows at 4" + F2 / F4 / F5 / F6 / A1 (unflagged non-frame targets with a catalogue
    # lens <= 5"); frame-or-positive buffer: PBUF (6" from C1) + the two N_unflagged frame units
    assert S["excluded_desi"] == 10 and S["purged"] == 15 and S["excluded_frame_or_positive"] == 3
    assert S["cat_matches"] == {"2": 3, "3": 5, "5": 15}
    assert S["unflagged_ok"] == 90 + 1 + 5 + 2          # P* + PBUF + F2 F4 F5 F6 A1 + N1 N2 (PFLG flagged, PNOC no_coverage)
    assert S["eligible"] == 98 - 15 - 3 - 10
    # quotas: largest remainder over the positives' (layout x field_class) mix, capped by the pool
    pos = btm.positives_table(src)[0]
    mix = pos.groupby(["layout", "field_class"]).size().to_dict()
    alloc, left = stn.allocate(mix, stn.eligible_pool(src["inspections"], src["targets"], cat,
                                                      src["frame"][["candidate_id", "ra_deg", "dec_deg"]], src["desi"], None)[0], 20)
    assert S["alloc"] == {f"{k[0]}|{k[1]}": v for k, v in alloc.items()} and sum(alloc.values()) == 20
    got = neg.groupby(["layout", "field_class"]).size().to_dict()
    assert got == alloc, (got, alloc)
    # halves: 10/10, within 1 per field_class and per cell
    assert neg["half"].value_counts().to_dict() == {"design": 10, "holdout": 10}
    cell = pd.crosstab([neg["layout"], neg["field_class"]], neg["half"])
    assert (abs(cell["design"] - cell["holdout"]) <= 1).all()
    # determinism, and a different seed changes the draw
    (neg2, _), _ = _quiet(stn.sample, u.src(), u.catalogue(), 20, _util.SEED)
    assert neg2.equals(neg)
    (neg3, _), _ = _quiet(stn.sample, u.src(), u.catalogue(), 20, 7)
    assert not neg3["candidate_id"].equals(neg["candidate_id"])


def test_sampler_unflagged_ok_count():
    u = _universe()
    src = u.src()
    ins = src["inspections"]
    n = int(((ins["status"] == "ok") & ~ins["flagged"].astype(bool)).sum())
    (neg, S), _ = _quiet(stn.sample, src, u.catalogue(), 20, _util.SEED)
    assert S["unflagged_ok"] == n


def test_sampler_assign_halves_keeps_systems_together_and_balances():
    # 11 single rows + one 2-row system (8" apart) in two cells -> 13 rows: 7/6 at best (odd total)
    ra = [100.0 + 0.01 * i for i in range(12)] + [100.0 + 0.01 * 11]
    dec = [0.0] * 12 + [8.0 * ARCSEC]
    neg = pd.DataFrame({"ra_deg": ra, "dec_deg": dec,
                        "layout": ["color"] * 13, "field_class": ["cluster"] * 8 + ["cosmos"] * 5})
    h = stn.assign_halves(neg, np.random.default_rng(1))
    assert h[11] == h[12], "a 10\" system must not straddle"
    vc = pd.Series(h).value_counts()
    assert abs(int(vc["design"]) - int(vc["holdout"])) <= 1
    # an even total lands exactly equal
    neg2 = neg.iloc[:12].copy()
    h2 = pd.Series(stn.assign_halves(neg2, np.random.default_rng(1))).value_counts()
    assert int(h2["design"]) == int(h2["holdout"]) == 6


def test_catalogue_union_reader_is_column_robust():
    with tempfile.TemporaryDirectory() as tmp:
        J = Path(tmp)
        (J / "data" / "lenscats" / "raw_misc").mkdir(parents=True)
        pd.DataFrame({"ra": [1.0, 400.0, np.nan], "dec": [1.0, 1.0, 1.0]}).to_csv(J / "data" / "lenscats" / "a.csv", index=False)
        pd.DataFrame({"RA": [2.0], "DEC": [2.0], "x": [1]}).to_csv(J / "data" / "lenscats" / "b.csv", index=False)
        pd.DataFrame({"foo": [1]}).to_csv(J / "data" / "lenscats" / "c.csv", index=False)
        pd.DataFrame({"ra": [3.0], "dec": [3.0], "main_id": ["s"], "otype": ["gLS"]}).to_csv(J / "data" / "simbad_lenses.csv", index=False)
        pd.DataFrame({"ra": [4.0], "dec": [4.0], "code": ["c"]}).to_csv(J / "data" / "lenscats" / "raw_misc" / "cowls_catalogue.csv", index=False)
        cat, text = _quiet(stn.load_catalogue_union, J, 4)
        assert len(cat) == 4 and set(cat["catalog"]) == {"a", "b", "simbad_lenses", "cowls_catalogue"}
        assert "no ra/dec columns" in text
        try:
            _quiet(stn.load_catalogue_union, J, 5)
            raise AssertionError("the >= min_rows assertion must fire")
        except AssertionError as e:
            assert "expected >= 5" in str(e)


# ============================================================================ split (synthetic)
def test_split_rules():
    u, src, cat, neg, S, man, notes = _manifest()
    splits, text = _quiet(st.assign, man, neg, src["frame"], _util.SEED)
    assert list(splits.columns) == SPLIT_COLS and len(splits) == len(man)
    s = splits.set_index("candidate_id")
    # forced: prior_exposure == 2 frame units, every anchor, and whatever shares their component
    for cid in ("R01", "R04", "R07", "R13", "R15", "R16", "R14"):
        assert s.at[cid, "half"] == "design" and s.at[cid, "forced"], cid
    assert "prior_exposure_2" in s.at["R01", "forced_reason"] and "anchor" in s.at["R15", "forced_reason"]
    assert s.at["R14", "forced_reason"] == "anchor" and s.at["R14", "system_id"] == s.at["R07", "system_id"]
    assert s.at["R17", "half"] == "design" and s.at["R17", "forced"] and "R16" in s.at["R17", "forced_reason"]
    # whole-proposal literature: 6882 (R04 forced) drags L2, L3 into design; 5594 (R16) likewise
    for cid in ("L2", "L3"):
        assert s.at[cid, "half"] == "design" and "R04" in s.at[cid, "forced_reason"]
    lit = man[man["truth_class"].isin(("lit_galaxy", "lit_cluster"))]
    by_prop = splits.set_index("candidate_id").loc[lit["name"]].assign(p=lit["proposal"].astype(str).to_numpy())
    assert (by_prop.groupby("p")["half"].nunique() == 1).all()
    # cowls are NOT proposal-grouped: the three 1727 controls can split
    assert (s.loc[["C1", "C2", "C3"], "forced"] == False).all()   # noqa: E712
    # systems never straddle; cells carry the band / substratum / field_class
    assert (splits.groupby("system_id")["half"].nunique() == 1).all()
    assert s.at["C1", "cell"] == "cowls:strong" and s.at["C3", "cell"] == "cowls:provenance"
    assert s.at["D1", "cell"] == "stress_D:merger" and s.at["U1", "cell"] == "stress_U:rank_101_300"
    assert s.at["D3", "cell"] == "anomalymatch" and s.at["R15", "cell"] == "anchor"
    assert s.at["L2", "cell"] == "lit_cluster" and s.at["L1", "cell"] == "lit_galaxy"
    # negatives: the sampler's half, 10/10, balanced per field_class cell
    n = splits[splits["truth_class"] == "negative"]
    assert n["half"].value_counts().to_dict() == {"design": 10, "holdout": 10}
    nh = neg.set_index("candidate_id")["half"]
    assert (n.set_index("candidate_id")["half"] == nh.loc[n["candidate_id"]]).all()
    assert n["cell"].str.startswith("negative:").all()
    st.check(splits)
    # firewall with the design/holdout labels: n_bad == 0 (and the overlap report runs)
    fw = splits.rename(columns={"candidate_id": "unit_id", "half": "split"})[["unit_id", "split"]]
    coords = pd.DataFrame({"unit_id": man["name"], "ra_deg": man["ra"], "dec_deg": man["dec"], "candidate_id": man["name"]})
    with tempfile.TemporaryDirectory() as tmp:
        ov, text = _quiet(split_halves.firewall, fw, coords, pool_dir=Path(tmp), halves=("design", "holdout"))
    assert "0 position collisions" in text and len(ov) == 0
    # determinism
    splits2, _ = _quiet(st.assign, man, neg, src["frame"], _util.SEED)
    assert splits2.equals(splits)
    # both halves got positives
    pos = splits[splits["truth_class"].isin(("cowls", "lit_galaxy", "lit_cluster"))]
    assert set(pos["half"]) == {"design", "holdout"}


def test_split_refuses_negative_in_a_positive_system_and_missing_half():
    u, src, cat, neg, S, man, notes = _manifest()
    # a negative with no sampler half
    try:
        _quiet(st.assign, man, neg.iloc[1:], src["frame"], _util.SEED)
        raise AssertionError("must refuse a negative without a sampler half")
    except AssertionError as e:
        assert "sampler half" in str(e)
    # a negative moved to within 10" of C1 would share its system
    man2 = man.copy()
    c1 = man2[man2["name"] == "C1"].iloc[0]
    i = man2.index[man2["truth_class"] == "negative"][0]
    man2.loc[i, "ra"] = c1["ra"] + 6 * ARCSEC; man2.loc[i, "dec"] = c1["dec"]
    try:
        _quiet(st.assign, man2, neg, src["frame"], _util.SEED)
        raise AssertionError("must refuse a negative inside a positive's 10\" system")
    except AssertionError as e:
        assert "share a 10" in str(e)


def test_split_check_catches_violations():
    good = pd.DataFrame({"candidate_id": ["a", "b", "c", "d", "e"], "system_id": [1, 1, 2, 2, 3],
                         "half": ["design", "design", "holdout", "holdout", "design"],
                         "forced": [True, True, False, False, False],
                         "forced_reason": ["anchor", "component_of:a", "", "", ""],
                         "truth_class": ["anchor", "cowls", "negative", "negative", "negative"],
                         "cell": ["anchor", "cowls:weak", "negative:cosmos", "negative:cosmos", "negative:cosmos"]})
    st.check(good)

    def _expect(df, msg):
        try:
            st.check(df); raise RuntimeError("expected failure")
        except AssertionError as e:
            assert msg in str(e), str(e)
    bad = good.copy(); bad.loc[3, "half"] = "design"                   # system 2 straddles (balance still ok)
    _expect(bad, "straddles")
    bad2 = good.copy(); bad2.loc[0, "half"] = "holdout"; bad2.loc[1, "half"] = "holdout"
    _expect(bad2, "forced")
    bad3 = good.copy(); bad3.loc[2, "half"] = "design"; bad3.loc[3, "half"] = "design"   # 3 vs 0 negatives
    _expect(bad3, "negatives")
    bad4 = good.copy(); bad4.loc[4, "half"] = "elsewhere"
    _expect(bad4, "unknown half")


# ============================================================================ split_halves.firewall halves=
def test_firewall_halves_argument():
    coords = pd.DataFrame({"unit_id": ["a", "b", "c"], "ra_deg": [10.0, 10.0, 20.0],
                           "dec_deg": [0.0, 1.5 * ARCSEC, 0.0], "candidate_id": ["a", "b", "c"]})
    with tempfile.TemporaryDirectory() as tmp:
        # default halves: a 1.5" pair across align/validate is a leak
        sp = pd.DataFrame({"unit_id": ["a", "b", "c"], "split": ["align", "validate", "align"]})
        try:
            _quiet(split_halves.firewall, sp, coords, pool_dir=Path(tmp))
            raise AssertionError("the 1.5\" align/validate pair must fail the firewall")
        except AssertionError as e:
            assert "leak audit FAILED" in str(e)
        # the same pair labelled design/holdout: with the default halves it used to be VACUOUS;
        # now the stray labels are refused, and with halves= it is tested
        sp2 = pd.DataFrame({"unit_id": ["a", "b", "c"], "split": ["design", "holdout", "design"]})
        try:
            _quiet(split_halves.firewall, sp2, coords, pool_dir=Path(tmp))
            raise AssertionError("design/holdout labels under the default halves must be refused, not ignored")
        except AssertionError as e:
            assert "not in halves" in str(e)
        try:
            _quiet(split_halves.firewall, sp2, coords, pool_dir=Path(tmp), halves=("design", "holdout"))
            raise AssertionError("the 1.5\" design/holdout pair must fail the firewall")
        except AssertionError as e:
            assert "leak audit FAILED" in str(e) and "design" in str(e)
        # a clean design/holdout split passes and reports per half
        sp3 = pd.DataFrame({"unit_id": ["a", "b", "c"], "split": ["design", "design", "holdout"]})
        ov, text = _quiet(split_halves.firewall, sp3, coords, pool_dir=Path(tmp), halves=("design", "holdout"))
        assert len(ov) == 0 and "firewall: design" in text and "firewall: holdout" in text
        try:
            split_halves.firewall(sp3, coords, pool_dir=Path(tmp), halves=("design",))
            raise AssertionError("halves must name two labels")
        except AssertionError as e:
            assert "two labels" in str(e)


# ============================================================================ registry.seed_from_frame
def test_registry_seed_then_mark_exposed_and_sync_preserves():
    frame = pd.DataFrame({"unit_id": ["u0001", "u0002", "u0003"], "candidate_id": ["J1", "J2", "J3"],
                          "desi_pool_overlap": [False, True, False]})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "golden_registry.csv"
        # pre-seed: mark_exposed must refuse (the documented KeyError)
        try:
            registry.mark_exposed(["u0001"], "truth_a1_sonnet_design_r1", "eval", path=path)
            raise AssertionError("mark_exposed on an empty registry must raise")
        except KeyError:
            pass
        reg, _ = _quiet(registry.seed_from_frame, frame, path)
        assert list(reg["unit_id"]) == ["u0001", "u0002", "u0003"] and (reg["split"] == "").all()
        assert list(reg["leak"]) == ["no", "desi_train", "no"] and (reg["render_version"] == registry.RENDER_VERSION).all()
        assert (reg["exposed_runs"] == "").all() and list(reg["candidate_id"]) == ["J1", "J2", "J3"]
        reg = registry.mark_exposed(["u0001", "u0003"], "truth_a1_sonnet_design_r1", "eval", path=path)
        r = reg.set_index("unit_id")
        assert r.at["u0001", "exposed_runs"] == "truth_a1_sonnet_design_r1" and r.at["u0002", "exposed_runs"] == ""
        # re-seeding is a no-op on existing rows (exposures kept), adds new units only
        frame2 = pd.concat([frame, pd.DataFrame({"unit_id": ["u0004"], "candidate_id": ["J4"], "desi_pool_overlap": [False]})])
        reg, _ = _quiet(registry.seed_from_frame, frame2, path)
        r = reg.set_index("unit_id")
        assert r.at["u0001", "exposed_runs"] == "truth_a1_sonnet_design_r1" and "u0004" in r.index
        # sync_from later: labelled units get split/grade, unlabelled seeded rows survive, exposures untouched
        labels = pd.DataFrame({"unit_id": ["u0001"], "candidate_id": ["J1"], "n_passes": [1], "grade_letter": ["B"], "render_sha": ["abc"]})
        splits = pd.DataFrame({"unit_id": ["u0001"], "split": ["validate"]})
        reg = registry.sync_from(labels, splits, frame=frame2, path=path)
        r = reg.set_index("unit_id")
        assert r.at["u0001", "split"] == "validate" and r.at["u0001", "grade_letter"] == "B"
        assert r.at["u0001", "exposed_runs"] == "truth_a1_sonnet_design_r1"
        assert set(r.index) == {"u0001", "u0002", "u0003", "u0004"} and r.at["u0003", "exposed_runs"] == "truth_a1_sonnet_design_r1"
        assert r.at["u0002", "split"] == "" and r.at["u0002", "leak"] == "desi_train"
        registry.assert_unexposed(["u0001", "u0003"], kinds=("fewshot", "sft"), path=path)     # eval exposure is allowed


def test_build_kit_collision_dirs_include_kits_truth():
    from lensjudge.golden import build_kit
    assert any(Path(d).name == "kits_truth" and Path(d).parent == _util.HERE for d in build_kit.DEFAULT_COLLISION_DIRS)
    with tempfile.TemporaryDirectory() as tmp:
        # a missing kits_truth dir is skipped; a byte-identical file raises
        items = Path(tmp) / "items"; items.mkdir()
        (items / "001.jpg").write_bytes(b"\xff\xd8same")
        _quiet(build_kit.assert_no_collision, items, [Path(tmp) / "absent"])
        kt = Path(tmp) / "kits_truth"; kt.mkdir(); (kt / "J1.jpg").write_bytes(b"\xff\xd8same")
        try:
            _quiet(build_kit.assert_no_collision, items, [kt])
            raise AssertionError("byte-identical truth JPEG must be refused")
        except AssertionError as e:
            assert "byte-identical" in str(e)


# ============================================================================ real data (skipped when absent)
def _have_J() -> bool:
    J = _util.JWST_REPO
    need = [J / "results" / "results.csv", J / "results" / "inspections.csv", J / "results" / "known_lens_recovery.csv",
            J / "data" / "targets.parquet", J / "data" / "simbad_lenses.csv",
            J / "data" / "lenscats" / "raw_misc" / "cowls_catalogue.csv", J / "data" / "lenscats" / "raw_misc" / "anomalymatch_jwst.csv",
            btm.OUTPUTS / "parity_train_pool.csv", _util.HERE / "frame.csv"]
    missing = [p for p in need if not p.exists()]
    if missing:
        print(f"  (skipped: missing {missing[0]})")
        return False
    return True


def test_real_catalogue_union_and_negatives_regression():
    if not _have_J():
        return
    cat, text = _quiet(stn.load_catalogue_union, _util.JWST_REPO)
    assert len(cat) >= stn.MIN_UNION and len(cat) == 195_818, len(cat)
    tracked = _util.HERE / "truth_negatives.csv"
    if not tracked.exists():
        print("  (truth_negatives.csv not built yet)")
        return
    src = btm.load_truth_sources(_util.JWST_REPO, btm.OUTPUTS, _util.HERE)
    (neg, S), _ = _quiet(stn.sample, src, cat, stn.N_TOTAL, _util.SEED)
    stn.check(neg)
    with tempfile.TemporaryDirectory() as tmp:
        _util.pin(neg, Path(tmp) / "n.csv")
        assert (Path(tmp) / "n.csv").read_text() == tracked.read_text(), \
            "rebuilt truth_negatives differs from golden/truth_negatives.csv — inputs or sampler changed; rebuild and re-pin"
    assert S["cat_matches"]["5"] == S["purged"] and S["cat_matches"]["2"] > 0
    assert neg["half"].value_counts().to_dict() == {"design": 200, "holdout": 200}


def test_real_manifest_and_splits_regression():
    if not _have_J():
        return
    tracked = _util.HERE / "truth_manifest.csv"
    if not tracked.exists() or not (_util.HERE / "truth_negatives.csv").exists():
        print("  (truth_manifest.csv not built yet)")
        return
    man = _util.read_pinned(tracked, dtype={"name": str, "unit_id": str}, float_precision="round_trip")
    assert list(man.columns) == CONTRACT_COLS
    assert int(man["is_positive"].sum()) == 87 and (man["truth_class"] == "negative").sum() == 400
    assert int(man["is_anchor"].sum()) == 5 and man["truth_class"].value_counts().to_dict()["cowls"] == 31
    assert man.loc[man["truth_class"] == "cowls", "cowls_band"].value_counts().to_dict() == \
        {"weak": 13, "marginal": 8, "strong": 5, "provenance": 5}
    # rebuilt without images -> identical except image_path / render_sha (and half when splits exist)
    with tempfile.TemporaryDirectory() as tmp:
        (out, notes), _ = _quiet(btm.build, _util.JWST_REPO, btm.OUTPUTS, _util.HERE, None, _util.HERE / "stamps",
                                 Path(tmp) / "kits_truth", None, True, Path(tmp) / "m.csv")
        a = man.drop(columns=["image_path"]); b = out.drop(columns=["image_path"])
        assert a.shape == b.shape
        same = _util.read_pinned(Path(tmp) / "m.csv", dtype={"name": str, "unit_id": str}, float_precision="round_trip").drop(columns=["image_path"])
        pd.testing.assert_frame_equal(a.reset_index(drop=True), same.reset_index(drop=True))
        # JPEGs under the temp kits dir carry the same bytes as the tracked ones
        for _, r in out[out["image_path"] != ""].head(5).iterrows():
            assert _util.sha_file(Path(r["image_path"])) == r["render_sha"]
    sp = _util.HERE / "truth_splits.csv"
    if sp.exists():
        splits = _util.read_pinned(sp, dtype={"candidate_id": str})
        st.check(splits)
        assert set(splits["candidate_id"]) == set(man["name"])
        assert (man.set_index("name").loc[splits["candidate_id"], "half"].to_numpy() == splits["half"].to_numpy()).all()
        neg = _util.read_pinned(_util.HERE / "truth_negatives.csv", dtype={"candidate_id": str})
        frame = _util.read_pinned(_util.HERE / "frame.csv", dtype={"unit_id": str})
        again, _ = _quiet(st.assign, man, neg, frame, _util.SEED)
        pd.testing.assert_frame_equal(again.fillna(""), splits.fillna(""), check_dtype=False)
        n = splits[splits["truth_class"] == "negative"]
        assert n["half"].value_counts().to_dict() == {"design": 200, "holdout": 200}


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
