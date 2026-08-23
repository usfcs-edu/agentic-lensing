#!/usr/bin/env python3
"""golden/build_truth_manifest.py — the JWST truth set (golden/truth_manifest.csv) for the
evidence-first grading scheme (Part 2), computed and pinned, never hand-summed.

WHAT. One row per cutout the truth evaluation may score, in eight `truth_class`es:

    cowls         the 31 blind COWLS positive controls (frame K_cowls); banded by the COWLS
                  six-grader ranking string (`cowls_band`, below)
    lit_galaxy    literature lens <= 2" from the cutout centre whose catalogue position is the
                  deflector / system (SIMBAD gLS, gLe, LeQ; a COWLS catalogue lens on a
                  non-control cutout; top-100 `discovery_status == known` rows the run's own
                  literature check calls galaxy-scale)
    lit_cluster   literature lens <= 2" whose catalogue position is an arc / knot (SIMBAD LeG,
                  LeI: cluster-scale lensing in the stamp, the centre is usually a cluster
                  member, not the deflector) + the top-100 known cluster arcs
    negative      golden/truth_negatives.csv (sample_truth_negatives.py): base-rate unflagged
                  targets purged 5" from the 195,818-position catalogue union
    stress_D      frame D_refuted (the pipeline's own refusals — machine-labelled)
    stress_U      frame U_tail (flagged, never verified — unlabelled, not negative)
    anomalymatch  AnomalyMatch JWST Class A/B/C candidates (human-voted, unconfirmed) on a
                  target within 2" that is not already a positive
    anchor        PI-derived design anchors (ranks 15, 7, 14, 16, 13) that are nothing else;
                  `is_anchor` is ALSO set on anchors that are positives (rank 16)

`is_positive` = cowls | lit_*  ("lensing anywhere in the 10\" stamp" — the primary label);
`centre_is_deflector` = the first secondary (cowls | gLS/gLe/LeQ | sep <= 1");
`is_stress` = stress_D | stress_U | anomalymatch (reported separately, never headline);
anchors NEVER count toward a truth endpoint (analyze_truth drops `is_anchor` rows first).

WHY the set is computed here. Design A and B both quoted positive counts (86 / 88) as hand
sums over overlapping lists (critique C9). This script derives every class from the run's
own tables, dedups by candidate_id and then at 2" by position (union-find; the rank-14
twin of rank 7 is the one deliberate exception: both anchor rows are kept because
|Δletter(7, 14)| on identical SW-only pixels is a pre-registered consistency check), and
pins the result, so the power tables use a number that exists on disk.

Positive-set rule (the one judgement call, stated): every cutout with a `known_lens_recovery`
match <= 2" is a positive UNLESS it is a top-100 row, where the run's own human literature
check (`discovery_status`) supersedes the raw crossmatch: `known` -> positive,
`field_match` / `possible` / `new` -> not (rank 48 sits 1.47" from the SIMBAD "VIRTUAL
PARENT" record of a lensing CLUSTER — that is the cluster, not a lens in the stamp). This
makes one frame U_tail unit with a SIMBAD lensed image at 1.31" a `lit_cluster` positive
rather than `stress_U` (printed as a note).

COWLS band (`cowls_band`), from the six-character ranking string (alphabet A B S U X Y,
one grader each; X = did not grade). The catalogue's own score code is
S = 2*nA + nB + nS (diag_truth §1.1: reproduces 417/418 catalogue rows; cross-checked here
against `control_recovery.score_x`). Bands, as in the approved plan and diag §4:
strong = nA >= 3 (== S >= 8 on every control), marginal = 4 <= S <= 7, weak = S <= 3 (incl.
all-U), provenance = no ranking (the M25 / P18 provenance-only codes). The work-package
text's "3*nA + 2*nB + nC" is NOT used: C is not in the alphabet and that formula reproduces
neither the catalogue score codes nor the plan's 5 / 8 / 13 / 5 band counts.

field_class (negative stratification covariate; also carried on every row): 1727 -> cosmos
(COSMOS-Web, where 28/31 COWLS sit); CLUSTER_PROPOSALS -> cluster (the four big cluster
programmes named in the plan plus every proposal whose fields.parquet target names are
galaxy-cluster designations — hard-coded after a regex review, `derive_cluster_proposals`
re-runs that review and the build prints any proposal the review would add); else blank.

Images: `image_path` = footer-cropped (y < 540) JPEG at quality 92 under
golden/kits_truth/<candidate_id>.jpg from golden/stamps/<id>/<id>_v1.jpg, encoded WITHOUT
Huffman optimisation (build_kit uses optimize=True for the kit items) so a truth JPEG is
pixel-identical to, but never byte-identical with, the kit item of the same unit —
build_kit.assert_no_collision (which now also scans kits_truth/) would otherwise refuse
every kit rebuild. Ids without a stamp get image_path '' and go to
golden/truth_fetch_list.csv (candidate_id, ra_deg, dec_deg, layout; pinned) for
`build_stamps.py --frame golden/truth_fetch_list.csv`; re-run this script afterwards.

`half` is filled from golden/truth_splits.csv when it exists and covers exactly this row
set (split_truth.py re-pins the manifest after splitting), else ''.

`image_path_v2r` / `render_sha_v2r` (the A3 arm's jwst_v2r render) are filled from the
pinned CSV golden/render_v2.py writes next to its JPEGs (`<kits_v2r>/render_v2r.csv`:
id, image_path_v2r, render_sha_v2r, status) whenever that file exists — render_v2.py never
rewrites the manifest itself, so re-running THIS script after it is the documented re-pin.
Paths are stored relative to lensjudge/ like image_path; rows whose render failed stay ''.

Run order:  sample_truth_negatives.py -> build_truth_manifest.py -> split_truth.py
            (-> build_stamps.py --frame golden/truth_fetch_list.csv -> build_truth_manifest.py
             -> render_v2.py --manifest golden/truth_manifest.csv --out-dir golden/kits_truth_v2r
             -> build_truth_manifest.py    # merges image_path_v2r / render_sha_v2r)

    python lensjudge/golden/build_truth_manifest.py [--jwst-repo J] [--golden-dir golden]
        [--negatives golden/truth_negatives.csv] [--stamps-dir golden/stamps]
        [--kits-dir golden/kits_truth] [--render-v2r golden/kits_truth_v2r/render_v2r.csv] [--no-images]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402
from lensjudge.golden import build_frame as bf  # noqa: E402
from lensjudge.golden.build_eval_manifest import EVAL_COLS  # noqa: E402

SEED = _util.SEED
GOLDEN = _util.HERE
OUTPUTS = _util.LENSJUDGE / "outputs"
STAMPS = GOLDEN / "stamps"
KITS_TRUTH = GOLDEN / "kits_truth"
KITS_TRUTH_V2R = GOLDEN / "kits_truth_v2r"          # render_v2.py --out-dir (render_v2r.csv lives there)
MANIFEST_CSV = GOLDEN / "truth_manifest.csv"
NEGATIVES_CSV = GOLDEN / "truth_negatives.csv"
SPLITS_CSV = GOLDEN / "truth_splits.csv"
FETCH_LIST_CSV = GOLDEN / "truth_fetch_list.csv"

SOURCE = "truth_jwst"
SURVEY_KEY = "jwst"
TRUTH_CLASSES = ("cowls", "lit_galaxy", "lit_cluster", "negative", "stress_D", "stress_U",
                 "anomalymatch", "anchor")
POSITIVE_CLASSES = ("cowls", "lit_galaxy", "lit_cluster")
STRESS_CLASSES = ("stress_D", "stress_U", "anomalymatch")
COWLS_BANDS = ("strong", "marginal", "weak", "provenance")
FIELD_CLASSES = ("cosmos", "cluster", "blank")
# the contract's column order: the 11 build_eval_set columns FIRST, then the truth extras
TRUTH_COLS = EVAL_COLS + [
    "unit_id", "image_path", "image_path_v2r", "render_sha", "render_sha_v2r",
    "truth_class", "is_positive", "is_stress", "is_anchor",
    "cowls_band", "cowls_ranking", "cowls_theta_E",
    "known_lens_name", "known_type", "known_sep_arcsec", "centre_is_deflector",
    "layout", "field_class", "proposal", "mag_r", "prior_exposure",
    "pipe_grade_passcount", "pipe_inspector_conf", "pipe_score", "in_frame", "half",
]
FETCH_COLS = ["candidate_id", "ra_deg", "dec_deg", "layout"]   # what build_stamps.load_frame reads

LIT_ARCSEC = 2.0             # literature match radius (build_frame.MATCH_ARCSEC)
DEDUP_ARCSEC = 2.0           # positional dedup among positives
CENTRE_ARCSEC = 1.0          # centre_is_deflector by separation
SYSTEM_ARCSEC = bf.SYSTEM_ARCSEC
ANCHOR_RANKS = (15, 7, 14, 16, 13)
# SIMBAD otype semantics (diag_truth §1.2): the position IS the deflector/system vs an arc/knot
GALAXY_TYPES = ("COWLS", "SIMBAD:gLS", "SIMBAD:gLe", "SIMBAD:LeQ")
CLUSTER_TYPES = ("SIMBAD:LeG", "SIMBAD:LeI")
CENTRE_TYPES = ("COWLS", "SIMBAD:gLS", "SIMBAD:gLe", "SIMBAD:LeQ")
# top-100 `discovery_status == known` rows: galaxy vs cluster scale read off the master
# `designation` text (rank: SL2S J02176 / DESI-307.9 / GDS J123730 = galaxy-scale lenses;
# [LRJ2007] 4 in A1689, RCSGA 032727 knots E2.2 / E1.1 = cluster arcs; MACS J0416 ID14 =
# galaxy-galaxy lens whose deflector is a cluster member -> galaxy). A known rank missing
# here falls back to the SIMBAD-type rule and prints a WARN.
TOP100_KNOWN_SCALE = {1: "galaxy", 4: "galaxy", 6: "galaxy", 9: "cluster", 16: "cluster",
                      17: "cluster", 21: "galaxy"}
TOP100_KNOWN_TYPE = "top100:known"

COSMOS_PROPOSALS = frozenset({"1727"})
# Galaxy-cluster programmes. Base set from the plan (6882 Abell/MACS sample, 5594 ACT/PSZ2
# clusters, 5890 A370/MACS0416/MACS1149, 2561 UNCOVER A2744) + every proposal whose
# J/data/fields.parquet target names are cluster designations (review of 2026-08-23 with
# CLUSTER_NAME_RE; star clusters / dwarf galaxies the regex also catches — 1256 Trapezium,
# 2045 Arches, 1334/4570 Draco II, 6796 "sub_clusters" — are deliberately left out).
CLUSTER_PROPOSALS = frozenset({
    "6882", "5594", "5890", "2561",                       # plan
    "1176", "1199", "1208", "1324", "1355", "1433", "1840", "2555", "2736", "2756", "2767",
    "2883", "3073", "3293", "3362", "3433", "3516", "3538", "3743", "4043", "4111", "4212",
    "4598", "4744", "4903", "5058", "5293", "5782", "6207", "6675", "9478",
})
CLUSTER_NAME_RE = re.compile(
    r"(ABELL|ACO|MACS|SMACS|CLUSTER|ACT-?CL|SPT\d|PLCK|RXC|RXJ|CLG|EL-?GORDO|BULLET|^A\d{3,4}\b|"
    r"^A\d{3,4}[-_]|M0416|MCS-J|SUNBURST|COOLJ|MS1358|RCS0224|MRG[MPS]\d|PSZ)", re.I)

# JPEG encoding of the truth images: same quality as the kit, NO Huffman optimisation (see
# the module docstring — pixel-identical to the kit item, never byte-identical)
TRUTH_JPEG_KW = dict(quality=92, optimize=False)


# ============================================================================ pure helpers
def cowls_score(ranking) -> tuple[int, int]:
    """(S, nA) from a COWLS ranking string: S = 2*nA + nB + nS (the catalogue's own score
    code). '' / NaN -> (-1, 0)."""
    if ranking is None or (isinstance(ranking, float) and np.isnan(ranking)):
        return -1, 0
    s = str(ranking).strip().upper()
    if not s or s in ("NAN", "NONE"):
        return -1, 0
    nA, nB, nS = s.count("A"), s.count("B"), s.count("S")
    return 2 * nA + nB + nS, nA


def cowls_band(ranking) -> str:
    """strong (nA >= 3) / marginal (4 <= S <= 7) / weak (S <= 3, incl. all-U) / provenance
    (no ranking string: the M25 / P18 provenance-only codes)."""
    S, nA = cowls_score(ranking)
    if S < 0:
        return "provenance"
    if nA >= 3:
        return "strong"
    if 4 <= S <= 7:
        return "marginal"
    return "weak"


def field_class_of(proposal) -> str:
    """cosmos / cluster / blank from the proposal id (str or int)."""
    p = "" if proposal is None or (isinstance(proposal, float) and np.isnan(proposal)) else str(proposal).strip()
    if p.endswith(".0"):
        p = p[:-2]
    if p in COSMOS_PROPOSALS:
        return "cosmos"
    if p in CLUSTER_PROPOSALS:
        return "cluster"
    return "blank"


def derive_cluster_proposals(fields: pd.DataFrame, min_frac: float = 0.5) -> dict[str, list[str]]:
    """The regex review behind CLUSTER_PROPOSALS: {proposal: matching target names} for every
    proposal where >= `min_frac` of its distinct fields.parquet target names match
    CLUSTER_NAME_RE. Used by the build to print drift against the hard-coded set."""
    out: dict[str, list[str]] = {}
    props = fields["proposal_id"].astype(str).str.strip()
    for p, g in fields.groupby(props):
        names = sorted(set(g["target_name"].astype(str)))
        hit = [n for n in names if CLUSTER_NAME_RE.search(n)]
        if names and len(hit) / len(names) >= min_frac:
            out[p] = hit
    return out


def literature_le2(known: pd.DataFrame, radius_arcsec: float = LIT_ARCSEC) -> pd.DataFrame:
    """known_lens_recovery rows within `radius` of the cutout centre, ANY source (COWLS
    included — build_frame.dedup_known_lenses drops COWLS rows because the controls enter
    the frame through control_recovery; here a COWLS lens on a non-control cutout is a
    literature positive like any other), one per cutout: the CLOSEST name wins."""
    k = known[known["sep_arcsec"].astype(float) <= radius_arcsec]
    k = k.sort_values(["sep_arcsec", "lens_name"], kind="stable")
    return k.drop_duplicates("cutout_id", keep="first").reset_index(drop=True)


def scale_of_type(known_type: str) -> str:
    """galaxy / cluster from a known_type string ('' -> galaxy, the conservative default)."""
    t = str(known_type or "")
    return "cluster" if t in CLUSTER_TYPES else "galaxy"


def short_designation(text) -> str:
    """The master's `designation` is a sentence; keep the leading catalogue name."""
    s = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text).strip()
    return re.split(r"\s+\(|\s+-+\s|\s+=\s|\s+--", s, maxsplit=1)[0].strip()


def dedup_positions(df: pd.DataFrame, priority: np.ndarray, radius_arcsec: float = DEDUP_ARCSEC):
    """Keep one row per 2" position group (union-find), the one with the LOWEST `priority`
    (ties: candidate_id). Returns (kept_df, dropped_df with a `dup_of` column)."""
    if len(df) == 0:
        return df, df.assign(dup_of=pd.Series(dtype=str))
    grp = bf.union_find_systems(df["ra_deg"], df["dec_deg"], radius_arcsec)
    d = df.assign(_grp=grp, _prio=np.asarray(priority)).sort_values(["_grp", "_prio", "candidate_id"], kind="stable")
    first = d.drop_duplicates("_grp", keep="first")
    keep_of = first.set_index("_grp")["candidate_id"]
    dropped = d[d.duplicated("_grp", keep="first")].assign(dup_of=lambda x: x["_grp"].map(keep_of))
    return (first.drop(columns=["_grp", "_prio"]).reset_index(drop=True),
            dropped.drop(columns=["_grp", "_prio"]).reset_index(drop=True))


# ============================================================================ sources
def load_truth_sources(jwst_repo: Path = _util.JWST_REPO, outputs: Path = OUTPUTS,
                       golden_dir: Path = GOLDEN) -> dict:
    """build_frame.load_sources (results, inspections, master, controls, known, targets,
    desi, render) + the pinned frame, the AnomalyMatch table and fields.parquet."""
    jwst_repo = Path(jwst_repo)
    src = bf.load_sources(jwst_repo, outputs)
    src["frame"] = _util.read_pinned(Path(golden_dir) / "frame.csv", dtype={"candidate_id": str, "unit_id": str})
    am = jwst_repo / "data" / "lenscats" / "raw_misc" / "anomalymatch_jwst.csv"
    src["anomalymatch"] = pd.read_csv(am) if am.exists() else None
    fp = jwst_repo / "data" / "fields.parquet"
    src["fields"] = pd.read_parquet(fp, columns=["proposal_id", "target_name"]) if fp.exists() else None
    return src


# ============================================================================ the classes
def select_positives(src: dict) -> tuple[pd.DataFrame, dict]:
    """cowls / lit_galaxy / lit_cluster rows (candidate_id, truth_class, known_*, cowls_*,
    rank_top100, in_frame, unit_id, ra_deg, dec_deg), deduped by id then at 2" by position.
    Returns (positives, notes)."""
    frame = src["frame"]; master = src["master"]; known = src["known"]; res = src["results"]
    fidx = frame.set_index("candidate_id")
    rk = res.set_index("id")
    top_ids = set(master["candidate_id"].astype(str))
    notes: dict = {"dropped_dups": [], "frame_lit_outside_L_known": [], "warn": []}
    rows: list = []

    def _frame_cols(cid):
        r = fidx.loc[cid]
        return dict(unit_id=str(r["unit_id"]), in_frame=True,
                    known_lens_name=str(r["known_lens_name"]) if pd.notna(r["known_lens_name"]) else "",
                    known_type=str(r["known_type"]) if pd.notna(r["known_type"]) else "",
                    known_sep_arcsec=float(r["known_sep_arcsec"]) if pd.notna(r["known_sep_arcsec"]) else np.nan,
                    cowls_ranking=str(r["cowls_ranking"]) if pd.notna(r["cowls_ranking"]) else "",
                    cowls_theta_E=float(r["cowls_theta_E"]) if pd.notna(r["cowls_theta_E"]) else np.nan,
                    rank_top100=float(r["rank_top100"]) if pd.notna(r["rank_top100"]) else np.nan,
                    prior_exposure=int(r["prior_exposure"]))

    # ---- cowls: every frame K_cowls unit (priority 0) --------------------------------------
    for cid in sorted(frame.loc[frame["stratum"] == "K_cowls", "candidate_id"].astype(str)):
        rows.append(dict(candidate_id=cid, truth_class="cowls", _prio=0, **_frame_cols(cid)))
    # ---- lit_*: frame L_known units (priority 1) --------------------------------------------
    for cid in sorted(frame.loc[frame["stratum"] == "L_known", "candidate_id"].astype(str)):
        fc = _frame_cols(cid)
        rows.append(dict(candidate_id=cid, truth_class=f"lit_{scale_of_type(fc['known_type'])}", _prio=1, **fc))
    placed = {r["candidate_id"] for r in rows}
    # ---- top-100 discovery_status == known (priority 1; the run's own literature check) -----
    kn = master[master["discovery_status"].astype(str).str.strip() == "known"].sort_values("rank")
    ms = master.set_index("candidate_id")
    for _, m in kn.iterrows():
        cid = str(m["candidate_id"])
        if cid in placed:
            continue
        if cid not in fidx.index:
            notes["warn"].append(f"top-100 known rank {int(m['rank'])} {cid} is not a frame row (alias?) — skipped")
            continue
        fc = _frame_cols(cid)
        rank = int(m["rank"])
        if rank in TOP100_KNOWN_SCALE:
            scale = TOP100_KNOWN_SCALE[rank]
        else:
            scale = scale_of_type(fc["known_type"])
            notes["warn"].append(f"top-100 known rank {rank} not in TOP100_KNOWN_SCALE; scale by SIMBAD type -> {scale}")
        if not fc["known_lens_name"]:
            fc["known_lens_name"] = short_designation(m.get("designation"))
            fc["known_type"] = TOP100_KNOWN_TYPE
        if not np.isfinite(fc["known_sep_arcsec"]):
            fc["known_sep_arcsec"] = float(m["nearest_sep_arcsec"]) if pd.notna(m.get("nearest_sep_arcsec")) else np.nan
        rows.append(dict(candidate_id=cid, truth_class=f"lit_{scale}", _prio=1, **fc))
        placed.add(cid)
    # ---- literature <= 2" on any non-top-100 cutout not yet placed (priority 2 off-frame) ----
    lit = literature_le2(known, LIT_ARCSEC)
    for _, k in lit.iterrows():
        cid = str(k["cutout_id"])
        if cid in placed or cid in top_ids:
            continue
        in_frame = cid in fidx.index
        base = dict(known_lens_name=str(k["lens_name"]), known_type=str(k["lens_src"]),
                    known_sep_arcsec=float(k["sep_arcsec"]), cowls_ranking="", cowls_theta_E=np.nan,
                    rank_top100=np.nan, prior_exposure=0, unit_id="", in_frame=False)
        if in_frame:
            base.update(_frame_cols(cid))
            # the frame's own literature columns are the same crossmatch; keep the closest name
            base.update(known_lens_name=str(k["lens_name"]), known_type=str(k["lens_src"]),
                        known_sep_arcsec=float(k["sep_arcsec"]))
            notes["frame_lit_outside_L_known"].append(
                f"{base['unit_id']} {cid} ({fidx.loc[cid, 'stratum']}) carries {k['lens_name']} "
                f"[{k['lens_src']}] at {float(k['sep_arcsec']):.2f}\" -> positive")
        rows.append(dict(candidate_id=cid, truth_class=f"lit_{scale_of_type(k['lens_src'])}",
                         _prio=1 if in_frame else 2, **base))
        placed.add(cid)

    pos = pd.DataFrame(rows)
    assert pos["candidate_id"].is_unique
    pos["ra_deg"] = rk.loc[pos["candidate_id"], "ra"].to_numpy(float)
    pos["dec_deg"] = rk.loc[pos["candidate_id"], "dec"].to_numpy(float)
    prio = pos.pop("_prio").to_numpy()
    kept, dropped = dedup_positions(pos, prio, DEDUP_ARCSEC)
    for _, d in dropped.iterrows():
        notes["dropped_dups"].append(f"{d['candidate_id']} ({d['truth_class']}) within {DEDUP_ARCSEC}\" of {d['dup_of']} — dropped")
    return kept.reset_index(drop=True), notes


def select_stress(src: dict, positive_ids: set) -> tuple[pd.DataFrame, dict]:
    """stress_D (frame D_refuted), stress_U (frame U_tail) — minus positives — and
    anomalymatch (Class A/B/C within 2" of a target, not a positive). Precedence when a
    target is both a frame stress unit and an AnomalyMatch candidate: anomalymatch (the
    human-voted label is the more informative one; printed)."""
    frame = src["frame"]; targets = src["targets"]; am = src.get("anomalymatch")
    notes: dict = {"overlaps": [], "anomalymatch_classes": {}}
    rows: list = []
    taken = set(positive_ids)
    fidx = frame.set_index("candidate_id")
    if am is not None and len(am):
        t = targets[["id", "ra", "dec"]].copy()
        t["id"] = t["id"].astype(str)
        idx, sep, hit = bf.sky_match(am["RA"], am["DEC"], t["ra"], t["dec"], LIT_ARCSEC)
        amh = am[hit].assign(candidate_id=t["id"].to_numpy()[idx[hit]], sep=sep[hit])
        amh = amh.sort_values(["sep", "candidate_id"], kind="stable").drop_duplicates("candidate_id")
        for _, a in amh.sort_values("candidate_id").iterrows():
            cid = str(a["candidate_id"])
            cls = str(a.get("voted_class", "")).strip().rstrip(".")
            if cid in taken:
                continue
            if cid in fidx.index:
                notes["overlaps"].append(f"{fidx.loc[cid, 'unit_id']} {cid} ({fidx.loc[cid, 'stratum']}) is AnomalyMatch {cls} -> anomalymatch")
            notes["anomalymatch_classes"][cls] = notes["anomalymatch_classes"].get(cls, 0) + 1
            rows.append(dict(candidate_id=cid, truth_class="anomalymatch",
                             known_lens_name=f"AnomalyMatch {cls}" + (f" {a['ID']}" if pd.notna(a.get("ID")) else ""),
                             known_type="anomalymatch", known_sep_arcsec=float(a["sep"]),
                             unit_id=str(fidx.loc[cid, "unit_id"]) if cid in fidx.index else "",
                             in_frame=cid in fidx.index,
                             prior_exposure=int(fidx.loc[cid, "prior_exposure"]) if cid in fidx.index else 0,
                             rank_top100=float(fidx.loc[cid, "rank_top100"]) if cid in fidx.index and pd.notna(fidx.loc[cid, "rank_top100"]) else np.nan))
            taken.add(cid)
    for stratum, cls in (("D_refuted", "stress_D"), ("U_tail", "stress_U")):
        for _, r in frame[frame["stratum"] == stratum].sort_values("candidate_id").iterrows():
            cid = str(r["candidate_id"])
            if cid in taken:
                continue
            rows.append(dict(candidate_id=cid, truth_class=cls, unit_id=str(r["unit_id"]), in_frame=True,
                             known_lens_name=str(r["known_lens_name"]) if pd.notna(r["known_lens_name"]) else "",
                             known_type=str(r["known_type"]) if pd.notna(r["known_type"]) else "",
                             known_sep_arcsec=float(r["known_sep_arcsec"]) if pd.notna(r["known_sep_arcsec"]) else np.nan,
                             prior_exposure=int(r["prior_exposure"]),
                             rank_top100=float(r["rank_top100"]) if pd.notna(r["rank_top100"]) else np.nan))
            taken.add(cid)
    cols = ["candidate_id", "truth_class", "unit_id", "in_frame", "known_lens_name", "known_type",
            "known_sep_arcsec", "prior_exposure", "rank_top100"]
    return (pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)), notes


def select_anchors(src: dict) -> pd.DataFrame:
    """ANCHOR_RANKS -> (candidate_id, rank_top100, unit_id, in_frame, prior_exposure). The
    rank-14 alias is not a frame row (collapsed onto rank 7 in the frame); it gets its own
    manifest row with prior_exposure by rank."""
    master = src["master"]; frame = src["frame"]
    fidx = frame.set_index("candidate_id")
    rows = []
    for rank in ANCHOR_RANKS:
        m = master[master["rank"] == rank]
        assert len(m) == 1, f"anchor rank {rank} not in the top-100 master"
        cid = str(m.iloc[0]["candidate_id"])
        in_frame = cid in fidx.index
        rows.append(dict(candidate_id=cid, rank_top100=float(rank),
                         unit_id=str(fidx.loc[cid, "unit_id"]) if in_frame else "", in_frame=in_frame,
                         prior_exposure=int(np.select([rank <= bf.DOCX_RANKS, rank <= bf.TOP_N], [2, 1], 0))))
    return pd.DataFrame(rows)


# ============================================================================ attach columns
def attach_run_columns(df: pd.DataFrame, src: dict) -> pd.DataFrame:
    """Positions (results.csv float64), pipeline verdict, observations / proposal / mag_r
    (targets.parquet), layout (build_frame.derive_layout with the run's render manifest),
    field_class, DESI leak flag, prior_exposure (frame value or by rank)."""
    res = src["results"].set_index("id"); tg = src["targets"].set_index("id")
    ids = df["candidate_id"].astype(str)
    missing = [c for c in ids if c not in res.index]
    assert not missing, f"ids absent from results.csv: {missing[:5]}"
    r = res.loc[ids]; t = tg.loc[ids]
    out = df.copy()
    out["candidate_id"] = ids.to_numpy()
    out["ra_deg"] = r["ra"].to_numpy(float)
    out["dec_deg"] = r["dec"].to_numpy(float)
    grade = r["grade"].fillna("").astype(str).to_numpy()
    flagged = grade != ""
    out["pipe_grade_passcount"] = grade
    out["pipe_inspector_conf"] = np.where(flagged, r["confidence"].to_numpy(float), np.nan)
    out["pipe_score"] = np.where(flagged, r["score"].to_numpy(float), np.nan)
    for c in ("sw_obs", "lw_obs", "sw_filter", "lw_filter", "proposal"):
        out[c] = t[c].fillna("").astype(str).to_numpy()
    out["mag_r"] = t["mag_r"].to_numpy(float)
    out["layout"] = bf.derive_layout(t["sw_obs"], t["lw_obs"], src.get("render"), ids)
    out["field_class"] = [field_class_of(p) for p in out["proposal"]]
    _, _, hit = bf.sky_match(out["ra_deg"], out["dec_deg"], src["desi"]["ra"], src["desi"]["dec"], bf.MATCH_ARCSEC)
    out["leak"] = np.where(hit, "desi_train", "no")
    if "prior_exposure" not in out.columns:
        out["prior_exposure"] = 0
    pe = pd.to_numeric(out["prior_exposure"], errors="coerce").fillna(0).astype(int).to_numpy()
    rk = pd.to_numeric(out.get("rank_top100", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(10 ** 9).to_numpy(int)
    out["prior_exposure"] = np.maximum(pe, np.select([rk <= bf.DOCX_RANKS, rk <= bf.TOP_N], [2, 1], 0)).astype(int)
    return out


def positives_table(src: dict) -> tuple[pd.DataFrame, dict]:
    """select_positives + run columns: what the negative sampler stratifies to."""
    pos, notes = select_positives(src)
    return attach_run_columns(pos, src), notes


# ============================================================================ assembly
def assemble(src: dict, negatives: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """All classes -> one DataFrame with TRUTH_COLS (image columns blank; see write_images).
    Precedence when one id qualifies twice: cowls > lit_* > anomalymatch > stress_D >
    stress_U > anchor > negative (anchors are flagged on whichever row they land on)."""
    pos, notes = select_positives(src)
    stress, snotes = select_stress(src, set(pos["candidate_id"]))
    anchors = select_anchors(src)
    notes.update(snotes)
    parts = [pos, stress]
    have = set(pos["candidate_id"]) | set(stress["candidate_id"])
    extra_anchor = anchors[~anchors["candidate_id"].isin(have)].assign(truth_class="anchor")
    parts.append(extra_anchor)
    have |= set(extra_anchor["candidate_id"])
    if negatives is not None and len(negatives):
        neg = negatives.copy()
        neg["candidate_id"] = neg["candidate_id"].astype(str)
        clash = sorted(set(neg["candidate_id"]) & have)
        assert not clash, f"truth_negatives overlap other classes: {clash[:5]}"
        parts.append(pd.DataFrame({"candidate_id": neg["candidate_id"], "truth_class": "negative",
                                   "unit_id": "", "in_frame": False, "prior_exposure": 0,
                                   "rank_top100": np.nan}))
    df = pd.concat(parts, ignore_index=True)
    assert df["candidate_id"].is_unique, "a candidate_id landed in two classes"
    df = attach_run_columns(df, src)
    # anchors + flags
    anchor_ids = set(anchors["candidate_id"])
    df["is_anchor"] = df["candidate_id"].isin(anchor_ids).to_numpy(bool)
    assert df["is_anchor"].sum() == len(ANCHOR_RANKS), "every anchor rank must have a manifest row"
    df["is_positive"] = df["truth_class"].isin(POSITIVE_CLASSES).to_numpy(bool)
    df["is_stress"] = df["truth_class"].isin(STRESS_CLASSES).to_numpy(bool)
    # literature columns: '' / NaN defaults for rows without one
    for c in ("known_lens_name", "known_type", "cowls_ranking", "unit_id"):
        df[c] = df[c].fillna("").astype(str) if c in df.columns else ""
    for c in ("known_sep_arcsec", "cowls_theta_E"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    # the COWLS catalogue writes einstein_radius = 0 where none was measured: a placeholder,
    # not a radius (it would enter the Spearman endpoint and the theta_E <= 1" stratum)
    df.loc[df["cowls_theta_E"] <= 0, "cowls_theta_E"] = np.nan
    # milli-arcsec is plenty, and a short repr survives the CSV round trip that split_truth's
    # re-pin of this file goes through (the sha must not depend on who wrote it last)
    df["known_sep_arcsec"] = df["known_sep_arcsec"].round(3)
    df["in_frame"] = df["in_frame"].fillna(False).astype(bool)
    df["cowls_band"] = [cowls_band(r) if tc == "cowls" else "" for r, tc in zip(df["cowls_ranking"], df["truth_class"])]
    # the secondary label is defined on positives only (an AnomalyMatch candidate centred on
    # its target is not "the centre is a known deflector")
    sep = df["known_sep_arcsec"].to_numpy(float)
    df["centre_is_deflector"] = (df["is_positive"] & ((df["truth_class"] == "cowls") | df["known_type"].isin(CENTRE_TYPES)
                                                      | (np.nan_to_num(sep, nan=np.inf) <= CENTRE_ARCSEC))).to_numpy(bool)
    # the negatives' sampler columns are authoritative for negatives (same sources; asserted)
    if negatives is not None and len(negatives):
        nn = negatives.set_index(negatives["candidate_id"].astype(str))
        m = df["truth_class"] == "negative"
        for c in ("layout", "field_class", "proposal"):
            if c in nn.columns:
                theirs = nn.loc[df.loc[m, "candidate_id"], c].astype(str).to_numpy()
                ours = df.loc[m, c].astype(str).to_numpy()
                assert (theirs == ours).all(), f"truth_negatives.{c} disagrees with the run tables"
    # eval columns
    df["name"] = df["candidate_id"]
    df["ra"] = df["ra_deg"]; df["dec"] = df["dec_deg"]
    df["survey_key"] = SURVEY_KEY
    df["grade_truth"] = ""
    df["binary_label"] = np.select([df["is_positive"], df["truth_class"] == "negative"], ["lens", "nonlens"], "")
    df["source"] = SOURCE
    df["region"] = df["proposal"].astype(str)
    df["tractor_type"] = ""
    df["p_meta"] = np.nan
    for c in ("image_path", "image_path_v2r", "render_sha", "render_sha_v2r", "half"):
        df[c] = ""
    # deterministic row order: class order, then candidate_id
    order = {c: i for i, c in enumerate(TRUTH_CLASSES)}
    df = df.assign(_o=df["truth_class"].map(order)).sort_values(["_o", "candidate_id"], kind="stable")
    df = df.drop(columns=["_o"]).reset_index(drop=True)
    # sanity: the 2" positional dedup holds among non-anchor rows of different ids
    na = df[~df["is_anchor"]]
    grp = bf.union_find_systems(na["ra_deg"], na["dec_deg"], DEDUP_ARCSEC)
    assert len(set(grp)) == len(na), "two non-anchor manifest rows within 2\" of each other"
    return df.reindex(columns=TRUTH_COLS), notes


# ============================================================================ images / half
def write_images(man: pd.DataFrame, stamps_dir: Path = STAMPS, kits_dir: Path = KITS_TRUTH) -> pd.DataFrame:
    """Footer-cropped q92 (optimize=False) JPEG per row with a v1 composite; image_path /
    render_sha filled, '' where no stamp exists yet."""
    from PIL import Image
    from lensjudge.common import jwst_fetch as jf
    kits_dir = Path(kits_dir); stamps_dir = Path(stamps_dir)
    paths, shas = [], []
    for cid in man["name"].astype(str):
        assert _util.safe_name(cid) == cid, f"candidate_id {cid!r} is not filesystem-safe"
        srcp = stamps_dir / cid / f"{cid}_v1.jpg"
        if not srcp.exists():
            paths.append(""); shas.append(""); continue
        dst = kits_dir / f"{cid}.jpg"
        im = Image.open(srcp).convert("RGB")
        if im.width != 752 or im.height < jf.FOOTER_Y:
            raise ValueError(f"{srcp}: expected a 752x562 composite, got {im.size}")
        kits_dir.mkdir(parents=True, exist_ok=True)
        jf.crop_footer(im, jf.FOOTER_Y).save(dst, **TRUTH_JPEG_KW)
        paths.append(str(dst.relative_to(_util.LENSJUDGE)) if dst.is_relative_to(_util.LENSJUDGE) else str(dst))
        shas.append(_util.sha_file(dst))
    out = man.copy()
    out["image_path"] = paths
    out["render_sha"] = shas
    return out


def attach_half(man: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    """Fill `half` from a truth_splits table that covers EXACTLY this row set; otherwise
    leave '' and say so (a stale split must never silently label a different manifest)."""
    out = man.copy()
    out["half"] = ""
    if splits is None or not len(splits):
        return out
    s = splits.set_index(splits["candidate_id"].astype(str))["half"].astype(str)
    ids = out["name"].astype(str)
    if set(s.index) != set(ids):
        print(f"NOTE: truth_splits covers {len(s)} ids, manifest has {len(ids)} — half left blank; "
              f"re-run split_truth.py", flush=True)
        return out
    out["half"] = ids.map(s).to_numpy()
    return out


def attach_v2r(man: pd.DataFrame, render_csv: Path | None) -> pd.DataFrame:
    """Fill `image_path_v2r` / `render_sha_v2r` from render_v2.py's pinned CSV (columns id,
    image_path_v2r, render_sha_v2r, status). Only `status == "ok"` rows with an existing
    JPEG count; paths become lensjudge-relative like image_path; everything else stays ''.
    The render CSV's sha is verified (it is pinned), and a CSV that names ids outside this
    manifest is reported, never silently dropped."""
    out = man.copy()
    out["image_path_v2r"] = ""
    out["render_sha_v2r"] = ""
    if render_csv is None or not Path(render_csv).exists():
        return out
    rv = _util.read_pinned(Path(render_csv), dtype=str).fillna("")
    rv = rv[(rv["status"] == "ok") & (rv["image_path_v2r"] != "")]
    rv = rv.set_index(rv["id"].astype(str))
    ids = out["name"].astype(str)
    stray = sorted(set(rv.index) - set(ids))
    if stray:
        print(f"NOTE: {render_csv} names {len(stray)} id(s) not in the manifest (ignored): {stray[:3]}", flush=True)
    csv_dir = Path(render_csv).resolve().parent      # render_v2.py writes <out-dir>/<id>.jpg next to its CSV
    paths, shas = [], []
    for cid in ids:
        if cid not in rv.index:
            paths.append(""); shas.append(""); continue
        given = Path(rv.at[cid, "image_path_v2r"])
        # absolute as written (render_v2.py resolves its out-dir), else lensjudge-relative, else
        # cwd-relative (an older CSV), else the JPEG next to the CSV by construction
        cands = [given] if given.is_absolute() else [_util.LENSJUDGE / given, Path.cwd() / given]
        cands.append(csv_dir / given.name)
        p = next((c.resolve() for c in cands if c.exists()), None)
        if p is None:
            paths.append(""); shas.append(""); continue
        paths.append(str(p.relative_to(_util.LENSJUDGE)) if p.is_relative_to(_util.LENSJUDGE) else str(p))
        shas.append(rv.at[cid, "render_sha_v2r"])
    out["image_path_v2r"] = paths
    out["render_sha_v2r"] = shas
    return out


def fetch_list(man: pd.DataFrame) -> pd.DataFrame:
    """Rows without a truth JPEG -> what build_stamps.py --frame needs (candidate_id, ra_deg,
    dec_deg; load_frame also reads unit_id / rank_top100 / layout when present)."""
    need = man[man["image_path"].astype(str) == ""]
    return pd.DataFrame({"candidate_id": need["name"].astype(str), "ra_deg": need["ra"].astype(float),
                         "dec_deg": need["dec"].astype(float), "layout": need["layout"].astype(str)}
                        ).reset_index(drop=True)[FETCH_COLS]


# ============================================================================ build / CLI
def build(jwst_repo: Path = _util.JWST_REPO, outputs: Path = OUTPUTS, golden_dir: Path = GOLDEN,
          negatives_path: Path | None = None, stamps_dir: Path = STAMPS, kits_dir: Path = KITS_TRUTH,
          splits_path: Path | None = None, images: bool = True, out: Path | None = None,
          render_v2r: Path | None = None) -> tuple[pd.DataFrame, dict]:
    golden_dir = Path(golden_dir)
    negatives_path = Path(negatives_path) if negatives_path else golden_dir / "truth_negatives.csv"
    splits_path = Path(splits_path) if splits_path else golden_dir / "truth_splits.csv"
    out = Path(out) if out else golden_dir / "truth_manifest.csv"
    src = load_truth_sources(jwst_repo, outputs, golden_dir)
    negatives = _util.read_pinned(negatives_path, dtype={"candidate_id": str}) if negatives_path.exists() else None
    if negatives is None:
        print(f"NOTE: {negatives_path} absent — manifest built WITHOUT negatives (run sample_truth_negatives.py)", flush=True)
    man, notes = assemble(src, negatives)
    if images:
        man = write_images(man, stamps_dir, kits_dir)
        # the v2r render (render_v2.py) is merged from its pinned CSV when it exists
        render_v2r = Path(render_v2r) if render_v2r else golden_dir / "kits_truth_v2r" / "render_v2r.csv"
        man = attach_v2r(man, render_v2r)
    splits = _util.read_pinned(splits_path, dtype={"candidate_id": str}) if splits_path.exists() else None
    man = attach_half(man, splits)
    # the COWLS score cross-check (the catalogue's own S-code vs our decode of the string)
    ctl = src["controls"]
    if "score_x" in ctl.columns:
        bad = []
        for _, c in ctl.iterrows():
            S, _ = cowls_score(c.get("ranking"))
            code = str(c["score_x"])
            if code.startswith("S") and S != int(code[1:]):
                bad.append(f"{c['id']} {c['ranking']} -> S{S:02d} != {code}")
        notes["cowls_score_mismatch"] = bad
    # drift of the hard-coded cluster set against the fields.parquet regex review
    if src.get("fields") is not None:
        derived = derive_cluster_proposals(src["fields"])
        seen = set(man["proposal"].astype(str))
        notes["cluster_review_would_add"] = sorted((set(derived) & seen) - CLUSTER_PROPOSALS)
    sha = _util.pin(man, out)
    fl = fetch_list(man)
    fsha = _util.pin(fl, golden_dir / "truth_fetch_list.csv")
    notes["sha"] = sha; notes["fetch_sha"] = fsha; notes["n_fetch"] = len(fl)
    return man, notes


def report(man: pd.DataFrame, notes: dict, out: Path) -> None:
    n_pos = int(man["is_positive"].sum())
    print(f"truth_manifest: {len(man)} rows -> {out} (sha {notes['sha']})")
    print("per truth_class:")
    print(man["truth_class"].value_counts().reindex(TRUTH_CLASSES, fill_value=0).to_string())
    print(f"positives (is_positive): {n_pos}   anchors: {int(man['is_anchor'].sum())}   "
          f"stress: {int(man['is_stress'].sum())}   negatives: {int((man['truth_class'] == 'negative').sum())}")
    pos = man[man["is_positive"]]
    print("positives by layout x field_class:")
    print(pd.crosstab(pos["layout"], pos["field_class"]).to_string())
    print("cowls_band:", man.loc[man["truth_class"] == "cowls", "cowls_band"].value_counts().reindex(COWLS_BANDS, fill_value=0).to_dict())
    print("centre_is_deflector among positives:", int(pos["centre_is_deflector"].sum()))
    print(f"images: {int((man['image_path'] != '').sum())} with JPEG, {notes['n_fetch']} to fetch "
          f"-> truth_fetch_list.csv (sha {notes['fetch_sha']}); v2r renders: {int((man['image_path_v2r'] != '').sum())}")
    print("half:", man["half"].value_counts(dropna=False).to_dict())
    for n in notes.get("frame_lit_outside_L_known", []):
        print("NOTE:", n)
    for n in notes.get("dropped_dups", []):
        print("NOTE:", n)
    for n in notes.get("overlaps", []):
        print("NOTE:", n)
    print("anomalymatch classes:", notes.get("anomalymatch_classes", {}))
    for n in notes.get("warn", []):
        print("WARN:", n)
    for n in notes.get("cowls_score_mismatch", []):
        print("WARN cowls score:", n)
    if notes.get("cluster_review_would_add"):
        print("WARN: fields.parquet regex review would also call cluster:", notes["cluster_review_would_add"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jwst-repo", type=Path, default=_util.JWST_REPO)
    ap.add_argument("--outputs", type=Path, default=OUTPUTS)
    ap.add_argument("--golden-dir", type=Path, default=GOLDEN)
    ap.add_argument("--negatives", type=Path, default=None, help="default <golden-dir>/truth_negatives.csv")
    ap.add_argument("--splits", type=Path, default=None, help="default <golden-dir>/truth_splits.csv (fills `half`)")
    ap.add_argument("--stamps-dir", type=Path, default=STAMPS)
    ap.add_argument("--kits-dir", type=Path, default=KITS_TRUTH)
    ap.add_argument("--render-v2r", type=Path, default=None,
                    help="render_v2.py's pinned CSV (default <golden-dir>/kits_truth_v2r/render_v2r.csv when present)")
    ap.add_argument("--no-images", action="store_true", help="skip the JPEG step (image_path stays '')")
    ap.add_argument("--out", type=Path, default=None, help="default <golden-dir>/truth_manifest.csv")
    a = ap.parse_args(argv)
    out = a.out or a.golden_dir / "truth_manifest.csv"
    man, notes = build(a.jwst_repo, a.outputs, a.golden_dir, a.negatives, a.stamps_dir, a.kits_dir,
                       a.splits, images=not a.no_images, out=out, render_v2r=a.render_v2r)
    report(man, notes, out)
    if not a.no_images:
        # a truth JPEG must never be byte-identical to a served kit item (build_kit convention)
        from lensjudge.golden.build_kit import assert_no_collision
        item_dirs = sorted(p for p in (a.golden_dir / "kits").glob("*/items") if p.is_dir())
        if item_dirs and Path(a.kits_dir).is_dir():
            assert_no_collision(a.kits_dir, item_dirs)


if __name__ == "__main__":
    main()
