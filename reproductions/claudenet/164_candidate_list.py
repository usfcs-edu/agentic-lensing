#!/usr/bin/env python3
"""164_candidate_list.py — Phase 160 (DR9 full sweep): the final FDR-controlled,
cross-matched candidate table + Lupton vetting pages + the sweep summary /
consistency JSON (runs LOCALLY, CPU).

Joins the three sweep products on row_id — stage-2 scores (162), crossmatch
status (163), group-conformal p/q/selection (165) — into
data/v2/sweep/candidates_v2.parquet: row_id, RA, DEC, footprint, brick,
p_stage1, p_final (the score), q_group (the per-candidate FULL-m conformal q;
q_pooled and the *_anticons diagnostics when present), status (crossmatch:
NEW/KNOWN_LOCAL/KNOWN_REMOTE), nearest_sep_arcsec/nearest_catalog/nearest_name,
the per-alpha selected flags, rank (p_final desc over ALL survivors) and
rank_new (rank among status==NEW, see the seen-row exclusion below; 0 = not
ranked). A trimmed CSV (rows selected at any PRIMARY alpha column or in the
top --n-vet NEW) is written for humans, headed by the guarantee note.

HEADLINE VALIDITY: the headline selection column sel_group_a<alpha> is 165's
PRIMARY full-m BH output (per-group FDR <= alpha; non-survivors censored at
p=1). The survivors-only-m *_anticons columns carry NO FDR guarantee and are
EXCLUDED from the headline, the printed summary and sweep_summary.json's
'headline' block (they remain in the table/selection counts, tagged
diagnostic). If the conformal table lacks the full-m column this script
ABORTS — it never manufactures a selection from q_group. 165's guarantee
note is propagated into the summary, the printed output, the CSV header and
the candidates parquet metadata.

SEEN-ROW FLAGS (the pipeline must not re-discover its own inputs): every
candidate gets boolean columns
  is_train_row                row_id in training_split_staged train/val
  is_mined_row                row_id in the 120/121 mined hard/random sets
                              (rows the retrained v2 members TRAINED on)
  is_negeval_calibration_row  row_id in the NegEval-1M manifest (threshold +
                              conformal calibration rows)
  is_v1_train_brick           (footprint, brick) of a v1 negative brick
                              (informational; brick-level, not row-level)
matched on the UNQUALIFIED row_id suffix (sweep row_ids are
footprint-qualified '<f>_<BRICKID>_<OBJID>' while the v1/110 tables use
'<BRICKID>_<OBJID>'; footprint is required to agree when both sides carry it)
AND on sky position within --seen-radius (1") as the fallback (catches the
DESI-* curated-positive ids). The NEW ranking and the vet list EXCLUDE rows
flagged by any of the three row-level flags by default (--include-seen
overrides); all counts go to sweep_summary.json.

Vetting pages (114's Lupton renderer REUSED by import, not copied): the top
--n-vet (default 1500) NEW candidates -> data/v2/sweep/vet_rowids.csv (feed to
111b_dump_rows.py on Perlmutter, rsync the npz back), then rerun with
--vet-npz present -> rank-labelled contact-sheet grids under
data/v2/sweep/vet/.

sweep_summary.json carries the numbers the plan demands:
  * counts: survivors / NEW / KNOWN_LOCAL / KNOWN_REMOTE, every 165 selection
    column's size + how many selected are NEW, per footprint, each tagged with
    its guarantee; seen-row flag counts;
  * recall of known lenses: 163's survivor-level accounting passed through,
    PLUS the catalog-entry recovery RECOMPUTED on the --alpha selected set
    (same radius, 163's catalog loaders imported, --extra-catalog passed
    through, numerator restricted to 163's exported in-coverage entries so
    numerator and denominator share one population);
  * THE consistency check: realized stage-1 pass rate (PRE-budget
    realized_pass_rate from --stage1-summary when present, else
    n_survivors/n_swept) vs the nominal stage-1 target FPR and vs the
    EVT-predicted FPR (the GPD tail F(t) = zeta*(1+xi*(t-u)/sigma)^(-1/xi)
    inverted from 113's fitted u/xi/sigma/zeta in --evt-json). The
    EVT-comparable quantity subtracts the KNOWN_LOCAL count
    (realized_rate_minus_known); in members mode the EVT prediction is the
    union bound sum_m F_m(thr_m) over the per-member thresholds recorded by
    162 (evt keys matched after removeprefix('member_')). The nominal
    union-bound comparison is ONE-SIDED (an upper bound: only realized >
    factor x nominal flags INCONSISTENT); the EVT comparison is two-sided
    (--consistency-factor). Override flags default to None so the
    --stage1-summary sidecar is never silently clobbered; 1e-4 is used only
    when BOTH the summary and the flag are absent (logged).

    python 164_candidate_list.py                 # table + summary + vet csv
    python 164_candidate_list.py --vet-npz data/v2/sweep/vet_topnew.npz  # +pages
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

V2 = C.DATA / "v2"
SWEEP = V2 / "sweep"
# alias keys accepted in --stage1-summary (the 161/162 sidecar)
S1_ALIASES = {"n_swept": ("n_swept", "n_scored", "n_total", "n_rows"),
              "n_survivors": ("n_survivors", "n_selected", "n_pass"),
              "n_pass_prebudget": ("n_pass_prebudget",),
              "realized_pass_rate": ("realized_pass_rate",),
              "stage1_thr": ("stage1_thr", "thr", "threshold"),
              "stage1_scorer": ("stage1_scorer", "scorer"),
              "stage1_fpr": ("stage1_fpr", "fpr", "target_fpr"),
              "per_scorer_fpr": ("per_member_fpr", "per_scorer_fpr"),
              "thresholds": ("thresholds",),
              "threshold_space": ("threshold_space",)}
# default per-scorer target FPR, used ONLY when both the summary and the
# override flag are absent (logged when it kicks in)
FALLBACK_FPR = 1e-4


# ===== stage-1 consistency (realized rate vs nominal vs EVT) =====================

def evt_fpr_at(thr: float, info: dict) -> float:
    """Invert 113's fitted GPD tail: P(X > thr) = zeta*(1+xi*(thr-u)/sigma)^(-1/xi)
    (log form as xi->0). NaN when thr <= u (below the POT threshold) or the
    fit is degenerate."""
    vals = [info.get(k) for k in ("u", "xi", "sigma", "zeta")]
    if any(v is None or not np.isfinite(v) for v in vals) or vals[2] <= 0:
        return float("nan")
    u, xi, sigma, zeta = (float(v) for v in vals)
    if thr <= u:
        return float("nan")
    z = (thr - u) / sigma
    if abs(xi) < 1e-12:
        return zeta * float(np.exp(-z))
    base = 1.0 + xi * z
    return 0.0 if base <= 0 else zeta * float(base ** (-1.0 / xi))


def stage1_info(args, n_stage2_rows: int) -> dict:
    """--stage1-summary json (alias-tolerant). CLI flags override ONLY when
    explicitly given (all default None — the sidecar is never silently
    clobbered); 1e-4 is the logged last-resort fpr fallback."""
    raw = {}
    p = Path(args.stage1_summary)
    if p.exists():
        raw = json.loads(p.read_text())
        print(f"[stage1] loaded {p}")
    else:
        print(f"[stage1] NOTE: {p} missing -> stage-1 metadata from flags only")
    info = {}
    for key, aliases in S1_ALIASES.items():
        info[key] = next((raw[a] for a in aliases if a in raw), None)
    for key, val in (("n_swept", args.n_swept), ("stage1_thr", args.stage1_thr),
                     ("stage1_scorer", args.stage1_scorer),
                     ("stage1_fpr", args.stage1_fpr)):
        if val is not None:                # None default == no override (B)
            info[key] = val
    if info["stage1_fpr"] is None:
        info["stage1_fpr"] = FALLBACK_FPR
        print(f"[stage1] NOTE: no stage1_fpr in summary or flags -> falling "
              f"back to {FALLBACK_FPR:g} (per-scorer default)")
    if info["per_scorer_fpr"] is None:     # union-bound fpr != per-scorer fpr
        info["per_scorer_fpr"] = (args.stage1_fpr if args.stage1_fpr is not None
                                  else FALLBACK_FPR)
    if info["n_survivors"] is None:
        info["n_survivors"] = n_stage2_rows
        info["n_survivors_note"] = ("fallback: post-merge stage-2 row count "
                                    "(post-apply_fits; pass --stage1-summary "
                                    "for the true pass counts)")
    # threshold fallback: the operating-points CSV row at the PER-SCORER fpr
    # (the union-bound stage1_fpr is a different quantity)
    if info["stage1_thr"] is None and info["stage1_scorer"] \
            and info["per_scorer_fpr"]:
        op = Path(args.operating_points)
        if op.exists():
            df = pd.read_csv(op)
            hit = df[(df.scorer == info["stage1_scorer"])
                     & np.isclose(df.fpr, float(info["per_scorer_fpr"]))]
            if len(hit):
                info["stage1_thr"] = float(hit.iloc[0]["thr"])
                print(f"[stage1] thr from {op.name}: {info['stage1_thr']:.6f} "
                      f"({info['stage1_scorer']} @ {info['per_scorer_fpr']:g})")
    return info


def find_evt_info(scorer: str, evt_jsons: str) -> dict | None:
    """First 'evt' block containing the scorer among the comma-listed jsons
    (113's thresholds_ci.json, 145's *_verdict.json share the schema). 145
    keys combiners 'v2:average' while score COLUMNS are 'v2_average', and 162
    records member thresholds under 'member_<scorer>' while 113 keys its evt
    blocks bare — the lookup tolerates colon/underscore and the member_
    prefix either way."""
    cands = list(dict.fromkeys((scorer, scorer.replace(":", "_"),
                                scorer.removeprefix("member_"),
                                f"member_{scorer}")))
    for f in (s for s in evt_jsons.split(",") if s):
        p = Path(f)
        if not p.exists():
            continue
        evt = json.loads(p.read_text()).get("evt", {})
        key = next((k for k in evt
                    if k in cands or k.replace(":", "_") in cands), None)
        if key is not None:
            print(f"[stage1] EVT params for {scorer!r} from {p.name} (key {key!r})")
            return evt[key]
    return None


def consistency_block(args, info: dict, n_known_local: int) -> dict:
    """realized stage-1 pass rate vs nominal target FPR vs EVT-predicted FPR —
    THE plan's consistency check. Apples-to-apples per the review: the
    PRE-budget pass rate is used when 162 recorded it; the EVT-comparable
    quantity subtracts the KNOWN_LOCAL crossmatches; the nominal union-bound
    comparison is ONE-SIDED (upper bound); members mode uses the union-bound
    EVT prediction sum_m F_m(thr_m)."""
    fac = args.consistency_factor
    blk = {**info, "n_known_local": int(n_known_local),
           "note": ("sweep keeps known lenses + v1-negative bricks, so the "
                    "realized rate sits above a pure FPR; "
                    "realized_rate_minus_known subtracts KNOWN_LOCAL for the "
                    "EVT comparison (two-sided factor "
                    f"{fac:g}); the nominal union bound is an UPPER bound -> "
                    f"one-sided (flag only realized > {fac:g} x nominal)")}
    if not info["n_swept"]:
        blk["verdict"] = "SKIPPED (n_swept unknown — pass --n-swept or --stage1-summary)"
        print(f"[stage1] consistency check {blk['verdict']}")
        return blk
    n_pass_pre = info.get("n_pass_prebudget") or info["n_survivors"]
    if info.get("realized_pass_rate") is not None:
        realized = float(info["realized_pass_rate"])
        blk["realized_source"] = "stage1_summary realized_pass_rate (pre-budget)"
    else:
        realized = n_pass_pre / info["n_swept"]
        blk["realized_source"] = "n_pass/n_swept (no realized_pass_rate in summary)"
    blk["realized_survivor_rate"] = realized
    realized_mk = max(n_pass_pre - n_known_local, 0) / info["n_swept"]
    blk["realized_rate_minus_known"] = realized_mk
    if info["stage1_fpr"]:
        blk["realized_over_nominal"] = realized / float(info["stage1_fpr"])

    # EVT prediction: single scorer (student) or per-member union (members)
    evt_pred, evt_kind = float("nan"), None
    if info["stage1_scorer"] and info["stage1_thr"] is not None:
        evt = find_evt_info(info["stage1_scorer"], args.evt_json)
        if evt is not None:
            blk["evt_params"] = {k: evt.get(k) for k in ("u", "xi", "sigma", "zeta")}
            evt_pred = evt_fpr_at(float(info["stage1_thr"]), evt)
            evt_kind = "single-scorer"
    elif isinstance(info.get("thresholds"), dict) and info["thresholds"]:
        contribs, missing = {}, []
        for col, thr in info["thresholds"].items():
            evt = find_evt_info(str(col).removeprefix("member_"), args.evt_json)
            f_m = evt_fpr_at(float(thr), evt) if evt is not None else float("nan")
            if np.isfinite(f_m):
                contribs[col] = f_m
            else:
                missing.append(col)
        blk["evt_member_contribs"] = contribs
        if missing:
            blk["evt_members_missing"] = missing
            print(f"[stage1] EVT union prediction incomplete (no usable params "
                  f"for {missing}) -> EVT leg skipped")
        elif contribs:
            evt_pred = float(sum(contribs.values()))
            evt_kind = f"union bound over {len(contribs)} members"
    blk["evt_predicted_fpr"] = evt_pred
    blk["evt_prediction_kind"] = evt_kind

    if np.isfinite(evt_pred) and evt_pred > 0:
        blk["realized_minus_known_over_evt"] = realized_mk / evt_pred
        blk["consistent"] = bool(1.0 / fac <= blk["realized_minus_known_over_evt"]
                                 <= fac)
        ref = f"EVT ({evt_kind}; known-lens-subtracted realized)"
    elif info["stage1_fpr"]:
        # union-bound nominal is an UPPER bound: one-sided check only
        blk["consistent"] = bool(realized <= fac * float(info["stage1_fpr"]))
        ref = "nominal upper bound (one-sided; EVT unavailable)"
    else:
        blk["verdict"] = "SKIPPED (no nominal fpr and no EVT prediction)"
        print(f"[stage1] consistency check {blk['verdict']}")
        return blk
    blk["verdict"] = ("CONSISTENT" if blk["consistent"] else "INCONSISTENT") + \
        f" vs {ref} within x{fac:g}"
    print(f"[stage1] realized {realized:.3e} (minus known {realized_mk:.3e}); "
          f"nominal {info['stage1_fpr']}; EVT-predicted {evt_pred:.3e} "
          f"-> {blk['verdict']}")
    return blk


# ===== seen-row flags (the pipeline must not re-discover its own inputs) ========

def _unqualified(rid: pd.Series) -> pd.Series:
    """Strip the 160 footprint qualifier: '<f>_<BRICKID>_<OBJID>' ->
    '<BRICKID>_<OBJID>' (v1/110 row_id scheme); other shapes pass through."""
    return rid.str.replace(r"^[a-z]_(?=\d+_\d+$)", "", regex=True)


def seen_flags(cand: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    """Boolean flag columns for rows the pipeline has already consumed:
    is_train_row / is_mined_row / is_negeval_calibration_row (row-level; gate
    the NEW ranking + vet list unless --include-seen) + is_v1_train_brick
    (brick-level, informational). Matching = unqualified row_id suffix
    (+ footprint when both sides carry it) OR sky position < --seen-radius."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    suffix = _unqualified(cand.row_id.astype(str))
    has_pos = {"RA", "DEC"} <= set(cand.columns)
    sky = (SkyCoord(ra=cand.RA.to_numpy(np.float64) * u.deg,
                    dec=cand.DEC.to_numpy(np.float64) * u.deg) if has_pos else None)
    has_foot = "footprint" in cand.columns
    flags = pd.DataFrame(index=cand.index)
    info: dict = {"radius_arcsec": args.seen_radius, "sources": {}}

    def match(ref: pd.DataFrame, what: str) -> np.ndarray:
        if has_foot and "footprint" in ref.columns:
            ck = pd.MultiIndex.from_arrays([cand.footprint.astype(str), suffix])
            rk = pd.MultiIndex.from_arrays([ref.footprint.astype(str),
                                            ref.row_id.astype(str)])
            m = np.asarray(ck.isin(rk))
        else:
            m = suffix.isin(set(ref.row_id.astype(str))).to_numpy()
        n_id = int(m.sum())
        n_pos = 0
        if sky is not None and {"RA", "DEC"} <= set(ref.columns) and len(ref):
            rsky = SkyCoord(ra=ref.RA.to_numpy(np.float64) * u.deg,
                            dec=ref.DEC.to_numpy(np.float64) * u.deg)
            _, sep, _ = sky.match_to_catalog_sky(rsky)
            mp = sep.to(u.arcsec).value < args.seen_radius
            n_pos = int((mp & ~m).sum())
            m = m | mp
        print(f"[seen] {what:34s} {int(m.sum()):6,} flagged "
              f"({n_id:,} by row_id suffix, +{n_pos:,} by position "
              f"<{args.seen_radius:g}\")")
        return m

    # 1. training rows (train/val; positives are DESI-* ids -> position match)
    p = Path(args.training_split)
    if p.exists():
        sp = pd.read_parquet(p)
        sp = sp[sp.split.isin(("train", "val"))]
        flags["is_train_row"] = match(sp, f"train/val rows ({len(sp):,})")
        info["sources"]["is_train_row"] = str(p)
    else:
        print(f"[seen] WARNING: {p} missing -> is_train_row all False")
        flags["is_train_row"] = False
        info["sources"]["is_train_row"] = None
    # 2. mined hard/random rows (the v2 members trained on these)
    mined, used = [], []
    for f in (s for s in args.mined_rowids.split(",") if s):
        mp = Path(f)
        if mp.exists():
            mined.append(pd.read_parquet(mp))
            used.append(str(mp))
        else:
            print(f"[seen] WARNING: {mp} missing -> skipped for is_mined_row")
    if mined:
        flags["is_mined_row"] = match(pd.concat(mined, ignore_index=True),
                                      f"mined hard/random rows")
    else:
        flags["is_mined_row"] = False
    info["sources"]["is_mined_row"] = used or None
    # 3. NegEval calibration rows (stage-1 thresholds + 165 conformal cal)
    p = Path(args.negeval_manifest)
    if p.exists():
        ne = pd.read_parquet(p)
        flags["is_negeval_calibration_row"] = match(
            ne, f"NegEval calibration rows ({len(ne):,})")
        info["sources"]["is_negeval_calibration_row"] = str(p)
    else:
        print(f"[seen] WARNING: {p} missing -> is_negeval_calibration_row all False")
        flags["is_negeval_calibration_row"] = False
        info["sources"]["is_negeval_calibration_row"] = None
    # 4. v1 negative bricks (brick-level, informational only)
    p = Path(args.negatives_extra)
    if p.exists() and has_foot and "brick" in cand.columns:
        ne = pd.read_parquet(p, columns=["footprint", "brick"]).drop_duplicates()
        ck = pd.MultiIndex.from_arrays([cand.footprint.astype(str),
                                        cand.brick.astype(str)])
        rk = pd.MultiIndex.from_arrays([ne.footprint.astype(str),
                                        ne.brick.astype(str)])
        flags["is_v1_train_brick"] = np.asarray(ck.isin(rk))
        info["sources"]["is_v1_train_brick"] = str(p)
        print(f"[seen] {'v1 negative bricks (brick-level)':34s} "
              f"{int(flags.is_v1_train_brick.sum()):6,} flagged (informational)")
    else:
        flags["is_v1_train_brick"] = False
        info["sources"]["is_v1_train_brick"] = str(p) if p.exists() else None
    info["counts"] = {c: int(flags[c].sum()) for c in flags.columns}
    return flags, info


# ===== recall on the selected set ================================================

def selected_recall(sel: pd.DataFrame, radius: float, recall_json: dict | None,
                    extra_csvs=()):
    """Catalog-entry recovery into the FINAL selected set, with the numerator
    RESTRICTED to 163's exported in-coverage entries (in_coverage_idx) so
    numerator and denominator share one population; --extra-catalog CSVs are
    passed through to the same loaders 163 used."""
    from astropy import units as u
    XM = C._load("cn_164_xmatch", C.ROOT / "163_crossmatch_known.py")
    cats = XM.load_catalogs(list(extra_csvs))
    out = {}
    sel_sky = XM.sky_of(sel) if len(sel) else None
    prior = (recall_json or {}).get("catalogs", {})
    for tag, cat in cats.items():
        rec = np.zeros(len(cat), bool)
        if sel_sky is not None:
            _, sep, _ = XM.sky_of(cat).match_to_catalog_sky(sel_sky)
            rec = sep.to(u.arcsec).value < radius
        ent = {"n_entries": int(len(cat)),
               "n_recovered_selected": int(rec.sum())}
        pri = prior.get(tag, {})
        if "in_coverage_idx" in pri and pri.get("n_entries") == len(cat):
            inc = np.zeros(len(cat), bool)
            inc[np.asarray(pri["in_coverage_idx"], dtype=int)] = True
            ent["n_in_coverage"] = int(inc.sum())
            ent["n_recovered_selected_in_coverage"] = int((rec & inc).sum())
            ent["recall_selected_in_coverage"] = (
                float((rec & inc).sum() / inc.sum()) if inc.any() else float("nan"))
        elif "n_in_coverage" in pri:
            ent["n_in_coverage"] = pri["n_in_coverage"]
            ent["note"] = ("163 json lacks a matching in_coverage_idx (rerun "
                           "163): in-coverage selected recall not computed — "
                           "an all-entries numerator over an in-coverage "
                           "denominator would mismatch populations")
        out[tag] = ent
    return out


# ===== main ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage2-scores", default=str(SWEEP / "stage2_scores.parquet"))
    ap.add_argument("--score-col", default="p_final")
    ap.add_argument("--crossmatch", default=str(SWEEP / "crossmatch.parquet"),
                    help="163 output")
    ap.add_argument("--recall-json", default=str(SWEEP / "crossmatch_recall.json"))
    ap.add_argument("--conformal", default=str(SWEEP / "conformal.parquet"),
                    help="165 output")
    ap.add_argument("--conformal-summary", default=str(SWEEP / "conformal_summary.json"))
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="headline FDR target (165's PRIMARY full-m "
                         "sel_group_a<alpha>; *_anticons never headlines)")
    ap.add_argument("--radius", type=float, default=5.0,
                    help="known-lens match radius for the selected-set recall")
    ap.add_argument("--extra-catalog", action="append", default=[],
                    help="extra known-lens CSV passed through to the "
                         "selected-set recall (use the same ones 163 got)")
    ap.add_argument("--stage1-summary", default=str(SWEEP / "stage1_summary.json"),
                    help="161/162 sidecar json (n_swept/n_survivors/thr/scorer/fpr)")
    ap.add_argument("--n-swept", type=int, default=None,
                    help="override: total rows scored at stage 1")
    ap.add_argument("--stage1-thr", type=float, default=None)
    ap.add_argument("--stage1-scorer", default=None)
    ap.add_argument("--stage1-fpr", type=float, default=None,
                    help="override the summary's nominal stage-1 fpr (default "
                         "None: the sidecar value wins; 1e-4 only when both "
                         "are absent)")
    # seen-row flag sources (is_train_row / is_mined_row / is_negeval_... )
    ap.add_argument("--training-split",
                    default=str(C.DATA / "training_split_staged.parquet"))
    ap.add_argument("--mined-rowids",
                    default=f"{V2 / 'mined_hard_rowids.parquet'},"
                            f"{V2 / 'mined_random_rowids.parquet'}",
                    help="comma list of 120/121 mined-row parquets")
    ap.add_argument("--negeval-manifest",
                    default=str(V2 / "negeval_manifest.parquet"))
    ap.add_argument("--negatives-extra",
                    default=str(C.DATA / "negatives_extra.parquet"),
                    help="v1 negative bricks (is_v1_train_brick, informational)")
    ap.add_argument("--seen-radius", type=float, default=1.0,
                    help="position-match radius (arcsec) for the seen-row flags")
    ap.add_argument("--include-seen", action="store_true",
                    help="keep seen-flagged rows in the NEW ranking + vet list "
                         "(default: excluded)")
    ap.add_argument("--operating-points", default=str(V2 / "operating_points_v2.csv"),
                    help="stage-1 threshold fallback lookup (scorer @ fpr)")
    ap.add_argument("--evt-json",
                    default=f"{V2 / 'thresholds_ci.json'},{V2 / 'ensemble_v2_verdict.json'}",
                    help="comma-list of jsons with an 'evt' block per scorer")
    ap.add_argument("--consistency-factor", type=float, default=3.0)
    ap.add_argument("--n-vet", type=int, default=1500,
                    help="how many top NEW candidates get vetting pages")
    ap.add_argument("--vet-npz", default=str(SWEEP / "vet_topnew.npz"),
                    help="111b npz of the vet cutouts (pages skipped if absent)")
    ap.add_argument("--grid", type=int, default=10, help="contact-sheet grid edge")
    ap.add_argument("--out-dir", default=str(SWEEP))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- join the three products on row_id ---------------------------------------
    cand = pd.read_parquet(args.stage2_scores)
    assert args.score_col in cand.columns, \
        f"{args.stage2_scores}: missing {args.score_col!r}"
    cand = cand.copy()
    cand["row_id"] = cand["row_id"].astype(str)
    n_surv = len(cand)
    for path, what in ((args.crossmatch, "163 crossmatch"),
                       (args.conformal, "165 conformal")):
        df = pd.read_parquet(path)
        df = df.copy()
        df["row_id"] = df["row_id"].astype(str)
        df = df.drop(columns=[c for c in df.columns
                              if c != "row_id" and c in cand.columns])
        n0 = len(cand)
        cand = cand.merge(df, on="row_id", how="inner", validate="one_to_one")
        if len(cand) < n0:
            print(f"[164] WARNING: {n0 - len(cand):,} rows dropped joining {what} "
                  f"({path}) — stale inputs?")
    for col in ("status", "q_group"):
        assert col in cand.columns, f"joined table lacks {col!r} (rerun 163/165)"
    print(f"[164] {len(cand):,} survivors joined across stage2 + crossmatch + conformal")

    # -- guarantee note (165's summary is the source of truth) --------------------
    conf_summary = None
    cs = Path(args.conformal_summary)
    if cs.exists():
        conf_summary = json.loads(cs.read_text())
    guarantee_note = (conf_summary or {}).get("guarantee_note") or (
        "sel_group_a* = per-group conformal BH with the FULL per-footprint "
        "sweep total as m (per-group FDR <= alpha, marginal); *_anticons "
        "(m = n_survivors) columns have NO FDR guarantee — diagnostic only; "
        "pooled rows are descriptive. [165 summary json missing — default note]")

    # -- rank + seen-row flags + headline selection --------------------------------
    cand = cand.sort_values(args.score_col, ascending=False,
                            kind="mergesort").reset_index(drop=True)
    for c in ("rank", "rank_new"):          # collision guard vs joined inputs
        if c in cand.columns:
            print(f"[164] WARNING: input tables already carry {c!r} -> dropped "
                  f"(164 recomputes it)")
            cand = cand.drop(columns=c)
    cand.insert(1, "rank", np.arange(1, len(cand) + 1))

    flags, seen_info = seen_flags(cand, args)
    for c in flags.columns:
        cand[c] = flags[c].to_numpy()
    seen = (cand.is_train_row | cand.is_mined_row
            | cand.is_negeval_calibration_row).to_numpy()
    seen_info["n_seen_rowlevel"] = int(seen.sum())
    seen_info["include_seen"] = bool(args.include_seen)

    is_new = (cand.status == "NEW").to_numpy()
    eligible_new = is_new if args.include_seen else (is_new & ~seen)
    if not args.include_seen:
        n_excl = int((is_new & seen).sum())
        seen_info["n_new_excluded_as_seen"] = n_excl
        print(f"[seen] {n_excl:,} NEW rows excluded from the ranking/vet list "
              f"as already-seen (--include-seen overrides)")
    rank_new = np.zeros(len(cand), dtype=np.int64)
    rank_new[eligible_new] = np.arange(1, int(eligible_new.sum()) + 1)
    cand.insert(2, "rank_new", rank_new)

    sel_cols = sorted(c for c in cand.columns if c.startswith("sel_"))
    primary_cols = [c for c in sel_cols if "anticons" not in c]
    anticons_cols = [c for c in sel_cols if "anticons" in c]
    head_col = f"sel_group_a{args.alpha:g}"
    assert "anticons" not in head_col
    if head_col not in cand.columns:
        raise SystemExit(
            f"[164] FATAL: {head_col!r} (165's PRIMARY full-m selection) is not "
            f"in the conformal table — rerun 165 (it now requires per-footprint "
            f"sweep totals and emits the full-m columns by default). Refusing "
            f"to manufacture a selection from q_group"
            + (f"; only anti-conservative {anticons_cols} present (NO FDR "
               f"guarantee)" if anticons_cols else "") + ".")
    head_sel = cand[head_col].to_numpy(bool)

    selections = {"guarantee_note": guarantee_note}
    for col in sel_cols:
        s = cand[col].to_numpy(bool)
        selections[col] = {
            "n_selected": int(s.sum()), "n_new": int((s & is_new).sum()),
            "guarantee": ("NONE — survivors-only m, anti-conservative "
                          "(diagnostic)" if "anticons" in col else
                          ("descriptive (pooled calibration)" if "pooled" in col
                           else "per-group FDR <= alpha (full-m conformal BH)")),
            "by_footprint": {g: int(s[(cand.footprint == g).to_numpy()].sum())
                             for g in sorted(cand.footprint.unique())}
            if "footprint" in cand.columns else {}}
    print(f"[164] selections (headline -> {head_col}; *_anticons diagnostics "
          f"OMITTED here — no FDR guarantee):")
    for col in primary_cols:
        e = selections[col]
        print(f"[164]   {col:24s} n={e['n_selected']:7,}  new={e['n_new']:7,}  "
              + " ".join(f"{g}={n:,}" for g, n in e["by_footprint"].items()))
    if anticons_cols:
        print(f"[164]   ({len(anticons_cols)} *_anticons diagnostic columns in "
              f"the table/json only)")
    print(f"[164] guarantee: {guarantee_note}")

    # -- write the table (guarantee note in parquet metadata + CSV header) --------
    out_pq = out_dir / "candidates_v2.parquet"
    M165 = C._load("cn_164_conf165", C.ROOT / "165_group_conformal.py")
    M165.parquet_with_note(cand, out_pq, guarantee_note)
    human = cand[np.column_stack([cand[c].to_numpy(bool)
                                  for c in primary_cols]).any(axis=1)
                 | ((rank_new >= 1) & (rank_new <= args.n_vet))]
    out_csv = out_dir / "candidates_v2.csv"
    with open(out_csv, "w") as fh:
        fh.write(f"# guarantee: {guarantee_note}\n")
        human.to_csv(fh, index=False)
    print(f"[164] wrote {out_pq} ({len(cand):,} rows) + {out_csv} "
          f"({len(human):,} selected/vet rows; read with comment='#')")

    # -- vetting: rowids csv always; Lupton pages once the 111b npz is back -------
    # (seen-flagged rows are already excluded from rank_new unless --include-seen)
    new_top = cand[eligible_new].head(args.n_vet)
    vet_csv = out_dir / "vet_rowids.csv"
    with open(vet_csv, "w") as fh:
        fh.write(f"# top NEW candidates; seen-flagged rows "
                 f"{'INCLUDED (--include-seen)' if args.include_seen else 'excluded'}\n")
        new_top[["row_id", "RA", "DEC", args.score_col, "rank", "rank_new"]].to_csv(
            fh, index=False)
    print(f"[164] wrote {vet_csv} ({len(new_top):,} top NEW candidates; feed to "
          f"111b_dump_rows.py --row-ids on Perlmutter; read with comment='#')")
    vet_pages = 0
    npz = Path(args.vet_npz)
    if npz.exists():
        AU = C._load("cn_164_audit", C.ROOT / "114_purity_audit.py")
        vet_dir = out_dir / "vet"
        vet_dir.mkdir(parents=True, exist_ok=True)
        vdf = pd.DataFrame({"row_id": new_top.row_id.to_numpy(),
                            "rank": new_top.rank_new.to_numpy(),
                            "score": new_top[args.score_col].to_numpy(),
                            "match5": False})
        AU.render_contact_sheets(
            vdf, npz, vet_dir, args.grid,
            title=f"DR9 sweep — top-{len(vdf)} NEW candidates by {args.score_col}",
            prefix="vet")
        vet_pages = (len(vdf) + args.grid ** 2 - 1) // args.grid ** 2
    else:
        print(f"[164] {npz} not found -> vetting pages pending (run 111b with "
              f"{vet_csv.name}, rsync the npz back, rerun)")

    # -- recall of known lenses (survivor level passthrough + selected level) -----
    recall_json = None
    rp = Path(args.recall_json)
    if rp.exists():
        recall_json = json.loads(rp.read_text())
    else:
        print(f"[164] WARNING: {rp} missing -> survivor-level recall unavailable")
    sel_recall = selected_recall(cand[head_sel], args.radius, recall_json,
                                 args.extra_catalog)
    print(f"[164] known-lens recovery in the {head_col} selected set:")
    for tag, e in sel_recall.items():
        line = f"[164]   {tag:14s} {e['n_recovered_selected']:6,} recovered"
        if "recall_selected_in_coverage" in e:
            line += (f" (in-coverage {e['n_recovered_selected_in_coverage']:,}"
                     f" / {e['n_in_coverage']:,} = "
                     f"{e['recall_selected_in_coverage']:.3f})")
        print(line)

    # -- stage-1 consistency + summary json ---------------------------------------
    status_counts = {k: int(v) for k, v in cand.status.value_counts().items()}
    s1 = stage1_info(args, n_surv)
    consistency = consistency_block(args, s1,
                                    status_counts.get("KNOWN_LOCAL", 0))

    summary = {
        "inputs": {"stage2_scores": str(args.stage2_scores),
                   "crossmatch": str(args.crossmatch),
                   "conformal": str(args.conformal),
                   "score_col": args.score_col, "alpha": args.alpha,
                   "headline_sel_col": head_col},
        "counts": {
            "n_survivors": int(len(cand)),
            "status": status_counts,
            "n_remote_queried": int(cand.remote_queried.sum())
            if "remote_queried" in cand.columns else 0,
            "by_footprint": {g: int((cand.footprint == g).sum())
                             for g in sorted(cand.footprint.unique())}
            if "footprint" in cand.columns else {},
            "seen_rows": seen_info,
            "selections": selections,
            # headline = the PRIMARY full-m column ONLY (never *_anticons)
            "headline": {"sel_col": head_col,
                         "n_selected": int(head_sel.sum()),
                         "n_selected_new": int((head_sel & is_new).sum()),
                         "n_selected_new_unseen":
                             int((head_sel & is_new & ~seen).sum()),
                         "guarantee_note": guarantee_note}},
        "recall_known_lenses": {
            "survivor_level": recall_json,
            "selected_level": sel_recall,
            "selected_level_sel_col": head_col},
        "stage1_consistency": consistency,
        "conformal_summary": conf_summary,
        "vetting": {"n_vet": int(len(new_top)), "vet_rowids_csv": str(vet_csv),
                    "npz": str(npz), "pages_rendered": vet_pages,
                    "seen_excluded": not args.include_seen},
    }
    out_json = out_dir / "sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[164] headline ({head_col}, per-group FDR <= {args.alpha:g} by "
          f"full-m conformal BH): {int(head_sel.sum()):,} selected "
          f"({int((head_sel & is_new).sum()):,} NEW, "
          f"{int((head_sel & is_new & ~seen).sum()):,} NEW-and-unseen) of "
          f"{len(cand):,} survivors")
    print(f"[164] wrote {out_json} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
