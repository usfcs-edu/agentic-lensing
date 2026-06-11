#!/usr/bin/env python3
"""165_group_conformal.py — Phase 160 (DR9 full sweep): GROUP-conformal
(north/south) FDR-controlled selection of the stage-2 sweep scores against the
NegEval-1M calibration negatives (runs LOCALLY, CPU).

Generalizes 50_conformal_selection.py (Jin & Candes conformal p-values + BH),
whose conformal_pvalues/bh_select are REUSED by import, not reimplemented.
Under north/south domain shift a POOLED calibration set gives the shifted
group anti-conservative p-values, so BH runs PER FOOTPRINT GROUP with that
group's own calibration negatives:

    p_j^g = (1 + #{cal_g >= s_j}) / (n_cal_g + 1)      (50's construction)
    BH within group g at alpha  ->  per-group FDR <= alpha

The guarantee is per-group and marginal (exchangeability of that group's
calibration/test negatives); the pooled-calibration row and the union counts
are reported for comparison, not as guarantees. BH q-values (adjusted
p-values; {q <= alpha} == bh_select at every alpha, asserted) are written per
row so 164 can carry a per-candidate "conformal q".

Calibration scores = NegEval-1M calibrated combiner scores: 145's
data/v2/ensemble_v2_pool_combined.parquet (--cal-col auto prefers v2_average)
— the column MUST be the same calibrate+combine path 162 applied for p_final.
When that file is absent the run ABORTS (run 145, or 145 --apply-roster on
the NegEval pool); the 113 v1-average fallback
(scores_negeval_pool_combined.parquet::average) is a DIFFERENT score function
whenever the v2 roster differs from the v1 five, so it is only allowed behind
--allow-v1-cal and only after the persisted fits' meta.json proves the v2
roster IS the v1 five with flagship == average. Calibration footprints join
from the negeval manifest. Test scores = the stage-2 sweep parquet's
--test-col (p_final).

HEADLINE VALIDITY (the m question): the test set is the stage-1 SURVIVOR
subset of the full sweep. Running BH with m = n_survivors conditions the
test set on a stage-1 selection computed from the SAME members as p_final,
which makes null survivors' p-values sub-uniform — NO FDR guarantee
(--synthetic-check demonstrates the violation numerically). The PRIMARY
columns (q_group / sel_group_a*) therefore use the FULL per-footprint sweep
totals as m: non-survivors are censored, never-selectable tests at p=1;
censoring can only RAISE a p-value, and the unconditional conformal p-value
of any null sweep row vs the NegEval calibration is exchangeable (both draw
from the same parent population), so per-group BH at alpha keeps FDR <=
alpha marginally. The totals are REQUIRED: --sweep-totals "north=N,south=M"
overrides; otherwise they are read from 162's stage1_summary.json
(per_footprint_swept — the exact finite-scored counts) or, failing that,
from 160's sweep_manifest_summary.json (per_footprint manifest counts; a
superset, hence conservative). The survivors-only-m variant is still emitted
as *_anticons columns — DIAGNOSTIC ONLY, excluded from every headline.

POWER FLOOR (surfaced in the summary): conformal p-values cannot go below
1/(n_cal_g+1), so full-m BH at alpha can select NOTHING unless at least
ceil(m_g / (alpha*(n_cal_g+1))) tests sit at that floor — grow the group's
calibration set, not the test set, for power.

NEGEVAL DOUBLE-USE (documented) + MITIGATION (implemented): the stage-1
thresholds (113/145 operating points) and this conformal calibration draw on
the same NegEval-1M pool. Mitigation: the calibration rows are split
deterministically (seed 2026, row_id-sorted permutation) into a
stage-1-threshold half and a disjoint conformal half, and ONLY the conformal
half calibrates here (--no-cal-split disables; the residual dependence —
historical operating points fitted on the full pool — is recorded in the
summary). Calibration rows that re-appear as sweep TEST rows (matched on the
unqualified row_id suffix + footprint; 160 row_ids are footprint-qualified)
are dropped from calibration.

Outputs: data/v2/sweep/conformal.parquet (per row: group/pooled conformal p,
PRIMARY full-m BH q + selected flags per alpha in {0.05,0.1,0.25}, *_anticons
diagnostics; the guarantee note is embedded in the parquet metadata) +
data/v2/sweep/conformal_summary.json.

    python 165_group_conformal.py
    python 165_group_conformal.py --sweep-totals "north=5040000,south=12250000"
    python 165_group_conformal.py --synthetic-check    # no real inputs: shifted
        # north nulls -> asserts per-group FDR <= alpha while POOLED calibration
        # inflates the shifted group's FDR; q<=alpha == bh_select equivalence;
        # null-p super-uniformity; padded full-m BH arithmetic; and the
        # TWO-STAGE check: under correlated stage-1 selection the survivor-m
        # (anticons) BH violates FDR while full-m BH controls it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

CONF = C._load("cn_165_conf50", C.ROOT / "50_conformal_selection.py")  # reuse 50
V2 = C.DATA / "v2"
SWEEP = V2 / "sweep"
ALPHAS = (0.05, 0.10, 0.25)
CAL_AUTO = ("ensemble_v2_pool_combined.parquet", "scores_negeval_pool_combined.parquet")
CAL_COL_AUTO = ("v2_average", "average")
V1_ROSTER = {"shielded_A", "effnet_B", "effnet_B3", "effnet_S2", "resnet46_C"}
CAL_SPLIT_SEED = 2026

GUARANTEE_NOTE = (
    "PRIMARY columns (q_group / sel_group_a*): per-group conformal BH with the "
    "FULL per-footprint sweep total as m — non-survivors are censored, "
    "never-selectable tests at p=1; censoring can only raise a p-value, and the "
    "unconditional conformal p-value of any null sweep row vs the NegEval "
    "calibration is exchangeable (same parent population), so per-group FDR <= "
    "alpha holds marginally. DIAGNOSTIC *_anticons columns use m = n_survivors, "
    "conditioning the test set on a stage-1 selection computed from the same "
    "members: NO FDR guarantee (anti-conservative; --synthetic-check "
    "demonstrates the violation). Pooled rows are descriptive comparisons, not "
    "guarantees. Power floor: p >= 1/(n_cal_g+1), so full-m BH at alpha can "
    "select nothing unless ceil(m_g/(alpha*(n_cal_g+1))) tests sit at the floor. "
    "NegEval double-use (stage-1 thresholds AND this calibration come from "
    "NegEval-1M) is mitigated by the deterministic seed-2026 split: only the "
    "conformal half calibrates here; calibration rows re-appearing as sweep "
    "tests are dropped.")


def parquet_with_note(df: pd.DataFrame, path: Path, note: str):
    """Write a parquet whose schema metadata carries the guarantee note —
    every output carrying a selection carries the note."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(t.schema.metadata or {})
    md[b"claudenet_guarantee_note"] = note.encode()
    pq.write_table(t.replace_schema_metadata(md), path)


# ===== BH extensions on top of 50's arithmetic ===================================

def bh_qvalues(pvals, m_total: int | None = None) -> np.ndarray:
    """BH-adjusted p-values: q_(k) = min_{j>=k} m*p_(j)/j (capped at 1), with
    m = len(pvals) or an external total test count m_total (>= n observed;
    the unobserved tests sit at p=1 and cannot change any cummin term)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    m = n if m_total is None else int(m_total)
    assert m >= n, f"m_total {m} < n observed {n}"
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, n + 1)
    q = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    out = np.empty(n)
    out[order] = q
    return out


def bh_select_m(pvals, alpha: float, m_total: int) -> np.ndarray:
    """50's bh_select loop with the k/m ratio taken over m_total tests
    (observed p-values padded by never-selectable p=1 tests; identical to
    bh_select on the explicitly padded array for any alpha < 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    n, m = p.size, int(m_total)
    assert m >= n, f"m_total {m} < n observed {n}"
    order = np.argsort(p)
    thresh = 0
    for k in range(1, n + 1):
        if p[order[k - 1]] <= k / m * alpha:
            thresh = k
    sel = np.zeros(n, bool)
    sel[order[:thresh]] = True
    return sel


def select_and_q(p, alphas, m_total=None, strict=False, label=""):
    """Authoritative BH selection (50's bh_select / the padded variant) at each
    alpha + q-values; asserts/warns {q<=alpha} == selection (float-boundary
    mismatches would show here)."""
    q = bh_qvalues(p, m_total)
    sels = {}
    for a in alphas:
        sel = (CONF.bh_select(p, a) if m_total is None
               else bh_select_m(p, a, m_total))
        mism = int((sel != (q <= a)).sum())
        if mism:
            msg = (f"q-vs-bh_select mismatch ({label}, alpha={a:g}): {mism} rows "
                   f"(float boundary)")
            if strict:
                raise AssertionError(msg)
            print(f"[conf] WARNING: {msg}")
        sels[a] = sel
    return q, sels


def parse_totals(spec: str) -> dict[str, int]:
    """'north=5040000,south=12250000' -> {'north': 5040000, ...}"""
    out = {}
    for part in spec.split(","):
        if not part:
            continue
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


# ===== the real run ==============================================================

def resolve_totals(args, groups) -> tuple[dict, str]:
    """REQUIRED full-sweep per-footprint totals (the BH m): --sweep-totals
    override > 162's stage1_summary.json per_footprint_swept (exact
    finite-scored counts) > 160's sweep_manifest_summary.json per_footprint
    (manifest counts; a superset of swept, hence conservative). FATAL when
    none is available — the survivors-only-m variant has no FDR guarantee and
    is never the primary output."""
    if args.sweep_totals:
        return parse_totals(args.sweep_totals), "--sweep-totals (CLI override)"
    p = Path(args.stage1_summary)
    if p.exists():
        t = json.loads(p.read_text()).get("per_footprint_swept")
        if t:
            return ({k: int(v) for k, v in t.items()},
                    f"{p} per_footprint_swept (finite-scored sweep rows)")
    p = Path(args.sweep_summary)
    if p.exists():
        t = json.loads(p.read_text()).get("per_footprint")
        if t:
            return ({k: int(v) for k, v in t.items()},
                    f"{p} per_footprint (160 manifest rows >= swept; conservative)")
    raise SystemExit(
        "[165] FATAL: full-m BH REQUIRES per-footprint sweep totals and none "
        "were found. Pass --sweep-totals 'north=N,south=M', or point "
        "--stage1-summary at 162's stage1_summary.json (per_footprint_swept), "
        "or --sweep-summary at 160's sweep_manifest_summary.json "
        "(per_footprint). Refusing to emit a survivors-only-m headline: it "
        "has NO FDR guarantee under two-stage selection.")


def resolve_calibration(args):
    """--calibration 'auto' -> 145's pool-combined parquet (the SAME score
    function as 162's p_final). The 113 v1-average fallback is a DIFFERENT
    function unless the v2 roster is the v1 five with flagship 'average', so
    it needs --allow-v1-cal AND a meta.json proof of that equality."""
    if args.calibration == "auto":
        p145, p113 = V2 / CAL_AUTO[0], V2 / CAL_AUTO[1]
        if p145.exists():
            cal_path = p145
        elif p113.exists():
            if not args.allow_v1_cal:
                raise SystemExit(
                    f"[165] FATAL: {p145} missing. The 113 fallback ({p113.name}, "
                    f"column 'average') is the V1-roster combiner — a DIFFERENT "
                    f"score function from 162's p_final whenever the v2 roster "
                    f"differs from the v1 five, which voids exchangeability and "
                    f"every q/selection column. Run 145 (or 145 --apply-roster "
                    f"on the NegEval pool) first; --allow-v1-cal permits the "
                    f"fallback ONLY after the fits meta.json proves roster "
                    f"equality.")
            mp = Path(args.fits)
            mp = mp / "meta.json" if mp.is_dir() else mp.parent / "meta.json"
            assert mp.exists(), (f"--allow-v1-cal needs {mp} to verify the v2 "
                                 f"roster equals the v1 five (run 145 first)")
            meta = json.loads(mp.read_text())
            assert set(meta["keys"]) == V1_ROSTER, \
                (f"--allow-v1-cal refused: persisted v2 roster {sorted(meta['keys'])} "
                 f"!= v1 roster {sorted(V1_ROSTER)} — the 113 'average' column is a "
                 f"different score function")
            assert meta.get("flagship_combiner") == "average", \
                (f"--allow-v1-cal refused: flagship combiner "
                 f"{meta.get('flagship_combiner')!r} != 'average'")
            print(f"[165] --allow-v1-cal: {mp} proves v2 roster == v1 five & "
                  f"flagship == average -> 113 fallback is the same function")
            cal_path = p113
        else:
            raise SystemExit(f"[165] FATAL: none of {CAL_AUTO} exist under {V2} — "
                             f"run 145 on NegEval-1M first")
    else:
        cal_path = Path(args.calibration)
        assert cal_path.exists(), f"--calibration {cal_path} missing"
    cal = pd.read_parquet(cal_path)
    if args.cal_col == "auto":
        col = next((c for c in CAL_COL_AUTO if c in cal.columns), None)
        assert col, (f"{cal_path}: none of {CAL_COL_AUTO} present "
                     f"(have {list(cal.columns)}); pass --cal-col")
    else:
        col = args.cal_col
        assert col in cal.columns, f"{cal_path}: no column {col!r}"
    print(f"[165] calibration = {cal_path.name}::{col} ({len(cal):,} rows)")
    return cal_path, cal, col


def main_real(args, alphas):
    t0 = time.time()
    SWEEP.mkdir(parents=True, exist_ok=True)
    cal_path, cal, cal_col = resolve_calibration(args)

    man = pd.read_parquet(args.cal_manifest, columns=["row_id", "footprint"])
    man["row_id"] = man["row_id"].astype(str)
    cal = cal.copy()
    cal["row_id"] = cal["row_id"].astype(str)
    n0 = len(cal)
    cal = cal.merge(man, on="row_id", how="inner", validate="one_to_one")
    assert len(cal), f"no calibration rows after joining {args.cal_manifest}"
    if len(cal) < n0:
        print(f"[165] WARNING: {n0 - len(cal):,} calibration rows missing from "
              f"the manifest -> dropped")
    fin = np.isfinite(cal[cal_col].to_numpy(np.float64))
    if (~fin).any():
        print(f"[165] dropping {int((~fin).sum()):,} non-finite calibration scores")
        cal = cal[fin]

    # -- NegEval double-use mitigation: deterministic seed-2026 split ------------
    cal_split = {"applied": False, "seed": CAL_SPLIT_SEED,
                 "note": ("stage-1 thresholds (113/145) and this calibration "
                          "draw on the same NegEval-1M pool; only the conformal "
                          "half of a deterministic seed-2026 split calibrates "
                          "here. Residual dependence: historical operating "
                          "points were fitted on the full pool.")}
    if not args.no_cal_split:
        cal = cal.sort_values("row_id", kind="mergesort").reset_index(drop=True)
        perm = np.random.default_rng(CAL_SPLIT_SEED).permutation(len(cal))
        half = len(cal) // 2
        keep = np.zeros(len(cal), bool)
        keep[perm[half:]] = True
        cal_split.update(applied=True, n_pool=int(len(cal)),
                         n_threshold_half=int(half),
                         n_conformal_half=int(keep.sum()))
        cal = cal[keep]
        print(f"[165] NegEval split (seed {CAL_SPLIT_SEED}): {half:,} rows reserved "
              f"for stage-1 thresholds, {len(cal):,} calibrate the conformal "
              f"p-values (--no-cal-split disables)")
    else:
        print("[165] WARNING: --no-cal-split — conformal calibration reuses the "
              "FULL NegEval pool the stage-1 thresholds were fitted on")

    test = pd.read_parquet(args.stage2_scores)
    for col in ("row_id", args.test_col, "footprint"):
        assert col in test.columns, f"{args.stage2_scores}: missing column {col!r}"
    test = test.copy()
    test["row_id"] = test["row_id"].astype(str)
    assert test.row_id.is_unique, "stage-2 scores: duplicate row_ids"
    fin = np.isfinite(test[args.test_col].to_numpy(np.float64))
    if (~fin).any():
        print(f"[165] dropping {int((~fin).sum()):,} non-finite test scores")
        test = test[fin]
    test = test.reset_index(drop=True)
    print(f"[165] test = {args.stage2_scores}::{args.test_col} ({len(test):,} rows)")

    # -- a calibration row must never be its own conformal test case -------------
    # (sweep row_ids are footprint-qualified '<f>_...' while the NegEval
    # manifest uses '<BRICKID>_<OBJID>': match on the unqualified suffix +
    # footprint)
    tkey = pd.MultiIndex.from_arrays(
        [test.footprint.astype(str),
         test.row_id.str.replace(r"^[a-z]_(?=\d+_\d+$)", "", regex=True)])
    ckey = pd.MultiIndex.from_arrays(
        [cal.footprint.astype(str), cal.row_id.astype(str)])
    in_test = ckey.isin(tkey)
    n_cal_in_test = int(in_test.sum())
    if n_cal_in_test:
        print(f"[165] dropping {n_cal_in_test:,} calibration rows that re-appear "
              f"as sweep test rows (calibration/test disjointness)")
        cal = cal[~np.asarray(in_test)]

    groups = sorted(test.footprint.unique())
    totals, totals_source = resolve_totals(args, groups)
    missing = [g for g in groups if g not in totals]
    assert not missing, f"sweep totals ({totals_source}) lack groups {missing}"
    print(f"[165] full-m totals from {totals_source}: "
          + ", ".join(f"{g}={totals[g]:,}" for g in groups))

    out = pd.DataFrame({"row_id": test.row_id, "footprint": test.footprint,
                        args.test_col: test[args.test_col].to_numpy(np.float64)})
    for c in ("p_conf_group", "q_group", "q_group_anticons",
              "p_conf_pooled", "q_pooled", "q_pooled_anticons"):
        out[c] = np.nan
    summary = {"alphas": list(alphas), "calibration": f"{cal_path}::{cal_col}",
               "cal_manifest": str(args.cal_manifest),
               "stage2_scores": str(args.stage2_scores), "test_col": args.test_col,
               "sweep_totals": totals, "sweep_totals_source": totals_source,
               "cal_split": cal_split, "n_cal_dropped_in_test": n_cal_in_test,
               "groups": {}, "pooled": {},
               "guarantee_note": GUARANTEE_NOTE}

    # -- per-group: 50's p-value construction + BH within the group --------------
    # PRIMARY = full-m (valid FDR); *_anticons = survivors-only m (DIAGNOSTIC,
    # no guarantee — see GUARANTEE_NOTE)
    for g in groups:
        gi = (test.footprint == g).to_numpy()
        cal_g = cal[cal.footprint == g][cal_col].to_numpy(np.float64)
        assert len(cal_g) >= args.min_cal, \
            f"group {g}: only {len(cal_g)} calibration negatives (< --min-cal)"
        s = out.loc[gi, args.test_col].to_numpy()
        p = CONF.conformal_pvalues(s, cal_g)
        m_g, n_test_g = int(totals[g]), int(gi.sum())
        assert m_g >= n_test_g, (f"group {g}: sweep total {m_g:,} < n_test "
                                 f"{n_test_g:,} — wrong totals source?")
        q, sels = select_and_q(p, alphas, m_total=m_g, label=f"group {g} (full-m)")
        qa, sels_a = select_and_q(p, alphas, label=f"group {g} (anticons)")
        out.loc[gi, "p_conf_group"] = p
        out.loc[gi, "q_group"] = q
        out.loc[gi, "q_group_anticons"] = qa
        min_p = 1.0 / (len(cal_g) + 1)
        ent = {"n_cal": int(len(cal_g)), "n_test": n_test_g, "m_total": m_g,
               "min_possible_p": min_p,
               "power_floor": {
                   "min_possible_p": min_p,
                   "note": ("conformal p >= 1/(n_cal_g+1): full-m BH at alpha "
                            "selects NOTHING unless min_k_for_any_selection "
                            "tests sit at the floor — grow n_cal_g for power"),
                   "alphas": {}},
               "alphas": {}, "anticons_diagnostic_alphas": {}}
        for a in alphas:
            sel, sel_a = sels[a], sels_a[a]
            col = f"sel_group_a{a:g}"
            acol = f"sel_group_anticons_a{a:g}"
            for c in (col, acol):
                if c not in out.columns:
                    out[c] = False
            out.loc[gi, col] = sel
            out.loc[gi, acol] = sel_a
            k_min = int(np.ceil(min_p * m_g / a))
            ent["power_floor"]["alphas"][f"{a:g}"] = {
                "min_k_for_any_selection": k_min,
                "selection_possible": bool(n_test_g >= k_min),
                "n_cal_needed_for_k1": int(np.ceil(m_g / a)) - 1}
            ent["alphas"][f"{a:g}"] = {
                "n_selected": int(sel.sum()),
                "sel_rate": float(sel.mean()),
                "p_cutoff": float(p[sel].max()) if sel.any() else None}
            ent["anticons_diagnostic_alphas"][f"{a:g}"] = {
                "n_selected": int(sel_a.sum()),
                "note": "m=n_survivors; NO FDR guarantee (diagnostic only)"}
            if n_test_g < k_min:
                print(f"[conf] power floor (group {g}, alpha={a:g}): needs "
                      f">={k_min:,} tests at p={min_p:.2e} but only "
                      f"{n_test_g:,} survivors exist -> NOTHING selectable; "
                      f"grow n_cal_{g} (need ~{int(np.ceil(m_g / a)) - 1:,} "
                      f"for k=1)")
        summary["groups"][g] = ent

    # -- pooled (50's original path; descriptive comparison) ---------------------
    cal_all = cal[cal_col].to_numpy(np.float64)
    m_all = int(sum(totals[g] for g in groups))
    p_pool = CONF.conformal_pvalues(out[args.test_col].to_numpy(), cal_all)
    q_pool, sels_pool = select_and_q(p_pool, alphas, m_total=m_all,
                                     label="pooled (full-m)")
    qpa, sels_pa = select_and_q(p_pool, alphas, label="pooled (anticons)")
    out["p_conf_pooled"] = p_pool
    out["q_pooled"] = q_pool
    out["q_pooled_anticons"] = qpa
    summary["pooled"] = {"n_cal": int(len(cal_all)), "n_test": int(len(out)),
                         "m_total": m_all, "alphas": {},
                         "anticons_diagnostic_alphas": {}}
    for a in alphas:
        out[f"sel_pooled_a{a:g}"] = sels_pool[a]
        out[f"sel_pooled_anticons_a{a:g}"] = sels_pa[a]
        summary["pooled"]["alphas"][f"{a:g}"] = {
            "n_selected": int(sels_pool[a].sum()),
            "by_group": {g: int(sels_pool[a][(test.footprint == g).to_numpy()].sum())
                         for g in groups}}
        summary["pooled"]["anticons_diagnostic_alphas"][f"{a:g}"] = {
            "n_selected": int(sels_pa[a].sum())}

    # -- report + write -----------------------------------------------------------
    print(f"\n[conf] PRIMARY = full-m BH (m = per-footprint sweep totals); "
          f"(anticons) = survivors-only m, NO FDR guarantee, diagnostic only")
    print(f"[conf] {'alpha':>6} {'group':>10} {'n_cal':>9} {'m_total':>11} "
          f"{'n_test':>9} {'n_sel':>7} {'p_cutoff':>10} {'(anticons)':>10}")
    for a in alphas:
        for g in groups:
            e = summary["groups"][g]
            ea = e["alphas"][f"{a:g}"]
            cut = f"{ea['p_cutoff']:.2e}" if ea["p_cutoff"] is not None else "-"
            n_anti = e["anticons_diagnostic_alphas"][f"{a:g}"]["n_selected"]
            print(f"[conf] {a:>6.2f} {g:>10} {e['n_cal']:>9,} {e['m_total']:>11,} "
                  f"{e['n_test']:>9,} {ea['n_selected']:>7,} {cut:>10} "
                  f"{n_anti:>10,}")
        ep = summary["pooled"]["alphas"][f"{a:g}"]
        union = sum(summary["groups"][g]["alphas"][f"{a:g}"]["n_selected"]
                    for g in groups)
        print(f"[conf] {a:>6.2f} {'UNION':>10} {'':>9} {m_all:>11,} {len(out):>9,} "
              f"{union:>7,} {'':>10} {'':>10}  "
              f"(pooled-cal comparison: {ep['n_selected']:,})")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    parquet_with_note(out, out_p, GUARANTEE_NOTE)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(f"\n[165] wrote {out_p} + {args.summary_out} ({time.time() - t0:.1f}s)")
    return 0


# ===== --synthetic-check =========================================================

def synthetic_check(args, alphas):
    """No real inputs (seed 2026): null scores ~ Beta(0.5,8) with the NORTH
    group shifted up by +0.08 (domain shift); signals ~ Beta(6,3)+shift.
    Asserts (1) {q<=alpha} == bh_select everywhere (strict), (2) the padded
    full-m BH == bh_select on the explicitly padded array, (3) null conformal
    p-values are super-uniform per group, (4) GROUP-conformal empirical FDR
    <= alpha + slack in every group, (5) POOLED calibration inflates the
    shifted group's FDR above both alpha and the group-conformal FDR, and
    (6) the TWO-STAGE check that motivates the full-m primary columns:
    correlated stage-1/stage-2 scores (rho=0.9), stage-1 upper-tail cut at
    1e-4, conformal p vs an unconditional null calibration -> survivors-only-m
    BH (the *_anticons variant) VIOLATES FDR at alpha while full-m BH
    controls it."""
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    SHIFT = 0.08
    n_cal = {"north": 60_000, "south": 120_000}
    n_null = {"north": 4_000, "south": 8_000}
    n_sig = {"north": 400, "south": 400}

    def null_scores(g, n):
        return rng.beta(0.5, 8.0, n) + (SHIFT if g == "north" else 0.0)

    cal = {g: null_scores(g, n) for g, n in n_cal.items()}
    test, is_null, grp = [], [], []
    for g in ("north", "south"):
        test.append(np.r_[null_scores(g, n_null[g]),
                          rng.beta(6.0, 3.0, n_sig[g]) + (SHIFT if g == "north" else 0.0)])
        is_null.append(np.r_[np.ones(n_null[g], bool), np.zeros(n_sig[g], bool)])
        grp += [g] * (n_null[g] + n_sig[g])
    test, is_null, grp = np.concatenate(test), np.concatenate(is_null), np.array(grp)
    print(f"[synth] cal={n_cal}, test nulls={n_null}, signals={n_sig}, "
          f"north shift=+{SHIFT}, alphas={list(alphas)}, seed={args.seed}")

    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print(f"[synth] {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    # (1)+(4) group-conformal: q==select (strict) + per-group FDR <= alpha+slack
    fdr_group = {}
    for g in ("north", "south"):
        gi = grp == g
        p = CONF.conformal_pvalues(test[gi], cal[g])
        q, sels = select_and_q(p, alphas, strict=True, label=f"synth {g}")
        null_p = p[is_null[gi]]
        check(f"null p super-uniform ({g})", null_p.mean() >= 0.45,
              f"mean null p = {null_p.mean():.3f} (>= 0.45)")
        for a in alphas:
            sel = sels[a]
            fdp = float(is_null[gi][sel].mean()) if sel.any() else 0.0
            fdr_group[(g, a)] = fdp
            check(f"group FDR <= alpha+0.06 ({g}, alpha={a:g})", fdp <= a + 0.06,
                  f"FDP {fdp:.3f} vs alpha {a:g} (n_sel {int(sel.sum())})")

    # (5) pooled calibration under shift: north FDR inflated
    cal_pool = np.concatenate([cal[g] for g in ("north", "south")])
    p_pool = CONF.conformal_pvalues(test, cal_pool)
    a0 = 0.10
    sel_pool = CONF.bh_select(p_pool, a0)
    ni = grp == "north"
    seln = sel_pool & ni
    fdp_north_pool = float(is_null[seln].mean()) if seln.any() else 0.0
    check("pooled cal inflates shifted-group FDR (north, alpha=0.1)",
          fdp_north_pool > a0 and fdp_north_pool > fdr_group[("north", a0)],
          f"pooled-north FDP {fdp_north_pool:.3f} > alpha {a0:g} and > "
          f"group-north FDP {fdr_group[('north', a0)]:.3f}")

    # (2) padded full-m BH == bh_select on the explicitly padded array
    # (slice straddling the null/signal boundary -> non-trivial selection)
    p_small = CONF.conformal_pvalues(test[ni][3700:4400], cal["north"])
    m_tot = 5_000
    ok_pad, n_sel_pad = True, 0
    for a in alphas:
        sel_fast = bh_select_m(p_small, a, m_tot)
        padded = np.r_[p_small, np.ones(m_tot - p_small.size)]
        sel_ref = CONF.bh_select(padded, a)[:p_small.size]
        ok_pad &= bool((sel_fast == sel_ref).all())
        n_sel_pad = max(n_sel_pad, int(sel_fast.sum()))
        assert not CONF.bh_select(padded, a)[p_small.size:].any(), \
            "padded p=1 tests selected?!"
    check("padded full-m BH == bh_select on padded array (all alphas, non-trivial)",
          ok_pad and n_sel_pad > 0,
          f"m_total={m_tot}, n={p_small.size}, max n_sel {n_sel_pad}")
    qf = bh_qvalues(p_small, m_tot)
    sel_q = qf <= a0
    check("full-m q<=alpha == full-m selection (alpha=0.1)",
          bool((sel_q == bh_select_m(p_small, a0, m_tot)).all()),
          f"n_sel {int(sel_q.sum())}")

    # (6) TWO-STAGE selection: survivor-m (anticons) BH violates FDR under
    # correlated stage-1 selection of the test rows; full-m BH controls it.
    rng2 = np.random.default_rng(args.seed + 1)
    rho, n_null2, n_sig2, n_cal2 = 0.9, 2_000_000, 300, 1_000_000
    z = rng2.standard_normal(n_null2)
    w = np.sqrt(1.0 - rho * rho)
    s1_null = rho * z + w * rng2.standard_normal(n_null2)
    s2_null = rho * z + w * rng2.standard_normal(n_null2)
    s1_sig = rng2.normal(5.0, 1.0, n_sig2)
    s2_sig = rng2.normal(5.0, 1.0, n_sig2)
    cal2 = rng2.standard_normal(n_cal2)          # UNCONDITIONAL null calibration
    thr = float(np.quantile(rng2.standard_normal(n_null2), 1.0 - 1e-4))
    sn, ss = s1_null >= thr, s1_sig >= thr       # stage-1 survivors @1e-4
    p2 = CONF.conformal_pvalues(np.r_[s2_null[sn], s2_sig[ss]], cal2)
    null2 = np.r_[np.ones(int(sn.sum()), bool), np.zeros(int(ss.sum()), bool)]
    sel_anti = CONF.bh_select(p2, a0)            # m = n_survivors (anticons)
    sel_full = bh_select_m(p2, a0, m_total=n_null2 + n_sig2)
    fdp_anti = float(null2[sel_anti].mean()) if sel_anti.any() else 0.0
    fdp_full = float(null2[sel_full].mean()) if sel_full.any() else 0.0
    check("two-stage: survivor-m (anticons) BH VIOLATES FDR",
          fdp_anti > a0 + 0.05,
          f"survivor-m FDP {fdp_anti:.3f} >> alpha {a0:g} "
          f"(n_sel {int(sel_anti.sum())}/{int(null2.size)} survivors)")
    check("two-stage: full-m BH controls FDR under stage-1 selection",
          bool(sel_full.any()) and fdp_full <= a0 + 0.03,
          f"full-m FDP {fdp_full:.3f} <= alpha {a0:g} "
          f"(n_sel {int(sel_full.sum())})")
    check("two-stage: full-m strictly less anti-conservative than survivor-m",
          fdp_full < fdp_anti and sel_full.sum() <= sel_anti.sum(),
          f"FDP {fdp_full:.3f} < {fdp_anti:.3f}; n_sel "
          f"{int(sel_full.sum())} <= {int(sel_anti.sum())}")

    n_ok = sum(checks)
    print(f"\n[synth] {n_ok}/{len(checks)} checks passed ({time.time() - t0:.1f}s) "
          f"-> {'SYNTHETIC-CHECK PASS' if n_ok == len(checks) else 'SYNTHETIC-CHECK FAIL'}")
    return 0 if n_ok == len(checks) else 1


# ===== cli =======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration", default="auto",
                    help="NegEval-1M combiner-scores parquet; 'auto' = first of "
                         f"{', '.join(CAL_AUTO)} under data/v2/")
    ap.add_argument("--cal-col", default="auto",
                    help="calibration score column; 'auto' prefers "
                         f"{'/'.join(CAL_COL_AUTO)}; MUST match the 162 p_final path")
    ap.add_argument("--cal-manifest", default=str(V2 / "negeval_manifest.parquet"),
                    help="row_id -> footprint map for the calibration rows")
    ap.add_argument("--stage2-scores", default=str(SWEEP / "stage2_scores.parquet"),
                    help="162 output (test scores; needs footprint column)")
    ap.add_argument("--test-col", default="p_final")
    ap.add_argument("--alphas", default="0.05,0.1,0.25",
                    help="comma-separated BH target FDRs")
    ap.add_argument("--sweep-totals", default="",
                    help="OVERRIDE for the REQUIRED full-sweep test counts per "
                         "group, e.g. 'north=5040000,south=12250000'; default: "
                         "read from --stage1-summary (per_footprint_swept) or "
                         "--sweep-summary (per_footprint)")
    ap.add_argument("--stage1-summary", default=str(SWEEP / "stage1_summary.json"),
                    help="162 summary json: per_footprint_swept totals source")
    ap.add_argument("--sweep-summary",
                    default=str(SWEEP / "sweep_manifest_summary.json"),
                    help="160 summary json: per_footprint totals fallback")
    ap.add_argument("--fits", default=str(V2 / "ensemble_v2_fits"),
                    help="persisted 145 fits dir; its meta.json gates "
                         "--allow-v1-cal (roster-equality proof)")
    ap.add_argument("--allow-v1-cal", action="store_true",
                    help="permit the 113 v1-average calibration fallback — only "
                         "honoured when meta.json proves v2 roster == v1 five "
                         "and flagship == average")
    ap.add_argument("--no-cal-split", action="store_true",
                    help="disable the seed-2026 NegEval threshold/conformal "
                         "split (calibrate on the FULL pool; double-use caveat)")
    ap.add_argument("--min-cal", type=int, default=1000,
                    help="minimum calibration negatives per group")
    ap.add_argument("--out", default=str(SWEEP / "conformal.parquet"))
    ap.add_argument("--summary-out", default=str(SWEEP / "conformal_summary.json"))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--synthetic-check", action="store_true",
                    help="run the BH/group-conformal arithmetic on synthetic "
                         "shifted groups + assert FDR behaviour")
    args = ap.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(","))
    if args.synthetic_check:
        return synthetic_check(args, alphas)
    return main_real(args, alphas)


if __name__ == "__main__":
    raise SystemExit(main())
