#!/usr/bin/env python3
"""145_refit_ensemble_v2.py — Phase 140: rebuild the flagship ensemble from a
CONFIGURABLE member roster and evaluate v2 vs v1 vs published, with paired CIs
on BOTH negative sets (runs LOCALLY, CPU only).

Mechanics (the v1 25/26/28 recipes + the 113 bootstrap machinery REUSED by
import, not reimplemented): per roster member the isotonic calibrator is refit
on the val split (_ensemble.make_calibrator, the 25 recipe; a stored `pc`
column is verified to < 1e-9 when present — for NEW members, where no stored
value exists, DETERMINISM is asserted instead by refitting twice); the
average/logistic/rf combiners are fit per roster on that roster's calibrated
val matrix (_ensemble.fit_combiner, the 26 recipe, refit-twice determinism
check; member ORDER in the roster file is the combiner feature order, so the
default v1 roster keeps 26's sorted-glob order). Then EVERY eval-roster
member, BOTH rosters' combiners, the leave-one-out average ensembles (the
member-admission report) and the Stage-D baselines go through
113_thresholds_ci_evt.run_suite in ONE call — one shared multinomial pool
resample + shared positive resample indices per rep — so the v2-vs-v1-flagship
delta, the delta-vs-baseline_meta and every member's marginal are all exactly
paired per rep; with brick-block sensitivity, the EVT cross-check and
old-testneg (6,501-row v1) continuity point estimates.

Roster: --roster is a JSON list of {name, scores_parquet, pool_column} entries
(scores_parquet = v1 schema [split,row_id,label,p(,pc)]; pool_column = that
member's RAW-score column in the merged --pool-scores parquet(s)).
--make-default-roster writes data/v2/roster_v2.json (the v1 6-member roster —
the starting point the orchestrator edits per the 122/132/141 gates) plus the
frozen data/v2/roster_v1.json reference; the v1 flagship reference
(--reference-combiner over --ref-roster) is computed IN the same run.

SHIP GATE (encoded exactly): SHIP iff v2 beats v1 flagship on >=3/4 of
(storfer,inchausti)x(1e-3: CI excludes 0; 1e-2: point).

Outputs: <out-prefix>_operating_points.csv (113 rows_from_results schema +
roster tag column), <out-prefix>_verdict.json (per-metric paired v2-vs-v1 and
v2-vs-meta deltas + CIs, member admission report, Pearson/Spearman member
correlation matrices on calibrated pool scores, the gate), and
<out-prefix>_pool_combined.parquet (combiner pool scores, 114 purity-audit
input). Default out-prefix: data/v2/ensemble_v2.

    /home2/benson/.venvs/claudenet/bin/python 145_refit_ensemble_v2.py --make-default-roster
    /home2/benson/.venvs/claudenet/bin/python 145_refit_ensemble_v2.py \\
        --roster data/v2/roster_v2.json \\
        --pool-scores data/v2/scores_negeval_pool.parquet[,<112 --only-extra out>,...]
    /home2/benson/.venvs/claudenet/bin/python 145_refit_ensemble_v2.py --synthetic-check
        # no real inputs: 5 synthetic members (correlated trio + harmful noisy
        # + known-better sharp) through the FULL file -> calibrate -> combine
        # -> run_suite -> gate path, in BOTH roster directions, asserting the
        # admission logic, the correlation ordering and the gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _ensemble as E

V2 = C.DATA / "v2"
COMBINERS = ("average", "logistic", "rf")
BASELINES = ("baseline_effnet", "baseline_meta")
POS_SPLITS = ("storfer", "inchausti")
GATE_TEXT = ("SHIP iff v2 beats v1 flagship on >=3/4 of "
             "(storfer,inchausti)x(1e-3: CI excludes 0; 1e-2: point)")
# v1 roster in 26's combiner feature order (= _combine.member_names() sorted glob)
V1_MEMBERS = ("aion", "effnet_B", "effnet_B3", "effnet_S2", "resnet46_C", "shielded_A")
V1_ROSTER = [{"name": n, "scores_parquet": f"data/scores_member_{n}.parquet",
              "pool_column": f"member_{n}"} for n in V1_MEMBERS]


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else C.ROOT / p


# ===== roster loading ============================================================

def load_roster_file(path: Path, fallback=None, what="roster"):
    if path.exists():
        entries = json.loads(path.read_text())
    elif fallback is not None:
        print(f"[145] NOTE: {path} missing -> built-in v1 default used as {what}")
        entries = [dict(e) for e in fallback]
    else:
        raise SystemExit(f"[145] FATAL: {what} {path} missing — run "
                         f"`python 145_refit_ensemble_v2.py --make-default-roster` first")
    assert isinstance(entries, list) and entries, f"{path}: roster must be a non-empty JSON list"
    for e in entries:
        missing = {"name", "scores_parquet", "pool_column"} - set(e)
        assert not missing, f"{path}: roster entry {e} missing keys {missing}"
        assert ":" not in e["name"], f"{path}: member name {e['name']!r} may not contain ':'"
    return entries


def load_members(rosters: dict, order: tuple[str, str]):
    """Unique member registry across both rosters. Key = name; a name collision
    with a DIFFERENT (parquet, pool_column) source gets '<name>@<tag>'.
    Returns (registry {key: dict}, {tag: [ordered keys]})."""
    reg, keys = {}, {t: [] for t in order}
    for tag in order:
        entries = rosters[tag]
        names = [e["name"] for e in entries]
        assert len(set(names)) == len(names), f"duplicate member names in {tag} roster"
        for e in entries:
            pq = _resolve(e["scores_parquet"])
            ident = (str(pq), e["pool_column"])
            key = e["name"]
            if key in reg and (str(reg[key]["parquet"]), reg[key]["pool_column"]) != ident:
                key = f"{e['name']}@{tag}"
                assert key not in reg, f"member key collision: {key}"
            if key not in reg:
                assert pq.exists(), f"{tag} roster member {e['name']}: {pq} missing"
                df = pd.read_parquet(pq)
                miss = {"split", "row_id", "label", "p"} - set(df.columns)
                assert not miss, f"{pq}: missing columns {miss} (need v1 scores_member schema)"
                df = df.copy()
                df["row_id"] = df["row_id"].astype(str)
                reg[key] = dict(name=e["name"], parquet=pq,
                                pool_column=e["pool_column"], df=df)
            keys[tag].append(key)
    return reg, keys


# ===== calibration + combiners (25/26 recipes; determinism instead of stored) ====

def fit_calibrators(reg) -> dict:
    """25's recipe per member: isotonic fit on the val split. Stored pc (when
    present) verified to < 1e-9; determinism asserted by refitting twice."""
    report = {}
    for k, m in reg.items():
        val = m["df"][m["df"].split == "val"]
        assert len(val) and val["label"].nunique() == 2, f"{k}: val split missing/degenerate"
        p, y = val["p"].to_numpy(np.float64), val["label"].to_numpy()
        assert np.isfinite(p).all(), f"{k}: non-finite val p"
        cal = E.make_calibrator("isotonic").fit(p, y)
        cal2 = E.make_calibrator("isotonic").fit(p, y)
        allp = m["df"]["p"].to_numpy(np.float64)
        ddet = float(np.max(np.abs(cal.transform(allp) - cal2.transform(allp))))
        assert ddet < 1e-9, f"{k}: isotonic refit not deterministic ({ddet:.3e})"
        rep = {"determinism_max_abs_diff": ddet}
        if "pc" in m["df"].columns:
            ref = m["df"]["pc"].to_numpy(np.float64)
            assert np.isfinite(ref).all(), f"{k}: stored pc has non-finite values"
            d = float(np.max(np.abs(cal.transform(allp) - ref)))
            print(f"[cal] {k:20s} refit isotonic vs stored pc: max|diff| = {d:.3e} (tol 1e-9)")
            assert d < 1e-9, f"calibration transfer FAILED for {k}: {d:.3e} >= 1e-9"
            rep["vs_stored_pc_max_abs_diff"] = d
        else:
            print(f"[cal] {k:20s} NEW member (no stored pc) -> determinism check "
                  f"only (max|diff| = {ddet:.0e})")
        m["cal"] = cal
        report[k] = rep
    return report


def roster_val_matrix(reg, keys):
    """Calibrated val matrix aligned across one ROSTER's members (26's recipe;
    inner join on row_id, non-finite raw rows dropped)."""
    base = None
    for k in keys:
        d = reg[k]["df"]
        s = d[d.split == "val"][["row_id", "label", "p"]].rename(columns={"p": k})
        base = s if base is None else base.merge(s.drop(columns="label"), on="row_id",
                                                 how="inner", validate="one_to_one")
    X = base[list(keys)].to_numpy(np.float64)
    base = base[np.isfinite(X).all(axis=1)].reset_index(drop=True)
    P = np.column_stack([reg[k]["cal"].transform(base[k].to_numpy(np.float64))
                         for k in keys])
    return P, base["label"].to_numpy().astype(int)


def fit_roster_combiners(reg, keys, tag):
    """26's recipe on one roster + the refit-twice determinism self-check
    (rosters change -> stored-value verification is impossible for NEW members)."""
    P, y = roster_val_matrix(reg, keys)
    print(f"[comb] {tag}: fitting combiners on val: {len(y)} rows "
          f"({int(y.sum())} pos) x {len(keys)} members")
    combs = {k: E.fit_combiner(k, P, y) for k in COMBINERS}
    combs2 = {k: E.fit_combiner(k, P, y) for k in COMBINERS}
    det = {}
    for k in COMBINERS:
        d = float(np.max(np.abs(combs[k](P) - combs2[k](P))))
        det[k] = d
        print(f"[comb] {tag}:{k:9s} determinism (refit twice) max|diff| = {d:.3e} (tol 1e-9)")
        assert d < 1e-9, f"combiner {tag}:{k} not deterministic across refits ({d:.3e})"
    return combs, det


# ===== aligned score tables ======================================================

def aligned_split(reg, split, base_df=None):
    """One aligned table for a v1 split across ALL unique members (+ baselines
    when base_df given): inner join on row_id, non-finite rows dropped, member
    columns isotonic-calibrated. Returns (row_ids, labels, {scorer: array})."""
    base = None
    for k, m in reg.items():
        d = m["df"]
        s = d[d.split == split][["row_id", "label", "p"]].rename(columns={"p": k})
        base = s if base is None else base.merge(s.drop(columns="label"), on="row_id",
                                                 how="inner", validate="one_to_one")
    if base_df is not None:
        b = base_df[base_df.split == split].copy()
        b["row_id"] = b["row_id"].astype(str)
        n0 = len(base)
        base = base.merge(b[["row_id", *BASELINES]], on="row_id", how="inner",
                          validate="one_to_one")
        if len(base) < n0:
            print(f"[145] {split}: {n0 - len(base)} rows dropped joining baseline scores")
    cols = list(reg) + (list(BASELINES) if base_df is not None else [])
    X = base[cols].to_numpy(np.float64)
    base = base[np.isfinite(X).all(axis=1)].reset_index(drop=True)
    vals = {k: reg[k]["cal"].transform(base[k].to_numpy(np.float64)) for k in reg}
    if base_df is not None:
        vals.update({b_: base[b_].to_numpy(np.float64) for b_ in BASELINES})
    return base["row_id"].to_numpy(), base["label"].to_numpy().astype(int), vals


def expand_scorers(d: dict, combs, keys, tag, rtag):
    """Add both rosters' combiner scores + the eval roster's leave-one-out
    average ensembles to a {scorer: array} dict of calibrated member scores."""
    d = dict(d)
    Q = {t: np.column_stack([d[k] for k in keys[t]]) for t in (tag, rtag)}
    for t in (tag, rtag):
        for cn, fn in combs[t].items():
            d[f"{t}:{cn}"] = fn(Q[t])
    avg = combs[tag]["average"]                 # parameterless: works at any width
    if len(keys[tag]) >= 2:
        for i, k in enumerate(keys[tag]):
            d[f"{tag}:avg_wo_{k}"] = avg(np.delete(Q[tag], i, axis=1))
    return d


def load_pool(paths):
    """Merge the comma-listed pool parquets on row_id (inner; 'ok' columns
    dropped; overlapping score columns are an error)."""
    pool = None
    for f in paths:
        fp = _resolve(f)
        assert fp.exists(), f"pool scores parquet missing: {fp}"
        df = pd.read_parquet(fp)
        assert "row_id" in df.columns, f"{fp}: no row_id column"
        df = df.copy()
        df["row_id"] = df["row_id"].astype(str)
        df = df.drop(columns=[c for c in df.columns if c == "ok"])
        if pool is None:
            pool = df
        else:
            overlap = (set(pool.columns) & set(df.columns)) - {"row_id"}
            assert not overlap, f"pool parquets share score columns {sorted(overlap)}"
            n0 = len(pool)
            pool = pool.merge(df, on="row_id", how="inner", validate="one_to_one")
            print(f"[pool] merged {fp.name}: {n0:,} x {len(df):,} -> {len(pool):,} rows")
        assert pool["row_id"].is_unique, f"{fp}: duplicate row_ids after merge"
    return pool


# ===== the refit + evaluation core ===============================================

def run_eval(args, fprs, rosters: dict, rng, write_pool_combined=True):
    """Full pipeline: calibrate -> combine (both rosters) -> ONE run_suite call
    over members + combiners + LOO + baselines -> CSV + verdict JSON + gate.
    Returns dict(df=..., R=..., verdict=...)."""
    t0 = time.time()
    T113 = C._load("cn_145_t113", C.ROOT / "113_thresholds_ci_evt.py")
    tag, rtag = args.tag, args.ref_tag
    assert tag != rtag, "--tag and --ref-tag must differ"
    assert {1e-2, 1e-3} <= set(fprs), f"gate needs FPRs 1e-2 AND 1e-3 in --fprs (got {fprs})"
    assert args.admission_fpr in fprs, f"--admission-fpr {args.admission_fpr:g} not in --fprs"
    flag, refc = f"{tag}:{args.flagship_combiner}", f"{rtag}:{args.reference_combiner}"

    reg, keys = load_members(rosters, (tag, rtag))
    eval_keys, ref_keys = keys[tag], keys[rtag]
    print(f"[145] {tag} roster ({len(eval_keys)}): {eval_keys}")
    print(f"[145] {rtag} roster ({len(ref_keys)}, reference): {ref_keys}")

    # -- 1. per-member isotonic (25) + per-roster combiners (26), determinism ---
    cal_report = fit_calibrators(reg)
    combs, det = {}, {}
    for t in (tag, rtag):
        combs[t], det[t] = fit_roster_combiners(reg, keys[t], t)

    # -- 2. pool: merge parquets, calibrate member columns, expand scorers ------
    pool = load_pool([s for s in args.pool_scores.split(",") if s])
    for k in reg:
        assert reg[k]["pool_column"] in pool.columns, (
            f"pool lacks column {reg[k]['pool_column']!r} for member {k} "
            f"(rerun 112 --extra-ckpt-dir / merge its output into --pool-scores; "
            f"for the escnn member, 142 --score-pool produces "
            f"data/v2/scores_pool_escnn_D4.parquet — add it to --pool-scores)")
    bases = [b for b in BASELINES if b in pool.columns]
    if set(bases) != set(BASELINES):
        print(f"[pool] WARNING: pool lacks baseline columns "
              f"{sorted(set(BASELINES) - set(bases))} -> delta-vs-meta unavailable")
    used = [reg[k]["pool_column"] for k in reg] + bases
    X = pool[used].to_numpy(np.float64)
    finite = np.isfinite(X).all(axis=1)
    if (~finite).any():
        print(f"[pool] dropping {(~finite).sum():,}/{len(pool):,} rows with "
              f"non-finite scores (failed cutouts)")
    pool = pool[finite].reset_index(drop=True)
    print(f"[pool] {len(pool):,} usable pool rows; baselines on pool: {bases}")

    mvals = {k: reg[k]["cal"].transform(pool[reg[k]["pool_column"]].to_numpy(np.float64))
             for k in reg}
    Pe = np.column_stack([mvals[k] for k in eval_keys])     # calibrated, for corr
    neg = expand_scorers({**mvals, **{b: pool[b].to_numpy(np.float64) for b in bases}},
                         combs, keys, tag, rtag)

    # -- 3. positives + old-testneg: union-aligned tables (shared rows = paired) -
    base_df = None
    if bases:
        bp = _resolve(args.baseline_scores)
        if not bp.exists():
            raise SystemExit(f"[145] FATAL: {bp} missing — run "
                             f"`python 113_thresholds_ci_evt.py --prep-baselines` first")
        base_df = pd.read_parquet(bp)
    pos, old = {}, None
    for sp in POS_SPLITS + ("testneg",):
        ids, _, vals = aligned_split(reg, sp, base_df)
        d = expand_scorers(vals, combs, keys, tag, rtag)
        if sp == "testneg":
            old = d
        else:
            pos[sp] = d
        print(f"[145] {sp}: {len(ids):,} aligned rows x {len(d)} scorers")

    # -- 4. bricks + ONE shared run_suite over every scorer ----------------------
    bricks = None
    mf = _resolve(args.manifest)
    if mf.exists():
        man = pd.read_parquet(mf)[["row_id", "brick"]]
        man["row_id"] = man["row_id"].astype(str)
        mb = pool[["row_id"]].merge(man, on="row_id", how="left")
        assert mb["brick"].notna().all(), "pool rows missing from manifest (brick map)"
        bricks = mb["brick"].to_numpy()
        print(f"[145] brick-block sensitivity enabled: {pd.unique(bricks).size:,} bricks")
    else:
        print(f"[145] WARNING: {mf} not found -> no brick-block sensitivity")
    R = T113.run_suite(neg, pos, old, fprs, args.reps, args.evt_reps, rng,
                       baseline="baseline_meta", bricks=bricks)

    # -- 5. operating-points CSV (113 schema + roster tag) -----------------------
    kinds, roster_of = {}, {}
    for k in reg:
        kinds[k] = "member"
        roster_of[k] = ("both" if k in eval_keys and k in ref_keys
                        else tag if k in eval_keys else rtag)
    for cn in COMBINERS:
        kinds[f"{tag}:{cn}"], roster_of[f"{tag}:{cn}"] = "combiner", tag
        kinds[f"{rtag}:{cn}"], roster_of[f"{rtag}:{cn}"] = "combiner_ref", rtag
    for k in eval_keys:
        s = f"{tag}:avg_wo_{k}"
        if s in neg:
            kinds[s], roster_of[s] = "loo_average", tag
    for b in bases:
        kinds[b], roster_of[b] = "baseline", "baseline"
    df = T113.rows_from_results(R, kinds, fprs, POS_SPLITS)
    df.insert(2, "roster", df["scorer"].map(roster_of))
    out_csv = Path(f"{args.out_prefix}_operating_points.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    T113.print_table(df, fprs, POS_SPLITS)

    # -- 6. verdict: paired deltas, admission, correlation, the SHIP gate --------
    def paired(a, b, sp, f):
        db = R[a]["rec_boot"][sp][f] - R[b]["rec_boot"][sp][f]
        d = {"delta": float(R[a]["rec"][sp][f] - R[b]["rec"][sp][f]),
             "lo": float(np.percentile(db, 2.5)), "hi": float(np.percentile(db, 97.5))}
        d["excludes_zero"] = bool(d["lo"] > 0.0 or d["hi"] < 0.0)
        d["positive_and_excludes_zero"] = bool(d["lo"] > 0.0)
        return d

    metrics = {}
    for sp in POS_SPLITS:
        for f in fprs:
            md = {"v2_rec": float(R[flag]["rec"][sp][f]),
                  "v1_rec": float(R[refc]["rec"][sp][f]),
                  "delta_v2_vs_v1": paired(flag, refc, sp, f)}
            if "baseline_meta" in R:
                md["meta_rec"] = float(R["baseline_meta"]["rec"][sp][f])
                md["delta_v2_vs_meta"] = paired(flag, "baseline_meta", sp, f)
            metrics[f"{sp}@{f:g}"] = md
    print(f"\n[verdict] flagship = {flag}, reference = {refc} (paired per-rep deltas):")
    for key, md in metrics.items():
        d1 = md["delta_v2_vs_v1"]
        line = (f"[verdict]   {key:18s} v2 {md['v2_rec']:.3f} v1 {md['v1_rec']:.3f} "
                f"d(v2-v1) {d1['delta']:+.4f} [{d1['lo']:+.4f},{d1['hi']:+.4f}]")
        if "delta_v2_vs_meta" in md:
            d2 = md["delta_v2_vs_meta"]
            line += f"  d(v2-meta) {d2['delta']:+.4f} [{d2['lo']:+.4f},{d2['hi']:+.4f}]"
        print(line)

    fA = args.admission_fpr
    withk, admission = f"{tag}:average", {}
    for k in eval_keys:
        wo = f"{tag}:avg_wo_{k}"
        if wo not in R:
            continue
        ent = {"fpr": fA, "combiner": "average"}
        for sp in POS_SPLITS:
            pdl = paired(withk, wo, sp, fA)
            ent[sp] = {**pdl, "rec_with": float(R[withk]["rec"][sp][fA]),
                       "rec_without": float(R[wo]["rec"][sp][fA]),
                       "helps": bool(pdl["lo"] > 0.0), "hurts": bool(pdl["hi"] < 0.0)}
        admission[k] = ent
    print(f"\n[admission] per-member marginal (average ensemble WITH - WITHOUT, "
          f"paired, @FPR {fA:g}):")
    for k, ent in admission.items():
        for sp in POS_SPLITS:
            e = ent[sp]
            v = "helps" if e["helps"] else "HURTS" if e["hurts"] else "neutral"
            print(f"[admission]   {k:24s} {sp:10s} {e['delta']:+.4f} "
                  f"[{e['lo']:+.4f},{e['hi']:+.4f}] -> {v}")

    corr = {"members": list(eval_keys),
            "pearson": np.round(E.score_correlation(Pe, "pearson"), 4).tolist(),
            "spearman": np.round(E.score_correlation(Pe, "spearman"), 4).tolist()}

    cells = {}
    for sp in POS_SPLITS:
        for f, rule in ((1e-3, "ci_excludes_zero"), (1e-2, "point")):
            pdl = paired(flag, refc, sp, f)
            ok = pdl["lo"] > 0.0 if rule == "ci_excludes_zero" else pdl["delta"] > 0.0
            cells[f"{sp}@{f:g}"] = {"rule": rule, **pdl, "pass": bool(ok)}
    n_pass = int(sum(c["pass"] for c in cells.values()))
    ship = bool(n_pass >= 3)
    gate = {"criterion": GATE_TEXT, "flagship": flag, "reference": refc,
            "cells": cells, "n_pass": n_pass, "n_cells": len(cells), "ship": ship}
    print(f"\n[gate] {GATE_TEXT}")
    for key, cl in cells.items():
        print(f"[gate]   {key:18s} rule={cl['rule']:16s} d={cl['delta']:+.4f} "
              f"[{cl['lo']:+.4f},{cl['hi']:+.4f}] -> {'PASS' if cl['pass'] else 'fail'}")
    print(f"[gate] cells passed: {n_pass}/{len(cells)} -> "
          f"{'SHIP' if ship else 'NO-SHIP'}")

    verdict = {"seed": args.seed, "reps": args.reps, "evt_reps": args.evt_reps,
               "fprs": list(fprs), "pool_scores": args.pool_scores,
               "n_pool_used": int(len(pool)),
               "rosters": {t: rosters[t] for t in (tag, rtag)},
               "members": {tag: eval_keys, rtag: ref_keys},
               "baselines_on_pool": bases,
               "calibration": cal_report, "combiner_determinism": det,
               "evt": {s: R[s]["evt_info"] for s in R},
               "metrics": metrics, "admission": admission,
               "correlation": corr, "gate": gate}
    out_json = Path(f"{args.out_prefix}_verdict.json")
    out_json.write_text(json.dumps(verdict, indent=2, default=float))

    if write_pool_combined:
        pc = pd.DataFrame({"row_id": pool["row_id"]})
        for s in neg:
            if kinds.get(s) in ("combiner", "combiner_ref"):
                pc[s.replace(":", "_")] = neg[s]
        pf = Path(f"{args.out_prefix}_pool_combined.parquet")
        pc.to_parquet(pf, index=False)
        print(f"[145] wrote {pf} (combiner pool scores; 114 purity-audit input)")
    print(f"[145] wrote {out_csv} + {out_json} ({time.time() - t0:.1f}s)")
    return {"df": df, "R": R, "verdict": verdict,
            "eval_keys": eval_keys, "ref_keys": ref_keys}


# ===== modes =====================================================================

def make_default_roster(args):
    """Emit the v1 6-member roster: --roster (the v2 starting point the
    orchestrator edits per the 122/132/141 gates) + the frozen --ref-roster."""
    V2.mkdir(parents=True, exist_ok=True)
    for path, what in ((_resolve(args.roster), "v2 starting point (EDIT per gates)"),
                       (_resolve(args.ref_roster), "frozen v1 reference")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(V1_ROSTER, indent=2) + "\n")
        print(f"[145] wrote {path} ({len(V1_ROSTER)} members; {what})")
    for e in V1_ROSTER:
        ok = _resolve(e["scores_parquet"]).exists()
        print(f"[145]   {e['name']:12s} {e['scores_parquet']:42s} "
              f"pool_column={e['pool_column']:18s} parquet={'OK' if ok else 'MISSING'}")
    return 0


def main_real(args, fprs):
    rng = np.random.default_rng(args.seed)
    eval_entries = load_roster_file(_resolve(args.roster), None, "--roster")
    ref_entries = load_roster_file(_resolve(args.ref_roster), V1_ROSTER, "--ref-roster")
    run_eval(args, fprs, {args.tag: eval_entries, args.ref_tag: ref_entries}, rng)
    return 0


def synthetic_check(args, fprs):
    """Full-path machinery check (seed 2026): 5 synthetic members over a latent
    lens score — a correlated trio (shared noise), a HARMFUL noisy member and a
    known-BETTER sharp member — written to real parquet/roster files and pushed
    through run_eval twice (v2 = trio+noisy+sharp vs v1 = trio+noisy, then the
    rosters swapped). Asserts the admission logic (sharp helps / noisy hurts,
    CIs exclude 0), the correlation ordering, the paired v2-vs-v1 delta
    direction in both runs and the SHIP gate in both directions."""
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    tmp = V2 / "synth145"
    tmp.mkdir(parents=True, exist_ok=True)
    MEMBERS = ("m_corr1", "m_corr2", "m_corr3", "m_noisy", "m_sharp")

    def scores_of(z, r):
        g = r.standard_normal(z.size)                       # shared -> correlated trio
        return {"m_corr1": z + 0.05 * g + 0.04 * r.standard_normal(z.size),
                "m_corr2": z + 0.05 * g + 0.04 * r.standard_normal(z.size),
                "m_corr3": z + 0.05 * g + 0.04 * r.standard_normal(z.size),
                "m_noisy": z + 0.30 * r.standard_normal(z.size),   # harmful
                "m_sharp": z + 0.01 * r.standard_normal(z.size),   # known better
                "baseline_effnet": z + 0.20 * r.standard_normal(z.size),
                "baseline_meta": z + 0.15 * r.standard_normal(z.size)}

    split_def = {"val": (6000, 1500), "testneg": (5000, 0),
                 "storfer": (0, 2500), "inchausti": (0, 2000)}
    frames, bframes = {m: [] for m in MEMBERS}, []
    for sp, (nn, npos) in split_def.items():
        z = np.r_[rng.beta(0.5, 8.0, nn), rng.beta(6.0, 3.0, npos)]
        y = np.r_[np.zeros(nn, int), np.ones(npos, int)]
        ids = np.array([f"{sp[:2].upper()}{i:06d}" for i in range(z.size)])
        sc = scores_of(z, rng)
        for m in MEMBERS:
            frames[m].append(pd.DataFrame({"split": sp, "row_id": ids, "label": y,
                                           "p": sc[m]}))
        if sp != "val":
            bframes.append(pd.DataFrame({"split": sp, "row_id": ids, "label": y,
                                         "baseline_effnet": sc["baseline_effnet"],
                                         "baseline_meta": sc["baseline_meta"]}))
    for m in MEMBERS:                       # NO pc column -> the NEW-member path
        pd.concat(frames[m], ignore_index=True).to_parquet(
            tmp / f"scores_member_{m}.parquet", index=False)
    pd.concat(bframes, ignore_index=True).to_parquet(
        tmp / "baseline_scores.parquet", index=False)

    NPOOL = 200_000
    zp = rng.beta(0.5, 8.0, NPOOL)
    ids = np.array([f"P{i:07d}" for i in range(NPOOL)])
    sc = scores_of(zp, rng)
    sc["m_noisy"][rng.choice(NPOOL, 40, replace=False)] = np.nan    # finite filter
    pd.DataFrame({"row_id": ids, **{f"raw_{m}": sc[m] for m in MEMBERS[:4]},
                  "baseline_effnet": sc["baseline_effnet"],
                  "baseline_meta": sc["baseline_meta"]}).to_parquet(
        tmp / "pool_main.parquet", index=False)
    pd.DataFrame({"row_id": ids, "ok": True, "raw_m_sharp": sc["m_sharp"]}
                 ).to_parquet(tmp / "pool_extra.parquet", index=False)  # merge path
    pd.DataFrame({"row_id": ids,
                  "brick": [f"b{i // 100:05d}" for i in range(NPOOL)]}).to_parquet(
        tmp / "manifest.parquet", index=False)

    def entry(m):
        return {"name": m, "scores_parquet": str(tmp / f"scores_member_{m}.parquet"),
                "pool_column": f"raw_{m}"}

    base4 = [entry(m) for m in MEMBERS[:4]]
    plus5 = base4 + [entry("m_sharp")]
    a = argparse.Namespace(**vars(args))
    a.pool_scores = f"{tmp / 'pool_main.parquet'},{tmp / 'pool_extra.parquet'}"
    a.baseline_scores = str(tmp / "baseline_scores.parquet")
    a.manifest = str(tmp / "manifest.parquet")
    a.reps, a.evt_reps = min(args.reps, 2000), min(args.evt_reps, 100)
    print(f"[synth] pool N={NPOOL:,}; members={list(MEMBERS)}; reps={a.reps:,}; "
          f"seed={args.seed}; files under {tmp}")

    print(f"\n[synth] ===== run A: {a.tag} = trio+noisy+SHARP vs {a.ref_tag} = trio+noisy =====")
    a.out_prefix = str(tmp / "fwd")
    resA = run_eval(a, fprs, {a.tag: plus5, a.ref_tag: base4},
                    np.random.default_rng(args.seed), write_pool_combined=False)
    print(f"\n[synth] ===== run B: rosters SWAPPED (gate must NOT ship) =====")
    a.out_prefix = str(tmp / "rev")
    resB = run_eval(a, fprs, {a.tag: base4, a.ref_tag: plus5},
                    np.random.default_rng(args.seed), write_pool_combined=False)

    print()
    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print(f"[synth] {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    adm = resA["verdict"]["admission"]
    check("admission: known-better member helps (CI > 0, both splits)",
          all(adm["m_sharp"][sp]["lo"] > 0 for sp in POS_SPLITS),
          ", ".join(f"{sp} {adm['m_sharp'][sp]['delta']:+.4f} "
                    f"[{adm['m_sharp'][sp]['lo']:+.4f},{adm['m_sharp'][sp]['hi']:+.4f}]"
                    for sp in POS_SPLITS))
    check("admission: harmful noisy member hurts (CI < 0, both splits)",
          all(adm["m_noisy"][sp]["hi"] < 0 for sp in POS_SPLITS),
          ", ".join(f"{sp} {adm['m_noisy'][sp]['delta']:+.4f} "
                    f"[{adm['m_noisy'][sp]['lo']:+.4f},{adm['m_noisy'][sp]['hi']:+.4f}]"
                    for sp in POS_SPLITS))
    names = resA["verdict"]["correlation"]["members"]
    Pn = np.asarray(resA["verdict"]["correlation"]["pearson"])
    Sn = np.asarray(resA["verdict"]["correlation"]["spearman"])
    i1, i2 = names.index("m_corr1"), names.index("m_corr2")
    i4, i5 = names.index("m_noisy"), names.index("m_sharp")
    check("correlation: shared-noise twin pair >> noisy-sharp pair (+0.2, both methods)",
          Pn[i1, i2] > Pn[i4, i5] + 0.2 and Sn[i1, i2] > Sn[i4, i5] + 0.2,
          f"pearson {Pn[i1, i2]:.3f} vs {Pn[i4, i5]:.3f}; "
          f"spearman {Sn[i1, i2]:.3f} vs {Sn[i4, i5]:.3f}")
    check("delta direction fwd: v2-v1 CI > 0 @1e-3 (both splits)",
          all(resA["verdict"]["metrics"][f"{sp}@0.001"]["delta_v2_vs_v1"]["lo"] > 0
              for sp in POS_SPLITS),
          ", ".join(f"{sp} [{resA['verdict']['metrics'][f'{sp}@0.001']['delta_v2_vs_v1']['lo']:+.4f},"
                    f"{resA['verdict']['metrics'][f'{sp}@0.001']['delta_v2_vs_v1']['hi']:+.4f}]"
                    for sp in POS_SPLITS))
    check("v2 beats baseline_meta @1e-3 (CI > 0, both splits)",
          all(resA["verdict"]["metrics"][f"{sp}@0.001"]["delta_v2_vs_meta"]["lo"] > 0
              for sp in POS_SPLITS),
          ", ".join(f"{sp} [{resA['verdict']['metrics'][f'{sp}@0.001']['delta_v2_vs_meta']['lo']:+.4f},"
                    f"{resA['verdict']['metrics'][f'{sp}@0.001']['delta_v2_vs_meta']['hi']:+.4f}]"
                    for sp in POS_SPLITS))
    gA, gB = resA["verdict"]["gate"], resB["verdict"]["gate"]
    check("gate fwd: better roster SHIPs 4/4",
          gA["ship"] and gA["n_pass"] == 4, f"n_pass={gA['n_pass']}/4 ship={gA['ship']}")
    check("delta direction rev: v2-v1 CI < 0 @1e-3 (both splits)",
          all(resB["verdict"]["metrics"][f"{sp}@0.001"]["delta_v2_vs_v1"]["hi"] < 0
              for sp in POS_SPLITS),
          ", ".join(f"{sp} [{resB['verdict']['metrics'][f'{sp}@0.001']['delta_v2_vs_v1']['lo']:+.4f},"
                    f"{resB['verdict']['metrics'][f'{sp}@0.001']['delta_v2_vs_v1']['hi']:+.4f}]"
                    for sp in POS_SPLITS))
    check("gate rev: swapped rosters do NOT ship (0/4)",
          (not gB["ship"]) and gB["n_pass"] == 0,
          f"n_pass={gB['n_pass']}/4 ship={gB['ship']}")

    n_ok = sum(checks)
    print(f"\n[synth] {n_ok}/{len(checks)} checks passed ({time.time() - t0:.1f}s) -> "
          f"{'SYNTHETIC-CHECK PASS' if n_ok == len(checks) else 'SYNTHETIC-CHECK FAIL'}")
    return 0 if n_ok == len(checks) else 1


# ===== cli =======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster", default=str(V2 / "roster_v2.json"),
                    help="v2 roster JSON: list of {name, scores_parquet, pool_column}")
    ap.add_argument("--ref-roster", default=str(V2 / "roster_v1.json"),
                    help="reference roster JSON (falls back to the built-in v1 6-member roster)")
    ap.add_argument("--pool-scores", default=str(V2 / "scores_negeval_pool.parquet"),
                    help="comma-list of NegEval-1M raw-score parquets, merged on row_id")
    ap.add_argument("--baseline-scores", default=str(V2 / "baseline_scores_v1splits.parquet"),
                    help="per-row Stage-D baseline scores on the v1 splits (113 --prep-baselines)")
    ap.add_argument("--manifest", default=str(V2 / "negeval_manifest.parquet"),
                    help="pool manifest (row_id->brick map for the block sensitivity)")
    ap.add_argument("--reference-combiner", default="average", choices=COMBINERS,
                    help="reference-roster combiner the gate compares against (v1 flagship)")
    ap.add_argument("--flagship-combiner", default="average", choices=COMBINERS,
                    help="eval-roster combiner shipped as the v2 flagship")
    ap.add_argument("--admission-fpr", type=float, default=1e-3,
                    help="FPR for the member-admission marginals")
    ap.add_argument("--tag", default="v2", help="eval roster tag (scorer prefix)")
    ap.add_argument("--ref-tag", default="v1", help="reference roster tag")
    ap.add_argument("--reps", type=int, default=10_000, help="bootstrap reps")
    ap.add_argument("--evt-reps", type=int, default=1_000, help="EVT parametric-bootstrap reps")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--fprs", default="0.01,0.001,0.0001", help="comma-separated target FPRs")
    ap.add_argument("--out-prefix", default=str(V2 / "ensemble_v2"),
                    help="output prefix -> _operating_points.csv / _verdict.json "
                         "/ _pool_combined.parquet")
    ap.add_argument("--make-default-roster", action="store_true",
                    help="write the v1 6-member roster to --roster and --ref-roster, then exit")
    ap.add_argument("--synthetic-check", action="store_true",
                    help="run the full path on synthetic members + assert admission/gate logic")
    args = ap.parse_args()
    fprs = tuple(float(x) for x in args.fprs.split(","))
    if args.make_default_roster:
        return make_default_roster(args)
    if args.synthetic_check:
        return synthetic_check(args, fprs)
    return main_real(args, fprs)


if __name__ == "__main__":
    raise SystemExit(main())
