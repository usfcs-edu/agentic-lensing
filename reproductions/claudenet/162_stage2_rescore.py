#!/usr/bin/env python3
"""162_stage2_rescore.py — Phase 160: stage-1 thresholding -> survivors
manifest, then the stage-2 FULL-v2 rescore of the survivors (both halves run
LOCALLY, CPU; the GPU work between them — survivors-only 111 extraction +
112/112b scoring — runs on Perlmutter and this script prints those commands).

  --make-survivors  (LOCAL, after the 161 stage-1 parquets are rsynced back)
    Concatenates every 161 output matching --stage1-scores (default glob is
    MODE-AWARE: <sweep-dir>/stage1/stage1_*_<stage1-mode>_s*.parquet, so a dir
    holding both members and student outputs never mixes them), audits
    coverage against the 160 part manifests (--manifests; missing rows abort
    unless --allow-partial), then selects stage-1 survivors.

    SCORE-SPACE COHERENCE (the contract): the 113/145 operating-point
    thresholds in --operating-points live in ISOTONIC-CALIBRATED space (113
    computes them on calibrator.transform(raw)), while 161/112 emit RAW
    probabilities — so the raw stage-1 columns are first transformed with the
    per-member calibrators from the PERSISTED 145 fits (--fits; FATAL if a
    needed calibrator is absent) and thresholding/margins/p_stage1 all happen
    in calibrated space, where thr, evt_thr and 164's EVT prediction are
    valid. Before any selection, the thresholds are verified against
    --stage1-fpr on the NegEval pool (--negeval-scores): each thr is first
    SNAPPED onto the calibrated pool grid (isotonic-plateau ULP guard, see
    check_negeval_fpr), then a realized pass fraction below target/2 aborts
    (recall loss = the space-mismatch failure mode) while a plateau-tie
    overshoot is recall-safe -> warned + recorded (the union rule and
    --survivor-budget cap the volume); --skip-negeval-check bypasses when the
    pool is absent (NOT recommended: it also skips the snap).
      * --stage1-mode student : p_stage1 = calibrated --student-col; survive
        iff p_stage1 >= thr where thr is the EVT-cross-checked --stage1-fpr
        (1e-4) threshold for --stage1-scorer in --operating-points
        (data/v2/operating_points_v2.csv). PREREQUISITE: a 113-style run over
        the student's NegEval scores must have appended that CSV row + 'evt'
        block, and the persisted 145 fits must hold the student's calibrator;
        otherwise this mode FATALs. --stage1-thr overrides with an explicit
        threshold — applied in calibrated space when the student's calibrator
        exists, else in RAW space (recorded; the CSV's calibrated-space
        evt_thr cross-check is then inapplicable and disabled).
      * --stage1-mode members : per-member 1e-4 thresholds thr_m from the same
        CSV; survive iff ANY calibrated member pc_m >= thr_m (the recall-safe
        UNION rule; nominal FPR <= n_members * fpr by the union bound,
        recorded as such). p_stage1 = max_m pc_m; margin = max_m (pc_m - thr_m).
    --survivor-budget (150,000) is the operational guard: if more rows pass,
    only the top-budget by margin are kept (cutoff recorded; the PRE-budget
    pass count and rate are persisted as n_pass_prebudget /
    realized_pass_rate — 164's consistency check consumes those, not the
    capped n_survivors). Rows with any non-finite stage-1 score (failed
    cutouts) are not selectable; n_swept counts the finite-scored rows.
    RAM BUDGET (documented, deliberate): the 17.3M-row concat holds the
    stage-1 frame (row_id + score cols only), the 160 manifest frame, the
    float64 score matrix (+ its calibrated copy) and a handful of masks
    simultaneously — ~10-15 GB peak on the local box; read_glob loads ONLY
    the needed columns to keep it there. Writes (all under data/v2/sweep/):
      survivors_manifest.parquet   111's manifest schema [row_id, RA, DEC,
                                   footprint, brick] — THE stage-2 extraction
                                   manifest (and the 162/163/164 join spine)
      stage1_survivors.parquet     row_id, footprint, raw stage-1 cols,
                                   calibrated cal_* cols, p_stage1
                                   (calibrated), stage1_margin
      stage1_summary.json          n_swept / n_survivors / n_pass_prebudget /
                                   realized_pass_rate / stage1_thr /
                                   stage1_scorer / stage1_fpr / per_member_fpr
                                   / threshold_space / negeval_fpr_check
                                   (164's consistency-check contract) +
                                   per-footprint counts (165's totals feed) +
                                   budget + per-member thresholds
    then prints the Perlmutter command block (orchestrator submits): 111 on
    the survivors manifest (101px grz; ~0.1-0.9% of 17.3M -> minutes), 112
    with --aion score (members + baselines + the degraded-AION member, merged
    into ONE parquet) + --extra-ckpt-dir for any v2-roster extras the
    persisted 145 fits need (read from <--fits>/meta.json when present), and
    with --native-griz the optional 160px-griz extraction + 131 embedding
    block for a native-AION member (only if the 132 gate shipped native —
    its scores parquet then joins --survivor-scores as another comma entry).

  --merge  (LOCAL, after the survivor scores parquet(s) are rsynced back)
    Merges the comma-listed --survivor-scores on row_id (145.load_pool), then
    applies the PERSISTED 145 calibrate+combine path (--fits, written by the
    145 fitting run; 145.load_fits/apply_fits — the exact objects, verified
    against stored reference outputs) and writes
      stage2_scores.parquet  [row_id, RA, DEC, footprint, brick, p_stage1,
                              <raw member/baseline cols>, pc_<member>...,
                              <tag>_<combiner>..., p_final]
    where p_final = the persisted flagship combiner column (--p-final-col
    overrides) — the SAME column family 145 wrote for NegEval-1M in
    ensemble_v2_pool_combined.parquet, so 165's calibration and test scores
    share one code path by construction. stage1_summary.json gains the
    stage-2 counts. 163/164/165 read stage2_scores.parquet.

    python 162_stage2_rescore.py --make-survivors --stage1-mode members
    # LEAN v2 roster (161 --stage1 lean outputs; members-mode UNION rule with
    # the v2lean fits/CSV — the defaults above are v1-era):
    python 162_stage2_rescore.py --make-survivors --stage1-mode members \\
        --stage1-scores 'data/v2/sweep/stage1/stage1_*_lean_s*.parquet' \\
        --members effnet_B,effnet_B3_hard,effnet_S2_hard,resnet46_C_hard,zoobot_N \\
        --operating-points data/v2/ensemble_v2lean_operating_points.csv \\
        --fits data/v2/ensemble_v2lean_fits \\
        --negeval-scores data/v2/scores_negeval_pool_full.parquet,data/v2/scores_negeval_member_variants.parquet
    python 162_stage2_rescore.py --merge \\
        --survivor-scores data/v2/sweep/survivor_scores.parquet
"""
from __future__ import annotations

import argparse
import glob as globlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

V2 = C.DATA / "v2"
SWEEP = V2 / "sweep"
MEMBERS_DEFAULT = "shielded_A,effnet_B,effnet_B3,effnet_S2,resnet46_C"


def atomic_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


def lookup_thr(op_csv: Path, scorer: str, fpr: float):
    """(resolved scorer name, empirical thr, evt_thr cross-check) from the 113
    operating-points CSV; tolerant of the member_ prefix either way."""
    assert op_csv.exists(), f"--operating-points {op_csv} missing (run 113/145 first)"
    df = pd.read_csv(op_csv)
    for name in dict.fromkeys((scorer, scorer.removeprefix("member_"),
                               f"member_{scorer}")):
        hit = df[(df.scorer == name) & np.isclose(df.fpr.astype(float), fpr)]
        if len(hit):
            row = hit.iloc[0]
            evt = row.get("evt_thr", float("nan"))
            evt = float(evt) if np.isfinite(float(evt)) else None
            return name, float(row["thr"]), evt
    raise SystemExit(f"[162] FATAL: no row (scorer={scorer!r}, fpr={fpr:g}) in "
                     f"{op_csv} (scorers there: {sorted(df.scorer.unique())})")


def read_glob(pattern: str, what: str, columns=None) -> tuple[pd.DataFrame, int]:
    """Concat a parquet glob, reading ONLY `columns` when given (the 17.3M-row
    RAM-budget guard); a missing column aborts with the available schema."""
    files = sorted(globlib.glob(pattern))
    assert files, f"no {what} match {pattern!r}"
    try:
        parts = [pd.read_parquet(f, columns=columns) for f in files]
    except Exception as e:
        import pyarrow.parquet as pq
        have = pq.read_schema(files[0]).names
        raise SystemExit(f"[162] FATAL: reading {what} columns {columns} failed "
                         f"({e}); {files[0]} has {have}")
    df = pd.concat(parts, ignore_index=True)
    df["row_id"] = df["row_id"].astype(str)
    print(f"[162] {what}: {len(files)} files -> {len(df):,} rows")
    assert df.row_id.is_unique, (f"{what}: duplicate row_ids across {pattern!r} "
                                 f"(overlapping shard ranges / stale files?)")
    return df, len(files)


def load_calibrators(fits_path) -> dict:
    """Persisted 145 fits -> {pool_column: isotonic calibrator}. The 113/145
    operating-point thresholds live in calibrated space, so stage-1 raw scores
    MUST go through these before any threshold comparison."""
    M145 = C._load("cn_162_145", C.ROOT / "145_refit_ensemble_v2.py")
    fp = Path(fits_path)
    f = fp / "ensemble_fits.joblib" if fp.is_dir() else fp
    if not f.exists():
        raise SystemExit(
            f"[162] FATAL: persisted 145 fits missing ({f}) — the operating-point "
            f"thresholds are in ISOTONIC-CALIBRATED space and raw 161/112 scores "
            f"cannot be thresholded against them. Run the fitting 145 first "
            f"(it writes <out-prefix>_fits/), or pass --fits.")
    fits, _ = M145.load_fits(fits_path)
    return {fits["pool_columns"][k]: fits["calibrators"][k] for k in fits["keys"]}


def check_negeval_fpr(args, cols, thrs, cals) -> dict:
    """Verify the (calibrated-space) thresholds against the NegEval pool BEFORE
    any sweep selection — the guard against the raw-vs-calibrated space mismatch
    (realized 3e-6 instead of 1e-4). Two realities of isotonic calibration are
    handled here (measured on the v2lean fits, 2026-06):
      * SNAP: the CSV thr is a type-7 quantile of the calibrated pool, but the
        CSV float roundtrip + calibrator re-application leave it ULP-fragile
        around the isotonic plateaus (zoobot_N's stored thr sits 2 ULP above
        its top plateau -> a naive >= passes NOTHING, silently disabling the
        member). Each thr is therefore snapped to the smallest calibrated pool
        value >= thr*(1-1e-9) — the operating point the quantile actually
        selected. `thrs` is updated IN PLACE; the selection and the summary use
        the snapped values.
      * TIES: a plateau straddling the target quantile makes the realized
        >=-FPR jump discretely (effnet_B: 5.6e-4 or 3e-6, nothing near 1e-4).
        Overshoot is RECALL-SAFE for stage-1 (the union rule + --survivor-budget
        cap the volume) -> WARN + record. Undershoot below target/2 is recall
        LOSS (the original space-mismatch failure mode) -> FATAL."""
    specs = [s for s in str(args.negeval_scores).split(",") if s]
    target = args.stage1_fpr
    absent = [s for s in specs if not (Path(s).exists() or globlib.glob(s))]
    if absent:
        msg = (f"{', '.join(absent)} missing — cannot verify the thresholds "
               f"reproduce fpr {target:g} on the NegEval pool")
        if args.skip_negeval_check:
            print(f"[162] WARNING: {msg} (--skip-negeval-check)")
            return {"verdict": "SKIPPED", "note": msg}
        raise SystemExit(f"[162] FATAL: {msg} (point --negeval-scores at the "
                         f"raw 112 NegEval parquet(s), or pass --skip-negeval-check)")
    if len(specs) == 1:
        pool, _ = read_glob(specs[0], "NegEval pool scores",
                            columns=["row_id", *[c for c in cols]])
    else:
        # comma-list (the lean roster's raw NegEval scores live in TWO 112
        # outputs, e.g. scores_negeval_pool_full + scores_negeval_member_
        # variants): merge on row_id, each needed column taken from the FIRST
        # listed parquet whose schema has it (mirrors 145.load_pool)
        import pyarrow.parquet as pq
        pool, need = None, list(cols)
        for s in specs:
            have = [c for c in need
                    if c in pq.read_schema(sorted(globlib.glob(s))[0]).names]
            if not have:
                continue
            df, _ = read_glob(s, f"NegEval pool scores [{Path(s).name}]",
                              columns=["row_id", *have])
            pool = df if pool is None else pool.merge(
                df, on="row_id", how="inner", validate="one_to_one")
            need = [c for c in need if c not in have]
        if need:
            raise SystemExit(f"[162] FATAL: --negeval-scores comma-list {specs} "
                             f"lacks stage-1 columns {need}")
    rep, bad = {}, []
    for c in cols:
        v = pool[c].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        vc = cals[c].transform(v) if cals.get(c) is not None else v
        thr_csv = thrs[c]
        above = vc[vc >= thr_csv * (1.0 - 1e-9)]
        if above.size:                       # SNAP (see docstring)
            t_eff = float(above.min())
            if t_eff != thr_csv:
                print(f"[162] negeval-snap {c}: thr {thr_csv!r} -> {t_eff!r} "
                      f"(smallest calibrated pool value at/above the CSV "
                      f"quantile; isotonic-plateau ULP guard)")
                thrs[c] = t_eff
        frac = float((vc >= thrs[c]).mean())
        slack = max(0.5 * target, 10.0 / max(len(v), 1))
        undershoot = frac < target - slack
        overshoot = frac > target + slack
        rep[c] = {"n_pool": int(len(v)), "realized_fpr": frac,
                  "target_fpr": target, "thr_csv": float(thr_csv),
                  "thr_effective": float(thrs[c]),
                  "ok": not undershoot, "overshoot": bool(overshoot)}
        verdict = ("UNDERSHOOT (recall loss)" if undershoot else
                   "OVERSHOOT (plateau ties; recall-safe, budget-capped)"
                   if overshoot else "OK")
        print(f"[162] negeval-fpr {c:28s} realized {frac:.3e} vs target "
              f"{target:g} -> {verdict}")
        if undershoot:
            bad.append(c)
    if bad:
        raise SystemExit(
            f"[162] FATAL: thresholds pass LESS than half the target fpr "
            f"{target:g} on the NegEval pool for {bad} — score-space mismatch "
            f"(raw vs calibrated) or a stale operating-points CSV would lose "
            f"recall. Refusing to run the sweep selection on invalid thresholds.")
    rep["verdict"] = "PASS"
    return rep


# ===== --make-survivors ==========================================================

def fits_extra_note(fits_dir: Path) -> str:
    """Which pool columns the persisted 145 fits will demand at --merge time
    (read from meta.json; tells the orchestrator which --extra-ckpt-dir /
    extra parquets the survivors 112 run must produce)."""
    mp = Path(fits_dir) / "meta.json"
    if not mp.exists():
        return (f"#   ({mp} not found yet — run the fitting 145 before --merge; "
                f"its meta.json lists the pool columns needed)")
    cols = sorted(set(json.loads(mp.read_text())["pool_columns"].values()))
    return f"#   persisted fits need pool columns: {', '.join(cols)}"


def print_perlmutter_block(args, n_surv: int):
    sm = args.survivors_manifest
    print(f"\n[162] next (PERLMUTTER, orchestrator submits; rsync {sm} up first):")
    print(f"# 1. survivors-only extraction ({n_surv:,} rows -> "
          f"~{n_surv * 3 * 101 * 101 * 4 / 1e9:.1f} GB, minutes on 60 workers):")
    print(f"sbatch --export=ALL,CMD='python 111_extract_cutouts_cfs.py "
          f"--manifest data/v2/sweep/survivors_manifest.parquet "
          f"--out-root $SCRATCH/claudenet/cutouts/sweep_survivors "
          f"--size 101 --bands grz --release dr9 --workers 60' nersc/shared_cpu.slurm")
    print(f"# 2. FULL v2 scoring (5 members + baselines + degraded-AION member "
          f"merged into ONE parquet; add --extra-ckpt-dir data/v2/ckpt for "
          f"v2-roster extras):")
    print(fits_extra_note(args.fits))
    print(f"sbatch --export=ALL,CMD='HF_HOME=$SCRATCH/claudenet/hf "
          f"python 112_score_pool.py "
          f"--cutout-root $SCRATCH/claudenet/cutouts/sweep_survivors "
          f"--out $SCRATCH/claudenet/scores/survivor_scores.parquet "
          f"--aion score' nersc/shared_gpu.slurm")
    print(f"# 3. rsync survivor_scores.parquet back to data/v2/sweep/ and run "
          f"`python 162_stage2_rescore.py --merge`")
    if args.native_griz:
        print(f"# OPTIONAL (only if the 132 gate shipped a NATIVE-griz AION "
              f"member in the v2 roster): 160px griz extraction + 131 "
              f"embeddings; probe-score them and join the resulting parquet "
              f"as another --survivor-scores comma entry at --merge:")
        print(f"sbatch --export=ALL,CMD='python 111_extract_cutouts_cfs.py "
              f"--manifest data/v2/sweep/survivors_manifest.parquet "
              f"--out-root $SCRATCH/claudenet/cutouts/sweep_survivors_griz "
              f"--size 160 --bands griz --release dr10 --workers 60' "
              f"nersc/shared_cpu.slurm")
        print(f"sbatch --export=ALL,CMD='HF_HOME=$SCRATCH/claudenet/hf "
              f"python 131_embed_aion_variants.py "
              f"--cutout-root $SCRATCH/claudenet/cutouts/sweep_survivors_griz "
              f"--out-root $SCRATCH/claudenet/emb/sweep_survivors_griz' "
              f"nersc/shared_gpu.slurm")


def make_survivors(args) -> int:
    t0 = time.time()
    Path(args.sweep_dir).mkdir(parents=True, exist_ok=True)
    if args.stage1_mode == "student":
        cols = [args.student_col]
        members = None
    else:
        members = [m for m in args.members.split(",") if m]
        cols = [f"member_{m}" for m in members]
    scored, n_files = read_glob(args.stage1_scores, "stage-1 scores",
                                columns=["row_id", *cols])
    man, _ = read_glob(args.manifests, "160 part manifests",
                       columns=["row_id", "RA", "DEC", "footprint", "brick"])

    # -- coverage audit: every manifest row must have a stage-1 score row -------
    in_scored = man.row_id.isin(pd.Index(scored.row_id))
    n_missing = int((~in_scored).sum())
    if n_missing:
        msg = (f"{n_missing:,}/{len(man):,} manifest rows have NO stage-1 score "
               f"(missing/failed 161 ranges?)")
        if not args.allow_partial:
            raise SystemExit(f"[162] FATAL: {msg} — rerun 161 or pass --allow-partial")
        print(f"[162] WARNING: {msg} (--allow-partial)")
    extra = ~scored.row_id.isin(pd.Index(man.row_id))
    assert not extra.any(), (f"{int(extra.sum()):,} stage-1 rows are not in the "
                             f"part manifests (wrong --manifests glob?)")

    # -- stage-1 columns, calibrators (SPACE COHERENCE), thresholds, margin ------
    # 113/145 thresholds are quantiles of ISOTONIC-CALIBRATED NegEval scores;
    # 161/112 emit raw probabilities -> calibrate BEFORE comparing.
    op_csv = Path(args.operating_points)
    thrs: dict[str, float] = {}
    evts: dict[str, float | None] = {}
    cals: dict[str, object] = {}
    cal_by_col = load_calibrators(args.fits)
    threshold_space = "calibrated (isotonic via persisted 145 fits)"
    if args.stage1_mode == "student":
        cals[cols[0]] = cal_by_col.get(cols[0])
        if args.stage1_thr is not None:
            name, thr, evt = args.stage1_scorer, float(args.stage1_thr), None
            if cals[cols[0]] is None:
                threshold_space = ("raw (explicit --stage1-thr, no calibrator "
                                   "for the student column; CSV thr/evt_thr "
                                   "are calibrated-space and NOT used)")
            print(f"[162] stage-1 thr OVERRIDE: {thr:.6f} ({threshold_space})")
        else:
            if cals[cols[0]] is None:
                raise SystemExit(
                    f"[162] FATAL: persisted fits ({args.fits}) hold no calibrator "
                    f"for {cols[0]!r} — the CSV thresholds are calibrated-space and "
                    f"cannot be applied to raw student scores. Rerun 145 with the "
                    f"student in the roster (after the 113-style NegEval run), or "
                    f"pass an explicit raw-space --stage1-thr.")
            name, thr, evt = lookup_thr(op_csv, args.stage1_scorer, args.stage1_fpr)
        thrs[cols[0]], evts[cols[0]] = thr, evt
        s1_scorer, s1_thr, nominal_fpr = name, thr, args.stage1_fpr
    else:
        for m, cl in zip(members, cols):
            if cal_by_col.get(cl) is None:
                raise SystemExit(
                    f"[162] FATAL: persisted fits ({args.fits}) hold no calibrator "
                    f"for {cl!r} — cannot threshold raw scores against the "
                    f"calibrated-space CSV thresholds. Rerun the fitting 145.")
            cals[cl] = cal_by_col[cl]
            name, thr, evt = lookup_thr(op_csv, m, args.stage1_fpr)
            thrs[cl], evts[cl] = thr, evt
        # union rule: no single scorer/threshold; nominal fpr = union bound
        s1_scorer, s1_thr = None, None
        nominal_fpr = len(cols) * args.stage1_fpr
    for cl in cols:
        ev = f"evt_thr={evts[cl]:.6f}" if evts[cl] is not None else "evt_thr=n/a"
        print(f"[162] thr {cl:28s} = {thrs[cl]:.6f} @ fpr {args.stage1_fpr:g} ({ev})")

    # -- assert the thresholds reproduce the target FPR on the NegEval pool ------
    negeval_check = check_negeval_fpr(args, cols, thrs, cals)

    P = scored[cols].to_numpy(np.float64)
    fin = np.isfinite(P).all(axis=1)
    n_swept = int(fin.sum())
    print(f"[162] {len(scored):,} rows scored, {n_swept:,} finite "
          f"({len(scored) - n_swept:,} failed cutouts excluded)")
    Pc = np.full_like(P, np.nan)                      # calibrated score matrix
    for j, cl in enumerate(cols):
        Pc[fin, j] = (cals[cl].transform(P[fin, j]) if cals.get(cl) is not None
                      else P[fin, j])
    margin = np.full(len(scored), -np.inf)
    margin[fin] = (Pc[fin] - np.array([thrs[c] for c in cols])).max(axis=1)
    p_stage1 = np.full(len(scored), np.nan)
    p_stage1[fin] = Pc[fin].max(axis=1)
    surv = margin >= 0.0
    n_pass = int(surv.sum())
    rate = n_pass / max(n_swept, 1)
    print(f"[162] stage-1 pass: {n_pass:,}/{n_swept:,} = {rate:.3e} "
          f"(nominal fpr {nominal_fpr:g}; known lenses are IN the sweep, so "
          f"the realized rate sits above a pure FPR — 164 checks consistency)")

    budget_applied, cutoff = False, None
    if n_pass > args.survivor_budget:
        order = np.argsort(-margin, kind="stable")[:args.survivor_budget]
        cutoff = float(margin[order[-1]])
        surv = np.zeros(len(scored), bool)
        surv[order] = True
        budget_applied = True
        print(f"[162] --survivor-budget {args.survivor_budget:,} applied: "
              f"margin cutoff {cutoff:+.6f} ({n_pass - args.survivor_budget:,} "
              f"above-threshold rows dropped)")

    out = scored.loc[surv, ["row_id", *cols]].reset_index(drop=True)
    out.insert(1, "p_stage1", p_stage1[surv])
    for j, cl in enumerate(cols):
        out[f"cal_{cl}"] = Pc[surv, j]              # the thresholded values
    out["stage1_margin"] = margin[surv]
    # merge (not set_index().loc) — survivors are <=150k of a 17.3M-row frame
    sm = out[["row_id"]].merge(man, on="row_id", how="left")
    assert not sm["footprint"].isna().any(), "survivor row_id missing from manifests"
    out.insert(2, "footprint", sm["footprint"].to_numpy())
    surv_man = (sm[["row_id", "RA", "DEC", "footprint", "brick"]]
                .sort_values(["footprint", "brick", "row_id"], ignore_index=True))
    atomic_parquet(surv_man, Path(args.survivors_manifest))
    atomic_parquet(out, Path(args.stage1_survivors))
    per_foot_swept = (man.loc[man.row_id.isin(pd.Index(scored.row_id[fin]))]
                      .footprint.value_counts().to_dict())
    per_foot_surv = surv_man.footprint.value_counts().to_dict()

    summary = {
        # -- 164's S1_ALIASES contract -------------------------------------------
        "n_swept": n_swept, "n_survivors": int(surv.sum()),
        "n_pass_prebudget": n_pass,
        "stage1_thr": s1_thr, "stage1_scorer": s1_scorer,
        "stage1_fpr": nominal_fpr,
        # -- extras ----------------------------------------------------------------
        "stage1_mode": args.stage1_mode, "stage1_cols": cols,
        "per_member_fpr": args.stage1_fpr,
        "thresholds": thrs, "evt_thresholds": evts,
        "threshold_space": threshold_space,
        "calibration_fits": str(args.fits),
        "negeval_fpr_check": negeval_check,
        "nominal_fpr_note": ("members mode: union rule over per-member "
                             "thresholds, nominal fpr = n_members * per-member "
                             "fpr (union bound)" if args.stage1_mode == "members"
                             else "single-scorer threshold"),
        "n_rows_total": int(len(scored)), "n_nonfinite": int(len(scored)) - n_swept,
        "n_manifest_rows": int(len(man)), "n_manifest_unscored": n_missing,
        "n_stage1_files": n_files, "stage1_scores_glob": args.stage1_scores,
        "realized_pass_rate": rate,
        "budget": int(args.survivor_budget), "budget_applied": budget_applied,
        "budget_margin_cutoff": cutoff,
        "per_footprint_swept": {k: int(v) for k, v in per_foot_swept.items()},
        "per_footprint_survivors": {k: int(v) for k, v in per_foot_surv.items()},
        "sweep_totals_arg": ",".join(f"{k}={int(v)}"
                                     for k, v in sorted(per_foot_swept.items())),
        "operating_points": str(op_csv),
    }
    sp = Path(args.summary)
    sp.write_text(json.dumps(summary, indent=2))
    print(f"[162] wrote {args.survivors_manifest} "
          f"({len(surv_man):,} rows), {args.stage1_survivors}, {sp}")
    print(f"[162] per-footprint swept (165 --sweep-totals): "
          f"{summary['sweep_totals_arg']}")
    print_perlmutter_block(args, len(surv_man))
    print(f"[162] --make-survivors done ({time.time() - t0:.1f}s)")
    return 0


# ===== --merge ===================================================================

def merge(args) -> int:
    t0 = time.time()
    M145 = C._load("cn_162_145", C.ROOT / "145_refit_ensemble_v2.py")
    fits, combs = M145.load_fits(args.fits)
    tag, flagship = fits["tag"], fits["flagship_combiner"]
    p_final_col = args.p_final_col or f"{tag}_{flagship}"

    paths = [s for s in args.survivor_scores.split(",") if s]
    pool = M145.load_pool(paths)         # merges on row_id, drops 'ok', asserts
    raw_cols = [c for c in pool.columns if c != "row_id"]
    print(f"[162] survivor scores: {len(pool):,} rows x {len(raw_cols)} raw cols")

    applied = M145.apply_fits(fits, combs, pool)     # row_id, pc_*, tag_* cols
    assert p_final_col in applied.columns, \
        (f"--p-final-col {p_final_col!r} not produced by the fits "
         f"(have {[c for c in applied.columns if c != 'row_id']})")

    man = pd.read_parquet(args.survivors_manifest)
    man["row_id"] = man["row_id"].astype(str)
    s1 = pd.read_parquet(args.stage1_survivors,
                         columns=["row_id", "p_stage1", "stage1_margin"])
    s1["row_id"] = s1["row_id"].astype(str)
    df = man.merge(s1, on="row_id", how="inner", validate="one_to_one")
    assert len(df) == len(man), "stage1_survivors does not cover the manifest"

    n0 = len(df)
    df = df.merge(pool, on="row_id", how="inner", validate="one_to_one")
    if len(df) < n0:
        msg = (f"{n0 - len(df):,}/{n0:,} survivors have no stage-2 scores "
               f"(incomplete 112 run?)")
        if not args.allow_partial:
            raise SystemExit(f"[162] FATAL: {msg} — rerun 112 or pass --allow-partial")
        print(f"[162] WARNING: {msg} (--allow-partial)")
    n1 = len(df)
    df = df.merge(applied, on="row_id", how="inner", validate="one_to_one")
    n_nonfinite = n1 - len(df)
    if n_nonfinite:
        print(f"[162] {n_nonfinite:,} survivors dropped by the finite-member "
              f"filter inside apply_fits (NaN member scores)")
    df["p_final"] = df[p_final_col].to_numpy(np.float64)
    df = df.sort_values("p_final", ascending=False,
                        kind="mergesort").reset_index(drop=True)

    out = Path(args.out)
    atomic_parquet(df, out)
    print(f"[162] wrote {out}: {len(df):,} rows x {df.shape[1]} cols "
          f"(p_final = {p_final_col}; top p_final = "
          + ", ".join(f"{v:.4f}" for v in df.p_final.head(5)) + ")")

    sp = Path(args.summary)
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    summary.update({"n_stage2": int(len(df)), "p_final_col": p_final_col,
                    "fits": str(args.fits),
                    "n_stage2_missing_scores": int(n0 - n1),
                    "n_stage2_nonfinite_members": int(n_nonfinite),
                    "stage2_scores": str(out),
                    "per_footprint_stage2":
                        {k: int(v) for k, v in
                         df.footprint.value_counts().items()}})
    sp.write_text(json.dumps(summary, indent=2))
    print(f"[162] updated {sp}; next: 163 (crossmatch) -> 165 (group conformal "
          f"with --sweep-totals from stage1_summary.json) -> 164 ("
          f"{time.time() - t0:.1f}s)")
    return 0


# ===== cli =======================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-survivors", action="store_true",
                      help="stage-1 thresholding -> survivors manifest + summary")
    mode.add_argument("--merge", action="store_true",
                      help="survivor 112(/112b) scores -> calibrate+combine -> "
                           "stage2_scores.parquet")
    ap.add_argument("--sweep-dir", default=str(SWEEP),
                    help="sweep artifact dir (default data/v2/sweep; all the "
                         "path defaults below resolve under it)")
    # --make-survivors inputs
    ap.add_argument("--stage1-scores", default=None,
                    help="glob of 161 output parquets (rsynced back; default is "
                         "MODE-AWARE: <sweep-dir>/stage1/stage1_*_<stage1-mode>_"
                         "s*.parquet so members/student files never mix)")
    ap.add_argument("--manifests", default=None,
                    help="glob of the 160 part manifests (default "
                         "<sweep-dir>/sweep_manifest_part*of*.parquet)")
    ap.add_argument("--stage1-mode", choices=("members", "student"),
                    help="stage-1 scorer set (must match what 161 ran)")
    ap.add_argument("--members", default=MEMBERS_DEFAULT,
                    help="members-mode roster (comma list, CSV scorer names)")
    ap.add_argument("--student-col", default="member_student_distilled",
                    help="student-mode score column in the 161 parquets")
    ap.add_argument("--stage1-scorer", default="student_distilled",
                    help="student-mode scorer name in --operating-points")
    ap.add_argument("--stage1-fpr", type=float, default=1e-4,
                    help="per-scorer target FPR for the threshold lookup")
    ap.add_argument("--stage1-thr", type=float, default=None,
                    help="student-mode explicit threshold override")
    ap.add_argument("--operating-points", default=str(V2 / "operating_points_v2.csv"))
    ap.add_argument("--negeval-scores", default=str(V2 / "scores_negeval_pool.parquet"),
                    help="raw 112 NegEval pool parquet (comma-list merged on "
                         "row_id when the roster's columns span several 112 "
                         "outputs, e.g. the LEAN v2 roster): the calibrated "
                         "thresholds must reproduce --stage1-fpr on it before "
                         "any selection")
    ap.add_argument("--skip-negeval-check", action="store_true",
                    help="proceed without the NegEval FPR reproduction check "
                         "(only when the pool parquet is unavailable)")
    ap.add_argument("--survivor-budget", type=int, default=150_000,
                    help="operational cap on survivors (top by margin kept)")
    ap.add_argument("--native-griz", action="store_true",
                    help="also print the optional native-AION 160px-griz "
                         "extraction/embedding block (132 gate shipped native)")
    # --merge inputs
    ap.add_argument("--survivor-scores", default=None,
                    help="comma-list of survivor score parquets (112 out incl. "
                         "member_aion; + extras, e.g. a native-AION parquet; "
                         "default <sweep-dir>/survivor_scores.parquet)")
    ap.add_argument("--fits", default=str(V2 / "ensemble_v2_fits"),
                    help="persisted 145 fits dir (or its ensemble_fits.joblib); "
                         "--make-survivors needs its CALIBRATORS (threshold "
                         "space coherence), --merge its calibrate+combine path")
    ap.add_argument("--survivors-manifest", default=None,
                    help="default <sweep-dir>/survivors_manifest.parquet")
    ap.add_argument("--stage1-survivors", default=None,
                    help="default <sweep-dir>/stage1_survivors.parquet")
    ap.add_argument("--p-final-col", default=None,
                    help="override the flagship column (default <tag>_<flagship> "
                         "from the persisted fits)")
    ap.add_argument("--out", default=None,
                    help="default <sweep-dir>/stage2_scores.parquet")
    # shared
    ap.add_argument("--summary", default=None,
                    help="default <sweep-dir>/stage1_summary.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="tolerate missing stage-1 ranges / stage-2 scores "
                         "(coverage gaps WARN instead of abort)")
    args = ap.parse_args()
    sd = Path(args.sweep_dir)
    s1_glob = f"stage1_*_{args.stage1_mode}_s*.parquet" if args.stage1_mode \
        else "*.parquet"      # mode-aware: never mix members/student outputs
    for key, val in (("stage1_scores", sd / "stage1" / s1_glob),
                     ("manifests", sd / "sweep_manifest_part*of*.parquet"),
                     ("survivor_scores", sd / "survivor_scores.parquet"),
                     ("survivors_manifest", sd / "survivors_manifest.parquet"),
                     ("stage1_survivors", sd / "stage1_survivors.parquet"),
                     ("out", sd / "stage2_scores.parquet"),
                     ("summary", sd / "stage1_summary.json")):
        if getattr(args, key) is None:
            setattr(args, key, str(val))
    if args.make_survivors:
        if not args.stage1_mode:
            ap.error("--make-survivors requires --stage1-mode")
        return make_survivors(args)
    return merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
