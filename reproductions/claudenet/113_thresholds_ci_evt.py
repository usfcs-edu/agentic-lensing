#!/usr/bin/env python3
"""113_thresholds_ci_evt.py — Phase 110 (NegEval-1M): CI-backed matched-FPR
operating points on the 1M-galaxy negative eval pool (runs LOCALLY, CPU).

For every scorer — the 6 v1 ensemble members (isotonic-calibrated), the v1
combiners (average / logistic / rf) and the reproduced Stage-D baselines
(baseline_effnet / baseline_meta) — this computes: thresholds at FPR
{1e-2, 1e-3, 1e-4} on NegEval-1M with the v1-verbatim quantile arithmetic
(thr = np.quantile(neg, 1-fpr); recovery = (cand >= thr).mean()); recovery of
the v1 held-out Storfer/Inchausti positives at each threshold; 10,000-rep
bootstrap percentile CIs (2.5/97.5) for threshold, recovery AND the paired
delta vs baseline_meta; a GPD/EVT cross-check of the extreme thresholds; and
continuity columns from the old 6,501-row v1 testneg.

CALIBRATION TRANSFER — 25/26 never persisted the fitted objects, so they are
REFIT here from the identical val-split inputs (same member parquets, same
seed) and VERIFIED to reproduce the stored values before touching the pool:
isotonic pc and the parameterless average to < 1e-9, logistic/rf combiner p
to < 1e-6 (actual maxima printed + saved in thresholds_ci.json).

BOOTSTRAP METHOD (exact joint multinomial, tail-evaluated) — per rep ONE
shared resample of the N pool rows is drawn as multiplicity weights
w = bincount(randint(0,N,N)) (an exact multinomial bootstrap; identical
resampled rows for EVERY scorer, so the delta-vs-baseline_meta CI is exactly
paired on the negative side as well as the positive side). Scorer s's
resampled threshold at FPR f is its r-th largest resampled value,
r = N - ceil((1-f)*N) + 1, located in O(T) on the scorer's precomputed
descending top-T tail (T ~ 3x the largest r; the searchsorted position is
asserted to stay inside the tail). Cost per rep: one 1M bincount + a 30K
cumsum per scorer — ~100s for 10,000 reps x 11 scorers, no 1M-row sort per
rep. Positives: all scorers share the SAME resample indices per rep.
Point estimates keep np.quantile (v1-verbatim type-7 interpolation); the
bootstrap order statistic differs from it by at most one inter-order-stat gap.
BRICK-BLOCK SENSITIVITY — pool rows are brick-clustered (up to 100 objects
per brick), so an iid bootstrap can understate threshold variance. A second
pass resamples whole BRICKS (multinomial over the ~17.6K bricks; per-row
weight = its brick's multiplicity; r scales with the rep's realized total
count) and reports the block thr/recovery CIs + width ratio vs iid. Deltas
keep the iid pairing; the block CIs are a reported sensitivity.

EVT CROSS-CHECK — peaks-over-threshold GPD (scipy.stats.genpareto, MLE,
floc=0) on exceedances over u = 99.5th pool percentile; closed-form extreme
quantile q(f) = u + (sigma/xi)*((zeta_u/f)**xi - 1) with zeta_u = P(X > u)
(log form as xi -> 0); parametric-bootstrap CI (resample the exceedance count
~ Binomial(N, zeta_u), simulate exceedances from the fitted GPD, refit).
Only meaningful for fpr < zeta_u (so 1e-3/1e-4; 1e-2 -> NaN). Members'
isotonic scores can be too tied/stepped for a GPD fit — degenerate tails are
skipped with a note rather than fitted dishonestly.

Inputs: --pool-scores parquet (from 112_score_pool.py on Perlmutter; columns:
row_id + one RAW-score column per scorer — member columns named either
member_<name> (112's convention) or bare <name> for the v1 members aion,
effnet_B, effnet_B3, effnet_S2, resnet46_C, shielded_A, plus baseline_effnet
and baseline_meta; missing members are tolerated but disable the combiners),
the v1 member/combined parquets in data/, and --baseline-scores (per-row
Stage-D baseline scores on the v1 splits, produced once by --prep-baselines).

Outputs: data/v2/operating_points_v2.csv, data/v2/thresholds_ci.json
(with the "verdicts" block: does the flagship-vs-meta delta CI exclude 0 at
1e-3 / 1e-4?), data/v2/scores_negeval_pool_combined.parquet (pool combiner
scores, reused by 111b/114 for the purity audit).

    python 113_thresholds_ci_evt.py --prep-baselines     # one-time: score the
        # v1 eval splits with the Stage-D baselines (mirrors
        # 03_reproduce_baseline) -> data/v2/baseline_scores_v1splits.parquet
    python 113_thresholds_ci_evt.py \\
        --pool-scores data/v2/scores_negeval_pool.parquet
    python 113_thresholds_ci_evt.py --synthetic-check    # no real inputs: 1M
        # synthetic negatives + 2k shifted positives through the FULL
        # machinery, with CI-coverage / pairing / EVT-agreement assertions
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _combine as CM
import _ensemble as E

FPRS = (1e-2, 1e-3, 1e-4)
COMBINERS = ("average", "logistic", "rf")
BASELINES = ("baseline_effnet", "baseline_meta")
V2 = C.DATA / "v2"


# ===== step 1: calibration transfer (refit 25/26 exactly, verify, reuse) =========

def refit_and_verify(scores):
    """Refit the v1 isotonic calibrators (25_calibrate_members) and combiners
    (26_fit_combiner) from the same val-split inputs and verify they reproduce
    the stored pc / combined p on every v1 split. Returns (cals, combs, names,
    report)."""
    report, cals = {}, {}
    for n, df in scores.items():
        val = df[df.split == "val"]
        cal = E.make_calibrator("isotonic").fit(val["p"].to_numpy(), val["label"].to_numpy())
        got, ref = cal.transform(df["p"].to_numpy()), df["pc"].to_numpy()
        assert np.isfinite(got).all() and np.isfinite(ref).all(), \
            f"calibration transfer {n}: non-finite pc values (refit {np.isfinite(got).sum()}, " \
            f"stored {np.isfinite(ref).sum()} finite of {len(ref)})"
        d = float(np.max(np.abs(got - ref)))
        print(f"[cal] {n:12s} refit isotonic vs stored pc: max|diff| = {d:.3e} (tol 1e-9)")
        assert d < 1e-9, f"calibration transfer FAILED for {n}: {d:.3e} >= 1e-9"
        report[f"isotonic_{n}"] = d
        cals[n] = cal

    _, y_v, Pv, names = CM.matrix(scores, "val", "pc")
    combs = {k: E.fit_combiner(k, Pv, y_v) for k in COMBINERS}
    stored = pd.read_parquet(C.DATA / "scores_combined.parquet")
    for k, fn in combs.items():
        worst = 0.0
        for split in ("val", "testneg", "storfer", "inchausti"):
            ids, _, P, _ = CM.matrix(scores, split, "pc")
            got = pd.DataFrame({"row_id": ids, "q": fn(P)})
            ref = stored[(stored.split == split) & (stored.combiner == k)][["row_id", "p"]]
            m = got.merge(ref, on="row_id", how="inner")
            assert len(m) == len(got), f"combiner verify: row_id misalignment ({k}/{split})"
            worst = max(worst, float(np.max(np.abs(m["q"].to_numpy() - m["p"].to_numpy()))))
        tol = 1e-9 if k == "average" else 1e-6
        print(f"[comb] {k:8s} refit vs stored scores_combined: max|diff| = {worst:.3e} (tol {tol:g})")
        assert worst < tol, f"combiner transfer FAILED for {k}: {worst:.3e} >= {tol:g}"
        report[f"combiner_{k}"] = worst
    return cals, combs, names, report


# ===== bootstrap core ============================================================

def _ranks(n: int, fprs) -> dict:
    """Descending rank r of the matched-FPR threshold among n values:
    the k-th ascending order statistic with k = ceil((1-f)*n) is the
    (n - k + 1)-th largest."""
    return {f: int(n - min(max(np.ceil((1.0 - f) * n), 1), n) + 1) for f in fprs}


def joint_threshold_boot(neg: dict, fprs, reps: int, rng,
                         bricks: np.ndarray | None = None) -> dict:
    """Exact joint bootstrap of matched-FPR thresholds, shared across scorers.

    neg: {scorer: (N,) scores}. Per rep one shared multinomial resample:
    iid mode  — w = bincount(randint(0, N, N)), total mass exactly N;
    block mode (bricks given as (N,) int codes) — multiplicity ~ multinomial
    over the B distinct bricks, per-row weight = its brick's multiplicity,
    rep total = dot(mult, brick_sizes) (varies), ranks recomputed per rep.
    Returns {scorer: {fpr: (reps,) thresholds}}.
    """
    scorers = list(neg)
    N = len(next(iter(neg.values())))
    r0 = _ranks(N, fprs)
    T = int(min(N, max(3 * max(r0.values()), 1000)))
    tails = {}
    for s in scorers:
        x = np.asarray(neg[s], dtype=np.float64)
        ord_desc = np.argsort(-x, kind="stable")[:T]
        tails[s] = (ord_desc, x[ord_desc])          # row indices + desc values
    out = {s: {f: np.empty(reps) for f in fprs} for s in scorers}
    if bricks is not None:
        code_of = pd.factorize(bricks)[0].astype(np.int64)
        B = int(code_of.max()) + 1
        bsizes = np.bincount(code_of, minlength=B).astype(np.int64)
    for rep in range(reps):
        if bricks is None:
            w = np.bincount(rng.integers(0, N, N), minlength=N)
            ranks = r0
        else:
            mult = np.bincount(rng.integers(0, B, B), minlength=B)
            w = None
            n_rep = int(mult @ bsizes)
            ranks = _ranks(n_rep, fprs)
        for s in scorers:
            idx, vals = tails[s]
            wt = (w[idx] if bricks is None else mult[code_of[idx]])
            cs = np.cumsum(wt)
            for f in fprs:
                pos = int(np.searchsorted(cs, ranks[f], side="left"))
                assert pos < len(cs), (
                    f"bootstrap tail too short (scorer {s}, fpr {f}, rep {rep}): "
                    f"resampled tail mass {cs[-1]} < rank {ranks[f]}")
                out[s][f][rep] = vals[pos]
    return out


def evt_block(xs_sorted, fprs, n_boot: int, rng, u_q: float = 0.995):
    """GPD peaks-over-threshold cross-check on one scorer's sorted pool scores.
    Returns ({fpr: (q, lo, hi)}, info)."""
    from scipy.stats import genpareto
    N = xs_sorted.size
    u = float(np.quantile(xs_sorted, u_q))
    exc = xs_sorted[xs_sorted > u] - u
    n_u = int(exc.size)
    zeta = n_u / N
    nanres = {f: (float("nan"),) * 3 for f in fprs}
    info = {"u": u, "n_exc": n_u, "zeta": zeta, "xi": float("nan"),
            "sigma": float("nan"), "n_boot_failed": 0, "note": ""}
    if n_u < 100 or np.unique(exc).size < 20:
        info["note"] = "degenerate tail (ties/atoms; e.g. isotonic steps) — GPD skipped"
        return nanres, info
    try:
        xi, _, sigma = genpareto.fit(exc, floc=0.0)
    except Exception as ex:
        info["note"] = f"GPD MLE failed: {ex}"
        return nanres, info
    info["xi"], info["sigma"] = float(xi), float(sigma)

    def q_of(f, xi_, sg_, zeta_):
        if not (0.0 < f < zeta_) or sg_ <= 0:
            return float("nan")
        if abs(xi_) < 1e-12:
            return u + sg_ * np.log(zeta_ / f)
        return u + sg_ / xi_ * ((zeta_ / f) ** xi_ - 1.0)

    qb = {f: [] for f in fprs}
    n_fail = 0
    for _ in range(n_boot):
        nb = int(rng.binomial(N, zeta))
        if nb < 30:
            n_fail += 1
            continue
        y = genpareto.rvs(xi, loc=0.0, scale=sigma, size=nb, random_state=rng)
        try:
            xb, _, sb = genpareto.fit(y, floc=0.0)
        except Exception:
            n_fail += 1
            continue
        for f in fprs:
            qb[f].append(q_of(f, xb, sb, nb / N))
    info["n_boot_failed"] = n_fail
    res = {}
    for f in fprs:
        pt = q_of(f, xi, sigma, zeta)
        arr = np.asarray([v for v in qb[f] if np.isfinite(v)], dtype=np.float64)
        if np.isfinite(pt) and arr.size >= 0.5 * n_boot:
            res[f] = (float(pt), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        else:
            res[f] = (float(pt) if np.isfinite(pt) else float("nan"), float("nan"), float("nan"))
    return res, info


def run_suite(neg: dict, pos: dict, old: dict | None, fprs, reps: int,
              evt_reps: int, rng, baseline: str, bricks: np.ndarray | None = None):
    """The matched-FPR + bootstrap + EVT machinery.

    neg:  {scorer: (N,) pool scores}            (same N, finite, same rows)
    pos:  {split: {scorer: (M_split,) scores}}  (aligned rows across scorers)
    old:  {scorer: old-testneg scores} or None  (continuity point estimates)
    Returns {scorer: results}; deltas are paired vs `baseline` per rep."""
    scorers = list(neg)
    N = len(next(iter(neg.values())))
    assert all(len(v) == N for v in neg.values()), "pool vectors must share rows"
    splits = list(pos)
    t0 = time.time()
    TB = joint_threshold_boot(neg, fprs, reps, rng)         # exact joint, iid rows
    print(f"[boot] joint multinomial threshold bootstrap: {len(scorers)} scorers x "
          f"{reps:,} reps ({time.time() - t0:.1f}s)")
    TBb = None
    if bricks is not None:
        t0 = time.time()
        TBb = joint_threshold_boot(neg, fprs, reps, rng, bricks=bricks)
        print(f"[boot] brick-BLOCK sensitivity bootstrap ({time.time() - t0:.1f}s)")
    idx = {sp: rng.integers(0, len(next(iter(pos[sp].values()))),
                            size=(reps, len(next(iter(pos[sp].values())))),
                            dtype=np.int64)
           for sp in splits}                                         # shared idx
    R = {}
    for s in scorers:
        t0 = time.time()
        x = np.asarray(neg[s], dtype=np.float64)
        xs = np.sort(x)
        ent = {"thr": {}, "thr_ci": {}, "thr_boot": {}, "thr_ci_block": {},
               "rec": {sp: {} for sp in splits}, "rec_ci": {sp: {} for sp in splits},
               "rec_boot": {sp: {} for sp in splits},
               "rec_ci_block": {sp: {} for sp in splits},
               "old_thr": {}, "old_rec": {sp: {} for sp in splits}}
        for f in fprs:
            ent["thr"][f] = E.fpr_threshold(xs, f)                   # v1 verbatim
            tb = TB[s][f]
            ent["thr_boot"][f] = tb
            ent["thr_ci"][f] = (float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5)))
            if TBb is not None:
                ent["thr_ci_block"][f] = (float(np.percentile(TBb[s][f], 2.5)),
                                          float(np.percentile(TBb[s][f], 97.5)))
        for sp in splits:
            p = np.asarray(pos[sp][s], dtype=np.float64)
            pr = p[idx[sp]]                                          # (reps, M)
            for f in fprs:
                ent["rec"][sp][f] = float((p >= ent["thr"][f]).mean())
                rb = (pr >= ent["thr_boot"][f][:, None]).mean(axis=1)
                ent["rec_boot"][sp][f] = rb
                ent["rec_ci"][sp][f] = (float(np.percentile(rb, 2.5)),
                                        float(np.percentile(rb, 97.5)))
                if TBb is not None:
                    rbb = (pr >= TBb[s][f][:, None]).mean(axis=1)
                    ent["rec_ci_block"][sp][f] = (float(np.percentile(rbb, 2.5)),
                                                  float(np.percentile(rbb, 97.5)))
            del pr
        if old is not None and s in old:
            for f in fprs:
                ent["old_thr"][f] = E.fpr_threshold(old[s], f)
                for sp in splits:
                    ent["old_rec"][sp][f] = float(
                        (np.asarray(pos[sp][s], dtype=np.float64) >= ent["old_thr"][f]).mean())
        ent["evt"], ent["evt_info"] = evt_block(xs, fprs, evt_reps, rng)
        R[s] = ent
        print(f"[boot] {s:16s} thresholds+recovery+EVT done ({time.time() - t0:.1f}s)")

    if baseline in R:
        for s in scorers:
            R[s]["delta"] = {sp: {} for sp in splits}
            for sp in splits:
                for f in fprs:
                    db = R[s]["rec_boot"][sp][f] - R[baseline]["rec_boot"][sp][f]
                    R[s]["delta"][sp][f] = (
                        R[s]["rec"][sp][f] - R[baseline]["rec"][sp][f],
                        float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5)))
    else:
        print(f"[boot] WARNING: baseline {baseline!r} not among scorers -> no deltas")
    return R


def rows_from_results(R: dict, kinds: dict, fprs, splits) -> pd.DataFrame:
    rows = []
    for s, ent in R.items():
        for f in fprs:
            row = {"scorer": s, "kind": kinds.get(s, "scorer"), "fpr": f,
                   "thr": ent["thr"][f],
                   "thr_lo": ent["thr_ci"][f][0], "thr_hi": ent["thr_ci"][f][1]}
            if ent.get("thr_ci_block"):
                row["thr_lo_block"] = ent["thr_ci_block"][f][0]
                row["thr_hi_block"] = ent["thr_ci_block"][f][1]
            for sp in splits:
                row[f"rec_{sp}"] = ent["rec"][sp][f]
                row[f"rec_{sp}_lo"] = ent["rec_ci"][sp][f][0]
                row[f"rec_{sp}_hi"] = ent["rec_ci"][sp][f][1]
                if ent.get("rec_ci_block") and ent["rec_ci_block"][sp]:
                    row[f"rec_{sp}_lo_block"] = ent["rec_ci_block"][sp][f][0]
                    row[f"rec_{sp}_hi_block"] = ent["rec_ci_block"][sp][f][1]
            for sp in splits:
                d = ent.get("delta", {}).get(sp, {}).get(f)
                row[f"delta_vs_meta_{sp}"] = d[0] if d else float("nan")
                row[f"delta_vs_meta_{sp}_lo"] = d[1] if d else float("nan")
                row[f"delta_vs_meta_{sp}_hi"] = d[2] if d else float("nan")
            evt = ent["evt"][f]
            row["evt_thr"], row["evt_thr_lo"], row["evt_thr_hi"] = evt
            for sp in splits:
                row[f"oldset_rec_{sp}"] = ent["old_rec"][sp].get(f, float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, fprs, splits):
    for f in fprs:
        sub = df[df.fpr == f]
        print(f"\n[113] ===== FPR = {f:g} (pool thresholds; CIs = bootstrap 2.5/97.5%) =====")
        hdr = f"{'scorer':>16} {'thr':>8} {'thr CI':>19}"
        for sp in splits:
            hdr += f" {'rec_' + sp[:6]:>9} {'CI':>15} {'d_meta':>7} {'d CI':>16}"
        print(hdr)
        for _, r in sub.iterrows():
            line = (f"{r.scorer:>16} {r.thr:>8.4f} "
                    f"[{r.thr_lo:>8.4f},{r.thr_hi:>8.4f}]")
            for sp in splits:
                line += (f" {r[f'rec_{sp}']:>9.3f} [{r[f'rec_{sp}_lo']:>6.3f},{r[f'rec_{sp}_hi']:>6.3f}]"
                         f" {r[f'delta_vs_meta_{sp}']:>+7.3f}"
                         f" [{r[f'delta_vs_meta_{sp}_lo']:>+6.3f},{r[f'delta_vs_meta_{sp}_hi']:>+6.3f}]")
            print(line)


# ===== --prep-baselines: per-row Stage-D baseline scores on the v1 splits ========

def prep_baselines(args):
    """Score the v1 eval splits (testneg/storfer/inchausti) with the reproduced
    Stage-D baselines, exactly as 03_reproduce_baseline did (whose functions are
    reused), and cache per-row scores -> --baseline-scores parquet."""
    import torch
    import _scorelib as SL
    R3 = C._load("cn_113_repro03", C.ROOT / "03_reproduce_baseline.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prep] device = {device}")
    sh, _, m_sh, s_sh, _ = SL.load_checkpoint_model(
        C.DATA / "checkpoint_best_shielded194k_staged.pt", device)
    ef, _, m_ef, s_ef, _ = SL.load_checkpoint_model(
        C.DATA / "checkpoint_best_efficientnet_staged.pt", device)
    meta = R3.load_meta(device)

    frames = []
    for split, fname in (("testneg", "eval_testneg.parquet"),
                         ("storfer", "eval_storfer.parquet"),
                         ("inchausti", "eval_inchausti.parquet")):
        ev = pd.read_parquet(C.DATA / fname)
        paths = [Path(r.fits_dir) / f"{r.row_id}.fits" for r in ev.itertuples()]
        pr = SL.score_paths(paths, sh, "shielded", m_sh, s_sh, device)
        pe = SL.score_paths(paths, ef, "efficientnet", m_ef, s_ef, device)
        pm = R3.meta_prob(meta, pr, pe, device)
        frames.append(pd.DataFrame({"split": split, "row_id": ev.row_id.astype(str),
                                    "label": ev.label.astype(int),
                                    "baseline_resnet": pr, "baseline_effnet": pe,
                                    "baseline_meta": pm}))
        print(f"[prep] {split}: scored {len(ev):,} cutouts")
    out = pd.concat(frames, ignore_index=True)

    # cross-check 1: matched-FPR recovery must reproduce meta_metrics_staged.json
    ref = json.load(open(C.DATA / "meta_metrics_staged.json"))["recovery_at_fpr"]
    neg = out[out.split == "testneg"]
    worst, tol = 0.0, 0.02
    for cat in ("storfer", "inchausti"):
        cand = out[out.split == cat]
        for mdl, col in (("effnet", "baseline_effnet"), ("meta", "baseline_meta")):
            rec = E.recovery_at_fpr(neg[col].to_numpy(), cand[col].to_numpy(), fprs=(0.01, 0.001))
            for f in (0.01, 0.001):
                d = abs(rec[f]["recovery"] - ref[f"{cat}|{mdl}|{f}"]["recovery"])
                worst = max(worst, d)
                print(f"[prep] check {cat}|{mdl}|{f}: ours {rec[f]['recovery']:.4f} "
                      f"stored {ref[f'{cat}|{mdl}|{f}']['recovery']:.4f} |d|={d:.4f}")
    assert worst <= tol, f"baseline re-score does NOT reproduce stored recoveries ({worst:.4f} > {tol})"

    # cross-check 2: per-row effnet vs the gate's stored staged-effnet scores
    gate = pd.read_parquet(C.DATA / "scores_gate_merged.parquet")[["split", "row_id", "p_effnet"]]
    m = out.merge(gate, on=["split", "row_id"], how="inner")
    d2 = float(np.nanmax(np.abs(m.baseline_effnet - m.p_effnet)))
    print(f"[prep] per-row effnet vs scores_gate_merged ({len(m):,} rows): max|diff| = {d2:.3e}")
    assert d2 < 1e-3, f"prep-baselines cross-check 2 FAILED: per-row effnet drift {d2:.3e} >= 1e-3"

    V2.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.baseline_scores, index=False)
    print(f"[prep] wrote {args.baseline_scores} ({len(out):,} rows)")
    return 0


# ===== real run ==================================================================

def build_split_table(scores, combs, names, base_df, split):
    """One aligned table for a v1 split: row_id + member pc + combiner p
    (+ baseline_* when base_df given), inner-joined across all scorers."""
    ids, _, P, nm = CM.matrix(scores, split, "pc")
    assert nm == names
    df = pd.DataFrame(P, columns=nm)
    df.insert(0, "row_id", ids)
    for k, fn in combs.items():
        df[k] = fn(P)
    if base_df is not None:
        b = base_df[base_df.split == split][["row_id", "baseline_effnet", "baseline_meta"]]
        n0 = len(df)
        df = df.merge(b, on="row_id", how="inner")
        if len(df) < n0:
            print(f"[113] {split}: {n0 - len(df)} rows dropped joining baseline scores")
    return df


def main_real(args, fprs):
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    V2.mkdir(parents=True, exist_ok=True)

    # -- 1. calibration transfer ------------------------------------------------
    scores = CM.load_scores()
    cals, combs, names, transfer = refit_and_verify(scores)

    # -- 2. pool scores -> calibrated member matrix -> combiner scores ----------
    pool = pd.read_parquet(args.pool_scores)
    # member raw-score columns: 112_score_pool writes member_<name>; accept bare too
    colmap = {n: (n if n in pool.columns
                  else f"member_{n}" if f"member_{n}" in pool.columns else None)
              for n in names}
    members = [n for n in names if colmap[n]]
    missing = [n for n in names if not colmap[n]]
    if missing:
        print(f"[pool] WARNING: pool lacks member columns {missing} -> combiners DISABLED")
    bases = [b for b in BASELINES if b in pool.columns]
    if set(bases) != set(BASELINES):
        print(f"[pool] WARNING: pool lacks baseline columns "
              f"{sorted(set(BASELINES) - set(bases))} -> those scorers skipped")
    used = [colmap[n] for n in members] + bases
    if not used:
        print(f"[pool] FATAL: no known scorer columns in {args.pool_scores} "
              f"(have: {list(pool.columns)})")
        return 1
    X = pool[used].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    if (~finite).any():
        print(f"[pool] dropping {(~finite).sum():,}/{len(pool):,} rows with non-finite "
              f"scores (failed cutouts)")
    pool = pool[finite].reset_index(drop=True)
    print(f"[pool] {len(pool):,} usable pool rows; scorers from pool: {used}")

    neg = {n: cals[n].transform(pool[colmap[n]].to_numpy(dtype=np.float64))
           for n in members}
    if not missing:
        Ppool = np.column_stack([neg[n] for n in names])
        for k, fn in combs.items():
            print(f"[pool] applying combiner {k} to {len(pool):,} rows ...")
            neg[k] = fn(Ppool)
        comb_out = pd.DataFrame({"row_id": pool["row_id"].astype(str)})
        for k in COMBINERS:
            comb_out[k] = neg[k]
        comb_path = V2 / "scores_negeval_pool_combined.parquet"
        comb_out.to_parquet(comb_path, index=False)
        print(f"[pool] wrote {comb_path} (input to 111b/114 purity audit)")
    for b in bases:
        neg[b] = pool[b].to_numpy(dtype=np.float64)

    # -- 3. positives + old-testneg tables (v1 parquets; baselines from cache) --
    base_df = None
    if Path(args.baseline_scores).exists():
        base_df = pd.read_parquet(args.baseline_scores)
    elif bases:
        print(f"[113] FATAL: {args.baseline_scores} missing — run "
              f"`python 113_thresholds_ci_evt.py --prep-baselines` first "
              f"(needed for baseline recovery + the paired delta-vs-meta claim)")
        return 1
    tabs = {sp: build_split_table(scores, combs, names, base_df, sp)
            for sp in ("storfer", "inchausti", "testneg")}
    scorer_list = list(neg)                      # members [+combiners] [+baselines]
    pos = {sp: {s: tabs[sp][s].to_numpy(dtype=np.float64) for s in scorer_list}
           for sp in ("storfer", "inchausti")}
    old = {s: tabs["testneg"][s].to_numpy(dtype=np.float64) for s in scorer_list}
    for sp in pos:
        print(f"[113] positives[{sp}]: {len(tabs[sp]):,} aligned rows; "
              f"old testneg: {len(tabs['testneg']):,} rows")

    # -- 4. bootstrap + EVT + old-set continuity ---------------------------------
    bricks = None
    if Path(args.manifest).exists():
        man = pd.read_parquet(args.manifest)[["row_id", "brick"]]
        man["row_id"] = man["row_id"].astype(str)
        mb = pool[["row_id"]].astype(str).merge(man, on="row_id", how="left")
        assert mb["brick"].notna().all(), "pool rows missing from manifest (brick map)"
        bricks = mb["brick"].to_numpy()
        print(f"[113] brick-block sensitivity enabled: {pd.unique(bricks).size:,} bricks")
    else:
        print(f"[113] WARNING: {args.manifest} not found -> no brick-block sensitivity")
    R = run_suite(neg, pos, old, fprs, args.reps, args.evt_reps, rng,
                  baseline="baseline_meta", bricks=bricks)
    kinds = {**{n: "member" for n in members}, **{k: "combiner" for k in COMBINERS},
             **{b: "baseline" for b in bases}}
    df = rows_from_results(R, kinds, fprs, ("storfer", "inchausti"))
    out_csv = V2 / "operating_points_v2.csv"
    df.to_csv(out_csv, index=False)
    print_table(df, fprs, ("storfer", "inchausti"))

    # -- 5. verdicts: flagship-vs-meta paired delta CI excludes 0? ---------------
    verdicts = {"flagship": "average", "baseline": "baseline_meta", "per_combiner": {}}
    have_deltas = (not missing) and ("baseline_meta" in neg)
    if have_deltas:
        for k in COMBINERS:
            verdicts["per_combiner"][k] = {}
            for sp in ("storfer", "inchausti"):
                for f in fprs:
                    d, lo, hi = R[k]["delta"][sp][f]
                    verdicts["per_combiner"][k][f"{sp}@{f:g}"] = {
                        "delta": d, "lo": lo, "hi": hi,
                        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                        "positive_and_excludes_zero": bool(lo > 0.0)}
        flag = verdicts["per_combiner"]["average"]
        verdicts["flagship_beats_meta"] = {
            key: v["positive_and_excludes_zero"] for key, v in flag.items()
            if key.endswith("@0.001") or key.endswith("@0.0001")}
        print("\n[verdict] flagship (average combiner) vs baseline_meta, paired delta CI:")
        for key, v in flag.items():
            print(f"[verdict]   {key:22s} delta = {v['delta']:+.4f} "
                  f"CI [{v['lo']:+.4f}, {v['hi']:+.4f}]  excludes 0: {v['excludes_zero']}")
    else:
        verdicts["note"] = "combiners or baseline_meta unavailable -> no verdicts"
        print("[verdict] SKIPPED (missing members or baseline_meta on the pool)")

    summary = {
        "seed": args.seed, "reps": args.reps, "evt_reps": args.evt_reps,
        "fprs": list(fprs), "pool_scores": str(args.pool_scores),
        "n_pool_used": int(len(pool)),
        "n_pos": {sp: int(len(tabs[sp])) for sp in ("storfer", "inchausti")},
        "n_old_testneg": int(len(tabs["testneg"])),
        "members": members, "members_missing_from_pool": missing,
        "baselines_on_pool": bases,
        "calibration_transfer_max_abs_diff": transfer,
        "evt": {s: R[s]["evt_info"] for s in R},
        "verdicts": verdicts,
    }
    out_json = V2 / "thresholds_ci.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[113] wrote {out_csv} + {out_json} ({time.time() - t0:.1f}s)")
    return 0


# ===== --synthetic-check =========================================================

def synthetic_check(args, fprs):
    """End-to-end machinery check on a fully synthetic pool (seed 2026):
    1M negatives (latent Beta(0.5,8) + scorer noise) and 2k shifted positives
    (latent Beta(6,3)) for three coupled scorers — `good`/`twin` (sd .02, equal
    quality) and `bad` (sd .10, the stand-in baseline). Asserts CI coverage,
    paired-delta behaviour and EVT/empirical agreement."""
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    N, M = 1_000_000, 2_000
    sds = {"good": 0.02, "twin": 0.02, "bad": 0.10}
    lat_neg = rng.beta(0.5, 8.0, N)
    lat_pos = rng.beta(6.0, 3.0, M)
    neg = {s: lat_neg + sd * rng.standard_normal(N) for s, sd in sds.items()}
    pos = {"synthpos": {s: lat_pos + sd * rng.standard_normal(M) for s, sd in sds.items()}}
    fresh_lat = rng.beta(0.5, 8.0, N)                    # independent draw, same law
    fresh = {s: fresh_lat + sd * rng.standard_normal(N) for s, sd in sds.items()}
    big_lat = rng.beta(0.5, 8.0, 4_000_000)              # 4M "truth" reference
    truth = {s: {f: float(np.quantile(big_lat + sd * rng.standard_normal(big_lat.size), 1 - f))
                 for f in fprs} for s, sd in sds.items()}
    print(f"[synth] pool N={N:,}, positives M={M:,}, scorers={list(sds)}, "
          f"reps={args.reps:,}, evt_reps={args.evt_reps:,}, seed={args.seed}")

    R = run_suite(neg, pos, None, fprs, args.reps, args.evt_reps, rng, baseline="bad")
    df = rows_from_results(R, {s: "synthetic" for s in sds}, fprs, ("synthpos",))
    V2.mkdir(parents=True, exist_ok=True)
    df.to_csv(V2 / "synthetic_operating_points.csv", index=False)
    print_table(df, fprs, ("synthpos",))
    print()

    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok)))
        print(f"[synth] {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    # 1. point estimates inside their own bootstrap CIs
    bad_pts = [(s, f) for s in sds for f in fprs
               if not (R[s]["thr_ci"][f][0] <= R[s]["thr"][f] <= R[s]["thr_ci"][f][1])]
    check("thr point in thr CI (9/9)", not bad_pts, f"violations: {bad_pts}")
    bad_rec = [(s, f) for s in sds for f in fprs
               if not (R[s]["rec_ci"]["synthpos"][f][0] - 1e-12
                       <= R[s]["rec"]["synthpos"][f]
                       <= R[s]["rec_ci"]["synthpos"][f][1] + 1e-12)]
    check("recovery point in recovery CI (9/9)", not bad_rec, f"violations: {bad_rec}")

    # 2. threshold CI covers the 4M-sample "truth" quantile (~95% nominal/cell)
    cov = sum(R[s]["thr_ci"][f][0] <= truth[s][f] <= R[s]["thr_ci"][f][1]
              for s in sds for f in fprs)
    check("thr CI covers truth quantile (>=7/9)", cov >= 7, f"covered {cov}/9")

    # 3. empirical FPR on an INDEPENDENT fresh pool at the point threshold
    ratios = {}
    ok_ratio, ok_bracket = True, True
    for s in sds:
        xf = fresh[s]
        for f in fprs:
            fp_pt = float((xf >= R[s]["thr"][f]).mean())
            fp_lo = float((xf >= R[s]["thr_ci"][f][1]).mean())   # upper thr -> lower fpr
            fp_hi = float((xf >= R[s]["thr_ci"][f][0]).mean())
            ratios[(s, f)] = fp_pt / f
            ok_ratio &= 0.5 <= fp_pt / f <= 2.0
            ok_bracket &= (fp_lo <= 1.25 * f) and (fp_hi >= 0.8 * f)
    check("fresh-pool FPR(thr) within x2 of target (9/9)", ok_ratio,
          "ratios: " + ", ".join(f"{s}@{f:g}={r:.2f}" for (s, f), r in ratios.items()))
    check("fresh-pool FPR(thr CI) brackets target (9/9)", ok_bracket, "tol x[0.8,1.25]")

    # 4. paired deltas: good-vs-bad must exclude 0 (real gap); good-vs-twin must not
    gb = {f: R["good"]["delta"]["synthpos"][f] for f in fprs}
    ok_gap = all(gb[f][1] > 0 for f in (1e-3, 1e-4) if f in fprs)
    check("delta(good - bad) CI excludes 0 at 1e-3 & 1e-4", ok_gap,
          ", ".join(f"{f:g}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]" for f, (d, lo, hi) in gb.items()))
    gt = {f: (float(np.percentile(R["good"]["rec_boot"]["synthpos"][f]
                                  - R["twin"]["rec_boot"]["synthpos"][f], 2.5)),
              float(np.percentile(R["good"]["rec_boot"]["synthpos"][f]
                                  - R["twin"]["rec_boot"]["synthpos"][f], 97.5)))
          for f in fprs}
    ok_twin = all(lo <= 0.0 <= hi for lo, hi in gt.values())
    check("delta(good - twin) CI contains 0 (all fprs)", ok_twin,
          ", ".join(f"{f:g}: [{lo:+.4f},{hi:+.4f}]" for f, (lo, hi) in gt.items()))

    # 5. pairing tightens the delta CI vs a rep-shuffled (unpaired) bootstrap
    f0 = 1e-3
    a = R["good"]["rec_boot"]["synthpos"][f0]
    b = R["twin"]["rec_boot"]["synthpos"][f0]
    perm = np.random.default_rng(args.seed + 1).permutation(len(b))
    w_pair = np.percentile(a - b, 97.5) - np.percentile(a - b, 2.5)
    w_unp = np.percentile(a - b[perm], 97.5) - np.percentile(a - b[perm], 2.5)
    check("paired delta CI narrower than unpaired", w_pair < w_unp,
          f"width paired {w_pair:.4f} < unpaired {w_unp:.4f}")

    # 6. EVT vs empirical threshold agreement (continuous tails -> GPD valid)
    rel = {(s, f): abs(R[s]["evt"][f][0] - R[s]["thr"][f]) / abs(R[s]["thr"][f])
           for s in sds for f in (1e-3, 1e-4) if f in fprs and np.isfinite(R[s]["evt"][f][0])}
    ok_evt = (len(rel) == 2 * len(sds)
              and all(v < (0.15 if f == 1e-3 else 0.25) for (s, f), v in rel.items()))
    check("EVT thr within 15%@1e-3 / 25%@1e-4 of empirical (all scorers)", ok_evt,
          ", ".join(f"{s}@{f:g}={v:.3f}" for (s, f), v in sorted(rel.items())))

    # 7. brick-block bootstrap: ~iid width under random brick assignment, wider
    #    under genuine within-brick correlation
    f0, w = 1e-3, lambda a: float(np.percentile(a, 97.5) - np.percentile(a, 2.5))
    rb = np.random.default_rng(args.seed + 2)
    bricks_rand = rb.integers(0, N // 100, N)
    t_iid = joint_threshold_boot({"g": neg["good"]}, (f0,), 2000, np.random.default_rng(7))
    t_rnd = joint_threshold_boot({"g": neg["good"]}, (f0,), 2000, np.random.default_rng(7),
                                 bricks=bricks_rand)
    ratio_rand = w(t_rnd["g"][f0]) / w(t_iid["g"][f0])
    brick_id = np.arange(N) // 100                       # 100-object bricks
    # tail membership must be BRICK-driven for real clustering: brick effect
    # (sd .10) dominates the per-row noise (sd .03), so the extreme tail
    # concentrates in a few high-draw bricks
    clust = (0.10 * rb.standard_normal(N // 100)[brick_id]
             + 0.03 * rb.standard_normal(N))
    t_ci = joint_threshold_boot({"c": clust}, (f0,), 2000, np.random.default_rng(8))
    t_cb = joint_threshold_boot({"c": clust}, (f0,), 2000, np.random.default_rng(8),
                                bricks=brick_id)
    ratio_clust = w(t_cb["c"][f0]) / w(t_ci["c"][f0])
    check("block boot ~iid when unclustered, wider when clustered",
          0.7 <= ratio_rand <= 1.4 and ratio_clust > 1.15,
          f"width ratio: random-bricks {ratio_rand:.2f} (in [0.7,1.4]), "
          f"clustered {ratio_clust:.2f} (>1.15)")

    n_ok = sum(ok for _, ok in checks)
    print(f"\n[synth] {n_ok}/{len(checks)} checks passed "
          f"({time.time() - t0:.1f}s) -> {'SYNTHETIC-CHECK PASS' if n_ok == len(checks) else 'SYNTHETIC-CHECK FAIL'}")
    return 0 if n_ok == len(checks) else 1


# ===== cli =======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool-scores", default=str(V2 / "scores_negeval_pool.parquet"),
                    help="NegEval-1M raw scores parquet (row_id + per-scorer columns)")
    ap.add_argument("--baseline-scores", default=str(V2 / "baseline_scores_v1splits.parquet"),
                    help="per-row Stage-D baseline scores on the v1 splits "
                         "(create once with --prep-baselines)")
    ap.add_argument("--manifest", default=str(V2 / "negeval_manifest.parquet"),
                    help="pool manifest (row_id->brick map for the block sensitivity)")
    ap.add_argument("--reps", type=int, default=10_000, help="bootstrap reps")
    ap.add_argument("--evt-reps", type=int, default=1_000, help="EVT parametric-bootstrap reps")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--fprs", default="0.01,0.001,0.0001",
                    help="comma-separated target FPRs")
    ap.add_argument("--prep-baselines", action="store_true",
                    help="one-time: score v1 eval splits with the Stage-D baselines")
    ap.add_argument("--synthetic-check", action="store_true",
                    help="run the full machinery on a synthetic pool + assert sanity")
    args = ap.parse_args()
    fprs = tuple(float(x) for x in args.fprs.split(","))
    if args.prep_baselines:
        return prep_baselines(args)
    if args.synthetic_check:
        return synthetic_check(args, fprs)
    return main_real(args, fprs)


if __name__ == "__main__":
    raise SystemExit(main())
