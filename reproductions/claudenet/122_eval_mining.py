#!/usr/bin/env python3
"""122_eval_mining.py — Phase 120: did 1M-pool hard-negative mining beat the
fixed-count random control and the v1 originals? (runs LOCALLY, CPU.)

Per retrained member (effnet_S2 / effnet_B3 / resnet46_C) this compares THREE
scorers — <m>_hard, <m>_random (121 retrains) and <m>_v1 (the original v1
member) — with 113's machinery imported by path (joint_threshold_boot/_ranks):
thresholds at each FPR on NegEval-1M (v1-verbatim quantile point estimates,
E.fpr_threshold), recovery of the v1 held-out Storfer/Inchausti positives, and
PAIRED bootstrap deltas (hard - random) and (hard - v1) — one shared multinomial
resample of the pool rows per rep across ALL scorers and shared positive
resample indices, so the delta CIs are exactly paired on both sides. Matched-FPR
recovery is invariant to monotone recalibration, so raw `p` columns are compared
throughout. Also reports old-testneg (6,501-row) continuity point estimates and
a mean-over-members aggregate (per-rep delta averaged across members).

GATE (plan): a mining round 2 is warranted only if the mean-over-members paired
delta (hard - random) on Inchausti recovery @ FPR 1e-3 is >= +0.015 with the 95%
CI excluding 0. Per-member gate flags are reported alongside.

Inputs:
  --negeval-variants  data/v2/scores_negeval_member_variants.parquet — per-
        variant member scores over NegEval-1M, produced by
        `112_score_pool.py --extra-ckpt-dir data/v2/ckpt --only-extra` on the
        121 checkpoints. Columns come out as the checkpoint file stems, e.g.
        member_effnet_S2_hard (the member_ prefix is accepted here, as is the
        bare <member>_<variant> form). Schema: row_id (str) [+ ok] + one float
        column per <member>_<variant>.
  --pool-scores       data/v2/scores_negeval_pool.parquet (v1 member_<m> columns)
  --negeval-manifest / --minepool-manifest
        data/v2/{negeval,minepool}_manifest.parquet — their brick sets are
        asserted DISJOINT up front (mined-negative -> eval-pool leakage guard).
  data/v2/scores_member_<m>_<variant>.parquet (121) + data/scores_member_<m>.parquet

Outputs: data/v2/mining_v2_results.csv + .json (with the round-2 verdict).

    /home2/benson/.venvs/claudenet/bin/python 122_eval_mining.py
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
MEMBERS = ("effnet_S2", "effnet_B3", "resnet46_C")
VARIANTS = ("hard", "random", "v1")
POS_SPLITS = ("storfer", "inchausti")


def find_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_pos_scores(member: str, variant: str) -> pd.DataFrame | None:
    """[split,row_id,label,p] for one scorer (v2 retrain or v1 original)."""
    f = (C.DATA / f"scores_member_{member}.parquet" if variant == "v1"
         else V2 / f"scores_member_{member}_{variant}.parquet")
    if not f.exists():
        print(f"[122] WARNING: {f} missing -> {member}_{variant} unavailable")
        return None
    df = pd.read_parquet(f)[["split", "row_id", "label", "p"]]
    df["row_id"] = df["row_id"].astype(str)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--negeval-variants",
                    default=str(V2 / "scores_negeval_member_variants.parquet"),
                    help="per-variant member scores on NegEval-1M (extra 112 pass)")
    ap.add_argument("--pool-scores", default=str(V2 / "scores_negeval_pool.parquet"),
                    help="v1 member scores on NegEval-1M (112 production pass)")
    ap.add_argument("--negeval-manifest", default=str(V2 / "negeval_manifest.parquet"),
                    help="NegEval-1M manifest (brick column; leakage guard)")
    ap.add_argument("--minepool-manifest", default=str(V2 / "minepool_manifest.parquet"),
                    help="MinePool-1M manifest (brick column; leakage guard)")
    ap.add_argument("--members", default=",".join(MEMBERS),
                    help="comma-separated retrained members to evaluate")
    ap.add_argument("--reps", type=int, default=10_000, help="bootstrap reps")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--fprs", default="0.01,0.001,0.0001")
    ap.add_argument("--gate-delta", type=float, default=0.015)
    ap.add_argument("--gate-fpr", type=float, default=1e-3)
    ap.add_argument("--gate-split", default="inchausti")
    ap.add_argument("--out-prefix", default=str(V2 / "mining_v2_results"))
    args = ap.parse_args()
    t0 = time.time()
    fprs = tuple(float(x) for x in args.fprs.split(","))
    if args.gate_fpr not in fprs:
        ap.error(f"--gate-fpr {args.gate_fpr:g} must be one of --fprs {args.fprs}")
    rng = np.random.default_rng(args.seed)
    T113 = C._load("cn_122_t113", C.ROOT / "113_thresholds_ci_evt.py")

    # -- 0. leakage guard: NegEval-1M and MinePool-1M must be brick-disjoint ----
    nev_bricks = set(pd.read_parquet(args.negeval_manifest, columns=["brick"]).brick)
    mp_bricks = set(pd.read_parquet(args.minepool_manifest, columns=["brick"]).brick)
    shared = nev_bricks & mp_bricks
    assert nev_bricks.isdisjoint(mp_bricks), (
        f"LEAKAGE: NegEval-1M and MinePool-1M share {len(shared)} bricks "
        f"(e.g. {sorted(shared)[:5]}) — mined training negatives would sit in "
        f"the evaluation pool; rebuild the manifests brick-disjoint "
        f"({args.negeval_manifest} vs {args.minepool_manifest})")
    print(f"[122] leakage guard: {len(nev_bricks):,} NegEval bricks disjoint "
          f"from {len(mp_bricks):,} MinePool bricks")

    # -- 1. NegEval-1M scores: one aligned table over every available scorer ----
    nev = pd.read_parquet(args.negeval_variants)
    nev["row_id"] = nev["row_id"].astype(str)
    poolv1 = pd.read_parquet(args.pool_scores)
    poolv1["row_id"] = poolv1["row_id"].astype(str)

    members, neg_col = [], {}
    for m in args.members.split(","):
        cols = {"hard": find_col(nev, (f"{m}_hard", f"member_{m}_hard")),
                "random": find_col(nev, (f"{m}_random", f"member_{m}_random")),
                "v1": find_col(poolv1, (f"member_{m}", m))}
        missing = [v for v, c in cols.items() if c is None]
        if missing:
            print(f"[122] WARNING: member {m} lacks NegEval columns for {missing} "
                  f"-> member skipped")
            continue
        members.append(m)
        for v, c in cols.items():
            neg_col[(m, v)] = c
    if not members:
        print("[122] FATAL: no member has a complete hard/random/v1 column set")
        return 1

    nev_cols = sorted({neg_col[(m, v)] for m in members for v in ("hard", "random")})
    v1_cols = sorted({neg_col[(m, "v1")] for m in members})
    tab = nev[["row_id"] + nev_cols].merge(poolv1[["row_id"] + v1_cols],
                                           on="row_id", how="inner")
    n_join = len(tab)
    X = tab[nev_cols + v1_cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    tab = tab[finite].reset_index(drop=True)
    print(f"[122] NegEval-1M: {len(nev):,} variant rows ∩ {len(poolv1):,} v1 rows "
          f"= {n_join:,}; {len(tab):,} finite across all "
          f"{len(nev_cols) + len(v1_cols)} scorer columns")
    N = len(tab)
    neg = {f"{m}_{v}": tab[neg_col[(m, v)]].to_numpy(dtype=np.float64)
           for m in members for v in VARIANTS}

    # -- 2. positives + old-testneg: aligned per-split matrices ------------------
    pos_raw = {(m, v): load_pos_scores(m, v) for m in members for v in VARIANTS}
    if any(df is None for df in pos_raw.values()):
        print("[122] FATAL: missing member score parquet(s) above")
        return 1
    pos, old = {}, {}
    for sp in POS_SPLITS + ("testneg",):
        base = None
        for (m, v), df in pos_raw.items():
            s = (df[df.split == sp][["row_id", "p"]]
                 .rename(columns={"p": f"{m}_{v}"}))
            base = s if base is None else base.merge(s, on="row_id", how="inner")
        base = base[np.isfinite(base.drop(columns="row_id")
                                .to_numpy(dtype=np.float64)).all(axis=1)]
        d = {s: base[s].to_numpy(dtype=np.float64) for s in neg}
        if sp == "testneg":
            old = d
        else:
            pos[sp] = d
        print(f"[122] {sp}: {len(base):,} aligned rows across {len(neg)} scorers")

    # -- 3. joint bootstrap: shared negative resample + shared positive indices --
    ranks = T113._ranks(N, fprs)
    print(f"[122] pool N={N:,}; threshold order-statistic ranks "
          + ", ".join(f"{f:g}->r={ranks[f]}" for f in fprs))
    tb0 = time.time()
    TB = T113.joint_threshold_boot(neg, fprs, args.reps, rng)
    print(f"[boot] joint multinomial threshold bootstrap: {len(neg)} scorers x "
          f"{args.reps:,} reps ({time.time() - tb0:.1f}s)")
    idx = {sp: rng.integers(0, len(next(iter(pos[sp].values()))),
                            size=(args.reps, len(next(iter(pos[sp].values())))),
                            dtype=np.int64) for sp in POS_SPLITS}

    thr = {s: {f: E.fpr_threshold(neg[s], f) for f in fprs} for s in neg}
    rec, rec_boot, old_rec = {}, {}, {}
    for s in neg:
        rec[s], rec_boot[s], old_rec[s] = {}, {}, {}
        for sp in POS_SPLITS:
            p = pos[sp][s]
            pr = p[idx[sp]]                                     # (reps, M)
            rec[s][sp] = {f: float((p >= thr[s][f]).mean()) for f in fprs}
            rec_boot[s][sp] = {f: (pr >= TB[s][f][:, None]).mean(axis=1) for f in fprs}
            old_rec[s][sp] = {f: float((p >= E.fpr_threshold(old[s], f)).mean())
                              for f in fprs}
            del pr

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    # -- 4. per-member + mean-over-members rows ----------------------------------
    rows = []
    mean_d = {(sp, f, c): [] for sp in POS_SPLITS for f in fprs
              for c in ("hard_random", "hard_v1")}
    for m in members + ["mean3"]:
        for sp in POS_SPLITS:
            for f in fprs:
                row = {"member": m, "split": sp, "fpr": f, "n_pool": N,
                       "n_pos": len(pos[sp][f"{members[0]}_hard"])}
                if m == "mean3":
                    for c, ref in (("hard_random", "random"), ("hard_v1", "v1")):
                        db = np.mean(mean_d[(sp, f, c)], axis=0)   # per-rep mean delta
                        row[f"d_{c}"] = float(np.mean(
                            [rec[f"{mm}_hard"][sp][f] - rec[f"{mm}_{ref}"][sp][f]
                             for mm in members]))
                        row[f"d_{c}_lo"], row[f"d_{c}_hi"] = ci(db)
                else:
                    for v in VARIANTS:
                        s = f"{m}_{v}"
                        row[f"thr_{v}"] = thr[s][f]
                        row[f"rec_{v}"] = rec[s][sp][f]
                        row[f"rec_{v}_lo"], row[f"rec_{v}_hi"] = ci(rec_boot[s][sp][f])
                        row[f"oldset_rec_{v}"] = old_rec[s][sp][f]
                    for c, ref in (("hard_random", "random"), ("hard_v1", "v1")):
                        db = (rec_boot[f"{m}_hard"][sp][f]
                              - rec_boot[f"{m}_{ref}"][sp][f])
                        row[f"d_{c}"] = rec[f"{m}_hard"][sp][f] - rec[f"{m}_{ref}"][sp][f]
                        row[f"d_{c}_lo"], row[f"d_{c}_hi"] = ci(db)
                        mean_d[(sp, f, c)].append(db)
                rows.append(row)
    res = pd.DataFrame(rows)

    # -- 5. the round-2 gate ------------------------------------------------------
    gf, gsp, gd = args.gate_fpr, args.gate_split, args.gate_delta
    crit = f"(hard - random) >= +{gd:g} @ FPR {gf:g} {gsp} recovery, 95% CI > 0"
    per_member = {}
    for m in members:
        r = res[(res.member == m) & (res.split == gsp) & (res.fpr == gf)].iloc[0]
        per_member[m] = {"delta": float(r.d_hard_random),
                         "lo": float(r.d_hard_random_lo),
                         "hi": float(r.d_hard_random_hi),
                         "pass": bool(r.d_hard_random >= gd and r.d_hard_random_lo > 0)}
    r = res[(res.member == "mean3") & (res.split == gsp) & (res.fpr == gf)].iloc[0]
    mean3 = {"delta": float(r.d_hard_random), "lo": float(r.d_hard_random_lo),
             "hi": float(r.d_hard_random_hi),
             "pass": bool(r.d_hard_random >= gd and r.d_hard_random_lo > 0)}
    gate = {"criterion": crit, "per_member": per_member, "mean3": mean3,
            "any_member_pass": any(v["pass"] for v in per_member.values()),
            "all_members_pass": all(v["pass"] for v in per_member.values()),
            "round2": mean3["pass"]}

    out_csv, out_json = Path(args.out_prefix + ".csv"), Path(args.out_prefix + ".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({
        "seed": args.seed, "reps": args.reps, "fprs": list(fprs),
        "negeval_variants": str(args.negeval_variants),
        "pool_scores": str(args.pool_scores),
        "n_pool_used": N, "members": members,
        "threshold_ranks": {f"{f:g}": ranks[f] for f in fprs},
        "gate": gate}, indent=2))

    for f in fprs:
        print(f"\n[122] ===== FPR = {f:g} (NegEval-1M thresholds; paired bootstrap CIs) =====")
        print(f"{'member':>12} {'split':>10} {'rec_v1':>7} {'rec_rnd':>7} {'rec_hard':>8} "
              f"{'d(h-r)':>7} {'d CI':>17} {'d(h-v1)':>8} {'d CI':>17}")
        for _, r in res[res.fpr == f].iterrows():
            if r.member == "mean3":
                print(f"{r.member:>12} {r.split:>10} {'':>7} {'':>7} {'':>8} "
                      f"{r.d_hard_random:>+7.3f} [{r.d_hard_random_lo:>+7.3f},{r.d_hard_random_hi:>+7.3f}] "
                      f"{r.d_hard_v1:>+8.3f} [{r.d_hard_v1_lo:>+7.3f},{r.d_hard_v1_hi:>+7.3f}]")
            else:
                print(f"{r.member:>12} {r.split:>10} {r.rec_v1:>7.3f} {r.rec_random:>7.3f} "
                      f"{r.rec_hard:>8.3f} "
                      f"{r.d_hard_random:>+7.3f} [{r.d_hard_random_lo:>+7.3f},{r.d_hard_random_hi:>+7.3f}] "
                      f"{r.d_hard_v1:>+8.3f} [{r.d_hard_v1_lo:>+7.3f},{r.d_hard_v1_hi:>+7.3f}]")

    print(f"\n[verdict] gate: {crit}")
    for m, v in per_member.items():
        print(f"[verdict]   {m:12s} d={v['delta']:+.4f} CI [{v['lo']:+.4f},{v['hi']:+.4f}] "
              f"-> {'PASS' if v['pass'] else 'fail'}")
    print(f"[verdict]   {'mean3':12s} d={mean3['delta']:+.4f} "
          f"CI [{mean3['lo']:+.4f},{mean3['hi']:+.4f}] -> "
          f"{'PASS' if mean3['pass'] else 'fail'}")
    print(f"[verdict] ROUND-2 {'WARRANTED' if gate['round2'] else 'NOT warranted'}")
    print(f"[122] wrote {out_csv} + {out_json} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
