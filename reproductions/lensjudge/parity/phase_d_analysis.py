#!/usr/bin/env python3
"""Phase D: the pre-specified parity analysis (single scored run, frozen predictors).

Endpoints (approved plan; FINDINGS.md):
  E2 (PRIMARY, truth-referenced): on arm1 decided rows (confirmed_lens vs non_lens),
      compare each machine score to the HUMAN ordinal grade (A=3,B=2,C=1) as paired
      predictors of truth. Non-inferiority framing: dAUC = AUC_machine - AUC_human
      with paired stratified bootstrap CI (2000, seed 2026) + DeLong p; margin
      delta0 = 0.05 (reader-study standard; power_check.py showed today's n decides
      only coarse margins - report estimation regardless).
  E1 (secondary, agreement): QWK(machine grade, consensus grade) on arm1 graded rows
      vs the human anchors (0.776 individual-grader upper bound / 0.42 mutual / 0.29
      independent team).
  Arm2 (context): truth-referenced AUC on the 44 SLDE-B-decided rows; the Euclid
      panel's expert_score as a resolution-privileged upper anchor (not a paired
      comparator); grade-stratified score distributions at scale.

Predictors (all frozen before this run):
  human    arm1 grade ordinal (the comparator)
  cnn      published NeuraLens probability (like-for-like p_pub; C1a)
  student  ckpt-750 grade-token logprob (preds_bench_arm{1,2}.parquet, Perlmutter)
  rep      C1b probe (outputs/rep_probe/bench_scores_arm{1,2}.csv)
  [c3]     documented reference only (gate HARD 0.487) - not run at bench scale.

  python lensjudge/parity/phase_d_analysis.py

Writes outputs/parity_phase_d.json + prints the report tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score, roc_auc_score

OUT = Path(__file__).resolve().parents[1] / "outputs"
RNG = np.random.default_rng(2026)
GMAP = {"A": 3, "B": 2, "C": 1, "D": 0}
DELTA0 = 0.05


def paired_boot(y, a, b, n=2000):
    """Paired stratified bootstrap of dAUC = AUC(a) - AUC(b)."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    y, a, b = y[ok], a[ok], b[ok]
    d0 = roc_auc_score(y, a) - roc_auc_score(y, b)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        bi = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        if len(np.unique(y[bi])) == 2:
            v.append(roc_auc_score(y[bi], a[bi]) - roc_auc_score(y[bi], b[bi]))
    return d0, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(ok.sum())


def auc_ci(y, s, n=2000):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    a = roc_auc_score(y, s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(n):
        bi = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        if len(np.unique(y[bi])) == 2:
            v.append(roc_auc_score(y[bi], s[bi]))
    return a, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(ok.sum())


def delong_p(y, a, b):
    """DeLong test for paired AUC difference (two-sided p)."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    y, a, b = y[ok], a[ok], b[ok]
    pos, neg = y == 1, y == 0

    def structural(s):
        # V10: for each positive, fraction of negatives it beats; V01 symmetric.
        sp, sn = s[pos], s[neg]
        v10 = np.array([(np.sum(x > sn) + 0.5 * np.sum(x == sn)) / len(sn) for x in sp])
        v01 = np.array([(np.sum(sp > x) + 0.5 * np.sum(sp == x)) / len(sp) for x in sn])
        return v10, v01

    v10a, v01a = structural(a)
    v10b, v01b = structural(b)
    auc_a, auc_b = v10a.mean(), v10b.mean()
    m, n_ = len(v10a), len(v01a)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n_
    if var <= 0:
        return float("nan")
    z = (auc_a - auc_b) / np.sqrt(var)
    return float(2 * norm.sf(abs(z)))


def load_scores():
    a1 = pd.read_csv(OUT / "parity_bench_arm1.csv")
    a2 = pd.read_csv(OUT / "parity_bench_arm2.csv")
    # student (Perlmutter, grade-token logprob)
    for arm, df in (("arm1", a1), ("arm2", a2)):
        p = OUT / f"preds_bench_{arm}.parquet"
        if p.exists():
            s = pd.read_parquet(p)[["name", "p_lens_logprob", "agent_grade"]]
            s = s.rename(columns={"p_lens_logprob": "s_student", "agent_grade": "g_student"})
            df = df.merge(s, on="name", how="left")
        else:
            df["s_student"], df["g_student"] = np.nan, None
            print(f"  [warn] {p.name} not staged yet")
        # rep probe (C1b)
        r = OUT / "rep_probe" / f"bench_scores_{arm}.csv"
        if r.exists():
            rp = pd.read_csv(r).rename(columns={"p_lens_rep": "s_rep"})
            df = df.merge(rp[["name", "s_rep"]], on="name", how="left")
        if arm == "arm1":
            a1 = df
        else:
            a2 = df
    # cnn: like-for-like PUBLISHED probability. The decided arm1 rows are mostly
    # Paper II (huang2021) candidates, whose published ResNet probability lives in
    # the huang2021 catalog, not the inchausti score CSVs; huang2020's catalog
    # publishes no probability at all (those rows stay NaN — reported as reduced n).
    from lensjudge import config  # noqa: PLC0415
    st = pd.read_csv(config.INCH_DATA / "candidate_scores_storfer.csv")[["name", "probability"]] \
        .rename(columns={"probability": "s_cnn"})
    inc = pd.read_csv(config.INCH_DATA / "candidate_scores_inchausti.csv")[["name", "p_meta"]] \
        .rename(columns={"p_meta": "s_cnn"})
    h21 = pd.read_csv(config.REPRO / "huang-2021" / "data" / "huang2021_published_catalog.csv")[
        ["name", "probability"]].rename(columns={"probability": "s_cnn"})
    pub = pd.concat([st, inc, h21]).dropna(subset=["s_cnn"]).drop_duplicates("name")
    a1 = a1.merge(pub, on="name", how="left")
    return a1, a2


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    a1, a2 = load_scores()
    res = {"delta0": DELTA0, "seed": 2026}

    # ---------- E2 primary: arm1 decided ----------
    dec = a1[(a1.arm == "arm1") & a1.truth_label.isin(["lens", "nonlens"])].copy()
    dec["s_human"] = dec.grade.map(GMAP).astype(float)
    y = (dec.truth_label == "lens").to_numpy(int)
    print(f"=== E2 PRIMARY: arm1 decided (n={len(dec)}: {y.sum()} lens / {(1-y).sum()} refuted) ===")
    e2 = {}
    for tag in ("human", "cnn", "student", "rep"):
        s = dec[f"s_{tag}"].to_numpy(float)
        ok = ~np.isnan(s)
        if len(np.unique(y[ok])) < 2:
            print(f"  AUC {tag:8s} SKIPPED (coverage leaves one truth class; n={ok.sum()})")
            continue
        a, lo, hi, n = auc_ci(y, s)
        e2[tag] = {"auc": a, "ci": [lo, hi], "n": n}
        print(f"  AUC {tag:8s} {a:.3f} [{lo:.3f}, {hi:.3f}]  (n={n})")
    print(f"\n  paired vs HUMAN (dAUC = machine - human; NI margin delta0 = {DELTA0}):")
    for tag in ("cnn", "student", "rep"):
        s = dec[f"s_{tag}"].to_numpy(float)
        if np.isnan(s).all():
            continue
        d, lo, hi, n = paired_boot(y, s, dec.s_human.to_numpy(float))
        p = delong_p(y, s, dec.s_human.to_numpy(float))
        ni = "NON-INFERIOR" if lo > -DELTA0 else ("inconclusive" if hi > -DELTA0 else "inferior")
        e2[f"d_{tag}"] = {"dauc": d, "ci": [lo, hi], "delong_p": p, "n_pairs": n, "verdict": ni}
        print(f"    {tag:8s} dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  DeLong p={p:.3f}  -> {ni}")
    res["e2"] = e2

    # ---------- E1 secondary: agreement on arm1 graded rows ----------
    print("\n=== E1 secondary: QWK vs consensus (arm1 graded truth rows) ===")
    g1 = a1[(a1.arm == "arm1") & a1.grade.isin(list("ABC"))].copy()
    e1 = {}
    if g1.g_student.notna().any():
        t = g1.grade.map(GMAP)
        pr = g1.g_student.astype(str).str.strip().str.upper().str[0].map(GMAP)
        ok = t.notna() & pr.notna()
        k = cohen_kappa_score(t[ok], pr[ok], weights="quadratic")
        e1["student_qwk"] = {"qwk": float(k), "n": int(ok.sum())}
        print(f"  student QWK {k:.3f} (n={int(ok.sum())}) | anchors: 0.776 grader-vs-consensus, "
              f"0.42 mutual, 0.29 independent team")
    res["e1"] = e1

    # ---------- Arm2: truth-referenced at scale ----------
    print("\n=== Arm2: Euclid-field context ===")
    t2 = a2[a2.truth_label.isin(["lens", "nonlens"])].copy()
    arm2 = {}
    if len(t2):
        y2 = (t2.truth_label == "lens").to_numpy(int)
        print(f"  decided rows n={len(t2)} ({y2.sum()}/{(1-y2).sum()})")
        for tag in ("student", "rep"):
            s = t2[f"s_{tag}"].to_numpy(float)
            if np.isnan(s).all():
                continue
            a, lo, hi, n = auc_ci(y2, s)
            arm2[tag] = {"auc": a, "ci": [lo, hi], "n": n}
            print(f"  AUC {tag:8s} {a:.3f} [{lo:.3f}, {hi:.3f}]")
        # Euclid expert score as the resolution-privileged anchor
        es = pd.to_numeric(t2.get("euclid_score"), errors="coerce").to_numpy(float)
        if not np.isnan(es).all() and len(np.unique(y2[~np.isnan(es)])) == 2:
            a, lo, hi, n = auc_ci(y2, es)
            arm2["euclid_expert"] = {"auc": a, "ci": [lo, hi], "n": n}
            print(f"  AUC euclid-expert(0.1\" anchor) {a:.3f} [{lo:.3f}, {hi:.3f}] (n={n})")
    # grade-stratified student score at scale
    if a2.s_student.notna().any():
        strat = a2.groupby("euclid_grade").s_student.agg(["mean", "median", "count"]).round(3)
        arm2["by_euclid_grade"] = json.loads(strat.to_json())
        print(f"  student score by Euclid grade:\n{strat.to_string()}")
    res["arm2"] = arm2

    with open(OUT / "parity_phase_d.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nsaved {OUT / 'parity_phase_d.json'}")


if __name__ == "__main__":
    main()
