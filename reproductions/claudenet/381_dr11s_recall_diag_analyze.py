#!/usr/bin/env python3
"""381_dr11s_recall_diag_analyze.py — recall-recoverability diagnosis, local side.

Consumes 380's extracts (per-member RAW stage-1 scores for known lenses + a DR11 NegEval
sample) and answers: how much of the DR9->DR11 grade-A recall drop (62%->41%) is recoverable
WITHOUT a DR11-native retrain?

It (1) calibrates raw scores with the persisted v2-lean isotonic fits (so thresholds apply in
the same space the pipeline used); (2) decomposes the lost grade-A lenses into
  out-of-parent (not searchable)  |  in-parent-but-cut (the model/threshold problem)  |  recovered;
(3) measures the threshold-free separation (AUC of in-parent grade-A vs NegEval) per member and
for the 5-member mean — high AUC + low operating-point recall == recoverable by re-threshold/
combiner, not a feature shift; (4) draws the recall-vs-survivor-budget curve for the union rule
(DR9 vs DR11 thresholds) and the mean/max combiners, and reads off recall at the 95k operating
budget and at 150k. Held-out: curated-trained positions and (optionally) the catalogs' own
training members are excluded from the recall/AUC denominators.

  /home2/benson/.venvs/claudenet/bin/python 381_dr11s_recall_diag_analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import _clib as C

DATA = C.DATA / "v3"
MEMBERS = ["member_effnet_B", "member_effnet_B3_hard", "member_effnet_S2_hard",
           "member_resnet46_C_hard", "member_zoobot_N"]
SHORT = {m: m.replace("member_", "") for m in MEMBERS}
M_TOTAL = 53_809_040
OP_BUDGET = 95_104
FITS = C.DATA / "v2" / "ensemble_v2lean_fits"


def load_calibrators():
    M162 = C._load("cn381_162", C.ROOT / "162_stage2_rescore.py")
    return M162.load_calibrators(str(FITS))


def cal_frame(df, cals):
    return pd.DataFrame({m: cals[m].transform(np.nan_to_num(df[m].to_numpy(float), nan=0.0))
                         for m in MEMBERS}, index=df.index)


def dr9_thresholds():
    op = pd.read_csv(C.DATA / "v2" / "ensemble_v2lean_operating_points.csv")
    op = op[(op.fpr == 1e-4) & (op.scorer.isin(SHORT.values()))].set_index("scorer")["thr"]
    return {s: float(op[s]) for s in SHORT.values()}


def dr11_thresholds():
    j = json.load(open(DATA / "recalibrate_dr11s_summary.json"))["thr_dr10"]
    return {s: float(j[s]) for s in SHORT.values()}


def union_pass(cal, thr):
    p = np.zeros(len(cal), bool)
    for m in MEMBERS:
        p |= cal[m].to_numpy() >= thr[SHORT[m]]
    return p


def fpr_of(neg_score, t):
    return float(np.mean(neg_score >= t))


def main():
    kp = DATA / "dr11s_diag_knownlens.parquet"
    npq = DATA / "dr11s_diag_negeval.parquet"
    assert kp.exists() and npq.exists(), f"run 380 + scp first ({kp}, {npq})"
    known = pd.read_parquet(kp)
    neg = pd.read_parquet(npq)
    cals = load_calibrators()
    survivors = set(pd.read_parquet(DATA / "survivors_dr11s_recal.parquet").row_id.astype(str))

    cal_k = cal_frame(known, cals)
    cal_n = cal_frame(neg, cals)
    thr9, thr11 = dr9_thresholds(), dr11_thresholds()
    print("[381] per-member 1e-4 thresholds  DR9 -> DR11:")
    for s in SHORT.values():
        print(f"   {s:18s} {thr9[s]:.4f} -> {thr11[s]:.4f}"
              + ("  (SATURATED)" if thr11[s] >= 1.0 else ""))

    # ---- leakage flag: a held-out catalog entry within 5" of a curated training lens ----
    cur = known[known.src == "curated"]
    cur_xyz = np.radians(cur[["match_ra", "match_dec"]].to_numpy(float)) if len(cur) else None
    # use catalog coords for leakage (curated positions)
    from scipy.spatial import cKDTree

    def to_xyz(ra, dec):
        ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
        return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    curm = known[known.src == "curated"]
    ctree = cKDTree(to_xyz(curm.ra, curm.dec))
    d, _ = ctree.query(to_xyz(known.ra, known.dec), k=1)
    known["leaked_train"] = (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600) < 5.0

    known["recovered"] = known.match_rowid.isin(survivors) & known.in_parent
    known["union_pass_dr11"] = union_pass(cal_k, thr11) & known.in_parent
    known["union_pass_dr9"] = union_pass(cal_k, thr9) & known.in_parent
    cal_k["mean5"] = cal_k[MEMBERS].mean(axis=1)
    cal_k["max5"] = cal_k[MEMBERS].max(axis=1)
    cal_n["mean5"] = cal_n[MEMBERS].mean(axis=1)
    cal_n["max5"] = cal_n[MEMBERS].max(axis=1)

    rep = {"m_total": M_TOTAL, "op_budget": OP_BUDGET, "thr9": thr9, "thr11": thr11,
           "decomp": {}, "auc": {}, "curves": {}, "ablations": {}}

    # ===== 1. decomposition (Inchausti A/B, Storfer A) =====
    def decomp(mask, label):
        sub = known[mask]
        n = len(sub)
        inp = int(sub.in_parent.sum())
        rec = int(sub.recovered.sum())
        d = dict(n=n, out_of_parent=n - inp, in_parent=inp, recovered=rec,
                 in_parent_cut=inp - rec,
                 recall_raw=rec / n if n else float("nan"),
                 recall_in_parent=rec / inp if inp else float("nan"))
        held = sub[~sub.leaked_train]
        d["n_heldout"] = len(held)
        d["recovered_heldout"] = int(held.recovered.sum())
        d["recall_heldout_raw"] = d["recovered_heldout"] / len(held) if len(held) else float("nan")
        rep["decomp"][label] = d
        print(f"\n[381] {label}: N={n}  out-of-parent={n-inp}  in-parent={inp}  "
              f"recovered={rec}  in-parent-but-CUT={inp-rec}")
        print(f"        recall_raw={d['recall_raw']:.3f}  "
              f"recall|in-parent(searchable)={d['recall_in_parent']:.3f}  "
              f"recall_heldout_raw={d['recall_heldout_raw']:.3f}")
        return d

    iA = (known.src == "inchausti") & (known.grade == "A")
    iB = (known.src == "inchausti") & (known.grade == "B")
    sA = (known.src == "storfer") & (known.grade == "A")
    decomp(iA, "inchausti_A")
    decomp(iB, "inchausti_B")
    decomp(sA, "storfer_A")

    # ===== 2. threshold-free AUC: in-parent held-out grade-A vs NegEval =====
    posmask = (known.src.isin(["inchausti", "storfer"]) & (known.grade == "A")
               & known.in_parent & ~known.leaked_train)
    print(f"\n[381] AUC positives (in-parent, held-out grade-A): {int(posmask.sum())}  "
          f"vs NegEval n={len(neg):,}")
    for col in MEMBERS + ["mean5", "max5"]:
        y = np.r_[np.ones(int(posmask.sum())), np.zeros(len(cal_n))]
        s = np.r_[cal_k.loc[posmask, col].to_numpy(), cal_n[col].to_numpy()]
        rep["auc"][SHORT.get(col, col)] = float(roc_auc_score(y, s))
    for k, v in rep["auc"].items():
        print(f"   AUC {k:18s} {v:.4f}")

    # ===== 3. recall-vs-budget curves (held-out in-parent grade-A) =====
    posA = cal_k[posmask]
    fprs = np.unique(np.r_[np.logspace(-5.3, -1.5, 60), OP_BUDGET / M_TOTAL, 150_000 / M_TOTAL])
    for combiner in ["mean5", "max5"]:
        ps, ns = posA[combiner].to_numpy(), cal_n[combiner].to_numpy()
        rows = []
        for f in fprs:
            t = float(np.quantile(ns, 1 - f))
            rows.append((f, f * M_TOTAL, float(np.mean(ps >= t))))
        rep["curves"][combiner] = rows
    # union curve is a single operating point per threshold-set (not a sweep)
    for tag, thr in [("union_dr9", thr9), ("union_dr11", thr11)]:
        pn = union_pass(cal_n, thr)
        pp = union_pass(posA, thr)
        rep["ablations"][tag] = dict(
            fpr=float(pn.mean()), est_survivors=float(pn.mean() * M_TOTAL),
            recall=float(pp.mean()))

    # recall of each combiner at the OP budget (95k) and at 150k
    def recall_at_budget(combiner, budget):
        ps, ns = posA[combiner].to_numpy(), cal_n[combiner].to_numpy()
        t = float(np.quantile(ns, 1 - budget / M_TOTAL))
        return float(np.mean(ps >= t))
    rep["ablations"]["mean5_at_95k"] = recall_at_budget("mean5", OP_BUDGET)
    rep["ablations"]["mean5_at_150k"] = recall_at_budget("mean5", 150_000)
    rep["ablations"]["max5_at_95k"] = recall_at_budget("max5", OP_BUDGET)
    rep["ablations"]["max5_at_150k"] = recall_at_budget("max5", 150_000)

    # drop the saturated resnet member from the union (DR11 thr)
    others = [m for m in MEMBERS if m != "member_resnet46_C_hard"]
    def union_pass_sub(cal, thr, mem):
        p = np.zeros(len(cal), bool)
        for m in mem:
            p |= cal[m].to_numpy() >= thr[SHORT[m]]
        return p
    rep["ablations"]["union_dr11_no_resnet"] = dict(
        fpr=float(union_pass_sub(cal_n, thr11, others).mean()),
        recall=float(union_pass_sub(posA, thr11, others).mean()))

    print("\n[381] ablations / operating points (held-out in-parent grade-A):")
    print(f"   union DR9  thr : recall={rep['ablations']['union_dr9']['recall']:.3f}  "
          f"FPR={rep['ablations']['union_dr9']['fpr']:.2e}  "
          f"est_survivors={rep['ablations']['union_dr9']['est_survivors']:,.0f}")
    print(f"   union DR11 thr : recall={rep['ablations']['union_dr11']['recall']:.3f}  "
          f"FPR={rep['ablations']['union_dr11']['fpr']:.2e}  "
          f"est_survivors={rep['ablations']['union_dr11']['est_survivors']:,.0f}")
    print(f"   mean5 @95k     : recall={rep['ablations']['mean5_at_95k']:.3f}")
    print(f"   mean5 @150k    : recall={rep['ablations']['mean5_at_150k']:.3f}")
    print(f"   max5  @95k     : recall={rep['ablations']['max5_at_95k']:.3f}")

    out = DATA / "dr11s_recall_diag_summary.json"
    json.dump(rep, open(out, "w"), indent=2, default=float)
    print(f"\n[381] wrote {out}")

    # ---- figure: recall vs survivor budget ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for combiner, c in [("mean5", "C0"), ("max5", "C1")]:
            arr = np.array(rep["curves"][combiner])
            ax.plot(arr[:, 1], arr[:, 2], "-", color=c, label=f"{combiner} combiner")
        ax.scatter([rep["ablations"]["union_dr11"]["est_survivors"]],
                   [rep["ablations"]["union_dr11"]["recall"]], color="k", zorder=5,
                   label="union DR11 (current op)")
        ax.scatter([rep["ablations"]["union_dr9"]["est_survivors"]],
                   [rep["ablations"]["union_dr9"]["recall"]], color="r", marker="x", zorder=5,
                   label="union DR9 thr")
        ax.axvline(OP_BUDGET, ls=":", c="gray"); ax.axvline(150_000, ls="--", c="gray")
        ax.set_xscale("log"); ax.set_xlabel("estimated survivors (FPR x 53.8M)")
        ax.set_ylabel("recall of held-out in-parent grade-A")
        ax.set_title("DR11-south stage-1 recall recoverability (no retrain)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(DATA / "dr11s_recall_diag.png", dpi=150)
        print(f"[381] wrote {DATA/'dr11s_recall_diag.png'}")
    except Exception as e:
        print(f"[381] (figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
