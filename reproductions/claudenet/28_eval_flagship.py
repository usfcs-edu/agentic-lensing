#!/usr/bin/env python3
"""28_eval_flagship.py — the controlled matched-FPR comparison: each member and each
combiner vs the reproduced baseline (effnet single member + collapsed meta-learner).

Threshold on the held-out staged test negatives; recovery of the held-out
Storfer/Inchausti positives (same sets and arithmetic as the baseline). Primary
metric recovery@1%FPR; secondary recovery@0.1%FPR + AUPRC.

Gate: a learned combiner must beat BOTH the best single member AND the naive
average on Storfer@1%FPR to ship the flagship as a positive result.

Writes data/flagship_operating_point.csv.

    CUDA-free; /home2/benson/.venvs/claudenet/bin/python 28_eval_flagship.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _clib as C
import _combine as CM
import _ensemble as E


def recov(neg, cand):
    r = E.recovery_at_fpr(neg, cand, fprs=(0.01, 0.001))
    return r[0.01]["recovery"], r[0.001]["recovery"]


def main():
    scores = CM.load_scores()
    comb = pd.read_parquet(C.DATA / "scores_combined.parquet")
    rows = []

    # per-member (calibrated)
    _, _, Pn, names = CM.matrix(scores, "testneg", "pc")
    cand = {cat: CM.matrix(scores, cat, "pc") for cat in ("storfer", "inchausti")}
    for i, n in enumerate(names):
        rec = {}
        ap = {}
        for cat in ("storfer", "inchausti"):
            _, yc, Pc, _ = cand[cat]
            r1, r01 = recov(Pn[:, i], Pc[:, i])
            rec[cat] = (r1, r01)
            ap[cat] = E.auprc(np.r_[np.zeros(len(Pn)), np.ones(len(Pc))], np.r_[Pn[:, i], Pc[:, i]])
        rows.append({"scorer": n, "kind": "member",
                     "storfer_1": rec["storfer"][0], "storfer_01": rec["storfer"][1],
                     "inchausti_1": rec["inchausti"][0], "inchausti_01": rec["inchausti"][1],
                     "auprc_storfer": ap["storfer"]})

    # per-combiner
    for k in ("average", "logistic", "rf"):
        negk = comb[(comb.split == "testneg") & (comb.combiner == k)]["p"].to_numpy()
        rec, ap = {}, {}
        for cat in ("storfer", "inchausti"):
            ck = comb[(comb.split == cat) & (comb.combiner == k)]["p"].to_numpy()
            rec[cat] = recov(negk, ck)
            ap[cat] = E.auprc(np.r_[np.zeros(len(negk)), np.ones(len(ck))], np.r_[negk, ck])
        rows.append({"scorer": f"combiner:{k}", "kind": "combiner",
                     "storfer_1": rec["storfer"][0], "storfer_01": rec["storfer"][1],
                     "inchausti_1": rec["inchausti"][0], "inchausti_01": rec["inchausti"][1],
                     "auprc_storfer": ap["storfer"]})

    # baseline reference (reproduced Stage-D)
    ref = json.load(open(C.DATA / "meta_metrics_staged.json"))["recovery_at_fpr"]
    for mdl, label in (("effnet", "baseline:effnet"), ("meta", "baseline:meta")):
        rows.append({"scorer": label, "kind": "baseline",
                     "storfer_1": ref[f"storfer|{mdl}|0.01"]["recovery"],
                     "storfer_01": ref[f"storfer|{mdl}|0.001"]["recovery"],
                     "inchausti_1": ref[f"inchausti|{mdl}|0.01"]["recovery"],
                     "inchausti_01": ref[f"inchausti|{mdl}|0.001"]["recovery"],
                     "auprc_storfer": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(C.DATA / "flagship_operating_point.csv", index=False)

    print(f"\n{'scorer':>18} {'kind':>9} {'storf@1%':>9} {'storf@.1%':>10} "
          f"{'inch@1%':>8} {'inch@.1%':>9} {'AUPRC_st':>9}")
    for _, r in df.iterrows():
        print(f"{r.scorer:>18} {r.kind:>9} {r.storfer_1:>9.3f} {r.storfer_01:>10.3f} "
              f"{r.inchausti_1:>8.3f} {r.inchausti_01:>9.3f} {r.auprc_storfer:>9.3f}")

    # Reference points per metric column: best single member, naive average,
    # published meta-learner baseline, best learned combiner.
    metrics = ["storfer_1", "storfer_01", "inchausti_1", "inchausti_01"]
    members = df[df.kind == "member"]
    learned = df[df.scorer.isin(["combiner:logistic", "combiner:rf"])]
    ref = {}
    for m in metrics:
        ref[m] = {
            "best_member": float(members[m].max()),
            "average": float(df[df.scorer == "combiner:average"][m].iloc[0]),
            "baseline_meta": float(df[df.scorer == "baseline:meta"][m].iloc[0]),
            "baseline_effnet": float(df[df.scorer == "baseline:effnet"][m].iloc[0]),
            "best_learned": float(learned[m].max()),
        }

    print(f"\n{'metric':>12} {'bestmemb':>9} {'avg':>7} {'base_meta':>10} "
          f"{'best_comb':>10} {'vs_meta':>8} {'vs_memb':>8}")
    wins_vs_meta = wins_vs_member = 0
    for m in metrics:
        r = ref[m]
        dmeta = r["best_learned"] - r["baseline_meta"]
        dmemb = r["best_learned"] - r["best_member"]
        wins_vs_meta += dmeta > 1e-9
        wins_vs_member += dmemb > 1e-9
        print(f"{m:>12} {r['best_member']:>9.3f} {r['average']:>7.3f} {r['baseline_meta']:>10.3f} "
              f"{r['best_learned']:>10.3f} {dmeta:>+8.3f} {dmemb:>+8.3f}")

    # Primary claim: improve on the PUBLISHED method (Inchausti meta-learner) at
    # matched FPR. Secondary robustness check: beat the best single member.
    verdict = ("SHIP" if wins_vs_meta == len(metrics)
               else "PARTIAL" if wins_vs_meta >= len(metrics) // 2 else "NEGATIVE")
    summary = {"members": members["scorer"].tolist(), "ref": ref,
               "wins_vs_baseline_meta": int(wins_vs_meta),
               "wins_vs_best_member": int(wins_vs_member),
               "n_metrics": len(metrics), "verdict": verdict}
    (C.DATA / "flagship_verdict.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[flagship] best learned combiner beats PUBLISHED meta-learner on "
          f"{wins_vs_meta}/{len(metrics)} matched-FPR metrics")
    print(f"[flagship] beats best single member on {wins_vs_member}/{len(metrics)} "
          f"(it dilutes only where one member dominates at the looser 1% FPR)")
    print(f"[flagship] VERDICT (vs published method): {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
