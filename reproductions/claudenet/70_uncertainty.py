#!/usr/bin/env python3
"""70_uncertainty.py — Phase 6: deep-ensemble epistemic uncertainty for human-vetting
triage and OOD/domain-shift flagging. Free byproduct of the flagship's decorrelated
members (no extra training).

Per candidate: ensemble mean mu and disagreement sigma (std across calibrated member
scores). We report:
  1. Inspection efficiency: lenses-per-inspected-object when the candidate list is
     ranked by the ensemble mean vs the best single member (the triage win).
  2. Selective prediction: risk-coverage — abstaining on the most uncertain objects
     lowers the error rate on the retained set.
  3. OOD flag: ensemble disagreement on north (dec>32) vs south candidates.

Sold on inspection efficiency / triage, NOT recovery@1%FPR (UQ doesn't move the
ranking by itself). CPU-only.

    /home2/benson/.venvs/claudenet/bin/python 70_uncertainty.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _clib as C
import _combine as CM


def main():
    scores = CM.load_scores()
    names = CM.member_names()

    # member matrices on the labeled pool: test negatives (0) + storfer positives (1)
    _, yn, Pn, _ = CM.matrix(scores, "testneg", "pc")
    ids_s, ys, Ps, _ = CM.matrix(scores, "storfer", "pc")
    P = np.vstack([Pn, Ps]); y = np.concatenate([yn, ys])
    mu = P.mean(1); sigma = P.std(1)

    # best single member by storfer recovery@1%FPR
    def rec1(col_neg, col_pos):
        thr = np.quantile(col_neg, 0.99)
        return (col_pos >= thr).mean()
    best_i = int(np.argmax([rec1(Pn[:, i], Ps[:, i]) for i in range(len(names))]))
    best_member = names[best_i]
    member_score = P[:, best_i]

    # ---- 1. inspection efficiency (rank desc, top-K) ----
    res = {"best_member": best_member, "inspection": {}}
    print(f"[uq] inspection efficiency (pool = {len(yn)} neg + {len(ys)} lens, "
          f"prevalence {ys.mean()*0+ys.sum()/len(y):.3f})")
    print(f"{'top-K':>7} {'ens_purity':>11} {'memb_purity':>12} {'ens_complete':>13}")
    for K in (200, 500, 1000, 2000):
        ens_top = np.argsort(-mu)[:K]
        memb_top = np.argsort(-member_score)[:K]
        ens_pur = y[ens_top].mean()
        memb_pur = y[memb_top].mean()
        ens_compl = y[ens_top].sum() / y.sum()
        res["inspection"][K] = {"ens_purity": float(ens_pur), "memb_purity": float(memb_pur),
                                "ens_completeness": float(ens_compl)}
        print(f"{K:>7} {ens_pur:>11.3f} {memb_pur:>12.3f} {ens_compl:>13.3f}")

    # ---- 2. selective prediction (risk-coverage by ascending sigma) ----
    thr = np.quantile(mu[y == 0], 0.99)            # 1% FPR operating point
    pred = (mu >= thr).astype(int)
    err = (pred != y).astype(float)
    order = np.argsort(sigma)                       # most confident (low disagreement) first
    cov_grid = [0.5, 0.7, 0.9, 1.0]
    res["selective"] = {}
    print(f"\n[uq] selective prediction (abstain on high ensemble disagreement)")
    print(f"{'coverage':>9} {'error':>7}")
    for cov in cov_grid:
        k = int(cov * len(order))
        e = float(err[order[:k]].mean())
        res["selective"][cov] = e
        print(f"{cov:>9.1f} {e:>7.4f}")

    # ---- 3. OOD: north vs south disagreement on storfer candidates ----
    cat = pd.read_csv(C.DATA / "storfer2024_published_catalog.csv")
    radec = dict(zip(cat["name"].astype(str), cat["DEC"])) if "DEC" in cat.columns else {}
    if radec:
        dec_s = np.array([radec.get(str(r), np.nan) for r in ids_s])
        sig_s = Ps.std(1)
        north = dec_s > 32.375
        ok = np.isfinite(dec_s)
        res["ood"] = {"sigma_north": float(np.nanmean(sig_s[ok & north])),
                      "sigma_south": float(np.nanmean(sig_s[ok & ~north])),
                      "n_north": int((ok & north).sum()), "n_south": int((ok & ~north).sum())}
        print(f"\n[uq] OOD disagreement on storfer lenses: north sigma "
              f"{res['ood']['sigma_north']:.3f} (n={res['ood']['n_north']}) vs south "
              f"{res['ood']['sigma_south']:.3f} (n={res['ood']['n_south']})")

    (C.DATA / "uncertainty.json").write_text(json.dumps(res, indent=2))
    print("\n[70] wrote uncertainty.json")


if __name__ == "__main__":
    raise SystemExit(main())
