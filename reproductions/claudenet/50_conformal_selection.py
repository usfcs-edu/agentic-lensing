#!/usr/bin/env python3
"""50_conformal_selection.py — Phase 4: distribution-free FDR-controlled candidate
selection on top of the flagship ensemble (Jin & Candes 2023 conformal selection +
Benjamini-Hochberg).

The repo's deployment failure was an operating-point problem (a fixed p>=0.5 flags
37-51% of random galaxies). Conformal selection turns the arbitrary threshold into
a CERTIFIED one: with a held-out calibration set of confirmed non-lenses, it returns
a candidate set whose false-discovery rate is provably <= a chosen target (under
exchangeability of calibration/test negatives).

Procedure (one-sided, high score = lens-like):
  conformal p-value  p_j = (1 + #{cal_neg score >= s_j}) / (n_cal + 1)
  BH at target FDR alpha -> selected set
We measure the EMPIRICAL FDR (fraction of selected that are non-lenses) and
completeness vs the nominal alpha, on a test set of held-out negatives + Storfer
positives. CPU-only.

    /home2/benson/.venvs/claudenet/bin/python 50_conformal_selection.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _clib as C

ALPHAS = [0.05, 0.10, 0.25, 0.50]


def conformal_pvalues(test_scores, cal_neg_scores):
    cal = np.sort(cal_neg_scores)
    n = len(cal)
    # #{cal >= s} = n - searchsorted(cal, s, 'left')
    ge = n - np.searchsorted(cal, test_scores, side="left")
    return (1 + ge) / (n + 1)


def bh_select(pvals, alpha):
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = 0
    for k in range(1, m + 1):
        if pvals[order[k - 1]] <= k / m * alpha:
            thresh = k
    sel = np.zeros(m, bool)
    sel[order[:thresh]] = True
    return sel


def main():
    comb = pd.read_parquet(C.DATA / "scores_combined.parquet")
    rng = np.random.default_rng(C.SEED)
    out = {}
    for combiner in ("average", "rf"):
        c = comb[comb.combiner == combiner]
        neg = c[c.split == "testneg"]["p"].to_numpy()
        pos = c[c.split == "storfer"]["p"].to_numpy()
        neg = neg[np.isfinite(neg)]; pos = pos[np.isfinite(pos)]
        # split negatives into calibration / test
        idx = rng.permutation(len(neg))
        half = len(neg) // 2
        cal_neg = neg[idx[:half]]
        test_neg = neg[idx[half:]]
        test_scores = np.concatenate([test_neg, pos])
        test_is_neg = np.concatenate([np.ones(len(test_neg), bool), np.zeros(len(pos), bool)])
        pv = conformal_pvalues(test_scores, cal_neg)
        rows = []
        for a in ALPHAS:
            sel = bh_select(pv, a)
            nsel = int(sel.sum())
            fdr = float(test_is_neg[sel].mean()) if nsel else 0.0
            compl = float((sel & ~test_is_neg).sum() / max((~test_is_neg).sum(), 1))
            rows.append({"alpha": a, "n_selected": nsel, "empirical_fdr": fdr,
                         "completeness": compl, "valid": fdr <= a + 0.02})
        out[combiner] = rows
        print(f"\n[conformal:{combiner}] test = {len(test_neg)} neg + {len(pos)} storfer pos "
              f"(prevalence {len(pos)/len(test_scores):.2f})")
        print(f"{'alpha':>6} {'n_sel':>6} {'emp_FDR':>8} {'complete':>9} {'valid':>6}")
        for r in rows:
            print(f"{r['alpha']:>6.2f} {r['n_selected']:>6d} {r['empirical_fdr']:>8.3f} "
                  f"{r['completeness']:>9.3f} {str(r['valid']):>6}")

    (C.DATA / "conformal_selection.json").write_text(json.dumps(out, indent=2))
    print("\n[50] wrote conformal_selection.json")
    print("[50] NOTE: the FDR guarantee is MARGINAL and assumes calibration/test "
          "negatives are exchangeable; under north/south or sim-to-real shift use "
          "group-conformal calibration (Phase 5 territory).")


if __name__ == "__main__":
    raise SystemExit(main())
