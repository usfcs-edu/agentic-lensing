#!/usr/bin/env python3
"""27_correlation_report.py — the diversity proof: pairwise member correlations and
Yule's Q. Low off-diagonals => decorrelated members => a combiner has something to
exploit (the opposite of the repo's collapsed meta-learner).

Writes data/diversity.json (+ prints matrices).

    /home2/benson/.venvs/claudenet/bin/python 27_correlation_report.py
"""
from __future__ import annotations

import json

import numpy as np

import _clib as C
import _combine as CM
import _ensemble as E


def show(mat, names, title):
    print(f"\n{title}")
    print("            " + "".join(f"{n[:10]:>11}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:11]:>11} " + "".join(f"{mat[i, j]:>11.3f}" for j in range(len(names))))


def main():
    scores = CM.load_scores()
    # score correlation on test negatives (where false positives live)
    _, _, Pn, names = CM.matrix(scores, "testneg", "pc")
    Sp = E.score_correlation(Pn, "pearson")
    Ss = E.score_correlation(Pn, "spearman")
    # error correlation + Q-statistic need both classes -> use val (pos+neg)
    _, yv, Pv, _ = CM.matrix(scores, "val", "pc")
    Ec = E.error_correlation(Pv, yv)
    Q = E.q_statistic(Pv, yv)

    show(Sp, names, "Pearson score correlation (test negatives)")
    show(Ss, names, "Spearman score correlation (test negatives)")
    show(Q, names, "Yule's Q (val; lower=more diverse)")

    def offdiag_mean(m):
        m = np.asarray(m); n = m.shape[0]
        return float((m.sum() - np.trace(m)) / (n * n - n)) if n > 1 else 0.0

    summary = {"members": names,
               "pearson_offdiag_mean": offdiag_mean(Sp),
               "spearman_offdiag_mean": offdiag_mean(Ss),
               "q_offdiag_mean": offdiag_mean(Q),
               "pearson": Sp.tolist(), "spearman": Ss.tolist(),
               "error_corr": Ec.tolist(), "q": Q.tolist()}
    (C.DATA / "diversity.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[27] mean off-diagonal: Pearson {summary['pearson_offdiag_mean']:.3f} "
          f"Spearman {summary['spearman_offdiag_mean']:.3f} Q {summary['q_offdiag_mean']:.3f}")
    print("[27] wrote diversity.json")


if __name__ == "__main__":
    main()
