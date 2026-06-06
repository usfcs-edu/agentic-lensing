#!/usr/bin/env python3
"""26_fit_combiner.py — fit ensemble combiners on the calibrated member probs.

Combiners (fit on the val split, the held-out-from-member-training set):
  average  — parameterless baseline (what the repo's meta-learner collapsed to)
  logistic — linear stacking
  rf       — random-forest stacking (DES 2510.23782: tree combiners beat averaging)

Writes data/scores_combined.parquet [split,row_id,label,combiner,p].

    /home2/benson/.venvs/claudenet/bin/python 26_fit_combiner.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _clib as C
import _combine as CM
import _ensemble as E


def main():
    scores = CM.load_scores()
    ids_v, y_v, Pv, names = CM.matrix(scores, "val", "pc")
    print(f"[combiner] members = {names}")
    print(f"[combiner] fitting on val: {len(y_v)} rows ({int(y_v.sum())} pos)")
    combs = {k: E.fit_combiner(k, Pv, y_v) for k in ("average", "logistic", "rf")}

    out = []
    for split in ("val", "testneg", "storfer", "inchausti"):
        ids, y, P, _ = CM.matrix(scores, split, "pc")
        for k, fn in combs.items():
            out.append(pd.DataFrame({"split": split, "row_id": ids, "label": y,
                                     "combiner": k, "p": fn(P)}))
    pd.concat(out, ignore_index=True).to_parquet(C.DATA / "scores_combined.parquet", index=False)
    print(f"[26] wrote scores_combined.parquet ({len(names)} members -> avg/logistic/rf)")


if __name__ == "__main__":
    main()
