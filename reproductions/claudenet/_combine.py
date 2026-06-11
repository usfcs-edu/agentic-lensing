#!/usr/bin/env python3
"""_combine.py — load per-member score parquets and build aligned member-score
matrices for the Phase-1 calibration / combiner / correlation / eval steps."""
from __future__ import annotations

import numpy as np
import pandas as pd

import _clib as C


def member_names():
    return [p.name[len("scores_member_"):-len(".parquet")]
            for p in sorted(C.DATA.glob("scores_member_*.parquet"))]


def load_scores():
    return {n: pd.read_parquet(C.DATA / f"scores_member_{n}.parquet") for n in member_names()}


def matrix(scores, split, col="pc"):
    """Return (row_ids, y, P[n_rows, n_members], names) for a split, aligned by
    row_id (inner join across members); rows with any non-finite score dropped."""
    names = list(scores.keys())
    base = None
    for n in names:
        s = scores[n]
        s = s[s.split == split][["row_id", "label", col]].rename(columns={col: n})
        base = s if base is None else base.merge(s.drop(columns="label"), on="row_id", how="inner")
    P = base[names].to_numpy(dtype=float)
    ok = np.isfinite(P).all(1)
    return base["row_id"].to_numpy()[ok], base["label"].to_numpy()[ok], P[ok], names
