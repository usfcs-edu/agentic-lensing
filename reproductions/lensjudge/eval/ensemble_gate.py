#!/usr/bin/env python3
"""eval/ensemble_gate.py — grade-distribution ensembling across configs (v5 Phase A).

Verified research basis (arXiv 2606.00415): ensembles of many (model x prompt) VLM configs, combined
on the CLASS-PROBABILITY level, reached human-expert accuracy on galaxy-merger classification with
zero fine-tuning. Our analog: each config's run_batch parquet now carries the grade-token
distribution (gp_A..gp_D from llm_client.extract_grade_probs); this tool scores each config and
their MEAN-distribution ensemble on the human-labeled bench, with the mandatory
beat-the-best-single-pass gate (TTA/ensembling can hurt — adopt only on a measured win).

  python -m lensjudge.eval.ensemble_gate --manifest outputs/lensbench_manifest.csv \
      --preds outputs/a.parquet outputs/b.parquet ... [--labels-col source]

Scores per config + ensemble: detection AUC (lens vs random_neg), lens-vs-mimic AUC, using
(a) P(A)+P(B) and (b) expected-grade score sum(w_g * P(g)), w = A1.0/B0.7/C0.4/D0.0.
Rows are joined on `name`; per-row distributions are renormalized over A-D before averaging.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

GP = ["gp_A", "gp_B", "gp_C", "gp_D"]
W_EXP = np.array([1.0, 0.7, 0.4, 0.0])   # expected-grade weights A/B/C/D


def _norm_gp(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row grade distribution over A-D, renormalized (missing letters -> 0)."""
    g = df.reindex(columns=GP).astype(float).fillna(0.0)
    s = g.sum(axis=1)
    g = g.div(s.where(s > 0, 1.0), axis=0)
    g[s <= 0] = np.nan   # no distribution captured for this row
    return g


def _scores(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "s_ab": g["gp_A"] + g["gp_B"],
        "s_exp": g.to_numpy() @ W_EXP,
    }, index=g.index)


def _auc(j: pd.DataFrame, col: str, pos_mask, neg_mask) -> float:
    p = j.loc[pos_mask, col].dropna()
    n = j.loc[neg_mask, col].dropna()
    if not len(p) or not len(n):
        return float("nan")
    return roc_auc_score([1] * len(p) + [0] * len(n), list(p) + list(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="bench manifest with name + source columns")
    ap.add_argument("--preds", nargs="+", required=True, help="run_batch parquets (one per config)")
    ap.add_argument("--labels-col", default="source")
    ap.add_argument("--pos-prefix", default="graded", help="source prefix marking lens rows")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)[["name", args.labels_col]]
    per_cfg, dists = [], []
    base = None
    for p in args.preds:
        d = pd.read_parquet(p)
        g = _norm_gp(d).set_index(d["name"])
        s = _scores(g)
        j = man.merge(s.reset_index().rename(columns={"index": "name"}), on="name", how="inner")
        src = j[args.labels_col].astype(str)
        pos = src.str.startswith(args.pos_prefix)
        rn, mn = src.eq("random_neg"), src.eq("mimic_neg")
        row = {"config": Path(p).stem, "n": len(j),
               "det_ab": _auc(j, "s_ab", pos, rn), "mimic_ab": _auc(j, "s_ab", pos, mn),
               "det_exp": _auc(j, "s_exp", pos, rn), "mimic_exp": _auc(j, "s_exp", pos, mn)}
        per_cfg.append(row)
        dists.append(g)
        base = man if base is None else base

    # mean-of-distributions ensemble on the intersection of names
    common = dists[0].index
    for g in dists[1:]:
        common = common.intersection(g.index)
    ens = sum(g.loc[common] for g in dists) / len(dists)
    s = _scores(ens)
    j = man.merge(s.reset_index().rename(columns={"index": "name"}), on="name", how="inner")
    src = j[args.labels_col].astype(str)
    pos, rn, mn = src.str.startswith(args.pos_prefix), src.eq("random_neg"), src.eq("mimic_neg")
    per_cfg.append({"config": f"ENSEMBLE(mean of {len(dists)})", "n": len(j),
                    "det_ab": _auc(j, "s_ab", pos, rn), "mimic_ab": _auc(j, "s_ab", pos, mn),
                    "det_exp": _auc(j, "s_exp", pos, rn), "mimic_exp": _auc(j, "s_exp", pos, mn)})

    out = pd.DataFrame(per_cfg)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    singles = out.iloc[:-1]
    ens_row = out.iloc[-1]
    for metric in ("det_ab", "mimic_ab", "det_exp", "mimic_exp"):
        best = singles[metric].max()
        verdict = "BEATS" if ens_row[metric] > best else "does NOT beat"
        print(f"gate[{metric}]: ensemble {ens_row[metric]:.3f} {verdict} best single {best:.3f}")


if __name__ == "__main__":
    main()
