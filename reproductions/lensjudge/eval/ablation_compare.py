#!/usr/bin/env python3
"""Compare original-rubric vs triage-rubric grading on the 48-system slice.

Original-rubric columns are the slice subsets of the full-manifest runs (the
slice is a strict subset of lensbench_manifest.csv), so they cost nothing new.

  python lensjudge/eval/ablation_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402

SLICE = pd.read_csv(config.OUT / "lensbench_slice48.csv")
NAMES = set(SLICE.name)

CONFIGS = [
    ("baseline (frozen)",      "preds_lensbench_lean.parquet",        True),
    ("Sonnet 4.6 original",    "preds_lean_sonnet46_think.parquet",   True),
    ("Opus 4.8 original",      "preds_lean_opus48_think.parquet",     True),
    ("Sonnet 4.6 triage",      "preds_slice48_sonnet46_triage.parquet", False),
    ("Opus 4.8 triage",        "preds_slice48_opus48_triage.parquet",  False),
]


def metrics(df: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    ok = df[df.parse_ok == True].copy()  # noqa: E712
    ok["is_lens"] = ok.grade_truth.isin(["A", "B", "C"]).astype(int)
    ok["pred_lens"] = ok.grade_pred.isin(["A", "B", "C"]).astype(int)
    out = {"n": len(df), "parse": f"{len(ok)}/{len(df)}"}
    try:
        out["AUC"] = round(float(roc_auc_score(ok.is_lens, ok.p_lens.fillna(0))), 3)
    except Exception:
        out["AUC"] = float("nan")
    out["escalation"] = round(float(ok.escalate.mean()), 2)
    out["p_lens_A"] = round(float(ok[ok.grade_truth == "A"].p_lens.mean()), 3)
    out["recovery_ABC"] = round(float(ok[ok.is_lens == 1].pred_lens.mean()), 2)
    d_pred = ok[ok.grade_pred == "D"]
    out["D_precision"] = (round(float((d_pred.grade_truth == "D").mean()), 2)
                          if len(d_pred) else float("nan"))
    out["D_rate"] = round(float((ok.grade_pred == "D").mean()), 2)
    out["grades"] = ok.grade_pred.value_counts().reindex(list("ABCD"), fill_value=0).to_dict()
    out["cost"] = round(float(df.cost_usd.mean()), 3)
    return out


FULL_CONFIGS = [
    ("baseline (frozen)",   "preds_lensbench_lean.parquet"),
    ("Sonnet 4.6 original", "preds_lean_sonnet46_think.parquet"),
    ("Opus 4.8 original",   "preds_lean_opus48_think.parquet"),
    ("Sonnet 4.6 triage",   "preds_lean_sonnet46_triage.parquet"),
    ("Opus 4.8 triage",     "preds_lean_opus48_triage.parquet"),
]


def strata(df: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Per-stratum behavior: kept-in-lens-bin rate (gold/graded ABC) and
    rejection rate (random_neg / graded_D)."""
    m = df.merge(manifest[["name", "source"]], on="name", how="left")
    ok = m[m.parse_ok == True].copy()  # noqa: E712
    ok["pred_lens"] = ok.grade_pred.isin(["A", "B", "C"])
    out = {}
    for src, want in (("gold", None), ("graded", None),
                      ("graded_D", False), ("random_neg", False)):
        d = ok[ok.source == src]
        if src == "gold":   # gold tier mixes confirmed lenses + 4 non-lenses
            d = d[d.grade_truth.isin(["A", "B", "C"])]
        if not len(d):
            continue
        kept = float(d.pred_lens.mean())
        out[src] = round(kept if want is None else 1 - kept, 2)
    return out


def main():
    full = "--full" in sys.argv
    manifest = pd.read_csv(config.OUT / "lensbench_manifest.csv")
    rows, strat_rows = {}, {}
    configs = ([(l, f, False) for l, f in FULL_CONFIGS] if full else CONFIGS)
    for label, fname, subset in configs:
        p = config.OUT / fname
        if not p.exists():
            print(f"[skip] {label}: {fname} missing")
            continue
        df = pd.read_parquet(p)
        if subset:
            df = df[df.name.isin(NAMES)]
        rows[label] = metrics(df)
        if full:
            strat_rows[label] = strata(df, manifest)

    keys = ["n", "parse", "AUC", "escalation", "p_lens_A", "recovery_ABC",
            "D_precision", "D_rate", "grades", "cost"]
    width = max(len(l) for l in rows) + 2
    print(f"{'config':{width}s} " + " ".join(f"{k:>13s}" for k in keys))
    for label, m in rows.items():
        print(f"{label:{width}s} " + " ".join(f"{str(m.get(k)):>13s}" for k in keys))
    if full and strat_rows:
        print("\nPer-stratum (kept-in-lens-bin for gold/graded; rejection rate for "
              "graded_D/random_neg):")
        skeys = ["gold", "graded", "graded_D", "random_neg"]
        print(f"{'config':{width}s} " + " ".join(f"{k:>11s}" for k in skeys))
        for label, s in strat_rows.items():
            print(f"{label:{width}s} " + " ".join(f"{str(s.get(k)):>11s}" for k in skeys))


if __name__ == "__main__":
    main()
