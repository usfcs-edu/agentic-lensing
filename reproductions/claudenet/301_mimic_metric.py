#!/usr/bin/env python3
"""301_mimic_metric.py — ClaudeNet v3: the lens-vs-MIMIC recovery metric.

The v1/v2 headline metric is recovery (TPR) at a matched FALSE-POSITIVE rate where
the FPR denominator is a *random galaxy* (NegEval-1M). The DR9 qualification campaign
showed that this null is the wrong one: the 737 candidates were ~0/601 real lenses,
dominated by lens-MIMICS (lrg_companion 49%, merger, blend, ring, spiral). A model can
score perfectly on recovery@random-FPR and still drown in mimics.

This module defines the v3 headline: **recovery @ matched-MIMIC-FPR** — identical
arithmetic to `_ensemble.recovery_at_fpr`, but the "negatives" are the lens-mimic bank
(300_build_mimic_bank.py) instead of random galaxies. So we REUSE recovery_at_fpr
verbatim (neg_scores := mimic_scores) and add: (a) Wilson + bootstrap CIs (the bank is
small, ~hundreds-to-tens-of-thousands, so report uncertainty), and (b) a per-contaminant
breakdown (threshold within each mimic type) so we see WHICH type a model fails on.

Headline phi = 0.05 (you cannot estimate a 1e-3 mimic quantile from a few-hundred-row
seed; phi tightens as the bank grows in A1). Imported by 302/331/345; CLI is for
ad-hoc inspection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import _ensemble as E

PHIS = (0.05, 0.01)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (recovery is a TPR)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def recovery_at_mimic_fpr(mimic_scores, pos_scores, phis=PHIS) -> dict:
    """recovery@matched-mimic-FPR. Thin wrapper over _ensemble.recovery_at_fpr with the
    mimic bank as the negative class, so the arithmetic is byte-identical to the v2
    headline. Adds a Wilson CI on each recovery (k = recovered positives)."""
    base = E.recovery_at_fpr(mimic_scores, pos_scores, fprs=phis)
    out = {}
    n_pos = int(np.isfinite(np.asarray(pos_scores, float)).sum())
    for phi, d in base.items():
        k = int(round(d["recovery"] * d["n_cand"])) if d["n_cand"] else 0
        lo, hi = wilson_ci(k, d["n_cand"])
        out[phi] = {**d, "n_pos": n_pos, "recovery_wilson_lo": lo, "recovery_wilson_hi": hi}
    return out


def bootstrap_recovery_ci(mimic_scores, pos_scores, phi: float, nboot: int = 2000,
                          seed: int = 2026) -> dict:
    """Paired bootstrap over BOTH positives (recovery sampling error) and mimics
    (threshold sampling error) — the bank is small so the threshold itself is noisy."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos_scores, float); pos = pos[np.isfinite(pos)]
    mim = np.asarray(mimic_scores, float); mim = mim[np.isfinite(mim)]
    if len(pos) == 0 or len(mim) == 0:
        return {"phi": phi, "recovery_lo": float("nan"), "recovery_hi": float("nan")}
    recs = np.empty(nboot)
    for b in range(nboot):
        mb = mim[rng.integers(0, len(mim), len(mim))]
        pb = pos[rng.integers(0, len(pos), len(pos))]
        thr = np.quantile(mb, 1.0 - phi)
        recs[b] = (pb >= thr).mean()
    return {"phi": phi, "recovery_lo": float(np.quantile(recs, 0.025)),
            "recovery_hi": float(np.quantile(recs, 0.975)),
            "recovery_boot_mean": float(recs.mean())}


def per_type_recovery(mimic_df: pd.DataFrame, pos_scores, score_col: str,
                      type_col: str = "mimic_type", phis=PHIS, min_n: int = 8) -> dict:
    """For each contaminant type t: set the threshold from type-t mimics only and report
    recovery of the positives above it. Reveals which mimic class the model cannot
    separate from real lenses. Types with < min_n members are reported with a flag."""
    out = {}
    pos = np.asarray(pos_scores, float); pos = pos[np.isfinite(pos)]
    for t, g in mimic_df.groupby(type_col):
        s = g[score_col].to_numpy(float); s = s[np.isfinite(s)]
        rec = recovery_at_mimic_fpr(s, pos, phis) if len(s) else {}
        out[str(t)] = {"n_mimic": int(len(s)), "small_n": bool(len(s) < min_n),
                       "recovery": {str(p): rec[p]["recovery"] for p in phis} if rec else {}}
    return out


def summarize(mimic_scores, pos_scores, mimic_df=None, score_col=None, label="",
              phis=PHIS, bootstrap=True) -> dict:
    """One-call summary used by 302/345: overall recovery@mimic-FPR (+CIs) and, if a
    typed mimic frame is given, the per-type breakdown."""
    res = {"label": label, "n_mimic": int(np.isfinite(np.asarray(mimic_scores, float)).sum()),
           "overall": {}}
    rec = recovery_at_mimic_fpr(mimic_scores, pos_scores, phis)
    for phi in phis:
        entry = {k: rec[phi][k] for k in ("threshold", "recovery", "n_pos",
                                          "recovery_wilson_lo", "recovery_wilson_hi")}
        if bootstrap:
            entry["bootstrap"] = bootstrap_recovery_ci(mimic_scores, pos_scores, phi)
        res["overall"][str(phi)] = entry
    if mimic_df is not None and score_col is not None:
        res["per_type"] = per_type_recovery(mimic_df, pos_scores, score_col, phis=phis)
    return res


def _fmt(res: dict) -> str:
    lines = [f"[mimic-metric] {res.get('label','')}  n_mimic={res['n_mimic']}"]
    for phi, d in res["overall"].items():
        b = d.get("bootstrap", {})
        ci = f"[{b.get('recovery_lo', float('nan')):.3f},{b.get('recovery_hi', float('nan')):.3f}]"
        lines.append(f"   phi={phi}: recovery={d['recovery']:.3f}  boot95{ci}  "
                     f"wilson[{d['recovery_wilson_lo']:.3f},{d['recovery_wilson_hi']:.3f}]  "
                     f"thr={d['threshold']:.4f}  n_pos={d['n_pos']}")
    if "per_type" in res:
        lines.append("   per-type recovery (phi=0.05 / 0.01):")
        for t, d in sorted(res["per_type"].items(), key=lambda kv: -kv[1]["n_mimic"]):
            r = d["recovery"]
            flag = " (small-n)" if d["small_n"] else ""
            lines.append(f"     {t:16s} n={d['n_mimic']:5d}  "
                         f"{r.get('0.05', float('nan')):.3f} / {r.get('0.01', float('nan')):.3f}{flag}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="ad-hoc recovery@matched-mimic-FPR")
    ap.add_argument("--mimic", required=True, help="parquet with the mimic-bank scores")
    ap.add_argument("--mimic-col", default="p_final")
    ap.add_argument("--type-col", default="mimic_type")
    ap.add_argument("--pos", required=True, help="parquet/npy of positive scores")
    ap.add_argument("--pos-col", default="score")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mdf = pd.read_parquet(args.mimic)
    pos = (pd.read_parquet(args.pos)[args.pos_col].to_numpy(float)
           if args.pos.endswith(".parquet") else np.load(args.pos))
    res = summarize(mdf[args.mimic_col].to_numpy(float), pos, mdf, args.mimic_col,
                    label=Path(args.mimic).stem)
    print(_fmt(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
