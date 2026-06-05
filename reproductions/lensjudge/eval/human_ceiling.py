#!/usr/bin/env python3
"""Inter-team human ceiling for strong-lens A/B/C grading.

Our DESI labels are single 2-author consensus (no per-rater votes), so a per-rater kappa
is not computable from them. But the external crossmatch gives the SAME candidates graded
INDEPENDENTLY by other expert groups (the SuGOHI ~9-grader committee, the Euclid ~10-expert
panel). The agreement between these independent teams is a legitimate human-ceiling estimate:
it bounds how well any automated grader can reproduce one team's grade. We also drop in
LensJudge's own grades to ask whether the agent agrees with the DESI consensus as well as an
independent expert team does.

  python lensjudge/eval/human_ceiling.py

Caveats (printed): cross-survey / cross-rubric / cross-resolution, biased to mutually-flagged
candidates (the C/D detection boundary is not exercised here), small n on the Euclid side.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parents[1] / "outputs"
GMAP = {"A": 3, "B": 2, "C": 1, "D": 0}
_RNG = np.random.default_rng(2026)


def _clean(s):
    return s.astype(str).str.strip().str.upper().str[0]


def _qwk(a, b):
    if len(np.unique(a)) < 2 and len(np.unique(b)) < 2:
        return np.nan
    try:
        return cohen_kappa_score(a, b, weights="quadratic")
    except Exception:
        return np.nan


def _boot_ci(a, b, fn, n=2000):
    a = np.asarray(a); b = np.asarray(b); m = len(a)
    vals = []
    for _ in range(n):
        idx = _RNG.integers(0, m, m)
        v = fn(a[idx], b[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def pair(label, g1, g2):
    g1, g2 = _clean(g1), _clean(g2)
    ok = g1.isin(GMAP) & g2.isin(GMAP)
    g1, g2 = g1[ok], g2[ok]
    a = g1.map(GMAP).to_numpy(); b = g2.map(GMAP).to_numpy()
    if len(a) < 3:
        return None
    qwk = _qwk(a, b)
    lo, hi = _boot_ci(a, b, _qwk)
    rho, _ = spearmanr(a, b) if len(np.unique(a)) > 1 and len(np.unique(b)) > 1 else (np.nan, None)
    return {"comparison": label, "n": int(len(a)), "qwk": round(qwk, 3),
            "qwk_lo": round(lo, 3), "qwk_hi": round(hi, 3),
            "linear_kappa": round(cohen_kappa_score(a, b, weights="linear"), 3),
            "exact_agree": round((a == b).mean(), 3),
            "within1": round((np.abs(a - b) <= 1).mean(), 3),
            "spearman": round(float(rho), 3) if not np.isnan(rho) else np.nan}


def main():
    sug = pd.read_csv(OUT / "desi_x_sugohi_matches.csv")
    euc = pd.read_csv(OUT / "xmatch_euclid_q1.csv")
    rows = []

    # --- human-vs-human (independent expert teams) ---
    rows.append(pair("HUMAN: DESI team vs SuGOHI ~9-grader committee", sug.desi_grade, sug.sugohi_grade))
    rows.append(pair("HUMAN: DESI team vs Euclid ~10-expert panel", euc.grade, euc.euclid_grade))
    # SuGOHI vs Euclid where a DESI candidate matched BOTH
    both = sug.merge(euc, left_on="desi_name", right_on="name", how="inner")
    if len(both) >= 3:
        rows.append(pair("HUMAN: SuGOHI committee vs Euclid panel (mutual)", both.sugohi_grade, both.euclid_grade))

    # --- agent vs each team, on the SAME objects (agent-at-ceiling test) ---
    psug = OUT / "preds_sugohi_lean.parquet"
    if psug.exists():
        p = pd.read_parquet(psug)[["name", "grade_pred"]].dropna()
        sa = sug.merge(p, left_on="desi_name", right_on="name", how="inner")
        rows.append(pair("AGENT: LensJudge vs DESI team (SuGOHI-matched set)", sa.desi_grade, sa.grade_pred))
        rows.append(pair("AGENT: LensJudge vs SuGOHI committee (same set)", sa.sugohi_grade, sa.grade_pred))
    pe = OUT / "euclid_paired_preds.parquet"
    if pe.exists():
        pp = pd.read_parquet(pe)
        if "agent_grade_desi" in pp:
            rows.append(pair("AGENT: LensJudge(DESI img) vs DESI team (Euclid-matched)", pp.desi_grade, pp.agent_grade_desi))
        if "agent_grade_euclid" in pp:
            rows.append(pair("AGENT: LensJudge(Euclid img) vs Euclid panel", pp.euclid_grade, pp.agent_grade_euclid))

    df = pd.DataFrame([r for r in rows if r])
    df.to_csv(OUT / "human_ceiling.csv", index=False)
    pd.set_option("display.width", 160, "display.max_colwidth", 60)
    print("\n=== Inter-team human ceiling for A/B/C lens grading (QWK = quadratic-weighted kappa) ===")
    print(df.to_string(index=False))
    print(f"\nsaved {OUT/'human_ceiling.csv'}")
    print("\nReading: QWK 0.0=chance, ~0.2-0.4 'fair', ~0.4-0.6 'moderate', >0.8 'almost perfect'.")
    print("Caveats: cross-survey/rubric/resolution; mutually-flagged candidates only (C/D detection")
    print("boundary not exercised); small n on Euclid. The DESI<->Euclid gap is partly REAL (Euclid")
    print("resolves more), so DESI<->SuGOHI (closer resolution) is the cleaner same-regime ceiling.")


if __name__ == "__main__":
    main()
