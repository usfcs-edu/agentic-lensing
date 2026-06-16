#!/usr/bin/env python3
"""302_score_mimic_bank.py — ClaudeNet v3: the v2-lean BASELINE on the mimic metric.

This produces the single number that motivates the whole v3 program: how badly does the
shipped v2-lean ensemble separate real lenses from lens-MIMICS? It scores the SAME
v2-lean flagship (`flagship_combiner == average` = mean of the 5 isotonic-calibrated
members) on three populations and reports recovery@matched-FPR against two nulls:

  * RANDOM null  (NegEval-1M, ensemble_v2lean_pool_combined.parquet::v2lean_average) —
    reproduces the v2 headline as a sanity check.
  * MIMIC  null  (data/v3/mimic_bank_seed.parquet::p_final) — the v3 headline, overall
    and per contaminant type (301_mimic_metric.summarize).

Positive scores = v2lean_average on the held-out Storfer/Inchausti positives, computed
as the mean of the stored calibrated member scores `pc` (== the average combiner output;
the member parquets' pc is asserted == refit-isotonic-on-val in 145). Both the positive
and the mimic scores are therefore "mean of isotonic-calibrated members" — the SAME
function — so they share one scale. An integrity check refits the isotonic calibrators
on the val split, applies them to the mimic bank's RAW member columns, and confirms the
reconstruction matches the persisted p_final (the apply-roster selection column).

    /home2/benson/.venvs/claudenet/bin/python 302_score_mimic_bank.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _ensemble as E
import importlib
mm = importlib.import_module("301_mimic_metric")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
V2 = DATA / "v2"
OUT = V2.parent / "v3"
POS_SPLITS = ("storfer", "inchausti")


def load_roster(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def positive_avg_scores(roster: list[dict]) -> dict[str, np.ndarray]:
    """v2lean_average (= mean of calibrated pc) on each held-out positive split, aligned
    by row_id across the 5 members (combiner order is irrelevant for the mean)."""
    per_split = {}
    for sp in POS_SPLITS:
        mats = None
        for m in roster:
            df = pd.read_parquet(ROOT / m["scores_parquet"])
            g = df[df["split"] == sp][["row_id", "pc"]].rename(columns={"pc": m["name"]})
            g["row_id"] = g["row_id"].astype(str)
            mats = g if mats is None else mats.merge(g, on="row_id", how="inner")
        cols = [m["name"] for m in roster]
        per_split[sp] = mats[cols].to_numpy(float).mean(axis=1)
        print(f"[302] positives {sp}: {len(mats)} rows, v2lean_average "
              f"[{per_split[sp].min():.3f},{per_split[sp].max():.3f}]")
    return per_split


def integrity_check(roster: list[dict], mimic: pd.DataFrame) -> float:
    """Refit isotonic on val per member, apply to the mimic bank's RAW member_* columns,
    mean -> reconstruct p_final; return max|reconstructed - p_final|."""
    recon = None
    n = 0
    for m in roster:
        col = "member_" + m["name"] if not m.get("pool_column") else m["pool_column"]
        if col not in mimic.columns:
            print(f"[302] integrity: mimic bank missing raw col {col}; skipping check")
            return float("nan")
        df = pd.read_parquet(ROOT / m["scores_parquet"])
        v = df[df["split"] == "val"]
        cal = E.IsotonicCalibrator().fit(v["p"].to_numpy(float), v["label"].to_numpy(int))
        c = cal.transform(mimic[col].to_numpy(float))
        recon = c if recon is None else recon + c
        n += 1
    recon = recon / n
    d = float(np.nanmax(np.abs(recon - mimic["p_final"].to_numpy(float))))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", default=str(V2 / "roster_v2lean.json"))
    ap.add_argument("--mimic", default=str(OUT / "mimic_bank_seed.parquet"))
    ap.add_argument("--neg-pool", default=str(V2 / "ensemble_v2lean_pool_combined.parquet"))
    ap.add_argument("--neg-col", default="v2lean_average")
    ap.add_argument("--out", default=str(OUT / "baseline_v2_mimic_metric.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    roster = load_roster(Path(args.roster))
    mimic = pd.read_parquet(args.mimic)
    mimic_scores = mimic["p_final"].to_numpy(float)
    neg = pd.read_parquet(args.neg_pool)[args.neg_col].to_numpy(float)
    pos = positive_avg_scores(roster)

    d = integrity_check(roster, mimic)
    print(f"[302] integrity: max|reconstructed p_final - stored p_final| = {d:.3e} "
          f"({'OK' if (np.isnan(d) or d < 1e-3) else 'WARN scale mismatch'})")

    report = {"integrity_max_abs_diff": d, "n_mimic": int(len(mimic_scores)),
              "splits": {}}
    # v2 verdict reference (random null @ fpr 0.001) for the sanity print
    ref = {"storfer": 0.895, "inchausti": 0.961}
    for sp, ps in pos.items():
        rand = E.recovery_at_fpr(neg, ps, fprs=(0.01, 0.001))
        got01 = rand[0.001]["recovery"]
        print(f"[302] {sp}: recovery@random-FPR 1e-2={rand[0.01]['recovery']:.3f} "
              f"1e-3={got01:.3f}  (v2 verdict ~{ref[sp]:.3f})")
        res = mm.summarize(mimic_scores, ps, mimic, "p_final", label=f"v2lean/{sp}")
        print(mm._fmt(res))
        report["splits"][sp] = {
            "recovery_random_fpr": {str(k): rand[k]["recovery"] for k in (0.01, 0.001)},
            "mimic_metric": res}

    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print(f"[302] wrote {args.out}")
    # headline
    for sp in POS_SPLITS:
        ov = report["splits"][sp]["mimic_metric"]["overall"]
        print(f"[302] HEADLINE {sp}: recovery@mimic-FPR(0.05)="
              f"{ov['0.05']['recovery']:.3f}  vs recovery@random-FPR(0.01)="
              f"{report['splits'][sp]['recovery_random_fpr']['0.01']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
