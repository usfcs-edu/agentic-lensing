#!/usr/bin/env python3
"""Score the parity-bench arms with the fitted representation probe (Phase C1b).

SCORING-ONLY by construction: reads nothing from the bench manifests except
name/ra/dec/survey_key (truth columns are never loaded), computes the Tier-1
engineered features for every row (shared cache with train_rep_probe.py), and
writes p_lens_rep for EVERY manifest row:

  outputs/rep_probe/bench_scores_arm1.csv   (name, p_lens_rep)
  outputs/rep_probe/bench_scores_arm2.csv

Rows whose cutout cannot be resolved keep their line with an empty score, so
coverage is auditable. Cutouts are all pre-staged in cache/cubes/ (Phase B);
any stragglers are fetched politely (<=3 workers) via common/fetch.

  python lensjudge/parity/score_bench_rep.py            # both arms
  python lensjudge/parity/score_bench_rep.py --arms 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.parity.train_rep_probe import (  # noqa: E402
    FETCH_WORKERS, LOCAL_WORKERS, MODEL_PATH, OUT, calibrated_p, compute_features_for)

SCORING_COLS = ["name", "ra", "dec", "survey_key"]   # truth columns are NEVER read


def score_arm(arm: int, model: dict, fetch_workers: int, local_workers: int) -> None:
    man_path = config.OUT / f"parity_bench_arm{arm}.csv"
    man = pd.read_csv(man_path, usecols=SCORING_COLS, dtype={"name": str})
    print(f"[arm{arm}] {len(man)} rows from {man_path.name} (scoring columns only)", flush=True)

    cache = compute_features_for(man, fetch_workers, local_workers)
    cols = model["feature_cols"]
    m = man.merge(cache[["name"] + cols], on="name", how="left")
    have = m[cols].notna().any(axis=1)
    p = np.full(len(m), np.nan)
    if have.any():
        X = m.loc[have, cols].fillna(0).values.astype(float)
        p[have.values] = calibrated_p(model["pipeline"], model["isotonic"], X)
    out = pd.DataFrame({"name": man.name, "p_lens_rep": np.round(p, 6)})
    out_path = OUT / f"bench_scores_arm{arm}.csv"
    out.to_csv(out_path, index=False)
    print(f"[arm{arm}] scored {int(have.sum())}/{len(man)} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="1,2", help="comma-separated arms to score")
    ap.add_argument("--fetch-workers", type=int, default=FETCH_WORKERS)
    ap.add_argument("--local-workers", type=int, default=LOCAL_WORKERS)
    args = ap.parse_args()

    model = joblib.load(MODEL_PATH)
    print(f"[probe] {MODEL_PATH.name}: scheme={model['scheme']} C={model['C']} "
          f"(valsel HARD AUC {model['valsel_hard_auc']:.4f})", flush=True)
    t0 = time.time()
    for arm in (int(a) for a in args.arms.split(",")):
        score_arm(arm, model, args.fetch_workers, args.local_workers)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
