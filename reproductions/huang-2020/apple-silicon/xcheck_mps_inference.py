#!/usr/bin/env python3
"""xcheck_mps_inference.py — validate MPS inference fidelity.

Compares the bounded MPS inference smoke output (11b run with the *phoenix*
checkpoint on a few hundred bricks) against phoenix's full score parquet for the
same row_ids. Same weights + same brick pixels ⇒ scores should agree to float
precision, isolating MPS-vs-CUDA numerical drift from training stochasticity.

Writes data/mps_xcheck.json {n, max_abs_delta, median_abs_delta}, consumed by
verify_against_reference.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default=str(DATA / "inference_scores_shard0.parquet"),
                    help="MPS smoke scores (11b --limit-bricks with phoenix ckpt)")
    ap.add_argument("--ref", default=str(DATA / "ref" / "inference_scores_dr9trained.parquet"),
                    help="phoenix full scores produced with the same checkpoint")
    args = ap.parse_args()

    smoke = pd.read_parquet(args.smoke)[["row_id", "score"]]
    ref = pd.read_parquet(args.ref)[["row_id", "score"]]
    j = smoke.merge(ref, on="row_id", suffixes=("_mps", "_phx"))
    if len(j) == 0:
        raise SystemExit("no overlapping row_ids between smoke and reference")
    d = np.abs(j["score_mps"].to_numpy() - j["score_phx"].to_numpy())
    out = {
        "n": int(len(j)),
        "max_abs_delta": float(d.max()),
        "median_abs_delta": float(np.median(d)),
        "n_smoke": int(len(smoke)),
        "n_overlap": int(len(j)),
    }
    (DATA / "mps_xcheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[xcheck] overlap={out['n']}  max|Δ|={out['max_abs_delta']:.3e}  "
          f"median|Δ|={out['median_abs_delta']:.3e}")


if __name__ == "__main__":
    main()
