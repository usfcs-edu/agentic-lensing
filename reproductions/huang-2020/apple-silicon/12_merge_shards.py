#!/usr/bin/env python3
"""
12_merge_shards.py — Phase 3b M2 post-step.

Concatenate per-shard parquets and manifests written by 11_stream_inference_dr7.py
into single unified files:

  inference_scores_shard{0,1}.parquet  -->  inference_scores.parquet
  inference_manifest_shard{0,1}.csv    -->  inference_manifest.csv

Idempotent: safe to re-run whenever the shards are updated. Reports totals.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main() -> None:
    score_files = sorted(DATA.glob("inference_scores_shard*.parquet"))
    if not score_files:
        raise SystemExit("no per-shard score parquets found")
    print(f"[init] {len(score_files)} score shards")
    dfs = [pd.read_parquet(p) for p in score_files]
    out_scores = pd.concat(dfs, ignore_index=True)
    out_scores = out_scores.sort_values("score", ascending=False).reset_index(drop=True)
    out_path = DATA / "inference_scores.parquet"
    out_scores.to_parquet(out_path, compression="snappy")
    print(f"[done] {len(out_scores):,} rows --> {out_path}")
    print(f"       score top-5:  {out_scores['score'].head().tolist()}")
    print(f"       p>=0.9:       {(out_scores['score'] >= 0.9).sum():,}")
    print(f"       p>=0.5:       {(out_scores['score'] >= 0.5).sum():,}")

    manifest_files = sorted(DATA.glob("inference_manifest_shard*.csv"))
    if manifest_files:
        cat = pd.concat(
            [pd.read_csv(p, header=None, names=["row_id", "status", "info"])
             for p in manifest_files], ignore_index=True)
        cat.to_csv(DATA / "inference_manifest.csv", index=False)
        n_fail = int((cat["status"] != "ok").sum())
        print(f"[done] manifest: {len(cat):,} rows  ({n_fail:,} failures)")


if __name__ == "__main__":
    main()
