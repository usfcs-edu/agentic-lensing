#!/usr/bin/env python3
"""
12_merge_shards.py — Phase 4b post-step.

Concatenate the per-shard, per-model score parquets written by
11b_brick_inference_dr8.py into one unified file PER MODEL:

  inference_scores_l18_shard*.parquet       -->  inference_scores_l18_dr8.parquet
  inference_scores_shielded_shard*.parquet  -->  inference_scores_shielded_dr8.parquet

Also writes the slim p>=keep subset for each (committed to git; the full files
are gitignored), and merges the brick manifest. Idempotent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = ("l18", "shielded")


def merge_model(kind: str, keep_thresh: float) -> None:
    shards = sorted(DATA.glob(f"inference_scores_{kind}_shard*.parquet"))
    if not shards:
        print(f"[{kind}] no shards found — skipping")
        return
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    out = DATA / f"inference_scores_{kind}_dr8.parquet"
    df.to_parquet(out, compression="snappy")
    slim = df[df["score"] >= keep_thresh]
    slim_path = DATA / f"inference_scores_{kind}_dr8_p_ge_{keep_thresh:g}.parquet"
    slim.to_parquet(slim_path, compression="snappy")
    print(f"[{kind}] {len(df):,} rows -> {out.name}  "
          f"(p>=0.1: {(df['score']>=0.1).sum():,}  "
          f"p>=0.5: {(df['score']>=0.5).sum():,}  "
          f"p>=0.9: {(df['score']>=0.9).sum():,})")
    print(f"[{kind}] slim p>={keep_thresh:g}: {len(slim):,} rows -> {slim_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-thresh", type=float, default=0.1)
    args = ap.parse_args()
    for kind in MODELS:
        merge_model(kind, args.keep_thresh)

    manifests = sorted(DATA.glob("brick_manifest_shard*.csv"))
    if manifests:
        cat = pd.concat(
            [pd.read_csv(p, header=None,
                         names=["key", "status", "n_gal", "n_kept", "info"])
             for p in manifests], ignore_index=True)
        cat.to_csv(DATA / "brick_manifest_dr8.csv", index=False)
        n_fail = int((cat["status"] != "ok").sum())
        n_gal = pd.to_numeric(cat["n_gal"], errors="coerce").fillna(0).sum()
        print(f"[manifest] {len(cat):,} units  ({n_fail:,} failed)  "
              f"{int(n_gal):,} galaxies scored")


if __name__ == "__main__":
    main()
