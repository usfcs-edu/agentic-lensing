#!/usr/bin/env python3
"""120_mine_hard_negatives.py — Phase 120: deployment-scale hard-negative mining
(runs LOCALLY, CPU). Redoes v1's Phase-2 mining (30_hard_negative_mining.py:
fixed-count protocol, PU skim, random control) at 1M-pool scale, with the STRONG
staged EfficientNet (112's `baseline_effnet` column) as the miner instead of a
freshly trained round-0 shielded net.

Logic (v1 discipline, fractional PU skim):
  * rank the pool's ok rows by miner score DESC (high score = the miner's own
    most lens-like false positives = HARD negatives);
  * drop the top --pu-skim-frac band (possible unlabeled REAL lenses; mirrors
    v1's PU_SKIM=100 absolute skim, made fractional for the 1M pool);
  * take the next --n-mine rows as the HARD set;
  * draw --n-mine RANDOM rows (seed 2026, without replacement, full ok pool —
    overlap with the hard set allowed, exactly as v1) as the quality-vs-count
    control: same count, same pool, no mining signal.

Inputs : --pool-scores  scores parquet from 112_score_pool.py on the MinePool
                        (row_id + ok + per-scorer columns; miner = --miner-col)
         --manifest     data/v2/minepool_manifest.parquet (row_id,RA,DEC,
                        footprint,brick) for the stats + downstream RA/DEC
Outputs: <out-prefix>_hard_rowids.parquet / <out-prefix>_random_rowids.parquet
         (row_id [str] + p_miner,RA,DEC,footprint,brick; row order = selection
         order, hardest first) + <out-prefix>_stats.json (score ranges,
         per-footprint counts, brick spread). The rowid parquets feed
         111b_dump_rows.py on Perlmutter -> npz -> 120b locally.

    /home2/benson/.venvs/claudenet/bin/python 120_mine_hard_negatives.py \
        --pool-scores data/v2/scores_minepool_pool.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

V2 = C.DATA / "v2"


def brick_spread(df: pd.DataFrame) -> dict:
    """Brick-clustering stats for a selected set (mined sets that pile up in a
    few bricks would inherit brick systematics rather than lens-like structure)."""
    if "brick" not in df.columns or df["brick"].isna().all():
        return {"n_bricks": None}
    vc = df["brick"].value_counts()
    return {"n_bricks": int(vc.size),
            "max_per_brick": int(vc.iloc[0]),
            "mean_per_brick": float(vc.mean()),
            "top5_bricks": {str(k): int(v) for k, v in vc.head(5).items()}}


def set_stats(df: pd.DataFrame) -> dict:
    s = df["p_miner"].to_numpy(dtype=np.float64)
    return {"n": int(len(df)),
            "score_min": float(np.min(s)), "score_max": float(np.max(s)),
            "score_mean": float(np.mean(s)), "score_median": float(np.median(s)),
            "per_footprint": {str(k): int(v) for k, v in
                              df["footprint"].value_counts().items()}
            if "footprint" in df.columns else {},
            "brick_spread": brick_spread(df)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool-scores", default=str(V2 / "scores_minepool_pool.parquet"),
                    help="112 output on the MinePool (row_id + ok + scorer columns)")
    ap.add_argument("--manifest", default=str(V2 / "minepool_manifest.parquet"),
                    help="MinePool manifest (row_id,RA,DEC,footprint,brick)")
    ap.add_argument("--miner-col", default="baseline_effnet",
                    help="pool-scores column used as the miner (default: the "
                         "strong staged EfficientNet)")
    ap.add_argument("--n-mine", type=int, default=10000,
                    help="negatives per set (HARD and RANDOM; fixed-count protocol)")
    ap.add_argument("--pu-skim-frac", type=float, default=0.0005,
                    help="fraction of the ok pool skipped at the very top "
                         "(possible real lenses; v1 PU_SKIM, fractional)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out-prefix", default=str(V2 / "mined"))
    args = ap.parse_args()
    t0 = time.time()

    pool = pd.read_parquet(args.pool_scores)
    if args.miner_col not in pool.columns:
        raise SystemExit(f"[mine] FATAL: miner column {args.miner_col!r} not in "
                         f"{args.pool_scores} (have: {list(pool.columns)})")
    pool["row_id"] = pool["row_id"].astype(str)
    assert pool["row_id"].is_unique, "duplicate row_ids in pool scores"

    man = pd.read_parquet(args.manifest)
    man["row_id"] = man["row_id"].astype(str)
    pool = pool.merge(man[["row_id", "RA", "DEC", "footprint", "brick"]],
                      on="row_id", how="left")
    n_unmatched = int(pool["brick"].isna().sum())
    if n_unmatched:
        print(f"[mine] WARNING: {n_unmatched:,} pool rows missing from the manifest "
              f"(no RA/DEC/brick attached)")

    score = pool[args.miner_col].to_numpy(dtype=np.float64)
    ok = pool["ok"].to_numpy(bool) if "ok" in pool.columns else np.ones(len(pool), bool)
    usable = ok & np.isfinite(score)
    sub = pool[usable].reset_index(drop=True)
    s = score[usable]
    print(f"[mine] pool {len(pool):,} rows -> {len(sub):,} usable "
          f"(ok & finite {args.miner_col}); miner p99={np.quantile(s, 0.99):.4f} "
          f"p99.9={np.quantile(s, 0.999):.4f}")

    n_skim = int(np.ceil(args.pu_skim_frac * len(sub)))
    if n_skim + args.n_mine > len(sub):
        raise SystemExit(f"[mine] FATAL: pool too small ({len(sub):,}) for "
                         f"skim {n_skim:,} + n_mine {args.n_mine:,}")

    # HARD: rank desc (stable -> deterministic under score ties), skim, take next n
    order = np.argsort(-s, kind="stable")
    skim_idx = order[:n_skim]
    hard_idx = order[n_skim:n_skim + args.n_mine]
    # RANDOM control: same count, drawn from the SAME post-skim population as
    # HARD (the PU skim restricts the population; drawing the control from the
    # full pool would bias hard-vs-random). Overlap with HARD allowed (v1).
    rng = np.random.default_rng(args.seed)
    rand_idx = rng.choice(order[n_skim:], size=args.n_mine, replace=False)

    cols = ["row_id", "p_miner", "RA", "DEC", "footprint", "brick"]
    sub = sub.rename(columns={args.miner_col: "p_miner"})
    hard = sub.iloc[hard_idx][cols].reset_index(drop=True)
    rand = sub.iloc[rand_idx][cols].reset_index(drop=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    f_hard = out_prefix.parent / f"{out_prefix.name}_hard_rowids.parquet"
    f_rand = out_prefix.parent / f"{out_prefix.name}_random_rowids.parquet"
    hard.to_parquet(f_hard, index=False)
    rand.to_parquet(f_rand, index=False)

    overlap = len(set(hard.row_id) & set(rand.row_id))
    stats = {
        "pool_scores": str(args.pool_scores), "manifest": str(args.manifest),
        "miner_col": args.miner_col, "seed": args.seed,
        "n_pool": int(len(pool)), "n_usable": int(len(sub)),
        "pu_skim_frac": args.pu_skim_frac, "n_skim": n_skim,
        "skim_score_range": [float(s[skim_idx].min()), float(s[skim_idx].max())]
        if n_skim else None,
        "n_mine": args.n_mine,
        "hard": set_stats(hard), "random": set_stats(rand),
        "hard_random_overlap": overlap,
    }
    f_stats = out_prefix.parent / f"{out_prefix.name}_stats.json"
    f_stats.write_text(json.dumps(stats, indent=2))

    print(f"[mine] skim top {n_skim:,} ({args.pu_skim_frac:.4%}) score range "
          f"[{stats['skim_score_range'][0]:.4f},{stats['skim_score_range'][1]:.4f}]"
          if n_skim else "[mine] skim disabled (frac=0)")
    print(f"[mine] HARD   {len(hard):,} rows, score "
          f"[{stats['hard']['score_min']:.4f},{stats['hard']['score_max']:.4f}], "
          f"{stats['hard']['brick_spread']['n_bricks']} bricks, "
          f"footprints {stats['hard']['per_footprint']}")
    print(f"[mine] RANDOM {len(rand):,} rows, score "
          f"[{stats['random']['score_min']:.4f},{stats['random']['score_max']:.4f}], "
          f"{stats['random']['brick_spread']['n_bricks']} bricks, "
          f"footprints {stats['random']['per_footprint']} "
          f"(overlap with HARD: {overlap})")
    print(f"[120] wrote {f_hard} + {f_rand} + {f_stats} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
