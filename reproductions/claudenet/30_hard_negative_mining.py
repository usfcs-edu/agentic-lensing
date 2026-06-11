#!/usr/bin/env python3
"""30_hard_negative_mining.py — Phase 2: does replacing random negatives with MINED
hard negatives (the model's own top false-positives) improve recovery@FPR at a
FIXED negative count? Isolates negative QUALITY from quantity, with a random-negative
control and an iterated round.

Design (all within the existing 45,507 staged-train negatives; in-RAM cached):
  round0 : shielded on pos + 10k RANDOM (R0)        -> score the remaining ~35k pool
  RANDOM : shielded on pos + 10k random-from-pool   (control: same count, random)
  HARD   : shielded on pos + 10k hardest-from-pool  (top model0 false-positives)
  HARD2  : shielded on pos + 10k hardest-by-HARD    (one iteration)
PU guard: known-catalog negatives already removed in 19; we additionally drop the
very top fraction of mined negatives (most likely to be unlabeled real lenses).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      /home2/benson/.venvs/claudenet/bin/python 30_hard_negative_mining.py

Writes data/mining_operating_point.csv + data/mining_summary.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C
import _ensemble as E
import _minelib as ML

N_VAR = 10000          # negatives per training variant (fixed -> isolates quality)
N_R0 = 10000           # round-0 random negatives (miner)
PU_SKIM = 100          # drop the top-N mined (most lens-like -> possible real lenses)
EPOCHS = 30


def remap_dir(p):
    return str(C.DATA / Path(str(p)).name)


def recov(model, cache, mean, std, device, neg_df, cat_df):
    neg = ML.score(model, neg_df, cache, mean, std, device)
    cand = ML.score(model, cat_df, cache, mean, std, device)
    r = E.recovery_at_fpr(neg, cand, fprs=(0.01, 0.001))
    return r[0.01]["recovery"], r[0.001]["recovery"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(C.SEED)

    split = pd.read_parquet(C.DATA / "training_split_staged.parquet")
    split["fits_dir"] = split["fits_dir"].apply(remap_dir)
    pos = split[(split.split == "train") & (split.label == 1)][["row_id", "label", "fits_dir"]]
    neg = split[(split.split == "train") & (split.label == 0)][["row_id", "label", "fits_dir"]].reset_index(drop=True)
    val = pd.read_parquet(C.DATA / "eval_val.parquet")[["row_id", "label", "fits_dir"]]
    val["fits_dir"] = val["fits_dir"].apply(remap_dir)
    testneg = pd.read_parquet(C.DATA / "eval_testneg.parquet")[["row_id", "label", "fits_dir"]]
    testneg["fits_dir"] = testneg["fits_dir"].apply(remap_dir)
    storfer = pd.read_parquet(C.DATA / "eval_storfer.parquet")[["row_id", "label", "fits_dir"]]
    inch = pd.read_parquet(C.DATA / "eval_inchausti.parquet")[["row_id", "label", "fits_dir"]]

    print(f"[mine] loading RAM cache (pos {len(pos)}, neg {len(neg)}, val {len(val)}, "
          f"testneg {len(testneg)}, cand {len(storfer)+len(inch)}) ...")
    t0 = time.time()
    cache = ML.load_cache(pos, neg, val, testneg, storfer, inch)
    print(f"[mine] cached {len(cache)} cutouts in {time.time()-t0:.0f}s")

    # round 0: random R0 negatives -> miner
    perm = rng.permutation(len(neg))
    r0_idx, pool_idx = perm[:N_R0], perm[N_R0:]
    r0_neg, pool = neg.iloc[r0_idx], neg.iloc[pool_idx].reset_index(drop=True)

    def train(neg_sub, seed):
        tr = pd.concat([pos, neg_sub], ignore_index=True)
        m, mean, std, auc = ML.train_shielded(tr, val, cache, device, epochs=EPOCHS, aug_seed=seed)
        return m, mean, std, auc

    results = {}

    def evaluate(tag, model, mean, std):
        rs1, rs01 = recov(model, cache, mean, std, device, testneg, storfer)
        ri1, ri01 = recov(model, cache, mean, std, device, testneg, inch)
        results[tag] = {"storfer_1": rs1, "storfer_01": rs01, "inchausti_1": ri1, "inchausti_01": ri01}
        print(f"[eval] {tag:8s} storfer@1%={rs1:.3f} @.1%={rs01:.3f}  "
              f"inchausti@1%={ri1:.3f} @.1%={ri01:.3f}")

    print("\n[round0] miner on pos + 10k random")
    m0, me0, sd0, a0 = train(r0_neg, 700)
    evaluate("round0", m0, me0, sd0)

    # score the pool with the miner -> rank
    pool_p = ML.score(m0, pool, cache, me0, sd0, device)
    order = np.argsort(-np.nan_to_num(pool_p, nan=-1))   # high score = hard
    hard_idx = order[PU_SKIM:PU_SKIM + N_VAR]            # skim top PU_SKIM (possible real lenses)
    rand_idx = rng.choice(len(pool), size=N_VAR, replace=False)

    print(f"\n[mine] pool {len(pool)}; hardest mined score range "
          f"[{pool_p[hard_idx].min():.3f},{pool_p[hard_idx].max():.3f}] "
          f"vs random [{np.nanmin(pool_p[rand_idx]):.3f},{np.nanmax(pool_p[rand_idx]):.3f}]")

    print("\n[RANDOM] control: pos + 10k random-from-pool")
    mr, mer, sdr, ar = train(pool.iloc[rand_idx], 701)
    evaluate("RANDOM", mr, mer, sdr)

    print("\n[HARD] pos + 10k hardest-from-pool")
    mh, meh, sdh, ah = train(pool.iloc[hard_idx], 702)
    evaluate("HARD", mh, meh, sdh)

    # iterate: re-score pool with HARD model, take next hardest
    pool_p2 = ML.score(mh, pool, cache, meh, sdh, device)
    order2 = np.argsort(-np.nan_to_num(pool_p2, nan=-1))
    hard2_idx = order2[PU_SKIM:PU_SKIM + N_VAR]
    print("\n[HARD2] iteration: pos + 10k hardest-by-HARD")
    mh2, meh2, sdh2, ah2 = train(pool.iloc[hard2_idx], 703)
    evaluate("HARD2", mh2, meh2, sdh2)

    # summary
    rows = [{"variant": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(C.DATA / "mining_operating_point.csv", index=False)
    d_hard = results["HARD"]["storfer_1"] - results["RANDOM"]["storfer_1"]
    d_hard01 = results["HARD"]["storfer_01"] - results["RANDOM"]["storfer_01"]
    summary = {"results": results, "N_per_variant": N_VAR,
               "hard_minus_random_storfer_1": d_hard,
               "hard_minus_random_storfer_01": d_hard01,
               "verdict": "QUALITY-HELPS" if d_hard > 1e-3 or d_hard01 > 1e-3 else "NO-EFFECT"}
    (C.DATA / "mining_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[mining] HARD - RANDOM (same 10k count): storfer@1% {d_hard:+.3f}, "
          f"storfer@0.1% {d_hard01:+.3f}")
    print(f"[mining] VERDICT: {summary['verdict']}  "
          f"(negative QUALITY {'matters' if summary['verdict']=='QUALITY-HELPS' else 'no effect'} "
          f"at fixed count)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
