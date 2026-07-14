#!/usr/bin/env python3
"""finetune/build_corpus_v5.py — assemble the v5 HSC training corpus (fully Claude-free).

Sources (fetched by eval/stage_hsc into the LENSJUDGE_HSC_CACHE-style cache `cache/hsc_v5`):
  batch-1  cache/v5_catalogs/hsc_v5_batch1.csv        1,597 positives (SuGOHI A/B ∪ HOLISMOKES-XVI
           G>=1.5, 2" deduped) + 2,910 expert-labeled negatives (XVI rejects: all 1,910 near-miss
           0.5<=G<1.5 + 1,000 far G<0.5)
  batch-2  cache/v5_catalogs/hsc_v5_batch2_gradeC.csv 1,764 SuGOHI grade-C (soft mid-band, TRAIN-only)

Targets are HUMAN-soft (the Claude-free replacement for teacher soft labels, per the v4 lesson that
forced-A targets teach indiscriminate arc-calling): SuGOHI A->0.95 B->0.80 C->0.45; XVI rows carry
their continuous committee score G/3.

Splits (all pixel-backed via hsc_fetch.cached, coordinate-deduped at 2" against the ESTABLISHED
72-object real-lens gate outputs/distill_hsc/hsc_test.csv, which keeps its frozen baselines):
  test2   NEW larger held-out gate: 120 pos + 240 neg (never trained on, never used for selection)
  valsel  HSC-only AUC-selection set: 60 pos + 120 neg   (standing rule: per-survey selection)
  train   everything else (incl. ALL grade-C soft-mid rows; C rows are excluded from eval splits —
          ambiguous by construction, they carry soft signal for training only)

  python -m lensjudge.finetune.build_corpus_v5           # writes outputs/corpus_v5/{train,valsel,test2}.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import hsc_fetch  # noqa: E402

SEED = 2026
CAT = config.HERE / "cache" / "v5_catalogs"
OUT = config.OUT / "corpus_v5"


def _sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation (arcsec) between one point and arrays."""
    ra2, dec2 = np.asarray(ra2, float), np.asarray(dec2, float)
    c = (np.sin(np.radians(dec1)) * np.sin(np.radians(dec2)) +
         np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) *
         np.cos(np.radians(ra1) - np.radians(ra2)))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(config.HERE / "cache" / "hsc_v5"))
    ap.add_argument("--n-test2-pos", type=int, default=120)
    ap.add_argument("--n-test2-neg", type=int, default=240)
    ap.add_argument("--n-valsel-pos", type=int, default=60)
    ap.add_argument("--n-valsel-neg", type=int, default=120)
    args = ap.parse_args()

    hsc_fetch.HSC_CACHE = Path(args.cache)   # point cached() at the v5 cache

    parts = [pd.read_csv(CAT / "hsc_v5_batch1.csv")]
    b2 = CAT / "hsc_v5_batch2_gradeC.csv"
    if b2.exists():
        parts.append(pd.read_csv(b2))
    df = pd.concat(parts, ignore_index=True)
    # HOLISMOKES-XVI reject rows carry NO catalog name -> synthesize a coordinate name;
    # dedup by ROUNDED POSITION (names are unreliable across catalogs), keep first.
    noname = df["name"].isna() | (df["name"].astype(str).str.strip() == "")
    df.loc[noname, "name"] = [f"{s}_{r:.6f}{d:+.6f}" for s, r, d in
                              zip(df.loc[noname, "src"], df.loc[noname, "ra"], df.loc[noname, "dec"])]
    df["_pos"] = df.ra.round(5).astype(str) + "_" + df.dec.round(5).astype(str)
    df = df.drop_duplicates(subset="_pos").drop(columns="_pos").reset_index(drop=True)
    print(f"[corpus] catalog rows: {len(df)} "
          f"({df.grade.value_counts().to_dict()})")

    # pixel-backed only
    df["covered"] = [hsc_fetch.cached(r.ra, r.dec) for r in df.itertuples()]
    df = df[df.covered].drop(columns="covered").reset_index(drop=True)
    print(f"[corpus] pixel-backed: {len(df)}")

    # exclude the established 72-object gate (frozen baselines) by 2" position match
    gate = config.OUT / "distill_hsc" / "hsc_test.csv"
    if gate.exists():
        g = pd.read_csv(gate)
        keep = []
        for r in df.itertuples():
            keep.append(_sep_arcsec(r.ra, r.dec, g.ra, g.dec).min() > 2.0)
        n0 = len(df)
        df = df[np.array(keep)].reset_index(drop=True)
        print(f"[corpus] excluded {n0 - len(df)} rows within 2\" of the frozen hsc_test gate")

    df["label"] = np.select(
        [df.grade.isin(["A", "B"]), df.grade.eq("REJ")], ["lens", "nonlens"], default="softmid")

    rng = np.random.RandomState(SEED)
    pos = df[df.label == "lens"].sample(frac=1, random_state=SEED)
    neg = df[df.label == "nonlens"].sample(frac=1, random_state=SEED)
    mid = df[df.label == "softmid"]

    t2 = pd.concat([pos.iloc[:args.n_test2_pos], neg.iloc[:args.n_test2_neg]])
    vs = pd.concat([pos.iloc[args.n_test2_pos:args.n_test2_pos + args.n_valsel_pos],
                    neg.iloc[args.n_test2_neg:args.n_test2_neg + args.n_valsel_neg]])
    tr = pd.concat([pos.iloc[args.n_test2_pos + args.n_valsel_pos:],
                    neg.iloc[args.n_test2_neg + args.n_valsel_neg:],
                    mid])                                    # soft-mid rows: TRAIN ONLY

    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in (("test2", t2), ("valsel", vs), ("train", tr)):
        d = d.sample(frac=1, random_state=SEED).reset_index(drop=True)
        p = OUT / f"{name}.csv"
        d.to_csv(p, index=False)
        print(f"[corpus] {name}: {len(d)} ({d.label.value_counts().to_dict()}) -> {p}")


if __name__ == "__main__":
    main()
