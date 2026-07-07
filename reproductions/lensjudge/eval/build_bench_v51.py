#!/usr/bin/env python3
"""eval/build_bench_v51.py — the v5.1 HIGH-POWER cross-pool bench for tier-2 HSC graders.

Purpose: convert the n=72 "exceeds the frozen Claude bar" claim into a definitive one, with the
pool-covariate diagnostic BUILT IN (the Phase-E lesson: single-pool negatives let models score on
construction signatures; per-pool rejection breakdowns expose that immediately).

Composition (all position-deduped at 2" against corpus_v5 TRAIN and internally):
  POSITIVES (~146)
    sugohi_heldout   test2's 120 SuGOHI A/B (held out from training; never used for selection)
    desi_xmatch      the frozen 72-gate's 26 lenses (a DIFFERENT selection function: DESI-bright)
  NEGATIVES (~1,180)
    xvi_reject       test2's 240 expert rejects (the hard, in-corpus-pool family)
    gate_mix         the frozen 72-gate's 46 negatives (mimic-bank + random, historical pool)
    random_field     500 in-footprint offset positions (deployment-like, no construction signature)
    galaxycruise     ~400 human-labeled morphology mimics (rings / tidal / spirals — a third pool)

Also assembles a dedicated pixel cache (bench_v51_cache/pdr3_wide) so one LENSJUDGE_HSC_CACHE serves
the whole bench, and writes the manifest with a `pool` column for per-pool scoring.

  python -m lensjudge.eval.build_bench_v51
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import hsc_fetch  # noqa: E402

SEED = 51
OUT = config.OUT / "bench_v51"
V5CACHE = config.HERE / "cache" / "hsc_v5"
ALLCACHE = config.HERE / "cache" / "hsc_all"
CAT = config.HERE / "cache" / "v5_catalogs"


def _sep_arcsec(ra1, dec1, ra2, dec2):
    ra2, dec2 = np.asarray(ra2, float), np.asarray(dec2, float)
    c = (np.sin(np.radians(dec1)) * np.sin(np.radians(dec2)) +
         np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) *
         np.cos(np.radians(ra1) - np.radians(ra2)))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600.0


def main():
    rows = []
    # positives
    t2 = pd.read_csv(config.OUT / "corpus_v5" / "test2.csv")
    for r in t2[t2.label == "lens"].itertuples():
        rows.append(dict(name=r.name, ra=r.ra, dec=r.dec, label="lens", pool="sugohi_heldout"))
    for r in t2[t2.label == "nonlens"].itertuples():
        rows.append(dict(name=r.name, ra=r.ra, dec=r.dec, label="nonlens", pool="xvi_reject"))
    gate = pd.read_csv(config.OUT / "distill_hsc" / "hsc_test.csv")
    for r in gate.itertuples():
        rows.append(dict(name=r.name, ra=r.ra, dec=r.dec,
                         label=("lens" if r.label == "lens" else "nonlens"),
                         pool=("desi_xmatch" if r.label == "lens" else "gate_mix")))
    # random-field: 500-sample of the 2,000 fetched (rest reserved for v5.1 training dilution)
    rf = pd.read_csv(CAT / "hsc_v51_randomfield.csv").sample(500, random_state=SEED)
    for r in rf.itertuples():
        rows.append(dict(name=r.name, ra=r.ra, dec=r.dec, label="nonlens", pool="random_field"))
    # galaxy cruise mimics (by kind)
    gc = pd.read_csv(CAT / "hsc_v51_galaxycruise.csv")
    for r in gc.itertuples():
        rows.append(dict(name=r.name, ra=r.ra, dec=r.dec, label="nonlens", pool=str(r.src)))
    df = pd.DataFrame(rows)

    # dedup: internal by position, then 2" against corpus TRAIN
    df["_pos"] = df.ra.round(5).astype(str) + "_" + df.dec.round(5).astype(str)
    df = df.drop_duplicates("_pos").drop(columns="_pos").reset_index(drop=True)
    tr = pd.read_csv(config.OUT / "corpus_v5" / "train.csv")
    keep = []
    for r in df.itertuples():
        keep.append(_sep_arcsec(r.ra, r.dec, tr.ra, tr.dec).min() > 2.0)
    n0 = len(df)
    df = df[np.array(keep)].reset_index(drop=True)
    print(f"[bench] dropped {n0 - len(df)} rows within 2\" of corpus TRAIN")

    # pixel-backed check + dedicated cache assembly (pull dirs from hsc_v5 first, then hsc_all)
    cache = OUT / "bench_v51_cache" / "pdr3_wide"
    cache.mkdir(parents=True, exist_ok=True)
    kept, miss = [], 0
    for r in df.itertuples():
        key = hsc_fetch._key(r.ra, r.dec)
        src = None
        for c in (V5CACHE, ALLCACHE):
            if (c / "pdr3_wide" / key).exists():
                src = c / "pdr3_wide" / key
                break
        if src is None:
            miss += 1
            kept.append(False)
            continue
        shutil.copytree(src, cache / key, dirs_exist_ok=True)
        kept.append(True)
    df = df[np.array(kept)].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "bench_v51.csv", index=False)
    print(f"[bench] final: {len(df)} rows ({miss} missing pixels)")
    print(df.groupby(["label", "pool"]).size().to_string())
    print(f"[bench] manifest -> {OUT/'bench_v51.csv'} | cache -> {cache.parent}")


if __name__ == "__main__":
    main()
