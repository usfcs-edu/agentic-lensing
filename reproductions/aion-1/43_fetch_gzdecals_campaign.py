"""
43 -- Sustained GZ-DECaLS cutout campaign (faithful tasks 7/8 corpus).

The LS cutout service rate-limits to ~20/min, so a faithful GZ-DECaLS retrieval
corpus (Walmsley+2022, 253k galaxies) is a multi-day, resumable fetch. We build
a *priority-ordered* target list so the run is useful early:

  1. high-confidence MERGERS   (merger vote fraction > 0.7; ~3.4k, the rarest +
     most important positives)
  2. high-confidence SPIRALS   (spiral-arms_yes > 0.8 & featured > 0.5; cap 20k)
  3. random DISTRACTORS        (the rest; cap 40k) -> makes positives rare

Cutouts land in the shared `_ls_cutout` disk cache (deduped, resumable across
restarts). Writes data/raw/gzdecals/targets.parquet (iauname, ra, dec, label,
priority) for 43_retrieve_gzdecals.py to consume.

Run: HF_HOME=... python 43_fetch_gzdecals_campaign.py [--workers 6]
       [--n_spiral 20000] [--n_distract 40000]
"""

import argparse

import numpy as np
import pandas as pd

import _config as C
import _ls_cutout as LS

OUT = C.RAW / "gzdecals"


def build_targets(n_spiral, n_distract):
    df = pd.read_parquet(OUT / "gz5.parquet")
    sp = df["has-spiral-arms_yes_fraction"]
    feat = df["smooth-or-featured_featured-or-disk_fraction"]
    mg = df["merging_merger_fraction"]
    is_merger = (mg > 0.7).to_numpy()
    is_spiral = ((sp > 0.8) & (feat > 0.5)).to_numpy() & ~is_merger
    rng = np.random.default_rng(C.SEED)

    rows = []
    rows.append(df[is_merger].assign(label="merger", priority=1))
    sp_df = df[is_spiral]
    if len(sp_df) > n_spiral:
        sp_df = sp_df.iloc[np.sort(rng.choice(len(sp_df), n_spiral, replace=False))]
    rows.append(sp_df.assign(label="spiral", priority=2))
    rest = df[~is_merger & ~is_spiral]
    if len(rest) > n_distract:
        rest = rest.iloc[np.sort(rng.choice(len(rest), n_distract, replace=False))]
    rows.append(rest.assign(label="distractor", priority=3))

    t = pd.concat(rows)[["iauname", "ra", "dec", "label", "priority"]].reset_index(drop=True)
    t = t.sort_values("priority").reset_index(drop=True)
    t.to_parquet(OUT / "targets.parquet")
    print("targets:", t["label"].value_counts().to_dict(), "total", len(t))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n_spiral", type=int, default=20000)
    ap.add_argument("--n_distract", type=int, default=40000)
    args = ap.parse_args()

    t = build_targets(args.n_spiral, args.n_distract)
    coords = list(zip(t["ra"].to_numpy(float), t["dec"].to_numpy(float)))
    print(f"campaign: fetching {len(coords)} GZ-DECaLS cutouts (priority order, resumable)")
    # fetch_one caches to disk; we only need the cache populated, not the arrays.
    from concurrent.futures import ThreadPoolExecutor
    done = {"n": 0, "ok": 0}

    def job(i):
        a = LS.fetch_one(coords[i][0], coords[i][1], layer="ls-dr10", size=160)
        done["n"] += 1
        done["ok"] += int(a is not None)
        if done["n"] % 200 == 0:
            print(f"  campaign {done['n']}/{len(coords)} ok={done['ok']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, range(len(coords))))
    print(f"GZDECALS_CAMPAIGN_OK fetched {done['ok']}/{len(coords)} into cache")


if __name__ == "__main__":
    main()
