#!/usr/bin/env python3
"""110_build_negpool_manifest.py — Phase 110 (NegEval-1M): brick-disjoint
negative-eval + mining-pool manifests (runs LOCALLY).

Builds TWO brick-disjoint ~1M-galaxy manifests from the Huang non-PSF parent
sample (data/parent_dr8.parquet, 17.3M DEV/COMP galaxies):

    NEGEVAL  -> data/v2/negeval_manifest.parquet   (held-out negative eval set)
    MINEPOOL -> data/v2/minepool_manifest.parquet  (hard-negative mining pool)

Both require grz coverage (NOBS_G/R/Z >= 1), exclude EVERY brick used by the
v1 negatives (negatives_extra.parquet — v1 train AND eval negatives came from
those 393 bricks, so brick-level disjointness is the leak guard), drop objects
within 10" of any known lens (published catalogs + v1 curated positives — the
lineage standard from inchausti-2025/20_build_negatives_brick_dr9.py; that
script's positives_huang2020.parquet is not in claudenet/data, so
positives_curated.parquet substitutes — it covers the v1 training positives,
which is the leak that matters here; re-verify coverage if catalogs change), and
keep north/south at their natural parent proportions (~29% north) via
footprint-stratified brick sampling. The two manifests share no brick, so
mined hard negatives can never leak into the eval set. The 10" mask is applied
AFTER sampling (sample with --margin, mask, trim) — masking all 17M rows at
once is wasteful.

    python 110_build_negpool_manifest.py --target 1000000 --per-brick 100
    python 110_build_negpool_manifest.py --parent /tmp/parent50k.parquet \
        --target 2000 --per-brick 10 --dry-run     # logic check, no writes
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

import _clib as C

LENS_CSVS = ("storfer2024_published_catalog.csv",
             "inchausti2025_published_catalog.csv",
             "huang2021_published_catalog.csv")


def lens_mask(df, lens_sky):
    """True for rows within 10" of a known lens (copied from
    inchausti-2025/20_build_negatives_brick_dr9.py — the lineage standard)."""
    sky = SkyCoord(ra=df.RA.values * u.deg, dec=df.DEC.values * u.deg)
    _, sep, _ = sky.match_to_catalog_sky(lens_sky)
    return sep.to(u.arcsec).value < 10.0


def load_lens_sky():
    """Known-lens catalog: 3 published CSVs + v1 curated training positives."""
    frames = [pd.read_parquet(C.DATA / "positives_curated.parquet")[["RA", "DEC"]]]
    for f in LENS_CSVS:
        frames.append(pd.read_csv(C.DATA / f)[["RA", "DEC"]])
    lens = pd.concat(frames, ignore_index=True).dropna()
    return SkyCoord(ra=lens.RA.values * u.deg, dec=lens.DEC.values * u.deg), len(lens)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent", default=str(C.DATA / "parent_dr8.parquet"),
                    help="parent parquet (override with a small sample for dry runs)")
    ap.add_argument("--target", type=int, default=1_000_000, help="rows per manifest")
    ap.add_argument("--per-brick", type=int, default=100, help="max objects per brick")
    ap.add_argument("--margin", type=float, default=1.05,
                    help="oversample factor before the 10\" lens mask + trim")
    ap.add_argument("--dry-run", action="store_true", help="run everything but skip writes")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)

    # 1-2. coverage filter + v1-brick exclusion -------------------------------
    cols = ["BRICKID", "BRICKNAME", "OBJID", "RA", "DEC",
            "NOBS_G", "NOBS_R", "NOBS_Z", "footprint"]
    p = pd.read_parquet(args.parent, columns=cols)
    n0 = len(p)
    p = p[(p.NOBS_G >= 1) & (p.NOBS_R >= 1) & (p.NOBS_Z >= 1)]
    v1_bricks = set(pd.read_parquet(C.DATA / "negatives_extra.parquet")["brick"].unique())
    p = p[~p.BRICKNAME.isin(v1_bricks)].reset_index(drop=True)
    print(f"[parent] {n0:,} rows -> {len(p):,} after NOBS_GRZ>=1 + "
          f"{len(v1_bricks)} v1-negative-brick exclusion")

    # 3. known-lens catalog (mask applied to SAMPLED rows only, below) --------
    lens_sky, n_lens = load_lens_sky()
    print(f"[lens] {n_lens:,} known-lens positions (10\" exclusion radius)")

    # 4. footprint-stratified disjoint brick sampling --------------------------
    frac = p.footprint.value_counts(normalize=True)
    foots = sorted(frac.index)
    quota, acc = {}, 0
    for i, f in enumerate(foots):
        q = args.target - acc if i == len(foots) - 1 else int(round(args.target * frac[f]))
        quota[f], acc = q, acc + q
    print("[quota] " + ", ".join(f"{f}={quota[f]:,} ({frac[f]:.1%} of parent)"
                                 for f in foots))

    gidx = p.groupby(["footprint", "BRICKNAME"], sort=True).indices  # key -> positions
    bricks_by_foot: dict[str, list[str]] = {}
    for (f, b) in gidx:
        bricks_by_foot.setdefault(f, []).append(b)

    used: set[str] = set()          # bricknames assigned to either set (global:
    sels = {"negeval": [], "minepool": []}  # a sky brick in both footprints
    for f in foots:                          # must not straddle the two sets)
        blist = bricks_by_foot.get(f, [])
        order = iter(rng.permutation(len(blist)))
        for name in ("negeval", "minepool"):
            need, got, nb = int(np.ceil(quota[f] * args.margin)), 0, 0
            for j in order:
                b = blist[j]
                if b in used:
                    continue
                pos = gidx[(f, b)]
                k = min(args.per_brick, len(pos))
                sel = pos if k == len(pos) else rng.choice(pos, size=k, replace=False)
                sels[name].append(np.asarray(sel))
                used.add(b)
                got, nb = got + k, nb + 1
                if got >= need:
                    break
            print(f"[sample] {name}/{f}: {got:,} rows from {nb:,} bricks "
                  f"(margin target {need:,})")
            if got < need:
                print(f"[warn] {name}/{f}: brick pool exhausted before margin target")

    # 5. lens-mask sampled rows, trim to quota, assert, write ------------------
    frames = {}
    for name in ("negeval", "minepool"):
        if not sels[name]:
            print(f"[fatal] {name}: no bricks sampled (pool exhausted)")
            return 1
        df = p.take(np.concatenate(sels[name]))
        n_pre = len(df)
        df = df[~lens_mask(df, lens_sky)]
        print(f"[mask] {name}: {n_pre - len(df):,}/{n_pre:,} rows within 10\" of a "
              f"known lens -> dropped")
        trimmed = []
        for f in foots:
            sub = df[df.footprint == f]
            if len(sub) < quota[f]:
                print(f"[warn] {name}/{f}: {len(sub):,} < quota {quota[f]:,} "
                      f"after masking (raise --margin)")
            trimmed.append(sub.head(quota[f]))
        df = pd.concat(trimmed, ignore_index=True)
        out = pd.DataFrame({
            "row_id": df.BRICKID.astype(int).astype(str) + "_"
                      + df.OBJID.astype(int).astype(str),
            "RA": df.RA.astype(float), "DEC": df.DEC.astype(float),
            "footprint": df.footprint.astype(str), "brick": df.BRICKNAME.astype(str),
        })
        assert out.row_id.is_unique, f"{name}: duplicate row_ids"
        frames[name] = out

    b_ne, b_mp = set(frames["negeval"].brick), set(frames["minepool"].brick)
    assert not (b_ne & b_mp), "NEGEVAL/MINEPOOL share bricks"
    assert not ((b_ne | b_mp) & v1_bricks), "overlap with v1 negative bricks"
    ev_ids = set(pd.read_parquet(C.DATA / "eval_testneg.parquet")["row_id"])
    for name, fdf in frames.items():
        assert not (set(fdf.row_id) & ev_ids), f"{name} overlaps eval_testneg row_ids"
    print("[assert] brick disjointness (negeval|minepool|v1) + eval_testneg "
          "row_id disjointness: OK")

    for name, fdf in frames.items():
        per_foot = ", ".join(f"{f}: {int((fdf.footprint == f).sum()):,}" for f in foots)
        print(f"[summary] {name}: {len(fdf):,} rows ({per_foot}); "
              f"{fdf.brick.nunique():,} bricks")

    if args.dry_run:
        print(f"[dry-run] skipping writes ({time.time() - t0:.1f}s)")
        return 0
    out_dir = C.DATA / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fdf in frames.items():
        path = out_dir / f"{name}_manifest.parquet"
        fdf.to_parquet(path, index=False)
        print(f"[write] {path} ({len(fdf):,} rows)")
    print(f"[done] {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
