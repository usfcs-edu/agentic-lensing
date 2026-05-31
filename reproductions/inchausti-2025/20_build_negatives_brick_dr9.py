#!/usr/bin/env python3
"""
20_build_negatives_brick_dr9.py — Phase-5 Stage C (negative scale-up).

Build a large random-galaxy NEGATIVE set with DR9 cutouts, to lift the
negative:positive ratio from Stage-B's ~2.5:1 toward the papers' ~33:1 (Storfer)
/ ~100:1 (Inchausti) — the single biggest remaining data difference, which sets
the false-positive rate / operating threshold.

Efficiency: instead of ~45K rate-limited endpoint cutouts (~8 h), we download a
few hundred DR9 brick coadds and slice 101x101 cutouts locally (~200x faster),
reusing the Phase-4 brick machinery repointed to dr9/. Negatives are real
DEV/COMP parent-sample galaxies (the same non-lens population the papers use),
drawn from the densest bricks for download efficiency and filtered to be >10"
from any known lens (our positives + the published catalogues).

Output:
  data/cutouts_fits_neg_dr9/<BRICKID>_<OBJID>.fits
  data/negatives_extra.parquet   row_id, RA, DEC, footprint, brick
"""
from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "cutouts_fits_neg_dr9"
BRICK_TMP = DATA / "brick_tmp_neg"
BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9"
COADD = {"south": f"{BASE}/south/coadd", "north": f"{BASE}/north/coadd"}
BANDS = ("g", "r", "z")
SIZE, HALF = 101, 50
TIMEOUT, RETRIES, RETRY_BACKOFF = 180, 3, 6.0


def brick_url(footprint, brick, band):
    return f"{COADD[footprint]}/{brick[:3]}/{brick}/legacysurvey-{brick}-image-{band}.fits.fz"


def download_brick(footprint, brick, dest):
    paths = {}
    for band in BANDS:
        out = dest / f"{brick}-{band}.fits.fz"
        if out.exists() and out.stat().st_size > 1024:
            paths[band] = out; continue
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(brick_url(footprint, brick, band), timeout=TIMEOUT, stream=True)
                if r.status_code == 404:
                    return {}
                r.raise_for_status()
                tmp = out.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for ch in r.iter_content(chunk_size=262144):
                        if ch:
                            f.write(ch)
                tmp.rename(out); paths[band] = out; break
            except Exception:
                if attempt < RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
                else:
                    return {}
    return paths


def load_brick(paths):
    arrs, wcs = [], None
    for band in BANDS:
        with fits.open(paths[band]) as h:
            hdu = h[1]
            arrs.append(np.asarray(hdu.data, dtype=np.float32))
            if wcs is None:
                wcs = WCS(hdu.header)
    return np.stack(arrs, 0), wcs


def extract(cube, wcs, ra, dec):
    px = wcs.world_to_pixel_values(ra, dec)
    cx, cy = int(round(float(px[0]))), int(round(float(px[1])))
    H, W = cube.shape[1], cube.shape[2]
    y0, y1, x0, x1 = cy - HALF, cy + HALF + 1, cx - HALF, cx + HALF + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return cube[:, y0:y1, x0:x1]


def to_bytes(cube):
    bio = io.BytesIO(); fits.PrimaryHDU(data=cube.astype(np.float32)).writeto(bio); return bio.getvalue()


def lens_mask(df, lens_sky):
    sky = SkyCoord(ra=df.RA.values * u.deg, dec=df.DEC.values * u.deg)
    _, sep, _ = sky.match_to_catalog_sky(lens_sky)
    return sep.to(u.arcsec).value < 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=45000)
    ap.add_argument("--per-brick", type=int, default=150)
    ap.add_argument("--brick-workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); BRICK_TMP.mkdir(parents=True, exist_ok=True)

    p = pd.read_parquet(DATA / "parent_dr8.parquet")
    p = p[p["TYPE"].astype(str).str.upper() != "PSF"]
    # known-lens exclusion set
    lens = [pd.read_parquet(DATA / "positives_huang2020.parquet")[["RA", "DEC"]]]
    for f in ("storfer2024_published_catalog.csv", "inchausti2025_published_catalog.csv",
              "huang2021_published_catalog.csv"):
        if (DATA / f).exists():
            lens.append(pd.read_csv(DATA / f)[["RA", "DEC"]])
    lens = pd.concat(lens, ignore_index=True).dropna()
    lens_sky = SkyCoord(ra=lens.RA.values * u.deg, dec=lens.DEC.values * u.deg)
    print(f"[init] {len(p):,} non-PSF parent galaxies; {len(lens)} known lenses to avoid")

    # densest bricks first (download efficiency); pre-filter the parent to the
    # top candidate bricks ONCE (avoid rescanning 17.3M rows per brick).
    sizes = p.groupby(["footprint", "BRICKNAME"]).size().sort_values(ascending=False)
    n_bricks_needed = int(args.target / max(1, args.per_brick)) + 1
    top = sizes.head(max(600, n_bricks_needed * 3))
    top_keys = set(top.index)
    p = p.assign(_key=list(zip(p.footprint, p.BRICKNAME)))
    cand = p[p["_key"].isin(top_keys)]
    groups = {k: g for k, g in cand.groupby(["footprint", "BRICKNAME"])}
    chosen, n_acc = [], 0
    for (foot, brick) in top.index:
        sub = groups.get((foot, brick))
        if sub is None:
            continue
        sub = sub[~lens_mask(sub, lens_sky)]
        if len(sub) == 0:
            continue
        sub = sub.sample(n=min(args.per_brick, len(sub)), random_state=2026)
        chosen.append((foot, brick, sub))
        n_acc += len(sub)
        if n_acc >= args.target:
            break
    print(f"[plan] {len(chosen)} bricks, ~{n_acc:,} candidate negatives")

    def process(item):
        foot, brick, sub = item
        tmp = BRICK_TMP / f"{foot}_{brick}"; tmp.mkdir(parents=True, exist_ok=True)
        paths = download_brick(foot, brick, tmp)
        if not paths:
            for q in tmp.glob("*"):
                q.unlink(missing_ok=True)
            tmp.rmdir(); return []
        cube, wcs = load_brick(paths)
        recs = []
        for _, r in sub.iterrows():
            ct = extract(cube, wcs, float(r.RA), float(r.DEC))
            if ct is None or ct.shape != (3, 101, 101):
                continue
            rid = f"{int(r.BRICKID)}_{int(r.OBJID)}"
            (OUT / f"{rid}.fits").write_bytes(to_bytes(ct))
            recs.append({"row_id": rid, "RA": float(r.RA), "DEC": float(r.DEC),
                         "footprint": foot, "brick": brick})
        for q in paths.values():
            q.unlink(missing_ok=True)
        tmp.rmdir()
        return recs

    all_recs = []
    with ThreadPoolExecutor(max_workers=args.brick_workers) as ex:
        futs = [ex.submit(process, it) for it in chosen]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="brick"):
            all_recs.extend(fut.result())
    neg = pd.DataFrame(all_recs)
    neg.to_parquet(DATA / "negatives_extra.parquet", index=False)
    print(f"[done] {len(neg):,} negative cutouts -> {OUT.name}; negatives_extra.parquet written")


if __name__ == "__main__":
    main()
