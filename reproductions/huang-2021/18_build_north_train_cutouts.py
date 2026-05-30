#!/usr/bin/env python3
"""
18_build_north_train_cutouts.py — Phase 4b north-calibration fix.

The Phase 4a models were trained on southern (DECaLS/DES) positives + south DR1
negatives. They saw ~160 northern (MzLS) lens POSITIVES but ZERO northern
NEGATIVES, so the decision boundary in BASS/MzLS imaging space is undefined and
the L18 model over-fires on north non-lenses (91% score >=0.1 vs 4.8% in south;
diagnosed 2026-05-29). Fix: add northern negatives (and re-grab the northern
positives at DR8 so all north training data matches the DR8 deployment imaging),
then retrain (05c_train_northaug.py).

Downloads DR8 grz cutouts via the legacysurvey viewer (layer=ls-dr8) into
data/cutouts_fits_north/:
  - north positives: the MzLS-region rows of positives_huang2020.parquet
    (already training positives — no NEW leakage to the 363 leak-free shielded
    discoveries).
  - north negatives: a random sample of north (footprint==north) galaxies from
    parent_dr8.parquet.

Writes data/positives_north.parquet and data/negatives_north.parquet.
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
from astropy.io import fits
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT_FITS = DATA / "cutouts_fits_north"
OUT_FITS.mkdir(parents=True, exist_ok=True)
VIEWER = "https://www.legacysurvey.org/viewer/fits-cutout"
SEED = 2026


def fetch_cutout(ra: float, dec: float, out: Path, size: int = 101,
                 pixscale: float = 0.262) -> str:
    if out.exists() and out.stat().st_size > 1024:
        return "skip"
    url = (f"{VIEWER}?ra={ra:.6f}&dec={dec:.6f}&size={size}&layer=ls-dr8"
           f"&pixscale={pixscale}&bands=grz")
    for attempt in range(1, 6):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(30); continue
            r.raise_for_status()
            if len(r.content) < 256:
                raise RuntimeError("too small")
            with fits.open(io.BytesIO(r.content), memmap=False) as h:
                d = h[0].data
            if d is None or d.ndim != 3 or d.shape[0] != 3:
                raise ValueError(f"shape {None if d is None else d.shape}")
            out.write_bytes(r.content)
            return "ok"
        except Exception as e:
            if attempt == 5:
                return f"fail:{str(e)[:40]}"
            time.sleep(4 * attempt)
    return "fail"


def download_set(items: list[tuple[str, float, float]], workers: int, label: str) -> int:
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_cutout, ra, dec, OUT_FITS / f"{rid}.fits"): rid
                for rid, ra, dec in items}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=label):
            st = fut.result()
            if st == "ok": n_ok += 1
            elif st == "skip": n_skip += 1
            else: n_fail += 1
    print(f"[{label}] ok={n_ok} skip={n_skip} fail={n_fail}")
    return n_ok + n_skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neg", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    # North positives = MzLS rows of the existing training positives.
    pos = pd.read_parquet(DATA / "positives_huang2020.parquet")
    npos = pos[pos["Region"] == "MzLS"][["Name", "RA", "DEC"]].copy()
    npos.to_parquet(DATA / "positives_north.parquet", index=False)
    print(f"[init] {len(npos)} north positives (MzLS)")

    # North negatives = random north galaxies from the parent sample.
    par = pd.read_parquet(DATA / "parent_dr8.parquet",
                          columns=["RA", "DEC", "BRICKID", "OBJID", "footprint"])
    north = par[par["footprint"] == "north"]
    samp = north.sample(n=min(args.n_neg, len(north)), random_state=SEED).copy()
    samp["row_id"] = samp["BRICKID"].astype(str) + "_" + samp["OBJID"].astype(str)
    nneg = samp[["row_id", "RA", "DEC"]].reset_index(drop=True)
    nneg.to_parquet(DATA / "negatives_north.parquet", index=False)
    print(f"[init] {len(nneg)} north negatives sampled from {len(north):,} north galaxies")

    download_set([(r.Name, r.RA, r.DEC) for r in npos.itertuples()],
                 args.workers, "north-pos")
    download_set([(r.row_id, r.RA, r.DEC) for r in nneg.itertuples()],
                 args.workers, "north-neg")
    print(f"[done] cutouts in {OUT_FITS}")


if __name__ == "__main__":
    main()
