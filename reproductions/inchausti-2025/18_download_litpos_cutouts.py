#!/usr/bin/env python3
"""
18_download_litpos_cutouts.py — Phase-5 Stage B.

Download DR9 grz cutouts (101x101, pixscale 0.262 — identical to training) for
the NEW literature lens positions assembled by 17. Positions outside the DESI
Legacy DR9 footprint (e.g. most KiDS / COSMOS / parts of HSC) return a blank
(near-zero) cutout from the service; we record each cutout's per-band flux std so
the Stage-B trainer can keep only the in-footprint (non-blank) literature
positives.

Output:
  data/cutouts_fits_litpos_dr9/<Name>.fits
  data/litpos_cutout_manifest.csv   Name, RA, DEC, source, is_new, status, flux_std
"""
from __future__ import annotations

import argparse
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
OUT = DATA / "cutouts_fits_litpos_dr9"
LAYER = "ls-dr9"
PIXSCALE, SIZE = 0.262, 101
TIMEOUT, RETRIES, RETRY_BACKOFF, RL_BACKOFF = 60, 5, 4.0, 30.0
BASE = "https://www.legacysurvey.org/viewer/fits-cutout"


def url(ra, dec):
    return (f"{BASE}?ra={ra:.6f}&dec={dec:.6f}&size={SIZE}&layer={LAYER}"
            f"&pixscale={PIXSCALE}&bands=grz")


def flux_std(path: Path) -> float:
    try:
        with fits.open(path) as h:
            cube = np.asarray(h[0].data, dtype=np.float32)
        return float(np.nanstd(cube))
    except Exception:
        return 0.0


def fetch_one(row):
    name = str(row["Name"])
    out = OUT / f"{name}.fits"
    if out.exists() and out.stat().st_size > 0:
        return (name, "skip", flux_std(out))
    last = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url(row["RA"], row["DEC"]), timeout=TIMEOUT, stream=True)
            if r.status_code == 429:
                time.sleep(RL_BACKOFF); continue
            r.raise_for_status()
            with open(out, "wb") as f:
                for ch in r.iter_content(chunk_size=65536):
                    if ch:
                        f.write(ch)
            if out.stat().st_size < 256:
                out.unlink(); raise RuntimeError("too-small")
            return (name, "ok", flux_std(out))
        except Exception as e:
            last = str(e)
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return (name, f"fail:{last}", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--new-only", action="store_true", default=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lit = pd.read_parquet(DATA / "positives_literature.parquet")
    if args.new_only:
        lit = lit[lit["is_new"]].reset_index(drop=True)
    print(f"[init] {len(lit)} literature positions -> {OUT.name} (layer {LAYER})")
    rows = lit.to_dict("records")
    recs, counts = [], {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, r) for r in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="cut"):
            name, status, fstd = fut.result()
            k = "ok" if status == "ok" else ("skip" if status == "skip" else "fail")
            counts[k] += 1
            recs.append({"Name": name, "status": status, "flux_std": fstd})
    mf = lit.merge(pd.DataFrame(recs), on="Name")
    mf.to_csv(DATA / "litpos_cutout_manifest.csv", index=False)
    nonblank = int((mf["flux_std"] > 1e-3).sum())
    print(f"[done] ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")
    print(f"[footprint] {nonblank}/{len(mf)} have real flux (in DR9 footprint); "
          f"the rest are blank/out-of-footprint")


if __name__ == "__main__":
    main()
