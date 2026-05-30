#!/usr/bin/env python3
"""
12_download_candidate_cutouts.py — Phase-5 targeted-recovery, track (ii).

Download a 101x101 grz FITS cutout for every published Storfer-2024 (DR9) and
Inchausti-2025 (DR10) candidate, at the SAME size/pixscale/bands as the training
cutouts (Huang+2020 §3.2: size=101, pixscale=0.262, bands=grz) so our models see
consistent inputs. These let us directly score "would our reproduction have
flagged this published lens" (13_score_candidates_direct.py).

  storfer    -> layer ls-dr9  -> data/cutouts_fits_candidates_storfer/<name>.fits
  inchausti  -> layer ls-dr10 -> data/cutouts_fits_candidates_inchausti/<name>.fits

Resumable (skips files already on disk), ThreadPoolExecutor, HTTP-429 backoff.
Only ~1,895 + 811 = 2,706 cutouts total.

Output manifests: data/candidate_cutout_manifest_<catalog>.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

PIXSCALE = 0.262
SIZE_PIX = 101
TIMEOUT = 60
RETRIES = 5
RETRY_BACKOFF = 4.0
RATELIMIT_BACKOFF = 30.0
BASE = "https://www.legacysurvey.org/viewer"

CATALOGS = {
    "storfer":   dict(csv="storfer2024_published_catalog.csv",   layer="ls-dr9",
                      out="cutouts_fits_candidates_storfer"),
    "inchausti": dict(csv="inchausti2025_published_catalog.csv", layer="ls-dr10",
                      out="cutouts_fits_candidates_inchausti"),
}


def make_url(ra: float, dec: float, layer: str) -> str:
    return (f"{BASE}/fits-cutout?ra={ra:.6f}&dec={dec:.6f}"
            f"&size={SIZE_PIX}&layer={layer}&pixscale={PIXSCALE}&bands=grz")


def fetch_one(row: dict, out_dir: Path, layer: str) -> tuple[str, str, int]:
    name = str(row["name"])
    out = out_dir / f"{name}.fits"
    if out.exists() and out.stat().st_size > 0:
        return (name, "skip", out.stat().st_size)
    url = make_url(row["RA"], row["DEC"], layer)
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code == 429:
                last_err = "HTTP 429"
                time.sleep(RATELIMIT_BACKOFF)
                continue
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            sz = out.stat().st_size
            if sz < 256:  # HTML error page
                out.unlink()
                last_err = f"too-small ({sz} B)"
                raise RuntimeError(last_err)
            return (name, "ok", sz)
        except Exception as e:
            last_err = str(e)
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return (name, f"fail: {last_err}", 0)


def run_catalog(key: str, workers: int, limit: int | None) -> None:
    cfg = CATALOGS[key]
    cat = pd.read_csv(DATA / cfg["csv"])
    if limit:
        cat = cat.head(limit)
    out_dir = DATA / cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{key}] {len(cat):,} candidates -> {out_dir.name} (layer {cfg['layer']})")
    rows = cat.to_dict("records")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    recs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, r, out_dir, cfg["layer"]) for r in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="cutout"):
            name, status, nbytes = fut.result()
            k = "ok" if status == "ok" else ("skip" if status == "skip" else "fail")
            counts[k] += 1
            if k == "fail":
                tqdm.write(f"  [fail] {name}: {status}")
            recs.append({"name": name, "status": status, "bytes": nbytes})
    mf = cat.merge(pd.DataFrame(recs), on="name")
    mf_path = DATA / f"candidate_cutout_manifest_{key}.csv"
    mf.to_csv(mf_path, index=False)
    print(f"[{key}] ok={counts['ok']:,} skip={counts['skip']:,} fail={counts['fail']:,} "
          f"-> {mf_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", choices=("storfer", "inchausti", "both"), default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    keys = ("storfer", "inchausti") if args.catalog == "both" else (args.catalog,)
    for k in keys:
        run_catalog(k, args.workers, args.limit)


if __name__ == "__main__":
    main()
