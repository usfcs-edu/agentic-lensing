#!/usr/bin/env python3
"""
09_download_dr10_cutouts.py

For each of our 13,530 reproduced groups, download a DESI Legacy Surveys DR10
(g, r, z) cutout centered on the group centroid via the Legacy Survey image
service.

Two output formats, controlled by --format:
  jpeg : 3-band composite, ~3-10 KB each (~50 MB total). Good for visual
         inspection and quick CNN baselines (PIL.open / torchvision).
  fits : 3-band float32 cube (g, r, z), ~80 KB each (~1.1 GB total). Retains
         photometric calibration — required for DimpleScout-style training.

URL pattern (verified 2026-05-26):
  https://www.legacysurvey.org/viewer/{jpeg|fits}-cutout
    ?ra=X&dec=Y&size=N&layer=ls-dr10&pixscale=0.27[&bands=grz]

Adaptive cutout size: max(80, sep_arcsec/0.27 + 80) pixels, capped at 200.
At 0.27″/pix this gives 22-54″ FoV — enough to see a typical strong-lens arc
(θ_E ~ 1-3″) with comfortable surrounding context.

Resumable: skips files that already exist on disk.
Parallel: ThreadPoolExecutor(workers=8) — polite default for legacysurvey.org.

Outputs:
  figs/cutouts_{format}_dr10/group_{group_id:08d}.{jpg|fits}
  data/cutouts_dr10_manifest.csv  (group_id, RA, DEC, size_pix, FoV_arcsec, path)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
PAIRS = DATA / "dr1_pairs.parquet"

PIXSCALE = 0.27  # arcsec/pix, DR10 default for DECam
MIN_SIZE_PIX = 80
MAX_SIZE_PIX = 200
LAYER = "ls-dr10"
WORKERS = 8
PER_WORKER_DELAY = 0.0   # the natural ~3s HTTP latency is rate limiting enough
RATELIMIT_BACKOFF = 30.0 # seconds to wait after HTTP 429
TIMEOUT = 60
RETRIES = 5
RETRY_BACKOFF = 4.0


def adaptive_size_pix(sep_arcsec: float) -> int:
    sz = int(sep_arcsec / PIXSCALE + 80)
    return max(MIN_SIZE_PIX, min(MAX_SIZE_PIX, sz))


def group_centroids(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute (RA, Dec) centroid + max within-group angular separation per group.

    Uses a small-angle approximation: members are within 3″ by construction
    (FoF link length), so cos(Dec)-corrected Euclidean distance is exact to
    <0.01″.
    """
    import numpy as np
    g = pairs.groupby("group_id")
    cen = g[["RA", "DEC"]].mean().reset_index()
    span_ra_deg  = (g["RA"].max() - g["RA"].min()).reset_index(drop=True)
    span_dec_deg = (g["DEC"].max() - g["DEC"].min()).reset_index(drop=True)
    cos_dec = np.cos(np.deg2rad(cen["DEC"].to_numpy()))
    sep_arcsec = np.sqrt((span_ra_deg.to_numpy() * cos_dec) ** 2 +
                          span_dec_deg.to_numpy() ** 2) * 3600.0
    cen["sep_arcsec"] = np.clip(sep_arcsec, 0.0, 6.0)  # FoF link is 3″, doubled for safety
    cen["size_pix"] = cen["sep_arcsec"].apply(adaptive_size_pix).astype(int)
    cen["fov_arcsec"] = cen["size_pix"] * PIXSCALE
    return cen


def make_url(ra: float, dec: float, size: int, fmt: str) -> str:
    base = "https://www.legacysurvey.org/viewer"
    endpoint = "jpeg-cutout" if fmt == "jpeg" else "fits-cutout"
    bands = "" if fmt == "jpeg" else "&bands=grz"
    return (
        f"{base}/{endpoint}"
        f"?ra={ra:.6f}&dec={dec:.6f}&size={size}&layer={LAYER}"
        f"&pixscale={PIXSCALE}{bands}"
    )


def fetch_one(row: dict, out_dir: Path, fmt: str) -> tuple[int, str, int, int]:
    """Return (group_id, status, bytes_written, attempts).

    Honors PER_WORKER_DELAY between requests in the same thread (polite to the
    Legacy Survey service). On HTTP 429 (rate-limited), sleeps for
    RATELIMIT_BACKOFF seconds before retry — generous because the service's own
    cooldown can be 10-30 s.
    """
    suffix = "jpg" if fmt == "jpeg" else "fits"
    out = out_dir / f"group_{int(row['group_id']):08d}.{suffix}"
    if out.exists() and out.stat().st_size > 0:
        return (int(row["group_id"]), "skip", out.stat().st_size, 0)
    url = make_url(row["RA"], row["DEC"], int(row["size_pix"]), fmt)
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code == 429:
                last_err = "HTTP 429 (rate limited)"
                time.sleep(RATELIMIT_BACKOFF)
                continue
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            sz = out.stat().st_size
            if sz < 256:  # HTML error pages are typically 200-300 bytes
                last_err = f"too-small ({sz} B)"
                out.unlink()
                raise RuntimeError(last_err)
            time.sleep(PER_WORKER_DELAY)
            return (int(row["group_id"]), "ok", sz, attempt)
        except Exception as e:
            last_err = str(e)
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return (int(row["group_id"]), f"fail: {last_err}", 0, RETRIES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["jpeg", "fits"], default="jpeg")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only fetch first N centroids (smoke test)")
    args = ap.parse_args()

    out_dir = FIGS / f"cutouts_{args.format}_dr10"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not PAIRS.exists():
        raise SystemExit(f"missing {PAIRS}; run 05_run_full_fof.py first")
    pairs = pq.read_table(PAIRS).to_pandas()
    cen = group_centroids(pairs)
    if args.limit:
        cen = cen.head(args.limit)
    print(f"[plan] {len(cen):,} centroids, format={args.format}, out_dir={out_dir}")
    print(f"[plan] size distribution: min={cen['size_pix'].min()}, "
          f"median={int(cen['size_pix'].median())}, max={cen['size_pix'].max()} pixels")
    print(f"[plan] FoV: min={cen['fov_arcsec'].min():.1f}″, "
          f"median={cen['fov_arcsec'].median():.1f}″, max={cen['fov_arcsec'].max():.1f}″")

    manifest_rows = []
    counts = {"ok": 0, "skip": 0, "fail": 0}
    bytes_total = 0
    rows = cen.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_one, r, out_dir, args.format) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="cutout"):
            gid, status, nbytes, attempts = fut.result()
            bytes_total += nbytes
            if status == "ok":
                counts["ok"] += 1
            elif status == "skip":
                counts["skip"] += 1
            else:
                counts["fail"] += 1
                tqdm.write(f"  [fail] group {gid}: {status}")
            manifest_rows.append({
                "group_id": gid, "status": status, "bytes": nbytes,
                "attempts": attempts,
            })

    print()
    print(f"[done] ok={counts['ok']:,}  skip={counts['skip']:,}  fail={counts['fail']:,}")
    print(f"[done] total bytes downloaded: {bytes_total/1e6:.1f} MB")

    # Build manifest joining centroid metadata + fetch status
    mf = (
        cen.merge(pd.DataFrame(manifest_rows), on="group_id")
        .assign(path=lambda d: d["group_id"].apply(
            lambda g: str(out_dir / f"group_{int(g):08d}."
                          f"{'jpg' if args.format == 'jpeg' else 'fits'}")
        ))
    )
    mf_path = DATA / f"cutouts_dr10_manifest_{args.format}.csv"
    mf.to_csv(mf_path, index=False)
    print(f"[save] wrote {mf_path}  ({len(mf):,} rows)")


if __name__ == "__main__":
    main()
