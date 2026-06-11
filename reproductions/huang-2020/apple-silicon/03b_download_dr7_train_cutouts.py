#!/usr/bin/env python3
"""
03_download_decals_cutouts.py

Download DESI Legacy Surveys cutouts (FITS + optional JPEG) at the (RA, Dec)
of every row in an input parquet. Mirrors the Phase-2 cutout puller (script 09)
but tuned to Huang+2020 §3.2 spec:

  size      = 101 pixels       (Huang+2020 §3.2)
  pixscale  = 0.262 arcsec/px  (DECam native; ~26.5" FoV)
  bands     = grz
  layer     = ls-dr9           (closest available analogue to Huang's DR5)

Modes:
  --tier positives  : reads data/positives_huang2020.parquet (949 L18 rows)
  --tier negatives  : reads data/negatives.parquet (built separately)

Outputs:
  data/cutouts_fits_dr9/<row_id>.fits   (grz float32 cube)
  data/cutouts_jpeg_dr9/<row_id>.jpg    (composite for visual inspection)
  data/cutout_manifest_<tier>.csv
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
FITS_DIR = DATA / "cutouts_fits_dr7_train"
JPEG_DIR = DATA / "cutouts_jpeg_dr7_train"
FITS_DIR.mkdir(parents=True, exist_ok=True)
JPEG_DIR.mkdir(parents=True, exist_ok=True)

LAYER = "decals-dr7"
PIXSCALE = 0.262
SIZE_PIX = 101
WORKERS = 8
TIMEOUT = 60
RETRIES = 5
RETRY_BACKOFF = 4.0
RATELIMIT_BACKOFF = 30.0


def make_url(ra: float, dec: float, fmt: str) -> str:
    base = "https://www.legacysurvey.org/viewer"
    endpoint = "jpeg-cutout" if fmt == "jpeg" else "fits-cutout"
    bands = "" if fmt == "jpeg" else "&bands=grz"
    return (f"{base}/{endpoint}"
            f"?ra={ra:.6f}&dec={dec:.6f}&size={SIZE_PIX}&layer={LAYER}"
            f"&pixscale={PIXSCALE}{bands}")


def fetch_one(row: dict, fmt: str) -> tuple[str, str, int, int]:
    rid = str(row["row_id"])
    out_dir = JPEG_DIR if fmt == "jpeg" else FITS_DIR
    suffix = "jpg" if fmt == "jpeg" else "fits"
    out = out_dir / f"{rid}.{suffix}"
    if out.exists() and out.stat().st_size > 0:
        return (rid, "skip", out.stat().st_size, 0)
    url = make_url(row["RA"], row["DEC"], fmt)
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
            if out.stat().st_size < 256:
                last_err = f"too-small ({out.stat().st_size} B)"
                out.unlink()
                raise RuntimeError(last_err)
            return (rid, "ok", out.stat().st_size, attempt)
        except Exception as e:
            last_err = str(e)
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return (rid, f"fail: {last_err}", 0, RETRIES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["positives", "negatives"], required=True)
    ap.add_argument("--formats", nargs="+", choices=["jpeg", "fits"],
                    default=["jpeg", "fits"])
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_path = DATA / f"{args.tier}_huang2020.parquet" if args.tier == "positives" \
              else DATA / "negatives.parquet"
    if not in_path.exists():
        raise SystemExit(f"missing {in_path}; run 02_filter_catalog.py "
                         "or build negatives parquet first")
    cat = pd.read_parquet(in_path)
    # row_id = stable per-row key (use Name for positives, index for negatives)
    if "row_id" not in cat.columns:
        if "Name" in cat.columns:
            cat["row_id"] = cat["Name"]
        else:
            cat["row_id"] = [f"neg_{i:08d}" for i in range(len(cat))]
    if args.limit:
        cat = cat.head(args.limit)
    print(f"[plan] tier={args.tier}, n={len(cat):,}, formats={args.formats}")

    manifests = {}
    for fmt in args.formats:
        rows = cat.to_dict("records")
        counts = {"ok": 0, "skip": 0, "fail": 0}
        total_bytes = 0
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(fetch_one, r, fmt) for r in rows]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            unit="cutout", desc=fmt):
                rid, status, nbytes, attempts = fut.result()
                total_bytes += nbytes
                if status == "ok":
                    counts["ok"] += 1
                elif status == "skip":
                    counts["skip"] += 1
                else:
                    counts["fail"] += 1
                    tqdm.write(f"  [fail] {rid}: {status}")
                results.append({"row_id": rid, "status": status,
                                "bytes": nbytes, "attempts": attempts})
        print(f"[done] {fmt}: ok={counts['ok']:,}  skip={counts['skip']:,}  "
              f"fail={counts['fail']:,}  bytes={total_bytes/1e6:.1f} MB")
        manifests[fmt] = pd.DataFrame(results)

    mf = cat.copy()
    for fmt, df in manifests.items():
        mf = mf.merge(df.add_suffix(f"_{fmt}").rename(
            columns={f"row_id_{fmt}": "row_id"}), on="row_id", how="left")
    out_path = DATA / f"cutout_manifest_{args.tier}_dr7.csv"
    mf.to_csv(out_path, index=False)
    print(f"[save] wrote {out_path}  ({len(mf):,} rows)")


if __name__ == "__main__":
    main()
