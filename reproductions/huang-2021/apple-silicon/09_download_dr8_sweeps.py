#!/usr/bin/env python3
"""
09_download_dr8_sweeps.py — Phase 4b step 1.

Download the DECaLS+BASS+MzLS DR8 sweep catalog files from the NERSC portal.
Huang+2021 deploys on the full DR8 footprint (~14,000 deg²), which DR8 splits
into two regions served from separate directories:

  south  DECaLS (δ ≲ +32°, DECam)     dr8/south/sweep/8.0/   (~437 files)
  north  BASS+MzLS (δ ≳ +32°)         dr8/north/sweep/8.0/   (~286 files)

Files land in data/dr8_sweep/{south,north}/ so 10_select_parent_sample_dr8.py
can tag each surviving row with its footprint (needed to route brick-image
downloads to the correct coadd directory in 11b).

Total ~1.3-1.5 TB. Resumable: skips files present with the correct
Content-Length. Clone of huang-2020/09_download_dr7_sweeps.py generalised to
two footprints.

Usage:
  ./09_download_dr8_sweeps.py --dry-run                 # list, no download
  ./09_download_dr8_sweeps.py --footprint south --workers 4
  ./09_download_dr8_sweeps.py --workers 4               # both footprints
"""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dr8_sweep"

BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr8"
INDEX_URLS = {
    "south": f"{BASE}/south/sweep/8.0/",
    "north": f"{BASE}/north/sweep/8.0/",
}
SWEEP_NAME_RE = re.compile(r"sweep-[0-9pm]+-[0-9pm]+\.fits")
WORKERS = 4
TIMEOUT = 120
RETRIES = 5
RETRY_BACKOFF = 8.0


def list_sweeps(index_url: str) -> list[str]:
    r = requests.get(index_url, timeout=TIMEOUT)
    r.raise_for_status()
    return sorted(set(SWEEP_NAME_RE.findall(r.text)))


def expected_size(url: str) -> int:
    r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers.get("content-length", "0"))


def fetch_one(index_url: str, out_dir: Path, name: str,
              force: bool) -> tuple[str, str, int]:
    url = index_url + name
    out = out_dir / name
    if out.exists() and not force:
        try:
            expected = expected_size(url)
        except Exception:
            expected = -1
        actual = out.stat().st_size
        if expected > 0 and actual == expected:
            return (name, "skip", actual)
        if expected <= 0 and actual > 0:
            return (name, "skip-no-headcheck", actual)
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            tmp = out.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.rename(out)
            return (name, "ok", out.stat().st_size)
        except Exception as e:
            last_err = str(e)
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return (name, f"fail: {last_err}", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--footprint", choices=("south", "north", "both"), default="both")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="for testing")
    args = ap.parse_args()

    footprints = ["south", "north"] if args.footprint == "both" else [args.footprint]

    # Build the full (footprint, name) work list.
    work: list[tuple[str, Path, str]] = []
    for fp in footprints:
        index_url = INDEX_URLS[fp]
        out_dir = DATA / fp
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] listing {fp} sweeps from {index_url}")
        names = list_sweeps(index_url)
        print(f"[init] {fp}: {len(names)} sweep files in index")
        if args.limit:
            names = names[: args.limit]
        for n in names:
            work.append((index_url, out_dir, n))

    if args.dry_run:
        for index_url, _, n in work[:6]:
            print("  ", index_url + n)
        if len(work) > 6:
            print(f"  ... and {len(work) - 6} more  (total {len(work)} files)")
        return

    n_ok = n_skip = n_fail = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, iu, od, n, args.force): n for iu, od, n in work}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="dr8 sweeps"):
            name, status, size = fut.result()
            total_bytes += size
            if status == "ok":
                n_ok += 1
            elif status.startswith("skip"):
                n_skip += 1
            else:
                n_fail += 1
                tqdm.write(f"FAIL {name}: {status}")

    print(f"[done] ok={n_ok} skip={n_skip} fail={n_fail} "
          f"total={total_bytes / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
