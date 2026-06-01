#!/usr/bin/env python3
"""
09_download_dr7_sweeps.py — Phase 3b M1 step 1.

Download all DECaLS DR7 sweep catalog files from the NERSC portal.
292 files, ~3 GB each, ~875 GB total. Used by 10_select_parent_sample.py
to build the ~5.7M-row parent sample for the Phase 3b ResNet sweep.

Resumable: skips files that exist with the correct Content-Length.

Usage:
  ./09_download_dr7_sweeps.py [--workers 4] [--dry-run]
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
DATA = HERE / "data" / "dr7_sweep"
DATA.mkdir(parents=True, exist_ok=True)

INDEX_URL = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr7/sweep/7.1/"
SWEEP_NAME_RE = re.compile(r"sweep-[0-9pm]+-[0-9pm]+\.fits")
WORKERS = 4
TIMEOUT = 120
RETRIES = 5
RETRY_BACKOFF = 8.0


def list_sweeps() -> list[str]:
    r = requests.get(INDEX_URL, timeout=TIMEOUT)
    r.raise_for_status()
    names = sorted(set(SWEEP_NAME_RE.findall(r.text)))
    return names


def expected_size(name: str) -> int:
    r = requests.head(INDEX_URL + name, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers.get("content-length", "0"))


def fetch_one(name: str, force: bool) -> tuple[str, str, int]:
    out = DATA / name
    expected = -1
    if out.exists() and not force:
        try:
            expected = expected_size(name)
        except Exception:
            expected = -1
        actual = out.stat().st_size
        if expected > 0 and actual == expected:
            return (name, "skip", actual)
        if expected <= 0 and actual > 0:
            return (name, "skip-no-headcheck", actual)
        # else: size mismatch -> re-download
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(INDEX_URL + name, timeout=TIMEOUT, stream=True)
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
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true", help="redownload even if file exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="for testing")
    args = ap.parse_args()

    print(f"[init] listing sweeps from {INDEX_URL}")
    names = list_sweeps()
    print(f"[init] {len(names)} sweep files in index")
    if args.limit:
        names = names[: args.limit]
        print(f"[init] limited to first {len(names)}")
    if args.dry_run:
        for n in names[:8]:
            print(" ", n)
        if len(names) > 8:
            print(f"  ... and {len(names) - 8} more")
        return

    n_ok = n_skip = n_fail = total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, n, args.force): n for n in names}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="sweeps"):
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
