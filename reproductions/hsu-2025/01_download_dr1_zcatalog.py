#!/usr/bin/env python3
"""
01_download_dr1_zcatalog.py

Download DESI DR1 redshift catalogs into data/.

Two files:
  - zpix-sv3-dark.fits  (~901 MB)  : small-region validation set (script 04)
  - zall-pix-iron.fits  (~22.4 GB) : full-DR1 all-sky catalog (script 05)

Uses HTTP Range to resume partial downloads on failure.

After download, prints HDU info + column list for each file so script 04/05 can
hard-code the column names we know are present.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
from astropy.io import fits


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1"
FILES = {
    "zpix-sv3-dark.fits": 901_103_040,    # small validation
    "zall-pix-iron.fits": 22_371_272_640, # full DR1
}

CHUNK = 16 * 1024 * 1024  # 16 MiB streaming chunk


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TiB"


def download_resumable(url: str, dest: Path, expected_size: int) -> None:
    """Stream-download `url` to `dest`. Resume from existing partial file via HTTP Range."""
    pos = dest.stat().st_size if dest.exists() else 0
    if pos == expected_size:
        print(f"[skip] {dest.name} already complete ({fmt_bytes(pos)})")
        return
    if pos > expected_size:
        print(f"[warn] {dest.name} larger than expected ({pos} > {expected_size}); truncating")
        dest.unlink()
        pos = 0

    headers = {"Range": f"bytes={pos}-"} if pos else {}
    print(f"[get ] {url}")
    print(f"[get ] -> {dest}  (resume_from={fmt_bytes(pos)}, target={fmt_bytes(expected_size)})")

    t0 = time.time()
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if pos and r.status_code == 200:
            print(f"[warn] server ignored Range (status 200) — restarting from 0")
            dest.unlink(missing_ok=True)
            pos = 0
        elif pos and r.status_code != 206:
            r.raise_for_status()
        else:
            r.raise_for_status()

        mode = "ab" if pos else "wb"
        with open(dest, mode) as f:
            last_report = time.time()
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                pos += len(chunk)
                now = time.time()
                if now - last_report >= 5.0:
                    pct = 100.0 * pos / expected_size
                    speed = pos / (now - t0) if now > t0 else 0.0
                    eta = (expected_size - pos) / speed if speed > 0 else 0.0
                    print(
                        f"[prog] {dest.name}: {fmt_bytes(pos)} / "
                        f"{fmt_bytes(expected_size)} ({pct:5.1f}%) "
                        f"@ {fmt_bytes(int(speed))}/s  ETA {eta/60:.1f} min",
                        flush=True,
                    )
                    last_report = now

    final = dest.stat().st_size
    if final != expected_size:
        print(f"[fail] {dest.name}: final size {final} != expected {expected_size}")
        sys.exit(1)
    print(f"[done] {dest.name}: {fmt_bytes(final)} in {(time.time()-t0)/60:.1f} min")


def inspect_fits(path: Path) -> None:
    print(f"\n[fits] {path.name}")
    with fits.open(path, memmap=True) as hdul:
        hdul.info()
        for i, hdu in enumerate(hdul):
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                cols = [c.name for c in hdu.columns]
                print(f"  HDU {i} ({hdu.name}) columns ({len(cols)}):")
                # Print in 6-column rows for readability
                for j in range(0, len(cols), 6):
                    print("    " + ", ".join(cols[j:j+6]))
                break


def main() -> None:
    only = os.environ.get("HSU_ONLY", "").split(",") if os.environ.get("HSU_ONLY") else None
    for fname, expected in FILES.items():
        if only and fname not in only:
            print(f"[skip] {fname} (HSU_ONLY filter)")
            continue
        url = f"{BASE}/{fname}"
        dest = DATA / fname
        download_resumable(url, dest, expected)
    print()
    for fname in FILES:
        if only and fname not in only:
            continue
        dest = DATA / fname
        if dest.exists():
            inspect_fits(dest)


if __name__ == "__main__":
    main()
