#!/usr/bin/env python3
"""
Pull a DESI healpix subset into a tree that matches the NERSC iron layout that
redshifty/nersc/build_dr1_index.py expects.

Output layout:

    <output-root>/spectro/redux/<production>/healpix/<survey>/<program>/<hp_group>/<pixel>/
        coadd-<survey>-<program>-<pixel>.fits
        redrock-<survey>-<program>-<pixel>.fits

where <production> is `fuji` for EDR or `iron` for DR1.

A companion ``zall-pix-iron.fits`` zcatalog is NOT written here — use
build_mini_zcatalog.py for that (codecs only).

Resumable: if both files for a pixel already exist + non-empty, the pixel is
skipped. Add ``--force`` to overwrite.

Usage:
    # EDR sv3 bright (default)
    download_desi_subset.py --n-files 50 -o /raid/benson/data/desi_dr1_medium
    # EDR sv3 dark
    download_desi_subset.py --program dark --n-files 200 -o /raid/benson/data/desi_dr1_medium
    # DR1 main bright (much larger pool than EDR; no sv1/sv2/sv3 there)
    download_desi_subset.py --release dr1 --survey main --program bright -o ...
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm


# Per-release URL prefix + the on-disk "production" directory name.
RELEASES = {
    "edr": {
        "url": "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/{survey}/{program}",
        "production": "fuji",
    },
    "dr1": {
        "url": "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/{survey}/{program}",
        "production": "iron",
    },
}


def list_available_pixels(base_url: str) -> list[tuple[str, str]]:
    """Return list of (hp_group, pixel) for every pixel under <survey>/<program>."""
    r = requests.get(base_url + "/", timeout=60)
    if r.status_code != 200:
        return []
    groups = re.findall(r'href="(\d+)/"', r.text)
    pixels: list[tuple[str, str]] = []
    for g in groups:
        r2 = requests.get(f"{base_url}/{g}/", timeout=60)
        if r2.status_code != 200:
            continue
        subs = re.findall(r'href="(\d+)/"', r2.text)
        for s in subs:
            pixels.append((g, s))
    return pixels


def download_pixel(base_url: str, survey: str, program: str,
                   hp_group: str, pixel: str, dest_dir: Path,
                   force: bool = False, skip_big: int | None = None) -> str:
    """Return one of: 'skipped', 'downloaded', 'too_big', 'error'."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    coadd_fn = f"coadd-{survey}-{program}-{pixel}.fits"
    redrock_fn = f"redrock-{survey}-{program}-{pixel}.fits"
    coadd_dest = dest_dir / coadd_fn
    redrock_dest = dest_dir / redrock_fn

    coadd_ok = coadd_dest.exists() and coadd_dest.stat().st_size > 0
    redrock_ok = redrock_dest.exists() and redrock_dest.stat().st_size > 0
    if not force and coadd_ok and redrock_ok:
        return "skipped"

    coadd_url = f"{base_url}/{hp_group}/{pixel}/{coadd_fn}"
    redrock_url = f"{base_url}/{hp_group}/{pixel}/{redrock_fn}"

    if skip_big is not None:
        try:
            head = requests.head(coadd_url, timeout=30, allow_redirects=True)
            sz = int(head.headers.get("content-length", 0))
            if sz > skip_big:
                return "too_big"
        except requests.RequestException:
            pass  # keep going; HEAD failing isn't fatal

    try:
        for url, dest in [(coadd_url, coadd_dest), (redrock_url, redrock_dest)]:
            if not force and dest.exists() and dest.stat().st_size > 0:
                continue
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.rename(dest)
        return "downloaded"
    except requests.RequestException as e:
        print(f"  ERROR {coadd_fn}: {e}", file=sys.stderr)
        # cleanup half-files
        for p in [coadd_dest.with_suffix(".fits.part"), redrock_dest.with_suffix(".fits.part")]:
            if p.exists():
                p.unlink()
        return "error"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", "-o", type=Path, required=True,
                    help="Root for the iron-style tree (e.g. /raid/benson/data/desi_dr1_medium)")
    ap.add_argument("--release", default="edr", choices=list(RELEASES),
                    help="Public release tier. EDR = fuji production (sv1/sv2/sv3 only). "
                         "DR1 = iron production (adds main survey, much larger pool).")
    ap.add_argument("--survey", default="sv3", choices=["sv1", "sv2", "sv3", "main"])
    ap.add_argument("--program", default="bright", choices=["bright", "dark", "backup", "other"])
    ap.add_argument("--n-files", type=int, default=50,
                    help="Cap on pixels to download (after --pixels list, if any). 0 = no cap.")
    ap.add_argument("--pixels", nargs="+", type=str, default=None,
                    help="Specific pixel numbers (skip discovery). Hp_group inferred from pixel/100.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files exist")
    ap.add_argument("--skip-big-bytes", type=int, default=200 * (1 << 20),
                    help="Skip pixels whose coadd is larger than this many bytes (0 to disable). "
                         "Default 200 MiB — keeps small smoke-medium runs fast.")
    args = ap.parse_args()

    release = RELEASES[args.release]
    if args.release == "edr" and args.survey == "main":
        ap.error("EDR (fuji) does not include the 'main' survey; use --release dr1 for that")
    if args.release == "dr1" and args.survey in ("sv1", "sv2"):
        # DR1 still ships sv1/sv2 data, just less common. allow but warn.
        print(f"WARN: {args.survey} in DR1 has fewer pixels than sv3/main; double-check coverage", file=sys.stderr)

    base_url = release["url"].format(survey=args.survey, program=args.program)
    dest_base = args.output_root / "spectro" / "redux" / release["production"] / "healpix" / args.survey / args.program

    if args.pixels:
        items = [(str(int(p) // 100), p) for p in args.pixels]
    else:
        print(f"Discovering pixels at {base_url}/ ...", file=sys.stderr)
        items = list_available_pixels(base_url)
        if args.n_files and args.n_files > 0:
            items = items[: args.n_files]

    print(f"Targeting {len(items)} pixels under {args.survey}/{args.program} -> {dest_base}", file=sys.stderr)

    counts = {"downloaded": 0, "skipped": 0, "too_big": 0, "error": 0}
    skip_big = args.skip_big_bytes if args.skip_big_bytes > 0 else None
    t0 = time.time()

    for grp, pix in tqdm(items, desc="pixels"):
        dest_dir = dest_base / grp / pix
        st = download_pixel(base_url, args.survey, args.program, grp, pix,
                            dest_dir, force=args.force, skip_big=skip_big)
        counts[st] += 1

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s.")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    # disk usage report
    try:
        total = sum(p.stat().st_size for p in dest_base.rglob("*.fits") if p.is_file())
        print(f"  tree size: {total / 2**30:.2f} GiB at {dest_base}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
