#!/usr/bin/env python3
"""Stage grz cutouts for the parity bench (fills common.fetch's cache/cubes/).

  python lensjudge/parity/stage_parity_cutouts.py --arm arm1 [--workers 6]
  python lensjudge/parity/stage_parity_cutouts.py --arm arm2 --workers 8

Idempotent: fetch.get_cube caches by name/coord key, on-disk FITS resolve first.
Prints a coverage summary; failures (off-footprint etc.) are listed, not fatal.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import fetch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def _one(row) -> tuple[str, bool]:
    cube = fetch.get_cube(name=str(row["name"]), ra=row["ra"], dec=row["dec"],
                          survey=str(row["survey_key"]))
    ok = cube is not None and cube.shape == (3, config.SIZE_PIX, config.SIZE_PIX)
    return str(row["name"]), ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["arm1", "arm2"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--truth-only", action="store_true",
                    help="arm1: stage only the truth_followup rows (skip augmentation)")
    args = ap.parse_args()

    man = pd.read_csv(OUT / f"parity_bench_{args.arm}.csv")
    if args.truth_only and args.arm == "arm1":
        man = man[man.source == "truth_followup"]
    man = man[man.ra.notna()]
    print(f"staging {len(man)} cutouts for {args.arm} ({args.workers} workers)")

    fails = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, r) for _, r in man.iterrows()]
        for f in as_completed(futs):
            name, ok = f.result()
            done += 1
            if not ok:
                fails.append(name)
            if done % 200 == 0:
                print(f"  {done}/{len(man)} ({len(fails)} failures)")
    print(f"done: {done - len(fails)}/{len(man)} staged; {len(fails)} failures")
    if fails:
        fp = OUT / f"parity_stage_failures_{args.arm}.txt"
        fp.write_text("\n".join(fails) + "\n")
        print(f"failures listed in {fp}")


if __name__ == "__main__":
    main()
