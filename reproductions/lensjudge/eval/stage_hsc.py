#!/usr/bin/env python3
"""eval/stage_hsc.py — pre-fetch (stage) HSC-SSP PDR3 cutouts for OFFLINE tier-2 grading.

The tier-2 HSC flow is naturally DECOUPLED because the two steps need different things:
  1. FETCH needs internet + HSC-SSP credentials (HSC_USER/HSC_PASSWORD) — the das_cutout service.
  2. GRADE needs the GPU (the open-weight VLM), and can run fully offline.

Perlmutter COMPUTE nodes have no internet, so run this staging step on an INTERNET host — gpu3, or
a Perlmutter LOGIN node — to warm the HSC cutout cache, then rsync the cache dir to the GPU host and
grade offline. A warm cache serves credential-free (see common/hsc_fetch.fetch_hsc_cutout / cached),
so the offline grade host needs neither creds nor internet.

  # on an internet host (creds set):
  HSC_USER=... HSC_PASSWORD=... \
    python -m lensjudge.eval.stage_hsc --manifest cands.csv --cache $SCRATCH/ljv5/hsc_cache
  # then: rsync -a $SCRATCH/ljv5/hsc_cache/ <grade-host>:$SCRATCH/ljv5/hsc_cache/
  # then grade offline: LENSJUDGE_HSC_CACHE=$SCRATCH/ljv5/hsc_cache LENSJUDGE_BACKEND=openai ...

The manifest is a CSV with `ra`,`dec` columns (degrees); an optional `name` column is echoed in the
report. Rows already cached are skipped (idempotent), so this is safe to re-run/resume.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

# LENSJUDGE_HSC_CACHE must be set BEFORE importing hsc_fetch (HSC_CACHE is read at import time).


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with ra,dec[,name] columns (degrees)")
    ap.add_argument("--cache", default=None,
                    help="cache dir to populate (default: $LENSJUDGE_HSC_CACHE or repo cache/hsc)")
    ap.add_argument("--limit", type=int, default=0, help="stage only the first N rows")
    args = ap.parse_args()

    if args.cache:
        os.environ["LENSJUDGE_HSC_CACHE"] = str(Path(args.cache).expanduser())
    from lensjudge.common import hsc_fetch  # noqa: E402  (import after cache env is set)
    if args.cache:  # belt-and-suspenders: also override the module const (HSC_CACHE is read at import)
        hsc_fetch.HSC_CACHE = Path(args.cache).expanduser()

    if not hsc_fetch.have_credentials():
        raise SystemExit("ABORT: staging needs HSC_USER + HSC_PASSWORD (env-only; never committed).")

    df = pd.read_csv(args.manifest)
    cols = {c.lower(): c for c in df.columns}
    if "ra" not in cols or "dec" not in cols:
        raise SystemExit(f"manifest needs ra,dec columns; got {list(df.columns)}")
    if args.limit:
        df = df.head(args.limit)

    print(f"[stage-hsc] cache={hsc_fetch.HSC_CACHE}  rows={len(df)}")
    covered = missing = skipped = 0
    for i, r in df.reset_index(drop=True).iterrows():
        ra, dec = float(r[cols["ra"]]), float(r[cols["dec"]])
        nm = str(r[cols["name"]]) if "name" in cols else f"{ra:.5f}{dec:+.5f}"
        if hsc_fetch.cached(ra, dec):
            skipped += 1
            continue
        bands = hsc_fetch.fetch_hsc_cutout(ra, dec)
        if bands:
            covered += 1
            print(f"  [{i + 1}/{len(df)}] {nm}: OK bands={''.join(sorted(bands))}")
        else:
            missing += 1
            print(f"  [{i + 1}/{len(df)}] {nm}: no coverage")
    print(f"\n[stage-hsc] fetched={covered}  no-coverage={missing}  already-cached={skipped}")
    print(f"[stage-hsc] rsync this dir to the offline grade host, then set "
          f"LENSJUDGE_HSC_CACHE={hsc_fetch.HSC_CACHE}")


if __name__ == "__main__":
    main()
