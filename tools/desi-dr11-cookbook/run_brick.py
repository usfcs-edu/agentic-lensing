#!/usr/bin/env python
"""
run_brick.py -- process a single DR11 brick into One_percent-style HDF5 (smoke test / debugging).

Example:
  python run_brick.py --region north --brick 2400p345 \
    --data /global/cfs/cdirs/cosmo/data/legacysurvey/dr11/north \
    --out  $SCRATCH/dr11_test/north
"""
import argparse
import time

import dr11_collect as dc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brick", required=True, help="brick name, e.g. 2400p345")
    ap.add_argument("--data", required=True, help="DR11 region root (…/dr11/<region>)")
    ap.add_argument("--out", required=True, help="output region root (…/<region>)")
    ap.add_argument("--region", default=None, help="region label for logging (north/south)")
    args = ap.parse_args()

    t0 = time.time()
    n = dc.process_brick(args.brick, args.data, args.out)
    dt = time.time() - t0
    rrr = dc.brick_rrr(args.brick)
    if n == 0:
        print(f"[{args.region or '?'}] {args.brick}: 0 surviving objects -> no files written ({dt:.1f}s)")
    else:
        print(f"[{args.region or '?'}] {args.brick}: {n} objects -> "
              f"{args.out}/tractor/{rrr}/{args.brick}.h5 + coadd/{rrr}/{args.brick}.h5 ({dt:.1f}s)")


if __name__ == "__main__":
    main()
