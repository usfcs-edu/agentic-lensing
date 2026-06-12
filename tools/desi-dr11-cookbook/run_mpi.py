#!/usr/bin/env python
"""
run_mpi.py -- MPI fan-out driver: derive One_percent-style HDF5 for whole RA strips.

One output file per brick => no collective/parallel HDF5 needed; MPI is pure task
fan-out (rank r processes bricks[r::size]).  Rank 0 also writes the two log files.

Launch (see submit_strip.sh):
  srun -n 256 python run_mpi.py --region north --batchids 0,120,240,359 \
    --data /global/cfs/cdirs/cosmo/data/legacysurvey/dr11/north \
    --out  $SCRATCH/dr11_onepercent/north
"""
import argparse
import os
import time
import traceback

from mpi4py import MPI

import dr11_collect as dc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=["north", "south"])
    ap.add_argument("--batchids", required=True,
                    help="comma-separated RA strips 0..359, e.g. 0,120,240,359")
    ap.add_argument("--data", required=True, help="DR11 region root (…/dr11/<region>)")
    ap.add_argument("--out", required=True, help="output region root (…/<region>)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    batchids = [int(x) for x in args.batchids.split(",") if x.strip() != ""]

    # rank 0 enumerates the full brick list, broadcasts it
    bricks = None
    if rank == 0:
        bricks = []
        for bid in batchids:
            bricks.extend(dc.bricks_for_batch(args.data, bid))
        os.makedirs(os.path.join(args.out, "log"), exist_ok=True)
    bricks = comm.bcast(bricks, root=0)

    t0 = time.time()
    my = bricks[rank::size]
    n_written = n_objects = n_empty = n_error = 0
    errors = []
    for brick in my:
        try:
            n = dc.process_brick(brick, args.data, args.out)
            if n == 0:
                n_empty += 1
            else:
                n_written += 1
                n_objects += n
        except Exception as e:                                   # noqa: BLE001
            n_error += 1
            errors.append(f"{brick}: {e!r}\n{traceback.format_exc()}")

    stats = comm.gather((n_written, n_objects, n_empty, n_error, len(my)), root=0)
    all_errors = comm.gather(errors, root=0)
    dt = time.time() - t0

    if rank == 0:
        tw = sum(s[0] for s in stats); to = sum(s[1] for s in stats)
        te = sum(s[2] for s in stats); terr = sum(s[3] for s in stats)
        flat_err = [line for sub in all_errors for line in sub]
        summary = (f"region={args.region} batchids={batchids} ranks={size}\n"
                   f"bricks_total={len(bricks)} written={tw} empty={te} errors={terr} "
                   f"objects={to} wall={dt:.1f}s\n")
        print(summary, end="")

        logdir = os.path.join(args.out, "log")
        with open(os.path.join(logdir, "logging.log"), "w") as f:
            f.write(summary)
            if flat_err:
                f.write("\n=== ERRORS ===\n")
                f.write("\n".join(flat_err))
        # config.log: copy the spec file next to this script if present
        spec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.xml")
        if os.path.exists(spec):
            with open(spec) as src, open(os.path.join(logdir, "config.log"), "w") as dst:
                dst.write(src.read())
        if terr:
            print(f"WARNING: {terr} brick(s) errored; see {logdir}/logging.log")


if __name__ == "__main__":
    main()
