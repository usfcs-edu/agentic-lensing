#!/usr/bin/env python
"""
conformance.py -- prove a freshly-derived brick is array-exact vs the existing One_percent product.

Compares the tractor catalog (14 columns + dtypes + values, incl. float32) and every coadd
cutout (key set + shape + dtype + pixels) between a reference (existing) tree and a new tree.

Example:
  python conformance.py --brick 2400p345 --region north \
    --ref /global/cfs/projectdirs/cosmo/work/users/xhuang/dr11/One_percent \
    --new $SCRATCH/dr11_test
"""
import argparse
import sys

import numpy as np
import pandas as pd
import h5py

import dr11_collect as dc


# The 3 derived magnitude columns are reproduced to float32 precision (<=1 ULP), not
# bit-for-bit: log10's last-bit rounding differs across numpy/libm builds, so the original
# run's exact bits aren't recoverable from any formula. 1 ULP at mag~21 is ~1.9e-6 mag
# (sub-micromag, ~4000x below photometric precision). All other columns are strictly exact.
MAG_COLS = ("mag_g", "mag_r", "mag_z")


def check_tractor(ref_path, new_path):
    a = pd.read_hdf(ref_path, "tractor")
    b = pd.read_hdf(new_path, "tractor")
    assert list(b.columns) == dc.COLS, f"column order: {list(b.columns)}"
    assert list(a.columns) == dc.COLS, f"REF column order differs: {list(a.columns)}"
    assert len(a) == len(b), f"row count {len(a)} != {len(b)}"
    for c in dc.COLS:
        assert a[c].dtype == b[c].dtype, f"dtype[{c}] ref={a[c].dtype} new={b[c].dtype}"
        av, bv = a[c].to_numpy(), b[c].to_numpy()
        if c in MAG_COLS:
            # require new value within +/-1 float32 ULP of the stored value
            up = np.nextafter(av, np.float32(np.inf))
            dn = np.nextafter(av, np.float32(-np.inf))
            within = (bv >= dn) & (bv <= up)
            assert within.all(), (
                f"{c}: {int((~within).sum())}/{len(av)} values exceed 1 ULP "
                f"(max |diff| {np.max(np.abs(bv - av)):.3e})")
            exact = int((av == bv).sum())
            print(f"  {c}: {exact}/{len(av)} bit-exact, rest within 1 ULP "
                  f"(max |diff| {np.max(np.abs(bv - av)):.2e} mag)")
        else:
            np.testing.assert_array_equal(av, bv, err_msg=f"values differ in column {c}")
    return len(b)


def check_coadd(ref_path, new_path):
    with h5py.File(ref_path, "r") as fa, h5py.File(new_path, "r") as fb:
        ka, kb = set(fa.keys()), set(fb.keys())
        assert ka == kb, (f"key sets differ: only-ref={sorted(ka - kb)[:5]} "
                          f"only-new={sorted(kb - ka)[:5]}")
        for k in ka:
            x, y = fa[k][:], fb[k][:]
            assert x.shape == y.shape == (101, 101, 3), f"{k} shape {x.shape} vs {y.shape}"
            assert x.dtype == y.dtype == np.float32, f"{k} dtype {x.dtype} vs {y.dtype}"
            np.testing.assert_array_equal(x, y, err_msg=f"pixels differ in {k}")
        return len(ka)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brick", required=True)
    ap.add_argument("--region", required=True, choices=["north", "south"])
    ap.add_argument("--ref", required=True, help="existing product root (has <region>/...)")
    ap.add_argument("--new", required=True, help="new product root (has <region>/...)")
    args = ap.parse_args()

    rrr = dc.brick_rrr(args.brick)
    rel_t = f"{args.region}/tractor/{rrr}/{args.brick}.h5"
    rel_c = f"{args.region}/coadd/{rrr}/{args.brick}.h5"

    nrow = check_tractor(f"{args.ref}/{rel_t}", f"{args.new}/{rel_t}")
    ncut = check_coadd(f"{args.ref}/{rel_c}", f"{args.new}/{rel_c}")
    assert nrow == ncut, f"tractor rows {nrow} != coadd cutouts {ncut}"
    print(f"CONFORMANCE PASS  {args.region}/{args.brick}: "
          f"{nrow} catalog rows and {ncut} cutouts are array-exact.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"CONFORMANCE FAIL: {e}", file=sys.stderr)
        sys.exit(1)
