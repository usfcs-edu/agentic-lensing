#!/usr/bin/env python3
"""385_assign_dr11_bricks.py — assign DR11-south bricks to the positive manifest (384 output),
producing the 111 input manifest. Runs ON PERLMUTTER (CFS bricks table + astropy).

Interval-matches each (RA, DEC) to the DR11-south survey-bricks table; membership in that
south-only table IS the footprint check — positions with no containing DR11-south brick are
out of coverage and dropped (the true-footprint enforcement). Carries weight/tier/source.

  module load python
  python 385_assign_dr11_bricks.py \
     --manifest dr11s_positive_manifest.parquet --out dr11s_positive_manifest_brick.parquet
Then 111_extract_cutouts_cfs.py --manifest <out> --release dr11 --bands grz --size 101 ...
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.spatial import cKDTree

BRICKS = "/global/cfs/cdirs/cosmo/data/legacysurvey/dr11/south/survey-bricks-dr11-south.fits.gz"


def xyz(ra, dec):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bricks", default=BRICKS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    man = pd.read_parquet(a.manifest)
    bt = Table.read(a.bricks)
    cols = {c.upper(): c for c in bt.colnames}
    bn = np.array([s.decode() if isinstance(s, bytes) else str(s)
                   for s in bt[cols["BRICKNAME"]]])
    RA1, RA2 = np.asarray(bt[cols["RA1"]]), np.asarray(bt[cols["RA2"]])
    DEC1, DEC2 = np.asarray(bt[cols["DEC1"]]), np.asarray(bt[cols["DEC2"]])
    bRA, bDEC = np.asarray(bt[cols["RA"]]), np.asarray(bt[cols["DEC"]])
    print(f"[385] {len(man):,} positives; {len(bn):,} DR11-south bricks")

    tree = cKDTree(xyz(bRA, bDEC))
    _, idx = tree.query(xyz(man.RA, man.DEC), k=1)
    ra, dec = man.RA.to_numpy(float), man.DEC.to_numpy(float)
    inside = (ra >= RA1[idx]) & (ra < RA2[idx]) & (dec >= DEC1[idx]) & (dec < DEC2[idx])
    man = man.assign(brick=bn[idx], footprint="south")
    n_drop = int((~inside).sum())
    man = man[inside].reset_index(drop=True)
    keep = ["row_id", "RA", "DEC", "footprint", "brick", "label", "weight", "tier", "source"]
    man[[c for c in keep if c in man.columns]].to_parquet(a.out, index=False)
    print(f"[385] in DR11-south coverage: {len(man):,}  (dropped out-of-footprint: {n_drop:,})")
    if "tier" in man.columns:
        print(f"[385] by tier (in-footprint): {man.tier.value_counts().to_dict()}")
    print(f"[385] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
