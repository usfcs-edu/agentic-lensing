#!/usr/bin/env python3
"""360_build_dr_parent.py — ClaudeNet v3 sweep deployment: build a DR10/DR11 parent
galaxy catalog from the Legacy Survey sweep catalogs, in the exact schema 160 consumes.

Runs on Perlmutter (CPU; the sweeps are CFS-local). The 160->165 sweep chain is
release-agnostic — 160 only reads [BRICKID, BRICKNAME, OBJID, RA, DEC, NOBS_G/R/Z,
footprint] and builds row_id "<f>_<BRICKID>_<OBJID>". This script OWNS the science
selection (160 only enforces NOBS>=1), matching the published DR10 search
(Inchausti+2025, arXiv:2508.20087) closely enough that recall of their 811 is meaningful:

  * galaxy morphology  TYPE in {SER, EXP, DEV, REX}   (headline; PSF excluded)
  * grz coverage       NOBS_G, NOBS_R, NOBS_Z >= --min-nobs   (published: >=3)
  * brightness         mag_z < --zmax  (mag = 22.5 - 2.5 log10 FLUX_Z; published: z<20)
  * finite RA/DEC, FLUX_{G,R,Z} > 0

It carries (ignored by 160, used downstream): NOBS_I / mag_i (the DR10/DR11-south native
i-band lever), TYPE, SERSIC, SHAPE_R, mag_z, and `is_devser` (TYPE in {DEV,SER} — the
DR9 DEV+COMP lineage proxy, since DR10/11 replaced COMP with SER) for the secondary
v2-lineage comparison. Reads only the columns it needs, one sweep file per worker, so
memory stays small (post-cut ~30k rows/file).

Usage (on Perlmutter, in nersc/shared_cpu.slurm):
    python 360_build_dr_parent.py --release dr10 --footprint south \
        --out $SCRATCH/claudenet/parent_dr10_south.parquet --workers 64
"""
from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SWEEP_VERSION = {"dr10": "10.0", "dr11": "11.0"}
LS_ROOT = "/global/cfs/cdirs/cosmo/data/legacysurvey"
GAL_TYPES = ("SER", "EXP", "DEV", "REX")
NEED = ["BRICKID", "BRICKNAME", "OBJID", "RA", "DEC", "TYPE",
        "FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z",
        "NOBS_G", "NOBS_R", "NOBS_I", "NOBS_Z", "SHAPE_R", "SERSIC", "MASKBITS"]

# module-level config for worker processes (set in main, read in _filter_file)
_CFG: dict = {}


def _mag(flux):
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, 22.5 - 2.5 * np.log10(flux), np.inf)


def _decode(a):
    return np.char.strip(np.asarray(a).astype(str))


def _filter_file(path: str) -> pd.DataFrame:
    import fitsio
    cfg = _CFG
    fits = fitsio.FITS(path)
    cols = fits[1].get_colnames()
    have = [c for c in NEED if c in cols]
    d = fits[1].read(columns=have)
    typ = _decode(d["TYPE"])
    keep = np.isin(typ, GAL_TYPES)
    for b in ("G", "R", "Z"):
        keep &= d[f"NOBS_{b}"] >= cfg["min_nobs"]
        keep &= d[f"FLUX_{b}"] > 0
    magz = _mag(d["FLUX_Z"])
    keep &= magz < cfg["zmax"]
    keep &= np.isfinite(d["RA"]) & np.isfinite(d["DEC"])
    if cfg["maskbits"] and "MASKBITS" in have:
        keep &= (d["MASKBITS"] & cfg["maskbits"]) == 0
    if not keep.any():
        return pd.DataFrame()
    has_i = "FLUX_I" in have
    out = pd.DataFrame({
        "BRICKID": d["BRICKID"][keep].astype(np.int64),
        "BRICKNAME": _decode(d["BRICKNAME"])[keep],
        "OBJID": d["OBJID"][keep].astype(np.int64),
        "RA": d["RA"][keep].astype(np.float64),
        "DEC": d["DEC"][keep].astype(np.float64),
        "NOBS_G": d["NOBS_G"][keep].astype(np.int16),
        "NOBS_R": d["NOBS_R"][keep].astype(np.int16),
        "NOBS_Z": d["NOBS_Z"][keep].astype(np.int16),
        "NOBS_I": (d["NOBS_I"][keep].astype(np.int16) if "NOBS_I" in have
                   else np.zeros(int(keep.sum()), np.int16)),
        "TYPE": typ[keep],
        "mag_z": magz[keep],
        "mag_i": (_mag(d["FLUX_I"])[keep] if has_i else np.full(int(keep.sum()), np.inf)),
        "SERSIC": d["SERSIC"][keep].astype(np.float32) if "SERSIC" in have else np.nan,
        "SHAPE_R": d["SHAPE_R"][keep].astype(np.float32) if "SHAPE_R" in have else np.nan,
        "footprint": cfg["footprint"],
    })
    out["is_devser"] = out["TYPE"].isin(["DEV", "SER"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", choices=["dr10", "dr11"], required=True)
    ap.add_argument("--footprint", default="south", choices=["south", "north"])
    ap.add_argument("--sweep-version", default=None, help="default 10.0/11.0 per release")
    ap.add_argument("--min-nobs", type=int, default=3, help="published DR10 search: >=3")
    ap.add_argument("--zmax", type=float, default=20.0, help="mag_z cut (published: <20)")
    ap.add_argument("--maskbits", type=int, default=0, help="bitmask to EXCLUDE (0=off)")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N sweep files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ver = args.sweep_version or SWEEP_VERSION[args.release]
    sweep_dir = f"{LS_ROOT}/{args.release}/{args.footprint}/sweep/{ver}"
    files = sorted(glob.glob(f"{sweep_dir}/sweep-*.fits"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"[360] no sweep files in {sweep_dir}")
    global _CFG
    _CFG = {"min_nobs": args.min_nobs, "zmax": args.zmax, "maskbits": args.maskbits,
            "footprint": args.footprint}
    print(f"[360] {args.release}/{args.footprint} sweep v{ver}: {len(files)} files, "
          f"cuts: TYPE in {GAL_TYPES}, NOBS_grz>={args.min_nobs}, mag_z<{args.zmax}, "
          f"workers={args.workers}")

    parts, done, total = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_set_cfg, initargs=(_CFG,)) as ex:
        futs = {ex.submit(_filter_file, f): f for f in files}
        for fut in as_completed(futs):
            df = fut.result()
            done += 1
            if len(df):
                parts.append(df); total += len(df)
            if done % 100 == 0 or done == len(files):
                print(f"[360] {done}/{len(files)} files, {total:,} galaxies so far",
                      flush=True)
    parent = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    parent.to_parquet(args.out, index=False)
    print(f"[360] wrote {args.out}: {len(parent):,} galaxies "
          f"({int(parent.is_devser.sum()):,} DEV/SER; "
          f"{int((parent.NOBS_I>0).sum()):,} with native i-band)")
    print(f"[360] TYPE: {parent.TYPE.value_counts().to_dict()}")
    print(f"[360] mag_z: [{parent.mag_z.min():.2f}, {parent.mag_z.max():.2f}], "
          f"median {parent.mag_z.median():.2f}")
    return 0


def _set_cfg(cfg):
    global _CFG
    _CFG = cfg


if __name__ == "__main__":
    raise SystemExit(main())
