#!/usr/bin/env python3
"""130b_assign_bricks.py — Phase 130: assign DR9 bricks + footprint +
source_release to the griz manifest (runs ON PERLMUTTER, CPU-only, login-node
OK; needs astropy + the CFS legacysurvey tree).

Takes 130's data/v2/griz_manifest_nobrick.parquet (row_id, RA, DEC, label,
source) and adds, per row:

  brick      — exact interval match RA1<=RA<RA2 & DEC1<=DEC<DEC2 against the
               DR9 survey-bricks table (--bricks-fits, default the CFS copy).
               If the table is unavailable the validated grid-math fallback is
               used WITH A LOUD WARNING: the DR9/DESI brick grid is 721 dec
               rows centered at -90+0.25*k (row index by HALF-UP rounding,
               floor(x+0.5) — the survey-bricks convention at exact half-grid
               boundaries like DEC=0.125, where banker's np.round mismatches);
               ncol(row) = 2*ceil(720*cos(declo))
               with declo = max(|dec_center|-0.125, 0) the equator-side dec
               edge; brickname = '<RRRR><p|m><DDD>' = truncated center*10.
               This math reproduces the catalog BRICKNAME for ALL 18.36M
               local ground-truth rows (parent_dr8 17.29M + negatives_extra
               65k + negeval_manifest 1M: zero mismatches). When the table IS
               available, the grid math is still computed as a cross-check and
               any disagreement is reported.
  footprint  — ground truth by coadd-dir existence: 'north' iff the brick dir
               exists under <cfs-root>/dr9/north/coadd/<bbb>/<brick> AND the
               brick center DEC >= 32.375 (the DR9 north/south split
               convention); else 'south'. If the CFS tree is absent (local
               smoke test) falls back, WITH A LOUD WARNING, to the documented
               rule: north iff DEC >= 32.375 and galactic b > 0 (NGC).
  source_release — 'dr10' if footprint==south (DR10 south carries native i),
               else 'dr9' (north never has i; 111 zero-fills + i_ok=False).

Output: TWO parquets split by source_release, the direct --manifest inputs of
the two 111 extraction runs documented in 130's docstring:

    data/v2/griz_manifest_south.parquet   (footprint=south, source_release=dr10)
    data/v2/griz_manifest_north.parquet   (footprint=north, source_release=dr9)

Also warns (counts only, rows kept) when an assigned south brick has no dir
under dr10/south/coadd — those rows will come back ok=False from 111.

    python 130b_assign_bricks.py                       # on Perlmutter (CFS defaults)
    python 130b_assign_bricks.py --bricks-fits /none --cfs-root /none   # fallback smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

CFS_ROOT = "/global/cfs/cdirs/cosmo/data/legacysurvey"     # as 111_extract_cutouts_cfs
BRICKS_FITS = CFS_ROOT + "/dr9/survey-bricks.fits.gz"
NORTH_DEC_MIN = 32.375                                     # DR9 north/south convention


def brickname_grid(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """DR9/DESI brick grid math (see module docstring; validated on 18.36M
    rows with zero mismatches). Vectorized RA/DEC [deg] -> brickname."""
    # dec-row index: HALF-UP rounding (floor(x+0.5)), matching the survey-bricks
    # table convention at exact half-grid boundaries — banker's np.round
    # mis-assigns e.g. DEC=0.125 (rounds 360.5 to 360, not 361)
    row = np.clip(np.floor((dec + 90.0) / 0.25 + 0.5).astype(int), 0, 720)
    dec_c = -90.0 + 0.25 * row
    declo = np.maximum(np.abs(dec_c) - 0.125, 0.0)         # equator-side dec edge
    ncol = (2 * np.ceil(720.0 * np.cos(np.radians(declo)))).astype(int)
    col = np.floor((ra % 360.0) * ncol / 360.0).astype(int)
    cen = (col + 0.5) * 360.0 / ncol
    ra4 = (np.trunc(cen * 10 + 1e-9).astype(int) % 3600).astype(str)
    d10 = np.trunc(np.abs(dec_c) * 10 + 1e-9).astype(int).astype(str)
    sign = np.where(dec_c < 0, "m", "p")
    return np.char.add(np.char.add(np.char.zfill(ra4, 4), sign), np.char.zfill(d10, 3))


def assign_from_table(man: pd.DataFrame, bricks_fits: Path, chunk: int) -> np.ndarray:
    """Exact RA1<=RA<RA2 & DEC1<=DEC<DEC2 match against survey-bricks (ground
    truth). Returns brickname per manifest row ('' where unmatched)."""
    from astropy.table import Table
    t = Table.read(str(bricks_fits))
    cols = {c.upper(): c for c in t.colnames}
    need = ["BRICKNAME", "RA1", "RA2", "DEC1", "DEC2"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"{bricks_fits}: missing columns {missing} (have {t.colnames})")
    name = np.char.strip(np.asarray(t[cols["BRICKNAME"]]).astype(str))
    ra1, ra2 = (np.asarray(t[cols[c]], float) for c in ("RA1", "RA2"))
    de1, de2 = (np.asarray(t[cols[c]], float) for c in ("DEC1", "DEC2"))
    print(f"[130b] survey-bricks: {len(name):,} bricks from {bricks_fits}")

    # group bricks into dec rows (unique DEC1 edges), each sorted by RA1
    row_edges, row_inv = np.unique(de1, return_inverse=True)
    rows = {}
    for j in range(len(row_edges)):
        sel = np.where(row_inv == j)[0]
        order = np.argsort(ra1[sel])
        sel = sel[order]
        rows[j] = (ra1[sel], ra2[sel], de2[sel], name[sel])

    out = np.full(len(man), "", dtype=object)
    RA, DEC = man.RA.to_numpy(float), man.DEC.to_numpy(float)
    for s in range(0, len(man), chunk):
        ra, dec = RA[s:s + chunk] % 360.0, DEC[s:s + chunk]
        j = np.searchsorted(row_edges, dec, side="right") - 1
        for jj in np.unique(j):
            m = np.where(j == jj)[0]
            if jj < 0:
                continue
            r1, r2, d2, nm = rows[jj]
            ok_dec = dec[m] < d2[0]                       # one DEC2 per row
            k = np.searchsorted(r1, ra[m], side="right") - 1
            valid = ok_dec & (k >= 0) & (ra[m] < r2[np.clip(k, 0, len(r2) - 1)])
            out[s + m[valid]] = nm[k[valid]]
    return out.astype(str)


def footprint_truth(bricks: pd.DataFrame, cfs_root: str) -> pd.Series:
    """bricks: index=brickname, col dec_c. North iff dr9/north coadd dir exists
    AND center DEC >= 32.375; else south. Falls back (loud warning) to the
    DEC>=32.375 & galactic-b>0 rule when the CFS tree is absent."""
    root = Path(cfs_root)
    if (root / "dr9").is_dir():
        def is_north(b, dec_c):
            return (dec_c >= NORTH_DEC_MIN and
                    (root / "dr9" / "north" / "coadd" / b[:3] / b).is_dir())
        fp = pd.Series({b: ("north" if is_north(b, d) else "south")
                        for b, d in bricks.dec_c.items()})
        # report south bricks missing from dr10/south (rows kept; 111 -> ok=False)
        miss = [b for b in fp.index[fp == "south"]
                if not (root / "dr10" / "south" / "coadd" / b[:3] / b).is_dir()]
        if miss:
            print(f"[130b] *** WARNING: {len(miss)} south bricks have NO dir under "
                  f"dr10/south/coadd (111 will mark their rows ok=False): "
                  f"{miss[:10]}{'...' if len(miss) > 10 else ''}")
        return fp
    print(f"[130b] *** WARNING: {cfs_root}/dr9 not found — footprint falls back to "
          f"the DEC>={NORTH_DEC_MIN} & galactic-b>0 rule (run on Perlmutter for "
          f"the coadd-dir ground truth) ***")
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    sky = SkyCoord(ra=bricks.ra_c.to_numpy() * u.deg, dec=bricks.dec_c.to_numpy() * u.deg)
    north = (bricks.dec_c.to_numpy() >= NORTH_DEC_MIN) & (sky.galactic.b.deg > 0)
    return pd.Series(np.where(north, "north", "south"), index=bricks.index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(C.DATA / "v2" / "griz_manifest_nobrick.parquet"))
    ap.add_argument("--bricks-fits", default=BRICKS_FITS,
                    help="DR9 survey-bricks table (ground truth); missing -> "
                         "validated grid-math fallback with a warning")
    ap.add_argument("--cfs-root", default=CFS_ROOT,
                    help="legacysurvey tree root for the footprint dir checks")
    ap.add_argument("--out-dir", default=str(C.DATA / "v2"))
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()
    t0 = time.time()

    man = pd.read_parquet(args.manifest)
    assert man.row_id.is_unique
    print(f"[130b] {len(man):,} manifest rows from {args.manifest}")

    # 1. brick assignment ------------------------------------------------------
    grid = brickname_grid(man.RA.to_numpy(float), man.DEC.to_numpy(float))
    bf = Path(args.bricks_fits)
    if bf.exists():
        brick = assign_from_table(man, bf, args.chunk)
        unmatched = int((brick == "").sum())
        assert unmatched == 0, f"{unmatched} rows match no survey-bricks interval"
        agree = float((brick == grid).mean())
        print(f"[130b] table-vs-grid-math agreement: {agree:.6f}"
              + ("" if agree == 1.0 else "  *** INVESTIGATE DISAGREEMENT ***"))
    else:
        print(f"[130b] *** WARNING: {bf} not found — using the grid-math fallback "
              f"(validated: 0 mismatches on 18.36M ground-truth rows) ***")
        brick = grid
    man["brick"] = brick

    # 2. footprint (dir-existence ground truth) + source_release ---------------
    ub = man.groupby("brick").agg(ra_c=("RA", "mean"), dec_c=("DEC", "mean"))
    # brick CENTER dec for the convention check (exact from the name encoding)
    sgn = np.where(ub.index.str[4] == "p", 1.0, -1.0)
    ub["dec_c"] = sgn * ub.index.str[5:].astype(int) / 10.0
    fp = footprint_truth(ub, args.cfs_root)
    man["footprint"] = man.brick.map(fp)
    man["source_release"] = np.where(man.footprint == "south", "dr10", "dr9")

    # 3. write the two 111 inputs ----------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["row_id", "RA", "DEC", "footprint", "brick", "source_release",
            "label", "source"]
    for foot in ("south", "north"):
        sub = man[man.footprint == foot][cols].reset_index(drop=True)
        path = out_dir / f"griz_manifest_{foot}.parquet"
        sub.to_parquet(path, index=False)
        per_src = ", ".join(f"{k}: {v:,}" for k, v in sub.source.value_counts().items())
        print(f"[130b] wrote {path} — {len(sub):,} rows, {sub.brick.nunique():,} "
              f"bricks, release={'dr10' if foot == 'south' else 'dr9'} ({per_src})")
    print(f"[130b] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
