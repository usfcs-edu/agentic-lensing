#!/usr/bin/env python3
"""
02_download_fastspecfit.py

Download DESI DR1 FastSpecFit VAC (v3.0) into data/fastspecfit/.

FastSpecFit is partitioned by survey × program × HEALPix nside1 tile. There is no
single merged file; we pull what we need by tier:

  tier=small  : fastspec-iron-sv3-dark.fits  (~2.7 GB)  for script 04 validation
  tier=full   : all 36 files                  (~79 GB)  for script 07 full classifier

Default tier=small. Switch with --tier=full when full-DR1 classification is needed.

After download, this script identifies the velocity-dispersion column (one of
VDISP, VDISP_FIXED, SIGMA_V, SIGMA_STAR) and writes a TARGETID -> sigma_v parquet
shard per file so script 07 can join cheaply.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy.io import fits


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fastspecfit"
DATA.mkdir(parents=True, exist_ok=True)

BASE = "https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs"

SMALL_FILES = {
    "fastspec-iron-sv3-dark.fits": 2_667_421_440,
}

ALL_FILES = {
    "fastspec-iron-cmx-other.fits": 12_545_280,
    "fastspec-iron-main-backup.fits": 66_525_120,
    "fastspec-iron-main-bright-nside1-hp00.fits": 447_563_520,
    "fastspec-iron-main-bright-nside1-hp01.fits": 4_468_049_280,
    "fastspec-iron-main-bright-nside1-hp02.fits": 5_968_638_720,
    "fastspec-iron-main-bright-nside1-hp03.fits": 1_773_118_080,
    "fastspec-iron-main-bright-nside1-hp04.fits": 4_312_356_480,
    "fastspec-iron-main-bright-nside1-hp05.fits": 1_322_732_160,
    "fastspec-iron-main-bright-nside1-hp06.fits": 5_919_598_080,
    "fastspec-iron-main-bright-nside1-hp07.fits": 2_959_801_920,
    "fastspec-iron-main-bright-nside1-hp08.fits": 359_242_560,
    "fastspec-iron-main-bright-nside1-hp09.fits": 293_886_720,
    "fastspec-iron-main-bright-nside1-hp10.fits": 200_378_880,
    "fastspec-iron-main-bright-nside1-hp11.fits": 293_852_160,
    "fastspec-iron-main-dark-nside1-hp00.fits": 1_550_954_880,
    "fastspec-iron-main-dark-nside1-hp01.fits": 4_922_671_680,
    "fastspec-iron-main-dark-nside1-hp02.fits": 7_476_324_480,
    "fastspec-iron-main-dark-nside1-hp03.fits": 944_686_080,
    "fastspec-iron-main-dark-nside1-hp04.fits": 7_653_182_400,
    "fastspec-iron-main-dark-nside1-hp05.fits": 2_547_901_440,
    "fastspec-iron-main-dark-nside1-hp06.fits": 12_548_456_640,
    "fastspec-iron-main-dark-nside1-hp07.fits": 4_541_656_320,
    "fastspec-iron-main-dark-nside1-hp08.fits": 953_916_480,
    "fastspec-iron-main-dark-nside1-hp09.fits": 688_584_960,
    "fastspec-iron-main-dark-nside1-hp10.fits": 327_824_640,
    "fastspec-iron-main-dark-nside1-hp11.fits": 178_871_040,
    "fastspec-iron-special-backup.fits": 2_609_280,
    "fastspec-iron-special-bright.fits": 189_846_720,
    "fastspec-iron-special-dark.fits": 65_655_360,
    "fastspec-iron-special-other.fits": 186_238_080,
    "fastspec-iron-sv1-backup.fits": 15_240_960,
    "fastspec-iron-sv1-bright.fits": 567_838_080,
    "fastspec-iron-sv1-dark.fits": 1_045_402_560,
    "fastspec-iron-sv1-other.fits": 152_979_840,
    "fastspec-iron-sv2-backup.fits": 662_400,
    "fastspec-iron-sv2-bright.fits": 208_823_040,
    "fastspec-iron-sv2-dark.fits": 236_347_200,
    "fastspec-iron-sv3-backup.fits": 7_030_080,
    "fastspec-iron-sv3-bright.fits": 1_195_600_320,
    "fastspec-iron-sv3-dark.fits": 2_667_421_440,
}

# Candidate velocity-dispersion column names — resolved at runtime from header.
VDISP_CANDIDATES = ("VDISP", "VDISP_FIXED", "SIGMA_V", "SIGMA_STAR", "VDISP_MEDIAN")
VDISP_ERR_CANDIDATES = ("VDISP_IVAR", "VDISP_ERR", "SIGMA_V_ERR", "SIGMA_STAR_ERR")

CHUNK = 16 * 1024 * 1024


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TiB"


def download_resumable(url: str, dest: Path, expected_size: int) -> None:
    pos = dest.stat().st_size if dest.exists() else 0
    if pos == expected_size:
        print(f"[skip] {dest.name} already complete ({fmt_bytes(pos)})")
        return
    if pos > expected_size:
        print(f"[warn] {dest.name} larger than expected; truncating")
        dest.unlink()
        pos = 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    print(f"[get ] {dest.name} ({fmt_bytes(expected_size)})")
    t0 = time.time()
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if pos and r.status_code == 200:
            dest.unlink(missing_ok=True)
            pos = 0
        r.raise_for_status()
        mode = "ab" if pos else "wb"
        with open(dest, mode) as f:
            last = time.time()
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                pos += len(chunk)
                now = time.time()
                if now - last >= 5.0:
                    pct = 100.0 * pos / expected_size
                    sp = pos / (now - t0) if now > t0 else 0
                    eta = (expected_size - pos) / sp if sp > 0 else 0
                    print(
                        f"[prog] {dest.name}: {fmt_bytes(pos)}/{fmt_bytes(expected_size)} "
                        f"({pct:5.1f}%) @ {fmt_bytes(int(sp))}/s ETA {eta/60:.1f} min",
                        flush=True,
                    )
                    last = now
    final = dest.stat().st_size
    if final != expected_size:
        print(f"[fail] {dest.name}: {final} != {expected_size}")
        sys.exit(1)
    print(f"[done] {dest.name}: {fmt_bytes(final)} in {(time.time()-t0)/60:.1f} min")


def resolve_vdisp_columns(cols: list[str]) -> tuple[str | None, str | None]:
    vd = next((c for c in VDISP_CANDIDATES if c in cols), None)
    ve = next((c for c in VDISP_ERR_CANDIDATES if c in cols), None)
    return vd, ve


def _find_hdu_with_column(hdul, want: str):
    """Return the first BinTable HDU containing `want`."""
    for h in hdul:
        if isinstance(h, (fits.BinTableHDU, fits.TableHDU)) and want in (
            c.name for c in h.columns
        ):
            return h
    return None


def build_sigma_v_parquet(fits_path: Path) -> Path | None:
    parquet = fits_path.with_suffix(".sigmav.parquet")
    if parquet.exists():
        print(f"[skip] {parquet.name} exists")
        return parquet
    with fits.open(fits_path, memmap=True) as hdul:
        # FastSpecFit splits across METADATA (HDU 1) and SPECPHOT (HDU 2). VDISP
        # lives in SPECPHOT; TARGETID, Z, ZWARN, SPECTYPE in METADATA. We pull
        # from whichever HDU has each column.
        h_meta = _find_hdu_with_column(hdul, "TARGETID")
        h_vdisp = None
        vdisp_col = None
        for cand in VDISP_CANDIDATES:
            h_vdisp = _find_hdu_with_column(hdul, cand)
            if h_vdisp is not None:
                vdisp_col = cand
                break
        if h_meta is None or h_vdisp is None:
            cols = [c.name for h in hdul if isinstance(h, fits.BinTableHDU) for c in h.columns]
            print(
                f"[warn] {fits_path.name}: missing required HDU "
                f"(meta_hdu={h_meta is not None}, vdisp_hdu={h_vdisp is not None})"
            )
            print(f"       sample columns: {cols[:40]}")
            return None
        meta_data = h_meta.data
        sp_data = h_vdisp.data
        if len(meta_data) != len(sp_data):
            print(
                f"[warn] {fits_path.name}: row count mismatch "
                f"({len(meta_data)} vs {len(sp_data)}); aborting"
            )
            return None
        meta_cols = {c.name for c in h_meta.columns}
        sp_cols = {c.name for c in h_vdisp.columns}
        err_col = next((c for c in VDISP_ERR_CANDIDATES if c in sp_cols), None)
        logmstar_col = "LOGMSTAR" if "LOGMSTAR" in sp_cols else None
        # FITS columns are big-endian on disk; convert to native (little-endian)
        # before going to pandas/pyarrow.
        def _native(arr, dtype):
            return np.ascontiguousarray(arr).astype(dtype, copy=False)

        out: dict[str, np.ndarray] = {
            "TARGETID": _native(meta_data["TARGETID"], np.int64),
            "VDISP": _native(sp_data[vdisp_col], np.float32),
        }
        if err_col is not None:
            # VDISP_IVAR is the inverse variance; downstream code computes the
            # standard deviation when it needs an error.
            out["VDISP_IVAR"] = _native(sp_data[err_col], np.float32)
        if "Z" in meta_cols:
            out["Z_FASTSPEC"] = _native(meta_data["Z"], np.float32)
        if logmstar_col:
            out["LOGMSTAR"] = _native(sp_data[logmstar_col], np.float32)
    df = pd.DataFrame(out)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet)
    print(
        f"[parq] {parquet.name}: {len(df):,} rows; "
        f"vdisp_col={vdisp_col!r}, err_col={err_col!r}, logmstar={logmstar_col!r}"
    )
    return parquet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["small", "full"], default="small")
    args = ap.parse_args()
    files = SMALL_FILES if args.tier == "small" else ALL_FILES
    print(f"[main] tier={args.tier}  ({len(files)} files, "
          f"{fmt_bytes(sum(files.values()))})")
    for fname, sz in files.items():
        url = f"{BASE}/{fname}"
        dest = DATA / fname
        download_resumable(url, dest, sz)
        build_sigma_v_parquet(dest)


if __name__ == "__main__":
    main()
