#!/usr/bin/env python3
"""
01_build_qso_sample_and_fof.py

Reproduce the Dawes et al. 2023 (ApJS 269:61) "autocorrelation" candidate
search, algorithmically (visual inspection / grading is OUT OF SCOPE, exactly
as the hsu-2025 reproduction handled VI).

ALGORITHM (Dawes+2023 Section 3):
  - Take the DESI Quasar Sample (~5M targets in the published paper).
  - "Autocorrelation": group quasar targets that lie within a separation cut
    of each other; recursively connect overlapping pairs into systems (doubles
    -> triples -> quads).  This is exactly Friends-of-Friends with a fixed
    angular linking length.
  - The paper first searches a 10" radius ("to err on the side of
    completeness"), then halves the cut to <5" because >95% of the
    "discoverable known" systems have images separated by <5".
  - 100% of the 94 "discoverable known" systems are recovered as
    recommendations.

PROXY CAVEAT (stated honestly):
  Dawes used the DESI Legacy Imaging Surveys *photometric* DESI Quasar Sample
  (~5M RF-selected QSO *targets* over 19,000 deg^2).  We do NOT have that target
  catalog on disk.  Instead we build the most directly available proxy: the DESI
  DR1 *spectroscopic* QSO sample from zall-pix-iron.fits
  (SPECTYPE=='QSO', ZWARN==0, ZCAT_PRIMARY==True).  This is a related but NOT
  identical sample: it is smaller (spectroscopically confirmed QSOs only, ~1.5M
  vs ~5M photometric targets) and selected differently.  Absolute candidate
  counts therefore should NOT be expected to match the paper's 436; the
  algorithm (FoF grouping at 5"/10") and the *recovery* of the published
  candidate positions are what we reproduce faithfully.

Mirrors reproductions/hsu-2025/05_run_full_fof.py (proven spherimatch FoF).

Outputs:
  data/qso_groups_5arcsec.parquet   one row per QSO in a kept (size>=2) group, link=5"
  data/qso_groups_10arcsec.parquet  same, link=10"
  data/fof_stats.json               count breakdowns at each stage / both links
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from spherimatch import fof


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# zcatalog lives in the hsu-2025 reproduction (22 GB, shared on disk)
ZCAT = HERE.parent / "hsu-2025" / "data" / "zall-pix-iron.fits"

OUT_5 = DATA / "qso_groups_5arcsec.parquet"
OUT_10 = DATA / "qso_groups_10arcsec.parquet"
OUT_STATS = DATA / "fof_stats.json"

LINK_5_DEG = 5.0 / 3600.0
LINK_10_DEG = 10.0 / 3600.0


def _native(arr, dtype):
    return np.ascontiguousarray(arr).astype(dtype, copy=False)


def load_qso(path: Path) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    print(f"[load] {path}")
    with fits.open(path, memmap=True) as hdul:
        zhdu = next(h for h in hdul if h.name == "ZCATALOG")
        nrows = zhdu.header["NAXIS2"]
        print(f"[load] {nrows:,} rows in ZCATALOG HDU")
        data = zhdu.data
        out = {
            "TARGETID":     _native(data["TARGETID"], np.int64),
            "RA":           _native(data["TARGET_RA"], np.float64),
            "DEC":          _native(data["TARGET_DEC"], np.float64),
            "Z":            _native(data["Z"], np.float64),
            "SPECTYPE":     np.char.strip(np.asarray(data["SPECTYPE"], dtype="U20")),
            "ZWARN":        _native(data["ZWARN"], np.int64),
            "ZCAT_PRIMARY": _native(data["ZCAT_PRIMARY"], np.bool_),
        }
    df = pd.DataFrame(out)
    print(f"[load] {len(df):,} rows loaded in {time.time()-t0:.1f} s")

    stats = {"raw_rows": int(len(df))}
    df = df[df["ZCAT_PRIMARY"]]
    stats["after_zcat_primary"] = int(len(df))
    df = df[df["ZWARN"] == 0]
    stats["after_zwarn_zero"] = int(len(df))
    df = df[df["SPECTYPE"] == "QSO"]
    stats["after_spectype_qso"] = int(len(df))
    df = df.reset_index(drop=True)
    print(f"[qso ] QSO sample: {len(df):,} spectra "
          f"(proxy for Dawes ~5M photometric QSO targets)")
    return df, stats


def run_fof(df: pd.DataFrame, link_deg: float, label: str) -> tuple[pd.DataFrame, dict]:
    radec = df[["RA", "DEC"]].to_numpy(dtype=np.float64)
    t0 = time.time()
    print(f"[fof ] spherimatch.fof N={len(radec):,} link={link_deg*3600:.1f}\"")
    res = fof(radec, link_deg)
    dt = time.time() - t0
    print(f"[fof ] done in {dt:.1f} s")

    group_df = res.get_group_dataframe().reset_index().rename(
        columns={"Group": "group_id", "Object": "obj_idx"}
    )
    sizes = group_df.groupby("group_id").size()
    big_ids = sizes[sizes >= 2].index
    big = group_df[group_df["group_id"].isin(big_ids)].copy()
    print(f"[fof ] {len(big_ids):,} groups size>=2 covering {len(big):,} spectra")

    df_idx = df.reset_index().rename(columns={"index": "obj_idx"})
    big = big.merge(df_idx, on="obj_idx", suffixes=("", "_df"))

    # multiplicity breakdown (size -> #groups)
    gsizes = big.groupby("group_id").size()
    mult = gsizes.value_counts().sort_index().to_dict()
    mult = {int(k): int(v) for k, v in mult.items()}
    n_doubles = int((gsizes == 2).sum())
    n_triples = int((gsizes == 3).sum())
    n_quads_plus = int((gsizes >= 4).sum())  # >=4 images -> "quad" by Dawes naming
    n_quads_exact = int((gsizes == 4).sum())

    stats = {
        "link_arcsec": link_deg * 3600.0,
        "fof_groups_total": int(group_df["group_id"].nunique()),
        "groups_size>=2": int(len(big_ids)),
        "spectra_in_groups": int(len(big)),
        "multiplicity_counts": mult,
        "n_doubles(size2)": n_doubles,
        "n_triples(size3)": n_triples,
        "n_quads(size>=4)": n_quads_plus,
        "n_quads(size==4)": n_quads_exact,
        "double_to_quadplus_ratio": (n_doubles / n_quads_plus) if n_quads_plus else None,
        "fof_wallclock_s": float(dt),
    }
    print(f"[fof ] {label}: doubles={n_doubles} triples={n_triples} "
          f"quads(>=4)={n_quads_plus} ratio(2:>=4)="
          f"{stats['double_to_quadplus_ratio']}")
    return big, stats


def save_groups(big: pd.DataFrame, out: Path) -> None:
    cols = ["group_id", "TARGETID", "RA", "DEC", "Z", "SPECTYPE"]
    tbl = pa.Table.from_pandas(big[cols].copy(), preserve_index=False)
    pq.write_table(tbl, out)
    print(f"[save] wrote {out} ({len(big):,} rows)")


def main() -> None:
    if not ZCAT.exists():
        raise SystemExit(f"missing {ZCAT}")
    df, pre_stats = load_qso(ZCAT)
    print(f"[pre ] {pre_stats}")

    big5, s5 = run_fof(df, LINK_5_DEG, "5arcsec")
    save_groups(big5, OUT_5)

    big10, s10 = run_fof(df, LINK_10_DEG, "10arcsec")
    save_groups(big10, OUT_10)

    pub = {
        "desi_qso_targets_paper": "~5,000,000 (photometric RF-selected; NOT used here)",
        "candidates_5arcsec_paper": 436,
        "recommendations_10arcsec_paper": ">27000 quasar targets",
        "recommendations_5arcsec_paper": "~6000 quasar targets",
        "double_to_quad_ratio_paper": "~21:1 (text: 24 quads identified by algorithm)",
        "discoverable_known_paper": 94,
    }
    report = {
        "proxy_note": (
            "DR1 spectroscopic QSO sample (SPECTYPE==QSO, ZWARN==0, "
            "ZCAT_PRIMARY) used as a proxy for Dawes' ~5M photometric DESI "
            "Quasar Sample targets. Absolute counts differ; algorithm + "
            "recovery are the reproduction targets."
        ),
        "prefilter": pre_stats,
        "fof_5arcsec": s5,
        "fof_10arcsec": s10,
        "published": pub,
    }
    OUT_STATS.write_text(json.dumps(report, indent=2))
    print(f"[save] wrote {OUT_STATS}")

    print("\n[verify]  metric                       PUBLISHED            OURS(proxy)")
    print(f"  QSO sample size            {'~5M targets':>20}    {pre_stats['after_spectype_qso']:>12,}")
    print(f"  candidate groups (5\")       {'436':>20}    {s5['groups_size>=2']:>12,}")
    print(f"  recommendations (10\")       {'>27000 (spectra)':>20}    {s10['spectra_in_groups']:>12,}")
    print(f"  recommendations (5\")        {'~6000 (spectra)':>20}    {s5['spectra_in_groups']:>12,}")
    print(f"  double:quad(>=4) ratio (5\") {'~21:1':>20}    "
          f"{s5['double_to_quadplus_ratio']}")


if __name__ == "__main__":
    main()
