#!/usr/bin/env python3
"""
06_xmatch_published_catalog.py

Cross-match our DR1 pair list against the published Hsu+2025 Grade A candidates.

The full Hsu+2025 candidate catalog (Table 4 schema, 2046+318 rows) is announced
in the paper Appendix A as "available on our project website and on Zenodo"
but as of the run date (2026-05-26) it has not appeared on:
  - sites.google.com/usfca.edu/neuralens/ (paper page lists arXiv only)
  - Zenodo (no 2509.16033 deposit)
The paper itself was submitted to ApJS in Sep 2025 and is still in review.

Until the machine-readable catalog is released we cross-match against the 20
Grade A candidates explicitly tabulated in the paper's Table 2 (the 20 "new"
Grade A systems). All 20 have published (RA, Dec) encoded in the
"DESI-{ra}{±dec}" naming convention.

For each Hsu Table 2 candidate we find the spatially-nearest group in our
data/dr1_pairs.parquet using a great-circle distance, and report:
  - mean angular offset (should be << 3″ since these candidates are in DR1 fiber
    pairs that pass spherimatch FoF with 3″ link)
  - recall = fraction of 20 with nearest-group offset < 3″
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PAIRS = DATA / "dr1_pairs.parquet"
OUT_REPORT = DATA / "xmatch_table2.json"

# Table 2 of Hsu+2025 — the 20 Grade A new candidates explicitly tabulated.
# Decoded from the "DESI-{RA}{±DEC}" naming convention; both coordinates in
# decimal degrees. Verified against the paper PDF (extracted via pypdf).
HSU_TABLE2 = [
    "DESI-004.5374+01.0382",
    "DESI-037.4253-03.3533",
    "DESI-043.3037+03.0729",
    "DESI-063.4488-18.1813",
    "DESI-066.3585-16.5116",
    "DESI-119.2449+42.4903",
    "DESI-132.2396-01.3110",
    "DESI-152.1067-03.2930",
    "DESI-162.5471+02.3538",
    "DESI-215.8346+35.7262",
    "DESI-227.5058+37.0509",
    "DESI-227.9553+36.6557",
    "DESI-228.8385+02.3565",
    "DESI-231.8344+01.6947",
    "DESI-249.7123+04.6103",
    "DESI-251.9827+32.5477",
    "DESI-259.9375+14.3606",
    "DESI-264.7547+30.3171",
    "DESI-285.9895+64.4143",
    "DESI-334.6393+00.6638",
]

NAME_RE = re.compile(r"^DESI-(\d{3}\.\d{4})([+-]\d{2}\.\d{4})$")


def parse_name(name: str) -> tuple[float, float]:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"bad name: {name!r}")
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    if not PAIRS.exists():
        raise SystemExit(f"missing {PAIRS}; run 05_run_full_fof.py first")
    pairs = pq.read_table(PAIRS).to_pandas()
    print(f"[load] {PAIRS}: {len(pairs):,} rows in "
          f"{pairs['group_id'].nunique():,} groups")

    # Group centroids — Hsu's "DESI-RA-Dec" name is the rounded centroid of the
    # fiber pair, so we compare to mean(RA), mean(Dec) per group.
    centroids = (
        pairs.groupby("group_id")[["RA", "DEC"]].mean().reset_index()
    )
    print(f"[ctrd] {len(centroids):,} group centroids")

    hsu_radec = np.array([parse_name(n) for n in HSU_TABLE2])
    hsu_sc = SkyCoord(ra=hsu_radec[:, 0] * u.deg, dec=hsu_radec[:, 1] * u.deg)
    our_sc = SkyCoord(
        ra=centroids["RA"].to_numpy() * u.deg,
        dec=centroids["DEC"].to_numpy() * u.deg,
    )

    # For each Hsu candidate: nearest group + offset
    idx, sep, _ = hsu_sc.match_to_catalog_sky(our_sc)
    rows = []
    for name, i_grp, off in zip(HSU_TABLE2, idx, sep):
        grp_id = int(centroids["group_id"].iloc[int(i_grp)])
        members = pairs[pairs["group_id"] == grp_id]
        rows.append({
            "name": name,
            "hsu_ra": float(parse_name(name)[0]),
            "hsu_dec": float(parse_name(name)[1]),
            "nearest_group_id": grp_id,
            "nearest_group_size": int(len(members)),
            "offset_arcsec": float(off.to_value(u.arcsec)),
            "member_targetids": [int(t) for t in members["TARGETID"].tolist()],
            "member_z": [float(z) for z in members["Z"].tolist()],
        })

    matches = pd.DataFrame(rows).sort_values("offset_arcsec")
    print()
    print(f"{'name':<26} {'offset″':>9} {'group':>9} {'size':>4}  z's")
    print("-" * 100)
    for _, r in matches.iterrows():
        z_str = ", ".join(f"{z:.3f}" for z in r["member_z"])
        print(f"{r['name']:<26} {r['offset_arcsec']:>9.3f} "
              f"{r['nearest_group_id']:>9d} {r['nearest_group_size']:>4d}  {z_str}")

    n_close = int((matches["offset_arcsec"] < 3.0).sum())
    n_total = len(matches)
    recall = n_close / n_total
    print()
    print(f"[recall over Hsu Table 2] {n_close}/{n_total} within 3″ "
          f"= {100.0 * recall:.1f}%")
    print(f"[recall over Hsu Table 2] {int((matches['offset_arcsec'] < 1.5).sum())}/{n_total} within 1.5″")
    print(f"[recall over Hsu Table 2] median offset = "
          f"{matches['offset_arcsec'].median():.3f}″")

    report = {
        "n_hsu_table2": n_total,
        "n_matched_within_3arcsec": n_close,
        "n_matched_within_1p5arcsec": int((matches["offset_arcsec"] < 1.5).sum()),
        "recall_3arcsec": recall,
        "median_offset_arcsec": float(matches["offset_arcsec"].median()),
        "max_offset_arcsec": float(matches["offset_arcsec"].max()),
        "matches": rows,
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[save] wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
