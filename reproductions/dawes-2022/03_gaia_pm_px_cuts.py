#!/usr/bin/env python3
"""
03_gaia_pm_px_cuts.py

Dawes+2023 Section 4.1: use Gaia EDR3 proper motion (PM) and parallax (PX) to
reject Milky Way stellar contaminants from the candidate quasar groups.

Dawes' significance definitions (Section 4.1):
  PMSIG = sqrt(PM_RA^2 + PM_DEC^2) / sqrt(sigma_PM_RA^2 + sigma_PM_DEC^2)
  PXSIG = parallax / sigma_parallax
Acceptance cut: PMSIG < 8 AND PXSIG < 3.5.
A candidate GROUP is rejected as stellar if ANY matched image has
PMSIG >= 8 OR PXSIG >= 3.5 (significant motion/parallax => Milky Way star).

We query Gaia EDR3 (gaiaedr3.gaia_source) via per-position synchronous cone
searches (astroquery.gaia, robust; the bulk TAP upload-join hung in testing)
for every QSO image in the RESOLVED candidate groups from 05
(data/resolved_candidates.parquet; ~3.2k images in ~1.5k groups). Checkpointed
to data/gaia_matches.parquet so it is resumable.

Dawes: 380/436 (~87%) candidates had Gaia info for >=1 image. Faint lensed
images mostly fall below Gaia's ~21 mag limit, so a large fraction of our
QSO images will have NO Gaia match -- that absence is itself the quasar
signature.

Outputs:
  data/gaia_matches.parquet   one row per QSO image with a Gaia match
  data/gaia_cuts.json         per-group pass/fail summary + counts
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CAND = DATA / "resolved_candidates.parquet"   # from 05
OUT_MATCH = DATA / "gaia_matches.parquet"
OUT_CUTS = DATA / "gaia_cuts.json"

MATCH_RADIUS_ARCSEC = 2.0
PMSIG_CUT = 8.0
PXSIG_CUT = 3.5
GAIA_COLS = ["source_id", "ra", "dec", "pmra", "pmra_error", "pmdec",
             "pmdec_error", "parallax", "parallax_error", "phot_g_mean_mag"]


def query_one(Gaia, ra: float, dec: float) -> pd.DataFrame:
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    for attempt in range(3):
        try:
            j = Gaia.cone_search_async(
                c, radius=MATCH_RADIUS_ARCSEC * u.arcsec, columns=GAIA_COLS)
            return j.get_results().to_pandas()
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  [warn] query failed at ({ra:.4f},{dec:.4f}): {e!r}")
                return pd.DataFrame(columns=GAIA_COLS)
            time.sleep(2)
    return pd.DataFrame(columns=GAIA_COLS)


def main() -> None:
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"
    Gaia.ROW_LIMIT = 50

    if not CAND.exists():
        raise SystemExit(f"missing {CAND}; run 05 first")
    g = pq.read_table(CAND).to_pandas().reset_index(drop=True)
    g["img_id"] = np.arange(len(g))
    print(f"[load] {len(g):,} QSO images in "
          f"{g['group_id'].nunique():,} resolved candidate groups", flush=True)

    # resume from checkpoint
    done_ids: set[int] = set()
    prev: list[pd.DataFrame] = []
    if OUT_MATCH.exists():
        old = pq.read_table(OUT_MATCH).to_pandas()
        done_ids = set(old["img_id"].unique())
        prev = [old]
        print(f"[resume] {len(done_ids):,} images already queried", flush=True)

    rows = []
    t0 = time.time()
    for _, r in g.iterrows():
        iid = int(r["img_id"])
        if iid in done_ids:
            continue
        res = query_one(Gaia, float(r["RA"]), float(r["DEC"]))
        if len(res):
            res = res.copy()
            res["img_id"] = iid
            res["group_id"] = int(r["group_id"])
            res["q_ra"] = float(r["RA"])
            res["q_dec"] = float(r["DEC"])
            rows.append(res)
        if iid % 200 == 0:
            elapsed = time.time() - t0
            print(f"[gaia] {iid}/{len(g)} images "
                  f"({elapsed:.0f}s, {len(rows)} matches so far)", flush=True)

    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    gm = pd.concat(prev + ([new] if len(new) else []), ignore_index=True)
    if not len(gm):
        raise SystemExit("no Gaia matches at all")

    # significances
    pm = np.hypot(gm["pmra"].to_numpy(dtype=float), gm["pmdec"].to_numpy(dtype=float))
    den = np.hypot(gm["pmra_error"].to_numpy(dtype=float),
                   gm["pmdec_error"].to_numpy(dtype=float))
    gm["PMSIG"] = pm / den
    gm["PXSIG"] = (gm["parallax"].to_numpy(dtype=float)
                   / gm["parallax_error"].to_numpy(dtype=float))
    gm["stellar_flag"] = ((gm["PMSIG"] >= PMSIG_CUT)
                          | (gm["PXSIG"] >= PXSIG_CUT)).fillna(False)

    pq.write_table(pa.Table.from_pandas(gm, preserve_index=False), OUT_MATCH)
    print(f"[save] wrote {OUT_MATCH} ({len(gm):,} Gaia matches)", flush=True)

    n_groups = int(g["group_id"].nunique())
    groups_with_gaia = int(gm["group_id"].nunique())
    stellar_groups = set(gm.loc[gm["stellar_flag"], "group_id"].unique())
    surviving = set(g["group_id"].unique()) - stellar_groups

    print(f"[gaia] groups with Gaia info for >=1 image: "
          f"{groups_with_gaia}/{n_groups} "
          f"({100*groups_with_gaia/n_groups:.0f}%) [Dawes ~87%]")
    print(f"[cut ] groups flagged stellar (PMSIG>=8 or PXSIG>=3.5): "
          f"{len(stellar_groups)}")
    print(f"[cut ] surviving candidate groups: {len(surviving)}/{n_groups}")

    report = {
        "match_radius_arcsec": MATCH_RADIUS_ARCSEC,
        "pmsig_cut": PMSIG_CUT, "pxsig_cut": PXSIG_CUT,
        "n_candidate_groups": n_groups,
        "n_qso_images": int(len(g)),
        "n_groups_with_gaia": groups_with_gaia,
        "frac_groups_with_gaia": groups_with_gaia / n_groups,
        "n_groups_flagged_stellar": len(stellar_groups),
        "n_groups_surviving": len(surviving),
        "published_frac_with_gaia": "380/436 ~ 0.87",
    }
    OUT_CUTS.write_text(json.dumps(report, indent=2))
    print(f"[save] wrote {OUT_CUTS}", flush=True)


if __name__ == "__main__":
    main()
