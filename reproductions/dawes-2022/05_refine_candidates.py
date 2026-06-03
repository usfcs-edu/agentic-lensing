#!/usr/bin/env python3
"""
05_refine_candidates.py

Refine the raw 5" FoF groups into Dawes-like candidate systems and re-measure
recovery + the double:quad ratio.

WHY: the raw FoF groups (01) are dominated by pairs at ~0" separation -- distinct
TARGETIDs sitting at the *same* sky position. These are the same physical QSO
observed under multiple DESI surveys/programs (SV1/SV3/main, repeat tiling) or
Tractor-split duplicates, NOT resolved close quasar pairs. The published Dawes
catalog has image separations of 0.3-5" (median ~1.9"). Dawes searched a
*photometric* target catalog where each image is a distinct detected source, so
they did not see this spectroscopic-duplicate pile-up.

We therefore define a "resolved candidate system" as a FoF group whose maximum
internal image separation is in [SEP_MIN, 5"]. SEP_MIN=0.3" matches the minimum
published Dawes separation (0.32"). We collapse near-zero-separation duplicates
by keeping, per group, the set of distinct sky positions (dedup within 0.2").

Outputs:
  data/resolved_candidates.parquet  members of resolved candidate groups
  data/refined_stats.json           counts + recovery + double:quad ratio
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
import astropy.units as u


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GROUPS = DATA / "qso_groups_5arcsec.parquet"
PUB = DATA / "dawes2023_vizier_table2.csv"
OUT_CAND = DATA / "resolved_candidates.parquet"
OUT_STATS = DATA / "refined_stats.json"

SEP_MIN = 0.3      # arcsec; min published Dawes image separation is 0.32"
SEP_MAX = 5.0      # arcsec; Dawes <5" cut
DEDUP = 0.2        # arcsec; positions closer than this are the same image
MATCH_TOL = 3.0    # arcsec; recovery match tolerance


def group_geometry(sub: pd.DataFrame) -> tuple[float, int]:
    """Return (max image separation in arcsec, number of distinct images)."""
    sc = SkyCoord(ra=sub["RA"].to_numpy() * u.deg,
                  dec=sub["DEC"].to_numpy() * u.deg)
    # dedup near-identical positions -> distinct images
    keep = []
    for k in range(len(sc)):
        if all(sc[k].separation(sc[m]).to_value(u.arcsec) > DEDUP for m in keep):
            keep.append(k)
    scd = sc[keep]
    n_img = len(scd)
    if n_img < 2:
        return 0.0, n_img
    i, j = np.triu_indices(n_img, 1)
    maxsep = scd[i].separation(scd[j]).to_value(u.arcsec).max()
    return float(maxsep), n_img


def main() -> None:
    g = pq.read_table(GROUPS).to_pandas()
    print(f"[load] {len(g):,} images in {g['group_id'].nunique():,} raw 5\" groups")

    rows = []
    for gid, sub in g.groupby("group_id"):
        maxsep, n_img = group_geometry(sub)
        rows.append({"group_id": gid, "max_sep_arcsec": maxsep,
                     "n_images": n_img,
                     "ra": float(sub["RA"].mean()),
                     "dec": float(sub["DEC"].mean())})
    geo = pd.DataFrame(rows)

    resolved = geo[(geo["max_sep_arcsec"] >= SEP_MIN) &
                   (geo["max_sep_arcsec"] <= SEP_MAX) &
                   (geo["n_images"] >= 2)].copy()
    print(f"[refine] resolved candidate systems "
          f"({SEP_MIN}\"<=sep<={SEP_MAX}\", >=2 distinct images): "
          f"{len(resolved):,}")

    n_doubles = int((resolved["n_images"] == 2).sum())
    n_triples = int((resolved["n_images"] == 3).sum())
    n_quads = int((resolved["n_images"] >= 4).sum())
    ratio = n_doubles / n_quads if n_quads else None
    print(f"[refine] doubles(2)={n_doubles} triples(3)={n_triples} "
          f"quads(>=4)={n_quads} double:quad(>=4)={ratio}")
    # Dawes naming: triples+quads are both 'multi-image'; their published Type
    # has 432 Double + 4 Quad. Compare doubles : (triples+quads).
    n_multi = n_triples + n_quads
    ratio_multi = n_doubles / n_multi if n_multi else None
    print(f"[refine] double:(triple+quad)={ratio_multi}")

    # recovery of published candidates against resolved candidate centroids
    pub = pd.read_csv(PUB)
    prim = pub[pub["Grade"].astype(str).str.strip().isin(["A", "B", "C"])]
    prim = prim.drop_duplicates("Name")
    pub_sc = SkyCoord(ra=prim["_RA"].to_numpy() * u.deg,
                      dec=prim["_DE"].to_numpy() * u.deg)
    cand_sc = SkyCoord(ra=resolved["ra"].to_numpy() * u.deg,
                       dec=resolved["dec"].to_numpy() * u.deg)
    _, sep, _ = pub_sc.match_to_catalog_sky(cand_sc)
    off = sep.to_value(u.arcsec)
    n_rec = int((off < MATCH_TOL).sum())
    print(f"[recover] {n_rec}/{len(prim)} published candidates within "
          f"{MATCH_TOL}\" of a resolved candidate "
          f"({100*n_rec/len(prim):.1f}%)")

    # of the 138 published candidates that have >=1 DR1 QSO within 5", how
    # many are recovered as a resolved pair? (the fair "ceiling" recovery)
    keep_members = g[g["group_id"].isin(resolved["group_id"])].copy()
    tbl = pa.Table.from_pandas(keep_members, preserve_index=False)
    pq.write_table(tbl, OUT_CAND)
    print(f"[save] wrote {OUT_CAND} ({len(keep_members):,} rows)")

    stats = {
        "sep_min_arcsec": SEP_MIN, "sep_max_arcsec": SEP_MAX,
        "dedup_arcsec": DEDUP, "match_tol_arcsec": MATCH_TOL,
        "n_raw_groups": int(g["group_id"].nunique()),
        "n_resolved_candidates": int(len(resolved)),
        "n_doubles": n_doubles, "n_triples": n_triples, "n_quads>=4": n_quads,
        "double_to_quad_ratio": ratio,
        "double_to_multi_ratio": ratio_multi,
        "n_published": int(len(prim)),
        "n_recovered_within_3arcsec": n_rec,
        "recovery_frac": n_rec / len(prim),
        "published_targets": {
            "candidates": 436, "doubles_quads_text": "24 quads identified",
            "double_quad_ratio_text": "~21:1", "type_col": "432 Double + 4 Quad",
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[save] wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
