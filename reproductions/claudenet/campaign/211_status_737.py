#!/usr/bin/env python3
"""211_status_737.py — merge the expanded-crossmatch status onto the 737 manifest.

Reads the 163 crossmatch output (run with --remote-services simbad +
--extra-catalog for the VizieR lens catalogs) and joins status / nearest-match /
remote-type columns onto manifest_737.parquet. 163 already bakes the precedence
KNOWN_LOCAL > KNOWN_REMOTE > NEW into its `status` column.

    /home2/benson/.venvs/claudenet/bin/python campaign/211_status_737.py \
        --crossmatch data/v2/campaign/crossmatch_737.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign"
KEEP = ["status", "nearest_sep_arcsec", "nearest_catalog", "nearest_name",
        "known_local", "remote_queried", "remote_lens", "remote_types"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossmatch", default=str(OUT / "crossmatch_737.parquet"))
    args = ap.parse_args()
    man = pd.read_parquet(OUT / "manifest_737.parquet")
    man["row_id"] = man["row_id"].astype(str)
    xm = pd.read_parquet(args.crossmatch)
    xm["row_id"] = xm["row_id"].astype(str)
    cols = ["row_id"] + [c for c in KEEP if c in xm.columns]
    m = man.merge(xm[cols], on="row_id", how="left")
    assert len(m) == len(man), "crossmatch join changed row count"
    # any row not in the crossmatch output stays NEW (shouldn't happen)
    m["status"] = m["status"].fillna("NEW")
    out = OUT / "manifest_737_xmatched.parquet"
    m.to_parquet(out, index=False)
    vc = m.status.value_counts().to_dict()
    print(f"[211] status over 737: {vc}")
    for st in ("KNOWN_LOCAL", "KNOWN_REMOTE"):
        sub = m[m.status == st]
        if len(sub):
            cats = sub["nearest_catalog"].value_counts().to_dict() if "nearest_catalog" in sub else {}
            print(f"[211]   {st}: {len(sub)} (nearest catalogs: {cats})")
    print(f"[211] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
