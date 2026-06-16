#!/usr/bin/env python3
"""300_build_mimic_bank.py — ClaudeNet v3: seed the lens-MIMIC bank from the DR9
qualification campaign's confirmed non-lenses.

The campaign (PR #12) graded 737 DR9 sweep candidates with two independent agentic
graders + a skeptic pass and found 0/601 genuinely-new lenses. Those 601 status==NEW
rejects are a GIFT: CNN-high-scoring, dual-agent-confirmed NON-lenses, each with a
*typed contaminant* label (lrg_companion / merger / blend / ring_galaxy / spiral /
star_halo / ...). They are exactly the population the v2 "random galaxy" conformal null
and random-pool hard-negative mining never targeted. This script joins:

  data/v2/campaign/consensus_full_737.parquet   (status, RA/DEC, p_final, grades)
  data/v2/campaign/visual_workflow_result.json  (grades[].contaminant — the type)
  data/v2/campaign/manifest_737.parquet         (the 5 v2-lean member scores, fits_path)

-> data/v3/mimic_bank_seed.parquet  (typed, ~601 rows), the seed for A1's mine->grade->
fold expansion. Belt-and-suspenders: re-dedup vs every known-lens catalog at 5"
(status==NEW already excludes the local catalogs + SIMBAD, so we expect ~0 drops; this
catches catalog drift). A `confident_mimic` flag marks grade==D with a named contaminant
(the high-confidence negatives A2 trains on; C-graded rows are carried but flagged
ambiguous).

    /home2/benson/.venvs/claudenet/bin/python 300_build_mimic_bank.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

import _clib as C

ROOT = Path(__file__).resolve().parent
CAMP = ROOT / "data" / "v2" / "campaign"
OUT = ROOT / "data" / "v3"
MEMBERS = ["member_effnet_B", "member_effnet_B3_hard", "member_effnet_S2_hard",
           "member_resnet46_C_hard", "member_zoobot_N"]


def load_visual_contaminants(path: Path) -> pd.DataFrame:
    """grades[] -> DataFrame[row_id, mimic_type, visual_grade, visual_p_lens, + criteria]."""
    d = json.loads(path.read_text())
    rows = []
    for g in d["grades"]:
        crit = g.get("criteria", {}) or {}
        rows.append({
            "row_id": str(g["row_id"]),
            "mimic_type": (g.get("contaminant") or "unknown"),
            "visual_grade": g.get("grade"),
            "visual_p_lens": g.get("p_lens"),
            "visual_confidence": g.get("confidence"),
            "visual_rationale": g.get("rationale"),
            **{f"crit_{k}": v for k, v in crit.items()},
        })
    return pd.DataFrame(rows)


def known_lens_skycoord() -> SkyCoord:
    ras, decs = [], []
    for p in C.known_lens_catalogs():
        df = pd.read_csv(p)
        ras.append(df["RA"].to_numpy(float)); decs.append(df["DEC"].to_numpy(float))
    return SkyCoord(np.concatenate(ras) * u.deg, np.concatenate(decs) * u.deg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", default=str(CAMP / "consensus_full_737.parquet"))
    ap.add_argument("--visual", default=str(CAMP / "visual_workflow_result.json"))
    ap.add_argument("--manifest", default=str(CAMP / "manifest_737.parquet"))
    ap.add_argument("--radius", type=float, default=5.0, help="known-lens dedup arcsec")
    ap.add_argument("--out", default=str(OUT / "mimic_bank_seed.parquet"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    con = pd.read_parquet(args.consensus)
    con["row_id"] = con["row_id"].astype(str)
    new = con[con["status"] == "NEW"].copy()
    print(f"[300] consensus: {len(con)} rows; status==NEW: {len(new)} "
          f"(KNOWN_LOCAL {int((con.status=='KNOWN_LOCAL').sum())}, "
          f"KNOWN_REMOTE {int((con.status=='KNOWN_REMOTE').sum())})")

    vis = load_visual_contaminants(Path(args.visual))
    man = pd.read_parquet(args.manifest)
    man["row_id"] = man["row_id"].astype(str)
    keep = ["row_id", "footprint", "brick", "fits_path", "p_stage1", "p_final"] + MEMBERS
    keep = [c for c in keep if c in man.columns]

    df = (new[["row_id", "RA", "DEC", "p_final", "q_group", "status", "my_grade",
               "lensjudge_grade", "consensus_p"]]
          .merge(vis, on="row_id", how="left")
          .merge(man[keep], on="row_id", how="left", suffixes=("", "_man")))
    if "p_final_man" in df.columns:           # prefer manifest p_final (same column)
        df["p_final"] = df["p_final"].fillna(df.pop("p_final_man"))

    # 5" belt-and-suspenders dedup vs all known-lens catalogs
    known = known_lens_skycoord()
    cand = SkyCoord(df["RA"].to_numpy(float) * u.deg, df["DEC"].to_numpy(float) * u.deg)
    _, sep, _ = cand.match_to_catalog_sky(known)
    hit = sep.to(u.arcsec).value < args.radius
    if hit.any():
        print(f"[300] WARNING dropping {int(hit.sum())} rows within {args.radius}\" of a "
              f"known lens (catalog drift since campaign)")
    df = df[~hit].copy()

    df["mimic_type"] = df["mimic_type"].fillna("unknown").replace({None: "unknown"})
    df["confident_mimic"] = (df["my_grade"].astype(str).eq("D")
                             & ~df["mimic_type"].isin(["unknown", "null", "other"]))
    df = df.reset_index(drop=True)
    df.to_parquet(args.out, index=False)

    print(f"[300] wrote {args.out}  ({len(df)} typed mimic-seed rows)")
    print("[300] mimic_type counts:")
    for t, n in df["mimic_type"].value_counts().items():
        print(f"        {t:18s} {n}")
    print("[300] visual grade (my_grade) among seed:",
          df["my_grade"].value_counts().to_dict())
    print(f"[300] confident_mimic (D + named type): {int(df.confident_mimic.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
