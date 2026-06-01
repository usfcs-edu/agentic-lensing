#!/usr/bin/env python3
"""
02_filter_catalog.py

Parse the NeuraLens combined catalog into per-paper positive lists.

Input: data/neuralens_catalog.csv — 1,312 rows pulled from
  https://drive.google.com/file/d/1_KbEHWhl8LeeTyXpXkWFbLRxt6o42wBg
Columns of interest: Name (DESI-RA±DEC encoded), Score, Probability,
ResNet Model ("L18" for Huang+2020, "shielded" for Huang+2021), Region.

Output: data/positives_huang2020.parquet  (L18-model rows in DECaLS region)
         + data/positives_all.parquet     (full ResNet-found catalog for later use)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

NAME_RE = re.compile(r"^DESI-(\d{3}\.\d{4})([+-]\d{2}\.\d{4})$")


def parse_name(name: str) -> tuple[float, float] | tuple[None, None]:
    m = NAME_RE.match(name.strip())
    if not m:
        return (None, None)
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    src = DATA / "neuralens_catalog.csv"
    df = pd.read_csv(src)
    df = df.rename(columns={"Unnamed: 0": "row_index", "ResNet Model": "resnet_model",
                              "Tractor Type": "tractor_type", "Score Diff": "score_diff",
                              "Photo-z": "photo_z", "g mag": "g_mag", "r mag": "r_mag",
                              "z mag": "z_mag"})
    radec = df["Name"].apply(parse_name)
    df["RA"] = radec.apply(lambda t: t[0])
    df["DEC"] = radec.apply(lambda t: t[1])
    bad = df["RA"].isna().sum()
    if bad:
        print(f"[warn] {bad} unparseable names dropped")
        df = df.dropna(subset=["RA", "DEC"]).reset_index(drop=True)

    print(f"[load] {len(df):,} rows after RA/Dec parse")
    print(f"  resnet_model: {df['resnet_model'].value_counts().to_dict()}")
    print(f"  region:       {df['Region'].value_counts().to_dict()}")
    print(f"  score:        {df['Score'].value_counts().sort_index().to_dict()}")

    # Save full catalog
    out_all = DATA / "positives_all.parquet"
    keep = ["row_index", "Name", "RA", "DEC", "tractor_type", "Score",
            "Probability", "resnet_model", "Region", "Spec", "photo_z",
            "g_mag", "r_mag", "z_mag"]
    df[keep].to_parquet(out_all, index=False)
    print(f"[save] {out_all}  ({len(df):,} rows)")

    # Huang 2020 = L18 ResNet model only; the paper says 335 candidates, the
    # online catalog now has 949 L18 rows after grading revisions and
    # additions through later collaboration work.
    df_2020 = df[df["resnet_model"] == "L18"].reset_index(drop=True)
    out_2020 = DATA / "positives_huang2020.parquet"
    df_2020[keep].to_parquet(out_2020, index=False)
    print(f"[save] {out_2020}  ({len(df_2020):,} rows; L18-model only)")

    # Grade A/B subset = score >= 3.0 (high-confidence, suitable for testing)
    df_high = df_2020[df_2020["Score"] >= 3.0].reset_index(drop=True)
    print(f"[info] high-confidence subset (score>=3.0): {len(df_high):,}")


if __name__ == "__main__":
    main()
