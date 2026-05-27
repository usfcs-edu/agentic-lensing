#!/usr/bin/env python3
"""
07_classify_einstein_dimple.py

For each FoF group in data/dr1_pairs.parquet, identify a (lens, source) pair as
(lower-z, higher-z) members and:

  1. Compute the angular fiber separation Δθ between lens and source.
  2. Look up the lens TARGETID in the FastSpecFit DR1 v3.0 VAC; extract σ_v.
  3. Compute the Einstein radius via Hsu+2025 eq. (1) (SIS):

       θ_E = 4π (σ_v/c)² × D_ds / D_s,    flat ΛCDM H0=70, Ω_m=0.3.

  4. Classify each pair:
       - "conventional"  if σ_v is reliably measured (per Hsu §4.1, the
         conventional Grade A criterion is "with an estimated Einstein radius
         (from velocity dispersion) available").
       - "dimple-candidate"  if σ_v is NOT available (per Hsu §4.4 + Fig. 6
         caption: "Velocity dispersion is not available for most of the dimple
         candidates; thus the estimated Einstein radius is not provided"). This
         is the only ALGORITHMIC analogue of the dimple class — the published
         318 number itself comes from visual inspection of imaging morphology
         (§4.4) which we explicitly do not reproduce.

Reports σ_v coverage, θ_E distribution, and dimple-proxy fraction. Saves
data/classified_pairs.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PAIRS = DATA / "dr1_pairs.parquet"
FASTSPEC_DIR = DATA / "fastspecfit"
OUT_PAIRS = DATA / "classified_pairs.parquet"
OUT_STATS = DATA / "classified_stats.json"

# Hsu+2025 §4.1: flat ΛCDM with H0 = 70 km/s/Mpc, Ω_m = 0.3
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
C_KM_S = 299_792.458  # km/s
ARCSEC_PER_RAD = 206_264.80624709636


def load_sigmav_master() -> pd.DataFrame:
    """Concatenate all per-file FastSpecFit σ_v parquets into a single table."""
    shards = sorted(FASTSPEC_DIR.glob("fastspec-iron-*.sigmav.parquet"))
    if not shards:
        raise SystemExit(
            f"no σ_v parquets in {FASTSPEC_DIR}; run 02_download_fastspecfit.py "
            f"--tier full first"
        )
    print(f"[load] concatenating {len(shards)} σ_v shards")
    tables = [pq.read_table(s) for s in shards]
    table = pa.concat_tables(tables, promote_options="default")
    df = table.to_pandas()
    print(f"[load] σ_v master: {len(df):,} rows across {df['TARGETID'].nunique():,} unique TARGETIDs")
    # FastSpecFit can have duplicate TARGETIDs across survey/program. Keep the
    # one with the largest VDISP_IVAR (most precise vdisp measurement).
    if "VDISP_IVAR" in df.columns:
        df = df.sort_values("VDISP_IVAR", ascending=False).drop_duplicates(
            "TARGETID", keep="first"
        )
    else:
        df = df.drop_duplicates("TARGETID", keep="first")
    print(f"[load] after dedup: {len(df):,} unique TARGETIDs")
    return df.set_index("TARGETID")


def assign_lens_source(pairs: pd.DataFrame) -> pd.DataFrame:
    """For each group, build a single (lens, source) row from the min-z and max-z
    members. Larger groups (triplets+) are decomposed to their (zmin, zmax) pair.
    """
    p = pairs.copy()
    # rank within group by Z
    p = p.sort_values(["group_id", "Z"])
    lens = p.groupby("group_id").first().reset_index().add_suffix("_lens")
    src = p.groupby("group_id").last().reset_index().add_suffix("_src")
    lens = lens.rename(columns={"group_id_lens": "group_id"})
    src = src.rename(columns={"group_id_src": "group_id"})
    out = lens.merge(src, on="group_id")
    size = pairs.groupby("group_id").size().rename("group_size").reset_index()
    out = out.merge(size, on="group_id")
    return out


def angular_sep(ra1, dec1, ra2, dec2) -> np.ndarray:
    sc1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    sc2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return sc1.separation(sc2).to_value(u.arcsec)


def theta_e_sis(sigma_v_kms: np.ndarray, z_l: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    """θ_E in arcsec for SIS lens. Vectorized."""
    # Equation 1: θ_E = 4π (σ/c)² × D_ds / D_s
    # astropy's angular_diameter_distance_z1z2 takes scalar arrays
    out = np.full_like(sigma_v_kms, np.nan, dtype=np.float64)
    finite = np.isfinite(sigma_v_kms) & (sigma_v_kms > 0) & (z_l < z_s)
    if not finite.any():
        return out
    zl = z_l[finite]
    zs = z_s[finite]
    s = sigma_v_kms[finite]
    d_s = COSMO.angular_diameter_distance(zs).to_value(u.Mpc)
    d_ds = COSMO.angular_diameter_distance_z1z2(zl, zs).to_value(u.Mpc)
    theta_rad = 4.0 * np.pi * (s / C_KM_S) ** 2 * (d_ds / d_s)
    out[finite] = theta_rad * ARCSEC_PER_RAD
    return out


def main() -> None:
    if not PAIRS.exists():
        raise SystemExit(f"missing {PAIRS}; run 05_run_full_fof.py first")
    pairs = pq.read_table(PAIRS).to_pandas()
    print(f"[pair] {PAIRS}: {len(pairs):,} rows / "
          f"{pairs['group_id'].nunique():,} groups")
    sigmav = load_sigmav_master()

    pls = assign_lens_source(pairs)
    pls["sep_arcsec"] = angular_sep(
        pls["RA_lens"].to_numpy(), pls["DEC_lens"].to_numpy(),
        pls["RA_src"].to_numpy(), pls["DEC_src"].to_numpy(),
    )
    # Look up σ_v for lens TARGETID
    pls = pls.merge(
        sigmav.reset_index()[["TARGETID", "VDISP", "VDISP_IVAR", "LOGMSTAR"]]
        .rename(columns={
            "TARGETID": "TARGETID_lens",
            "VDISP": "sigma_v_lens",
            "VDISP_IVAR": "sigma_v_ivar_lens",
            "LOGMSTAR": "logmstar_lens",
        }),
        on="TARGETID_lens", how="left",
    )
    # FastSpecFit returns VDISP = 250.0 km/s for failed fits (the LRG-template
    # cap) with VDISP_IVAR = 0. Require a positive inverse variance for a real
    # measurement — this is how Hsu's Table 4 distinguishes "Vd ± e_Vd" rows
    # from null entries.
    has_sigmav = (
        np.isfinite(pls["sigma_v_lens"])
        & (pls["sigma_v_lens"] > 0)
        & np.isfinite(pls["sigma_v_ivar_lens"])
        & (pls["sigma_v_ivar_lens"] > 0)
    )
    sigma_v_for_te = np.where(has_sigmav, pls["sigma_v_lens"].to_numpy(), np.nan)
    pls["theta_E_arcsec"] = theta_e_sis(
        sigma_v_for_te,
        pls["Z_lens"].to_numpy(),
        pls["Z_src"].to_numpy(),
    )

    # Classification: "conventional" needs σ_v; "dimple_proxy" lacks σ_v.
    # Note this is an ALGORITHMIC proxy for §4.4 — Hsu's published 318 came
    # from human visual inspection of imaging, not from a σ_v cut.
    pls["class_algo"] = np.where(has_sigmav, "conventional", "dimple_proxy")

    n_total = len(pls)
    n_conv = int(has_sigmav.sum())
    n_dimp = n_total - n_conv

    theta_E_valid = pls.loc[has_sigmav, "theta_E_arcsec"]
    sigma_v_valid = pls.loc[has_sigmav, "sigma_v_lens"]

    print()
    print(f"[stats] total pairs:                {n_total:,}")
    print(f"[stats] with σ_v (conventional):    {n_conv:,} ({100.0*n_conv/n_total:.1f}%)")
    print(f"[stats] without σ_v (dimple proxy): {n_dimp:,} ({100.0*n_dimp/n_total:.1f}%)")
    if n_conv:
        print()
        print(f"[σ_v ]   median = {sigma_v_valid.median():.1f} km/s")
        print(f"[σ_v ]   (16, 50, 84) = ({sigma_v_valid.quantile(0.16):.1f}, "
              f"{sigma_v_valid.quantile(0.50):.1f}, "
              f"{sigma_v_valid.quantile(0.84):.1f}) km/s")
        print(f"[θ_E ]   median = {theta_E_valid.median():.3f}″")
        print(f"[θ_E ]   (16, 50, 84) = ({theta_E_valid.quantile(0.16):.3f}, "
              f"{theta_E_valid.quantile(0.50):.3f}, "
              f"{theta_E_valid.quantile(0.84):.3f})″")
        # Hsu Table 2 sanity: σ_v in {242, 260, 270, 280, 284, 300, ...}.
        # Expect a peak around 250–400 km/s for conventional Grade A lenses.

    # Save
    keep = [
        "group_id", "group_size",
        "TARGETID_lens", "RA_lens", "DEC_lens", "Z_lens",
        "TARGETID_src",  "RA_src",  "DEC_src",  "Z_src",
        "sep_arcsec",
        "sigma_v_lens", "sigma_v_ivar_lens", "logmstar_lens",
        "theta_E_arcsec", "class_algo",
    ]
    table = pa.Table.from_pandas(pls[keep], preserve_index=False)
    pq.write_table(table, OUT_PAIRS)
    print(f"\n[save] wrote {OUT_PAIRS}  ({len(pls):,} rows)")

    OUT_STATS.write_text(
        json.dumps(
            {
                "n_pairs_total": int(n_total),
                "n_with_sigma_v":  int(n_conv),
                "n_without_sigma_v": int(n_dimp),
                "frac_with_sigma_v": float(n_conv / n_total),
                "sigma_v_kms": {
                    "n": int(len(sigma_v_valid)),
                    "p16": float(sigma_v_valid.quantile(0.16)) if n_conv else None,
                    "p50": float(sigma_v_valid.quantile(0.50)) if n_conv else None,
                    "p84": float(sigma_v_valid.quantile(0.84)) if n_conv else None,
                },
                "theta_E_arcsec": {
                    "n": int(len(theta_E_valid)),
                    "p16": float(theta_E_valid.quantile(0.16)) if n_conv else None,
                    "p50": float(theta_E_valid.quantile(0.50)) if n_conv else None,
                    "p84": float(theta_E_valid.quantile(0.84)) if n_conv else None,
                },
                "published_dimple_count": 318,
                "note": (
                    "The published 318 dimple count (Hsu+2025 §4.4) comes from "
                    "human visual inspection of imaging morphology, not from a "
                    "σ_v cut. Our 'dimple_proxy' column is the algorithmic "
                    "analogue (pairs lacking FastSpecFit σ_v on the lens) but is "
                    "NOT expected to match 318 exactly."
                ),
            },
            indent=2,
        )
    )
    print(f"[save] wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
