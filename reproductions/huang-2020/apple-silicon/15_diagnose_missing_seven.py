#!/usr/bin/env python3
"""
15_diagnose_missing_seven.py — Why are 7 of 342 published candidates
not in our DR7 parent sample? For each missing candidate, find the
nearest match in the raw DR7 sweeps (NOT pre-filtered by our cuts) and
report TYPE, FLUX_Z -> z-mag, NOBS_G/R/Z. That tells us which paper-cut
they fail (TYPE != DEV/COMP, z-mag > 20, or NOBS < 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SWEEP_DIR = DATA / "dr7_sweep"

# 7 missing candidates (from recovery_matched_dr9trained.csv where in_scored_set=False).
MISSING = pd.DataFrame([
    {"name": "DESI-009.9772-12.2100", "RA":   9.9772, "DEC": -12.2100, "grade": "B"},
    {"name": "DESI-010.3630-01.1298", "RA":  10.3630, "DEC":  -1.1298, "grade": "B"},
    {"name": "DESI-016.3969+00.1169", "RA":  16.3969, "DEC":   0.1169, "grade": "B"},
    {"name": "DESI-036.3915-05.0365", "RA":  36.3915, "DEC":  -5.0365, "grade": "B"},
    {"name": "DESI-031.8778-14.8046", "RA":  31.8778, "DEC": -14.8046, "grade": "C"},
    {"name": "DESI-034.3281-05.1331", "RA":  34.3281, "DEC":  -5.1331, "grade": "C"},
    {"name": "DESI-153.0462-00.8142", "RA": 153.0462, "DEC":  -0.8142, "grade": "C"},
])


def sweep_contains(name: str, ra: float, dec: float) -> bool:
    """sweep-RRRsDDD-RRRsDDD.fits — RRR are integer-degree RA, sDDD are
    sign-encoded integer-degree Dec (p=positive, m=negative). Each sweep
    covers RA [RA1, RA2] x Dec [Dec1, Dec2] in degrees."""
    parts = name.replace("sweep-", "").replace(".fits", "").split("-")
    def parse_coord(s):
        for sign_char, sign in (("p", 1), ("m", -1)):
            if sign_char in s:
                i = s.index(sign_char)
                return float(s[:i]), sign * float(s[i+1:])
        raise ValueError(s)
    ra1, dec1 = parse_coord(parts[0])
    ra2, dec2 = parse_coord(parts[1])
    return (min(ra1, ra2) <= ra <= max(ra1, ra2)
            and min(dec1, dec2) <= dec <= max(dec1, dec2))


def main() -> None:
    sweeps = sorted(SWEEP_DIR.glob("sweep-*.fits"))
    if not sweeps:
        raise SystemExit(f"no sweeps in {SWEEP_DIR}")

    out_rows = []
    miss_sky = SkyCoord(ra=MISSING["RA"].values * u.deg,
                        dec=MISSING["DEC"].values * u.deg)
    for _, m in MISSING.iterrows():
        ra, dec = float(m["RA"]), float(m["DEC"])
        # Find the relevant sweep file(s)
        relevant = [s for s in sweeps if sweep_contains(s.name, ra, dec)]
        if not relevant:
            out_rows.append({**m, "found": "no-sweep",
                             "nearest_sep_arcsec": np.nan})
            continue

        best_sep = np.inf
        best_row = None
        for path in relevant:
            with fits.open(path, memmap=True) as hdul:
                t = hdul[1].data
                src_sky = SkyCoord(ra=t["RA"] * u.deg, dec=t["DEC"] * u.deg)
                seps = src_sky.separation(SkyCoord(ra=ra * u.deg, dec=dec * u.deg)).arcsec
                idx = int(np.argmin(seps))
                if seps[idx] < best_sep:
                    best_sep = float(seps[idx])
                    # Materialize attrs
                    fz = float(t["FLUX_Z"][idx])
                    zmag = (22.5 - 2.5 * np.log10(fz)) if fz > 0 else np.nan
                    typ = str(t["TYPE"][idx]).strip()
                    best_row = {
                        "found": "ok",
                        "nearest_sep_arcsec": best_sep,
                        "TYPE": typ,
                        "FLUX_Z": fz,
                        "z_mag": zmag,
                        "NOBS_G": int(t["NOBS_G"][idx]),
                        "NOBS_R": int(t["NOBS_R"][idx]),
                        "NOBS_Z": int(t["NOBS_Z"][idx]),
                    }
        if best_row is None:
            out_rows.append({**m, "found": "no-source",
                             "nearest_sep_arcsec": np.nan})
        else:
            row = dict(m)
            row.update(best_row)
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    print("\n[diagnostic] nearest raw-sweep source for each of the 7 missing candidates:\n")
    cols = ["name", "grade", "found", "nearest_sep_arcsec",
            "TYPE", "z_mag", "NOBS_G", "NOBS_R", "NOBS_Z"]
    with pd.option_context("display.float_format", "{:.3f}".format,
                            "display.width", 200):
        print(df[cols].to_string(index=False))

    # Annotate which cut each failed (informative when found=ok)
    def explain(row):
        if row["found"] != "ok":
            return row["found"]
        if row["nearest_sep_arcsec"] > 5:
            return f"too-far ({row['nearest_sep_arcsec']:.1f}″)"
        reasons = []
        if row["TYPE"] not in ("DEV", "COMP"):
            reasons.append(f"TYPE={row['TYPE']}")
        if (row["NOBS_G"] < 3 or row["NOBS_R"] < 3 or row["NOBS_Z"] < 3):
            reasons.append(f"NOBS<{3}")
        if row["z_mag"] != row["z_mag"] or row["z_mag"] > 20:  # NaN or > cut
            reasons.append(f"z_mag={row['z_mag']:.2f}>20")
        return ", ".join(reasons) if reasons else "passes-cuts(?)"

    df["why_missing"] = df.apply(explain, axis=1)
    print("\n[diagnostic] reason each candidate is missing:\n")
    with pd.option_context("display.width", 200):
        print(df[["name", "grade", "why_missing"]].to_string(index=False))

    out_path = DATA / "missing_seven_diagnostic.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
