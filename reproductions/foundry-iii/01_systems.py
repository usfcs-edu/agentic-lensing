#!/usr/bin/env python
"""
Foundry III (Agarwal+2025, DESI Strong Lens Foundry III: Keck NIRES) reproduction.

Step 01: Canonical reference tables parsed from the paper
(papers/Agarwal_2025_DESI_Foundry_III.pdf, extracted to /tmp/foundry3.txt via PyMuPDF).

Two tables:
  - SYSTEMS:  Table 1 (NIRES observations) + Table 2 (redshift results),
              merged, with which emission lines the paper used per system (Sec 4.1-4.6).
  - REST_WAVELENGTHS: vacuum rest-frame wavelengths (Angstrom) of the NIR/optical
              nebular emission lines named in the paper.

These are the *ground truth* the rest of the pipeline reproduces.

Run:  python 01_systems.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---------------------------------------------------------------------------
# Rest-frame VACUUM wavelengths (Angstrom). Standard nebular line values.
# (Halpha 6562.8, Hbeta 4861.3 etc. -- the paper does not tabulate the exact
#  rest values it used; these are the canonical vacuum wavelengths. Air vs vac
#  shifts are ~1-2 A and cancel in a ratio-free single-z fit to O(1e-4) in z.)
# ---------------------------------------------------------------------------
REST_WAVELENGTHS = {
    "[OII]3727": 3727.42,   # [O II] doublet (3726.0/3728.8), blended centroid
    "[OII]3729": 3729.88,
    "Hgamma": 4341.68,
    "Hbeta": 4862.68,
    "[OIII]4959": 4960.30,
    "[OIII]5007": 5008.24,  # brighter of the [O III] doublet
    "Halpha": 6564.61,      # vacuum (air 6562.8)
    "[NII]6548": 6549.86,
    "[NII]6584": 6585.27,
    "[SII]6717": 6718.29,
    "[SII]6731": 6732.67,
    "MgII2796": 2796.35,
    "MgII2803": 2803.53,
}

# ---------------------------------------------------------------------------
# The 8 NIRES-observed systems. zs from Table 2; zs_source notes who measured it.
# 'lines' = emission features the paper explicitly reports detecting (Sec 4.x).
# 'fit_lines' = the two lines the paper says it Gaussian-fit for the redshift
#   (Sec 3.3: "two emission lines ... either the strongest ... or strongest from
#    two different elements, e.g. O and H"). For most systems Halpha + [OIII]5007
#   are the two cleanest; J215 is explicitly Halpha + brighter [OIII] (Sec 4.6).
# ---------------------------------------------------------------------------
SYSTEMS = [
    {
        "name": "DESI J006.3643+10.1853", "ra": 6.3643, "dec": 10.1853,
        "ut_night": "2022-11-15", "hst_night": "Nov 13, 2022", "exptime": 1200,
        "zd": 0.4631, "zs": 2.39688, "zs_err": 0.00004, "zs_source": "NIRES",
        "lines": ["Halpha", "[OII]3727", "[OIII]5007", "Hbeta", "[NII]6584"],
        "fit_lines": ["Halpha", "[OIII]5007"],
        "note": "Brightest source in program; 2nd lensed source at z=1.3143 from DESI [OII].",
    },
    {
        "name": "DESI J023.0157-16.0040", "ra": 23.0157, "dec": -16.0040,
        "ut_night": "2022-11-15", "hst_night": "Nov 13, 2022", "exptime": 600,
        "zd": None, "zs": 1.5818, "zs_err": 0.00007, "zs_source": "DESI",
        "lines": [], "fit_lines": [],
        "note": "NIRES non-detection (600s, airmass 1.6). zs later from DESI [OII] doublet.",
    },
    {
        "name": "DESI J024.1631+00.1384", "ra": 24.1631, "dec": 0.1384,
        "ut_night": "2022-11-15", "hst_night": "Nov 13, 2022", "exptime": 600,
        "zd": 0.3445, "zs": None, "zs_err": None, "zs_source": None,
        "lines": [], "fit_lines": [],
        "note": "NIRES non-detection (600s, airmass 1.6). zs not yet secured.",
    },
    {
        "name": "DESI J094.5639+50.3059", "ra": 94.5639, "dec": 50.3059,
        "ut_night": "2022-11-15", "hst_night": "Nov 13, 2022", "exptime": 3600,
        "zd": 0.522, "zs": 3.33185, "zs_err": 0.00010, "zs_source": "NIRES",
        "lines": ["[OII]3727", "[OIII]5007", "Hbeta", "MgII2796"],
        "fit_lines": ["[OIII]5007", "Hbeta"],
        "note": "Highest zs in paper. zd from Lick Kast. Halpha redshifted out of K band.",
    },
    {
        "name": "DESI J133.3800+23.3652", "ra": 133.3800, "dec": 23.3652,
        "ut_night": "2023-01-11", "hst_night": "Jan 10, 2023", "exptime": 2400,
        "zd": 0.3053, "zs": 2.18858, "zs_err": 0.00002, "zs_source": "NIRES",
        "lines": ["Halpha", "Hbeta", "[OII]3727", "[OIII]5007", "[NII]6584", "[SII]6717"],
        "fit_lines": ["Halpha", "[OIII]5007"],
        "note": "High SNR; manual PypeIt extraction (auto detection failed).",
    },
    {
        "name": "DESI J154.5307-00.1368", "ra": 154.5307, "dec": -0.1368,
        "ut_night": "2022-11-15", "hst_night": "Nov 13, 2022", "exptime": 2400,
        "zd": 0.3718, "zs": 1.73810, "zs_err": 0.00005, "zs_source": "NIRES",
        # two star-forming knots at slightly different z:
        "zs_knots": [1.73885, 1.73735], "zs_knots_err": [0.00004, 0.00006],
        "lines": ["Halpha", "[NII]6584", "Hbeta", "[OII]3727"],
        "fit_lines": ["Halpha", "Hbeta"],
        "note": "Two knots; zs = mean of 1.73885 & 1.73735. No clear [OIII].",
    },
    {
        "name": "DESI J165.4754-06.0423", "ra": 165.4754, "dec": -6.0423,
        "ut_night": "2023-01-11", "hst_night": "Jan 10, 2023", "exptime": 3600,
        "zd": 0.4834, "zs": 1.67511, "zs_err": 0.00005, "zs_source": "NIRES",
        "lines": ["Halpha", "[NII]6584", "Hbeta", "[OIII]5007", "[SII]6717", "Hgamma"],
        "fit_lines": ["Halpha", "[OIII]5007"],
        "note": "Lowest zs in work; [OII] just past DESI optical edge.",
    },
    {
        "name": "DESI J215.2654+00.3719", "ra": 215.2654, "dec": 0.3719,
        "ut_night": "2023-01-11", "hst_night": "Jan 10, 2023", "exptime": 2400,
        "zd": 0.6566, "zs": 2.20645, "zs_err": 0.00035, "zs_source": "NIRES",
        "lines": ["Halpha", "[OIII]5007"],
        "fit_lines": ["Halpha", "[OIII]5007"],
        "note": "Faintest knots; only Halpha + brighter [OIII] detected (Sec 4.6).",
    },
]

# The six systems with a NIRES-measured source redshift (the reproduction targets):
NIRES_ZS_TARGETS = [s for s in SYSTEMS if s["zs_source"] == "NIRES"]


def main():
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "systems.json"), "w") as f:
        json.dump({"systems": SYSTEMS, "rest_wavelengths": REST_WAVELENGTHS}, f, indent=2)
    print(f"Wrote {len(SYSTEMS)} systems -> data/systems.json")
    print(f"\n{len(NIRES_ZS_TARGETS)} systems have a NIRES source redshift "
          f"(the reproduction targets):")
    print(f"  {'system':28s} {'zs':>9s} {'sigma_zs':>10s}  fit lines")
    for s in NIRES_ZS_TARGETS:
        fl = " + ".join(s["fit_lines"])
        print(f"  {s['name']:28s} {s['zs']:9.5f} {s['zs_err']:10.5f}  {fl}")
    zmin = min(s["zs"] for s in NIRES_ZS_TARGETS)
    zmax = max(s["zs"] for s in NIRES_ZS_TARGETS)
    print(f"\n  zs range: {zmin:.5f} - {zmax:.5f}  (paper: 1.675 - 3.332)")


if __name__ == "__main__":
    main()
