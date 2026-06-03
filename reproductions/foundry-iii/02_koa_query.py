#!/usr/bin/env python
"""
Step 02: Query the Keck Observatory Archive (KOA) for the NIRES frames of the
8 Foundry-III systems, and locate each system's science exposures.

KOA (koa.ipac.caltech.edu) is PUBLIC for these 2022-11 / 2023-01 observations
(18-month proprietary period elapsed). We query via pyKOA (TAP/cgi services).

KEY FINDINGS (verified, see README):
  * The "Nov 13, 2022" half-night (paper Table 1) is UT 2022-11-15 at Keck
    (HST = UT-10; an evening-of-Nov-13 night runs into UT Nov 14/15).
  * The "Jan 10, 2023" half-night is UT 2023-01-11.
  * Program PI in KOA headers is "Schlegel" (DESI Strong Lens program).
  * The 300 s SPEC frames (KOAID prefix "NR.") are the lensed-source spectra;
    the OBJECT header carries the real target (e.g. "DESI-006+10"). The 10 s
    "NI." frames are imaging acquisitions.
  * NIRES has ONLY Level-0 (raw) data in KOA -- there is a single TAP table
    `koa_nires`, all filehand paths are /lev0/, and lev1file=1 download returns
    "Instrument [NIRES] does not have level1 data". PypeIt-reduced 1D spectra
    are NOT archived. (See README for the implication for this reproduction.)

This script does the queries and writes a per-system manifest of science frames.
By default it does NOT download the ~19 MB/frame raw data (set DOWNLOAD=1 env to
fetch a sample). Network requires sandbox disabled in this environment.

Run:  python 02_koa_query.py
      DOWNLOAD=1 python 02_koa_query.py    # also fetch one sample raw frame
"""
import json
import os
import collections

from astropy.io import ascii
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# The two science UT nights (resolved from the paper's HST-local dates):
UT_NIGHTS = {
    "2022-11-15": "Nov 13, 2022 half-night",
    "2023-01-11": "Jan 10, 2023 half-night",
}


def query_nights():
    """Query KOA for all NIRES frames on the two science nights. Cached to disk."""
    from pykoa.koa import Koa
    for ut in UT_NIGHTS:
        out = os.path.join(DATA, f"koa_nires_{ut}.tbl")
        if os.path.exists(out):
            print(f"  cached: {out}")
            continue
        print(f"  querying KOA NIRES {ut} ...")
        Koa.query_datetime("nires", f"{ut} 00:00:00/{ut} 23:59:59", out, format="ipac")


def build_manifest(systems):
    """Match each system to its 300 s SPEC science frames by RA/Dec proximity."""
    # load both nights' tables (prefer the scan_* names produced during dev)
    tables = {}
    for ut in UT_NIGHTS:
        for cand in (f"koa_nires_scan_{ut}.tbl", f"koa_nires_{ut}.tbl"):
            p = os.path.join(DATA, cand)
            if os.path.exists(p):
                tables[ut] = ascii.read(p, format="ipac")
                break
    manifest = {}
    for s in systems:
        ut = s["ut_night"]
        if ut not in tables:
            continue
        t = tables[ut]
        sci = t[[abs(float(x) - 300) < 1 for x in t["elaptime"]]]  # 300s SPEC frames
        # match by sky position (acquisition guide stars sit on the science field)
        ra, dec = s["ra"], s["dec"]
        cosd = np.cos(np.deg2rad(dec))
        dd = np.array([np.hypot((float(r["ra"]) - ra) * cosd,
                                float(r["dec"]) - dec) * 3600.0 for r in sci])
        sel = sci[dd < 60.0]  # within 1 arcmin
        frames = [{"koaid": str(r["koaid"]), "filehand": str(r["filehand"]),
                   "object": None, "ra": float(r["ra"]), "dec": float(r["dec"]),
                   "elaptime": float(r["elaptime"])} for r in sel]
        manifest[s["name"]] = {"ut_night": ut, "n_frames": len(frames), "frames": frames}
    return manifest


def main():
    systems = json.load(open(os.path.join(DATA, "systems.json")))["systems"]
    print("Querying KOA for the two NIRES science nights ...")
    try:
        query_nights()
    except Exception as e:
        print(f"  (query skipped/failed -- using cached tables if present): {e}")

    manifest = build_manifest(systems)
    with open(os.path.join(DATA, "koa_frame_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nScience-frame manifest (300 s SPEC frames within 1' of target):")
    for name, m in manifest.items():
        print(f"  {name:28s} UT {m['ut_night']}  {m['n_frames']:2d} frames")
    total = sum(m["n_frames"] for m in manifest.values())
    print(f"  total: {total} science frames located across both nights")
    print("\nWrote data/koa_frame_manifest.json")

    if os.environ.get("DOWNLOAD") == "1":
        from pykoa.koa import Koa
        # download one sample raw frame for the brightest system
        name = "DESI J006.3643+10.1853"
        fr = manifest[name]["frames"][0]
        one = ascii.read(os.path.join(DATA, f"koa_nires_scan_{manifest[name]['ut_night']}.tbl"),
                         format="ipac")
        one = one[[str(x) == fr["koaid"] for x in one["koaid"]]]
        tmp = os.path.join(DATA, "_sample_frame.tbl")
        ascii.write(one, tmp, format="ipac", overwrite=True)
        print(f"\nDownloading 1 sample raw frame for {name}: {fr['koaid']} (~19 MB) ...")
        Koa.download(tmp, "ipac", os.path.join(DATA, "raw_sample"), lev0file=1)
        os.remove(tmp)


if __name__ == "__main__":
    main()
