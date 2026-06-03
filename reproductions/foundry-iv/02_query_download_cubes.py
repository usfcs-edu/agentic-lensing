"""02 - Cross-match the public ESO/MUSE Phase-3 datacubes to the confirmed Foundry-IV
systems and download cubes for a few of them.

The five MUSE programs (109.238W.004, 111.24UJ.008, 111.24P8.001, 112.2614.001,
113.267Q.001; PI: Cikota/Bian) all have a 12-month proprietary period that has long
elapsed for the 2022-2024 runs, so the *reduced datacubes* are public Phase-3 products
in the ESO archive. We query them via the public ESO TAP service (ivoa.ObsCore) using
pyvo, cross-match by coordinate (the cubes are archived under generic "LensNN" target
names, NOT DESI names), and download the matched cubes by their access_url.

Reduced cubes (dataproduct_type='cube') save us from running the heavy MUSE ESOREX
pipeline on raw frames — exactly what the paper used (MUSE pipeline v2.2 + ZAP).

Usage:
    python 02_query_download_cubes.py            # query + write match table, download default targets
    python 02_query_download_cubes.py --list     # just print all available cubes, no download
    python 02_query_download_cubes.py --targets Lens16,Lens22,Lens24

Outputs:
    data/cube_match_table.csv     all archive cubes x nearest confirmed system
    data/cubes/<dpid>.fits        downloaded datacubes (gitignored)
"""
from pathlib import Path
import argparse
import csv
import sys
import urllib.request

import numpy as np
import pyvo
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse as parse_votable
import astropy.units as u

REPRO = Path(__file__).parent
DATA = REPRO / "data"
CUBES = DATA / "cubes"
CUBES.mkdir(parents=True, exist_ok=True)

TAP_URL = "http://archive.eso.org/tap_obs"
PROGS = ["109.238W.004", "111.24UJ.008", "111.24P8.001", "112.2614.001", "113.267Q.001"]
MATCH_RADIUS = 30.0  # arcsec; MUSE FoV is 60"x60", lens is near center

# Default download set: 3 dual-redshift MVP systems (lens absorption + source emission/UV),
# all with the longest exposure cube available, picked for in-band features.
DEFAULT_TARGETS = ["Lens16", "Lens22", "Lens24"]  # prog 109.238W.004 / 111.24P8.001


def load_catalog():
    rows = []
    with (DATA / "confirmed_catalog.csv").open() as f:
        for r in csv.DictReader(f):
            r["ra_deg"] = float(r["ra_deg"])
            r["dec_deg"] = float(r["dec_deg"])
            rows.append(r)
    return rows


def query_all_cubes():
    tap = pyvo.dal.TAPService(TAP_URL)
    out = []
    for p in PROGS:
        q = f"""SELECT dp_id, target_name, s_ra, s_dec, t_exptime, obs_release_date,
                proposal_id, access_url, em_min, em_max
                FROM ivoa.ObsCore
                WHERE proposal_id = '{p}' AND instrument_name='MUSE'
                AND dataproduct_type='cube'"""
        t = tap.search(q).to_table()
        for row in t:
            out.append(dict(
                dp_id=str(row["dp_id"]),
                target=str(row["target_name"]),
                ra=float(row["s_ra"]),
                dec=float(row["s_dec"]),
                exptime=float(row["t_exptime"]) if row["t_exptime"] is not None else 0.0,
                release=str(row["obs_release_date"]),
                prog=p,
                access_url=str(row["access_url"]),
            ))
    return out


def build_match_table(cubes, catalog):
    cat_coords = SkyCoord([c["ra_deg"] for c in catalog] * u.deg,
                          [c["dec_deg"] for c in catalog] * u.deg)
    rows = []
    for cb in cubes:
        cc = SkyCoord(cb["ra"] * u.deg, cb["dec"] * u.deg)
        seps = cc.separation(cat_coords).arcsec
        j = int(np.argmin(seps))
        sep = float(seps[j])
        matched = catalog[j] if sep <= MATCH_RADIUS else None
        rows.append(dict(
            cube_target=cb["target"], prog=cb["prog"], dp_id=cb["dp_id"],
            cube_ra=cb["ra"], cube_dec=cb["dec"], exptime=cb["exptime"],
            release=cb["release"], access_url=cb["access_url"],
            match_name=matched["name"] if matched else "",
            match_sep_arcsec=round(sep, 2) if matched else "",
            z_lens=matched["z_lens"] if matched else "",
            z_source=matched["z_source"] if matched else "",
        ))
    return rows


def resolve_datalink(datalink_url):
    """The ObsCore access_url is an ESO DataLink endpoint returning a VOTable.
    Parse it and return the '#this' direct-download URL (the science cube itself)
    plus its advertised size in bytes."""
    tmp = CUBES / "_datalink.tmp.xml"
    urllib.request.urlretrieve(datalink_url, tmp)
    vt = parse_votable(str(tmp))
    tab = vt.get_first_table().to_table()
    tmp.unlink(missing_ok=True)
    for row in tab:
        if str(row["semantics"]) == "#this" and str(row["access_url"]).startswith("http"):
            size = row["content_length"] if "content_length" in tab.colnames else None
            return str(row["access_url"]), (int(size) if size not in (None, "--") else None)
    raise RuntimeError("no #this link in DataLink response")


def download_cube(access_url, dp_id):
    dest = CUBES / f"{dp_id}.fits"
    if dest.exists() and dest.stat().st_size > 100_000_000:  # a real cube is GBs
        print(f"    already have {dest.name} ({dest.stat().st_size/1e9:.2f} GB)")
        return dest
    direct, size = resolve_datalink(access_url)
    sz = f"{size/1e9:.2f} GB" if size else "? GB"
    print(f"    downloading {dp_id} ({sz}) from dataPortal ...", flush=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(direct, tmp)
    tmp.rename(dest)
    print(f"    -> {dest.name} ({dest.stat().st_size/1e9:.2f} GB)")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="only list matched cubes")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="comma list of cube target names to download")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    catalog = load_catalog()
    print(f"Loaded {len(catalog)} confirmed systems.")
    print("Querying ESO TAP for MUSE Phase-3 cubes (5 programs) ...")
    cubes = query_all_cubes()
    print(f"  found {len(cubes)} reduced cubes total.")

    rows = build_match_table(cubes, catalog)
    matched = [r for r in rows if r["match_name"]]
    print(f"  {len(matched)} cubes fall within {MATCH_RADIUS}\" of a confirmed system.")

    mt = DATA / "cube_match_table.csv"
    with mt.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote match table -> {mt}")

    print("\nMatched cubes (cube_target -> confirmed system):")
    for r in sorted(matched, key=lambda x: x["match_name"]):
        print(f"  {r['cube_target']:22} {r['prog']:14} sep={r['match_sep_arcsec']:>5}\" "
              f"exp={r['exptime']:5.0f}s  {r['match_name']:26} zL={r['z_lens']} zS={r['z_source']}")

    if args.list or args.no_download:
        return

    want = [t.strip() for t in args.targets.split(",") if t.strip()]
    print(f"\nDownloading cubes for targets: {want}")
    for t in want:
        cands = [r for r in matched if r["cube_target"] == t]
        if not cands:
            print(f"  {t}: no matched cube found, skipping")
            continue
        best = max(cands, key=lambda x: x["exptime"])  # longest exposure
        print(f"  {t}  ({best['match_name']}, exp={best['exptime']:.0f}s):")
        try:
            download_cube(best["access_url"], best["dp_id"])
        except Exception as e:
            print(f"    DOWNLOAD FAILED: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
