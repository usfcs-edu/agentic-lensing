#!/usr/bin/env python3
"""94_external_xmatch.py -- literature crossmatch (NED + SIMBAD) for candidates the
Huang/Storfer/Inchausti-lineage crossmatch calls NEW.

163_crossmatch_known.py / _clib.known_lens_catalogs only check the LINEAGE catalogues
(Storfer 2024, Inchausti 2025, Huang 2020/2021). A candidate absent from those can
still be a strong lens already published by another survey -- CASSOWARY, DES, AGEL,
SuGOHI, SLACS, ... -- which is exactly what bit the v3 C-vet "NEW grade-A" set: all
three lineage-NEW survivors turned out to be previously-found lenses. This runs a cone
search against NED and SIMBAD (the literature aggregators) and writes the lens-typed
matches to v3/external_lens_catalog.csv, which _clib.known_lens_catalogs() now folds
into the crossmatch so a literature-known lens is never miscounted as NEW again.

Run this BEFORE claiming any candidate is net-new. Requires astroquery + network.

    python 94_external_xmatch.py [--radius 15] [--candidates v3/cv3_new_gradeA.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "v3"

# SIMBAD lens otypes (mirrors 163_crossmatch_known.SIMBAD_LENS_OTYPES) + any "lens".
SIMBAD_LENS = {"gLe", "gLS", "LeI", "LeG", "LeQ", "Le?", "LI?", "LS?"}


def is_lens(t) -> bool:
    t = str(t)
    return t in SIMBAD_LENS or "lens" in t.lower()


# id-prefix -> (survey, representative reference) for human-readable attribution.
SURVEY = [
    ("CSWA", ("CASSOWARY", "Belokurov et al. 2009")),
    ("AGEL", ("AGEL", "Tran et al. 2022")),
    ("DES J", ("DES", "O'Donnell et al. 2022")),
    ("SLACS", ("SLACS", "Bolton et al. 2008")),
    ("SL2S", ("SL2S", "Sonnenfeld et al. 2013")),
]


def survey_of(name: str) -> tuple[str, str]:
    for pre, sr in SURVEY:
        if name.upper().lstrip("[").startswith(pre.upper()):
            return sr
    return "literature (NED/SIMBAD)", ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", default=str(V3 / "cv3_new_gradeA.csv"),
                    help="CSV with a 'name' column (the lineage-NEW survivors)")
    ap.add_argument("--coords", default=str(V3 / "manifests_d2_newA.csv"),
                    help="CSV with name,ra,dec for those candidates")
    ap.add_argument("--radius", type=float, default=15.0, help="cone radius (arcsec)")
    ap.add_argument("--out", default=str(V3 / "external_lens_catalog.csv"))
    args = ap.parse_args()

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.ned import Ned
    from astroquery.simbad import Simbad
    Ned.TIMEOUT = 90

    names = [r["name"] for r in csv.DictReader(open(args.candidates))]
    want = set(names)
    coords = {r["name"]: (float(r["ra"]), float(r["dec"]))
              for r in csv.DictReader(open(args.coords)) if r["name"] in want}

    sim = Simbad()
    for f in ("otype", "ra", "dec"):
        try:
            sim.add_votable_fields(f)
        except Exception:
            pass

    def col(t, *ns):
        m = {c.lower(): c for c in t.colnames}
        return next((m[n] for n in ns if n in m), None)

    rows = []
    for name in names:
        if name not in coords:
            print(f"[94] no coords for {name}", file=sys.stderr)
            continue
        ra, dec = coords[name]
        ctr = SkyCoord(ra * u.deg, dec * u.deg)
        matches = []  # (sep_arcsec, ext_name, otype, source)

        try:  # SIMBAD (ra/dec are decimal degrees in astroquery >= 0.4.8)
            t = sim.query_region(ctr, radius=args.radius * u.arcsec)
            if t is not None and len(t):
                mc, rc, dc, oc = (col(t, "main_id"), col(t, "ra"),
                                  col(t, "dec"), col(t, "otype"))
                for row in t:
                    if not is_lens(row[oc]):
                        continue
                    sky = SkyCoord(float(row[rc]) * u.deg, float(row[dc]) * u.deg)
                    matches.append((ctr.separation(sky).arcsec,
                                    str(row[mc]).strip(), str(row[oc]), "SIMBAD"))
        except Exception as e:
            print(f"[94] SIMBAD {name}: {type(e).__name__}: {e}", file=sys.stderr)

        try:  # NED (supplementary; flaky/slow -- SIMBAD usually suffices)
            t = Ned.query_region(ctr, radius=args.radius * u.arcsec)
            if t is not None and len(t):
                nc, tc, rc, dc = (col(t, "object name"), col(t, "type"),
                                  col(t, "ra"), col(t, "dec"))
                for row in t:
                    ty = row[tc]
                    ty = ty.decode() if isinstance(ty, bytes) else str(ty)
                    if "lens" not in ty.lower():
                        continue
                    nm = row[nc]
                    nm = nm.decode() if isinstance(nm, bytes) else str(nm)
                    sky = SkyCoord(float(row[rc]) * u.deg, float(row[dc]) * u.deg)
                    matches.append((ctr.separation(sky).arcsec, nm.strip(), ty, "NED"))
        except Exception as e:
            print(f"[94] NED {name}: {type(e).__name__}: {e}", file=sys.stderr)

        if not matches:
            print(f"[94] {name}: NO literature lens within {args.radius:g}\" "
                  "-> genuinely new?")
            continue
        # prefer a recognised-survey id (CSWA/DES/AGEL/...), nearest among those.
        recognised = [m for m in matches if survey_of(m[1])[0] != "literature (NED/SIMBAD)"]
        sep, ext, otype, src = min(recognised or matches, key=lambda m: m[0])
        survey, ref = survey_of(ext)
        rows.append({"name": ext, "RA": f"{ra:.6f}", "DEC": f"{dec:.6f}",
                     "survey": survey, "ref": ref, "otype": otype,
                     "sep_arcsec": f"{sep:.2f}", "source": src, "desi_candidate": name})
        print(f"[94] {name} -> {ext}  ({survey}, {ref})  {sep:.2f}\"  [{otype}, {src}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "RA", "DEC", "survey", "ref", "otype",
              "sep_arcsec", "source", "desi_candidate"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[94] wrote {len(rows)} external (non-lineage) lens matches -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
