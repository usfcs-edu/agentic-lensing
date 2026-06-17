#!/usr/bin/env python3
"""crosscheck_candidates.py -- is each campaign candidate already a published lens?

Literature crosscheck (the same one applied to the v3 DR10 set):
  1. LINEAGE + external: nearest neighbour (haversine, 5") in the
     Huang+2020/2021, Storfer 2024, Inchausti 2025 catalogues + claudenet's
     external_lens_catalog.csv (CASSOWARY/DES literature matches). Mirrors
     claudenet/_clib.known_lens_catalogs but reads the sibling-reproduction copies
     that exist off-Perlmutter (claudenet/data is staged only on the cluster).
  2. LITERATURE (SIMBAD): cone search (15"); a lens-typed object within 6" marks the
     candidate KNOWN, with the survey inferred from the catalogue id.
  3. LITERATURE (NED), opt-in via --ned: for candidates still unmatched after SIMBAD,
     a NED cone search (15"); a NED object whose type contains "lens" within 6" marks
     it KNOWN. NED is slower/flakier than SIMBAD, so it is off by default.

This is the step the DR11 campaign's lineage-only crossmatch skipped (it also
omitted Huang+2020), which is why many already-published DES/AGEL/Huang lenses
were listed as new. Writes <out> with per-candidate status + the matched lens.
Requires astroquery + network for stages 2--3.

    python crosscheck_candidates.py --ned \
        --candidates data/dr11s_desi_resolution_AB.csv \
        --out data/dr11s_resolution_xmatch.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent          # reproductions/dr11-campaign
REPRO = ROOT.parent

LINEAGE = [
    ("Huang+2020", REPRO / "huang-2020/data/huang2020_published_catalog.csv"),
    ("Huang+2021", REPRO / "huang-2021/data/huang2021_published_catalog.csv"),
    ("Storfer+2024", REPRO / "inchausti-2025/data/storfer2024_published_catalog.csv"),
    ("Inchausti+2025", REPRO / "inchausti-2025/data/inchausti2025_published_catalog.csv"),
    ("external", REPRO / "claudenet/v3/external_lens_catalog.csv"),
]
SIMBAD_LENS = {"gLe", "gLS", "LeI", "LeG", "LeQ", "Le?", "LI?", "LS?"}
# id-prefix -> survey for SIMBAD hits (checked after stripping a leading "[REF] ").
SURVEY = [("CSWA", "CASSOWARY"), ("AGEL", "AGEL"), ("DES J", "DES"),
          ("DELS", "DECaLS/Huang"), ("SOGRAS", "SOGRAS"), ("SLACS", "SLACS"),
          ("DESI-", "DESI lens-finders")]


def sep_arcsec(ra1, d1, ra2, d2):
    r1, r2 = math.radians(ra1), math.radians(ra2)
    a1, a2 = math.radians(d1), math.radians(d2)
    a = math.sin((a2 - a1) / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def _col(fields, *names):
    m = {f.lower().strip(): f for f in fields if f}
    return next((m[n] for n in names if n in m), None)


def load(path):
    if not path.exists():
        return []
    rd = csv.DictReader(open(path))
    ra, de, nm = _col(rd.fieldnames, "ra"), _col(rd.fieldnames, "dec", "de"), _col(rd.fieldnames, "name", "id")
    out = []
    for r in rd:
        try:
            out.append((str(r.get(nm, "")), float(r[ra]), float(r[de])))
        except (TypeError, ValueError):
            pass
    return out


def survey_of(name):
    s = str(name).lstrip("[").upper()
    # skip a leading reference tag like "DBL2017] " before the catalogue id
    if "] " in str(name):
        s = str(name).split("] ", 1)[1].upper()
    for pre, sv in SURVEY:
        if s.startswith(pre.upper()):
            return sv
    return "literature (SIMBAD)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", default=str(ROOT / "data/dr11s_desi_resolution_AB.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/dr11s_resolution_xmatch.csv"))
    ap.add_argument("--lineage-radius", type=float, default=5.0)
    ap.add_argument("--simbad-radius", type=float, default=15.0)
    ap.add_argument("--simbad-match", type=float, default=6.0)
    ap.add_argument("--ned", action="store_true",
                    help="stage 3: also query NED for candidates still unmatched after SIMBAD")
    ap.add_argument("--ned-radius", type=float, default=15.0)
    ap.add_argument("--ned-match", type=float, default=6.0)
    args = ap.parse_args()

    cats = [(tag, load(p)) for tag, p in LINEAGE]
    rows = list(csv.DictReader(open(args.candidates)))
    rc = _col(rows[0].keys(), "ra")
    dc = _col(rows[0].keys(), "dec", "de")
    nc = _col(rows[0].keys(), "name", "id")
    gc = _col(rows[0].keys(), "grade_pred", "grade")
    pc = _col(rows[0].keys(), "p_lens")

    # stage 1: lineage + external
    results = []
    unmatched = []
    for r in rows:
        ra, dec = float(r[rc]), float(r[dc])
        best = (1e9, None, None)
        for tag, entries in cats:
            for cn, cra, cde in entries:
                if abs(cde - dec) > 0.05:
                    continue
                s = sep_arcsec(ra, dec, cra, cde)
                if s < best[0]:
                    best = (s, tag, cn)
        rec = {"name": str(r[nc]), "ra": f"{ra:.6f}", "dec": f"{dec:.6f}",
               "grade": r.get(gc, ""), "p_lens": r.get(pc, "")}
        if best[0] < args.lineage_radius:
            sv = best[1] if best[1] != "external" else survey_of(best[2])
            rec.update(status="known", counterpart=best[2], survey=sv,
                       sep_arcsec=f"{best[0]:.2f}", source="lineage")
            results.append(rec)
        else:
            unmatched.append((rec, ra, dec))

    # stage 2: SIMBAD literature for the lineage-unmatched
    if unmatched:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad
        sim = Simbad()
        for f in ("otype", "ra", "dec"):
            try:
                sim.add_votable_fields(f)
            except Exception:
                pass
        if args.ned:
            from astroquery.ipac.ned import Ned
            Ned.TIMEOUT = 120
        for rec, ra, dec in unmatched:
            ctr = SkyCoord(ra * u.deg, dec * u.deg)
            hit, src = None, None
            try:
                t = sim.query_region(ctr, radius=args.simbad_radius * u.arcsec)
                if t is not None and len(t):
                    mc, rcc, dcc, oc = (_col(t.colnames, "main_id"), _col(t.colnames, "ra"),
                                        _col(t.colnames, "dec"), _col(t.colnames, "otype"))
                    for row in t:
                        if str(row[oc]) not in SIMBAD_LENS and "lens" not in str(row[oc]).lower():
                            continue
                        s = ctr.separation(SkyCoord(float(row[rcc]) * u.deg, float(row[dcc]) * u.deg)).arcsec
                        if s < args.simbad_match and (hit is None or s < hit[0]):
                            hit, src = (s, str(row[mc]).strip(), str(row[oc])), "SIMBAD"
            except Exception as e:
                print(f"[xc] SIMBAD {rec['name']}: {type(e).__name__}", file=sys.stderr)
            # stage 3: NED for candidates still unmatched after SIMBAD (opt-in via --ned)
            if args.ned and hit is None:
                try:
                    nt = Ned.query_region(ctr, radius=args.ned_radius * u.arcsec)
                    if nt is not None and len(nt):
                        ncn, ntc, nrc, ndc = (_col(nt.colnames, "object name"), _col(nt.colnames, "type"),
                                              _col(nt.colnames, "ra"), _col(nt.colnames, "dec"))
                        for row in nt:
                            ty = row[ntc]
                            ty = ty.decode() if isinstance(ty, bytes) else str(ty)
                            if "lens" not in ty.lower():
                                continue
                            s = ctr.separation(SkyCoord(float(row[nrc]) * u.deg, float(row[ndc]) * u.deg)).arcsec
                            if s < args.ned_match and (hit is None or s < hit[0]):
                                nm = row[ncn]
                                nm = nm.decode() if isinstance(nm, bytes) else str(nm)
                                hit, src = (s, nm.strip(), ty), "NED"
                except Exception as e:
                    print(f"[xc] NED {rec['name']}: {type(e).__name__}", file=sys.stderr)
            if hit:
                rec.update(status="known", counterpart=hit[1], survey=survey_of(hit[1]),
                           sep_arcsec=f"{hit[0]:.2f}", source=src)
            else:
                rec.update(status="new", counterpart="", survey="", sep_arcsec="", source="")
            results.append(rec)

    # preserve input order by name
    order = {str(r[nc]): i for i, r in enumerate(rows)}
    results.sort(key=lambda x: order.get(x["name"], 1e9))
    fields = ["name", "ra", "dec", "grade", "p_lens", "status", "counterpart", "survey", "sep_arcsec", "source"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    nk = sum(r["status"] == "known" for r in results)
    by_survey = {}
    for r in results:
        if r["status"] == "known":
            by_survey[r["survey"]] = by_survey.get(r["survey"], 0) + 1
    print(f"[xc] {len(results)} candidates: {nk} KNOWN, {len(results) - nk} new")
    print(f"[xc] known by survey: {by_survey}")
    print(f"[xc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
