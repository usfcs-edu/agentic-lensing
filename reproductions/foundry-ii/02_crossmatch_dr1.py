#!/usr/bin/env python
"""Foundry II - cross-match the 73 Table-2 systems to the DESI DR1 (Iron)
redshift catalog within 1.5".

Foundry II used DESI EDR (Fuji). DR1/Iron is a SUPERSET that re-reduces the
same SV/EDR tiles plus Year-1 main-survey tiles, so most EDR fibers reappear
in Iron (sometimes with a slightly different reduction).  Some EDR fibers /
special tiles may not be present in DR1 -> coverage caveat.

For each system we keep the closest fiber within 1.5".  Because both lens and
source were often observed on SEPARATE fibers a fraction of an arcsec apart,
we ALSO keep ALL fibers within 3" and tag whether any has a redshift matching
the published z_d or z_s (to within 0.005) so we can recover both.

Reads only the needed columns from the 22 GB FITS via a coarse RA/Dec box
prefilter, then astropy SkyCoord matching.
"""
import os, csv
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = os.path.dirname(__file__)
CAT = "/raid/benson/git/agentic-lensing/reproductions/hsu-2025/data/zall-pix-iron.fits"
TAB = os.path.join(HERE, "data/foundry_ii_table2.csv")

MATCH_RADIUS = 1.5 * u.arcsec   # primary fiber match (closest fiber = the lens)
# The lensed-source fiber is deliberately offset onto the arc and can sit up to
# ~5" from the lens centre (e.g. J149.8209 source fiber at 3.83", J179.0547 at
# 4.45"), so we use a wide radius to recover the SOURCE-bearing companion fiber.
WIDE_RADIUS = 5.0 * u.arcsec


def load_systems():
    rows = list(csv.DictReader(open(TAB)))
    for r in rows:
        r["ra_deg"] = float(r["ra_deg"])
        r["dec_deg"] = float(r["dec_deg"])
        for k in ("z_lens_pub", "z_source_pub", "z_s2"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
    return rows


def main():
    systems = load_systems()
    sys_ra = np.array([s["ra_deg"] for s in systems])
    sys_dec = np.array([s["dec_deg"] for s in systems])
    sys_coord = SkyCoord(sys_ra * u.deg, sys_dec * u.deg)

    # coarse box prefilter over the catalog (read RA/Dec columns only)
    print("Opening catalog (columns RA/DEC for prefilter)...")
    with fits.open(CAT, memmap=True) as f:
        h = f[1]
        ra_all = h.data["TARGET_RA"]
        dec_all = h.data["TARGET_DEC"]
        ra_all = np.asarray(ra_all, dtype=np.float64)
        dec_all = np.asarray(dec_all, dtype=np.float64)

        # box: within 3" -> ~0.001 deg; use generous 0.01 deg dec pad, ra/cos(dec)
        pad_dec = 0.01
        good = np.zeros(len(ra_all), dtype=bool)
        for ra0, dec0 in zip(sys_ra, sys_dec):
            pad_ra = pad_dec / max(np.cos(np.deg2rad(dec0)), 1e-3)
            m = (np.abs(dec_all - dec0) < pad_dec) & (np.abs(((ra_all - ra0 + 180) % 360) - 180) < pad_ra)
            good |= m
        idx = np.where(good)[0]
        print(f"Box prefilter: {len(idx)} candidate catalog rows near the 73 systems")

        # now read the other needed columns only for those rows
        sub = h.data[idx]
        cra = np.asarray(sub["TARGET_RA"], dtype=np.float64)
        cdec = np.asarray(sub["TARGET_DEC"], dtype=np.float64)
        cz = np.asarray(sub["Z"], dtype=np.float64)
        czerr = np.asarray(sub["ZERR"], dtype=np.float64)
        czwarn = np.asarray(sub["ZWARN"])
        cspectype = np.asarray(sub["SPECTYPE"])
        cprimary = np.asarray(sub["ZCAT_PRIMARY"])
        csurvey = np.asarray(sub["SURVEY"])
        cprogram = np.asarray(sub["PROGRAM"])
        ctargetid = np.asarray(sub["TARGETID"])

    cat_coord = SkyCoord(cra * u.deg, cdec * u.deg)

    out_rows = []
    for s, sc in zip(systems, sys_coord):
        sep = sc.separation(cat_coord)
        within = np.where(sep <= WIDE_RADIUS)[0]
        if len(within) == 0:
            out_rows.append({**_base(s), "n_fibers_3as": 0, "matched": False})
            continue
        # all fibers within 3", sorted by separation
        order = within[np.argsort(sep[within].arcsec)]
        fibers = []
        for j in order:
            fibers.append({
                "sep_arcsec": float(sep[j].arcsec),
                "z": float(cz[j]),
                "zerr": float(czerr[j]),
                "zwarn": int(czwarn[j]),
                "spectype": cspectype[j].strip() if isinstance(cspectype[j], str) else str(cspectype[j]),
                "primary": bool(cprimary[j]),
                "survey": csurvey[j].strip() if isinstance(csurvey[j], str) else str(csurvey[j]),
                "program": cprogram[j].strip() if isinstance(cprogram[j], str) else str(cprogram[j]),
                "targetid": int(ctargetid[j]),
            })
        # primary fiber = closest within 1.5"
        prim = next((fb for fb in fibers if fb["sep_arcsec"] <= MATCH_RADIUS.value), None)

        # recover z_lens / z_source: best fiber whose z matches the published value
        def best_match(zpub, tol=0.005):
            if zpub is None:
                return None
            cands = [fb for fb in fibers if abs(fb["z"] - zpub) <= tol]
            if not cands:
                return None
            return min(cands, key=lambda fb: abs(fb["z"] - zpub))

        zl_fb = best_match(s["z_lens_pub"])
        zs_fb = best_match(s["z_source_pub"])
        zs2_fb = best_match(s["z_s2"])

        row = _base(s)
        row.update({
            "matched": prim is not None,
            "n_fibers_3as": len(fibers),
            "closest_sep_arcsec": round(fibers[0]["sep_arcsec"], 3),
            "prim_z": prim["z"] if prim else None,
            "prim_zwarn": prim["zwarn"] if prim else None,
            "prim_spectype": prim["spectype"] if prim else None,
            "prim_primary": prim["primary"] if prim else None,
            "prim_survey": prim["survey"] if prim else None,
            "prim_program": prim["program"] if prim else None,
            # recovered-z matching to published
            "z_lens_dr1": zl_fb["z"] if zl_fb else None,
            "z_lens_dr1_zwarn": zl_fb["zwarn"] if zl_fb else None,
            "z_lens_dr1_sep": round(zl_fb["sep_arcsec"], 3) if zl_fb else None,
            "z_lens_dr1_spectype": zl_fb["spectype"] if zl_fb else None,
            "z_lens_match": zl_fb is not None,
            "z_source_dr1": zs_fb["z"] if zs_fb else None,
            "z_source_dr1_zwarn": zs_fb["zwarn"] if zs_fb else None,
            "z_source_dr1_sep": round(zs_fb["sep_arcsec"], 3) if zs_fb else None,
            "z_source_match": zs_fb is not None,
            "z_s2_dr1": zs2_fb["z"] if zs2_fb else None,
            "z_s2_match": zs2_fb is not None,
        })
        out_rows.append(row)

    # write
    out = os.path.join(HERE, "data/foundry_ii_dr1_crossmatch.csv")
    cols = list(out_rows[0].keys())
    # ensure all rows have all cols
    allcols = []
    for r in out_rows:
        for k in r:
            if k not in allcols:
                allcols.append(k)
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=allcols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k) for k in allcols})

    _summary(out_rows)
    print(f"\nWrote {out}")


def _base(s):
    return {
        "name": s["name"], "section": s["section"],
        "ra_deg": s["ra_deg"], "dec_deg": s["dec_deg"],
        "z_lens_pub": s["z_lens_pub"], "z_source_pub": s["z_source_pub"],
        "z_s2_pub": s["z_s2"],
        "sigma_v_pub": s["sigma_v_pub"], "sigma_v_err_pub": s["sigma_v_err_pub"],
    }


def _summary(rows):
    n = len(rows)
    matched = sum(1 for r in rows if r.get("matched"))
    any_fiber = sum(1 for r in rows if r.get("n_fibers_3as", 0) > 0)
    zl_pub = sum(1 for r in rows if r["z_lens_pub"] is not None)
    zs_pub = sum(1 for r in rows if r["z_source_pub"] is not None)
    zl_rec = sum(1 for r in rows if r.get("z_lens_match"))
    zs_rec = sum(1 for r in rows if r.get("z_source_match"))
    print("\n=== DR1 cross-match summary ===")
    print(f"systems total                : {n}")
    print(f"matched within 1.5\"          : {matched}")
    print(f"any fiber within 3\"          : {any_fiber}")
    print(f"published z_lens             : {zl_pub}")
    print(f"  recovered in DR1 (|dz|<0.005): {zl_rec}")
    print(f"published z_source           : {zs_pub}")
    print(f"  recovered in DR1 (|dz|<0.005): {zs_rec}")


if __name__ == "__main__":
    main()
