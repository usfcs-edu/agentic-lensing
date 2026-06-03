#!/usr/bin/env python
"""Foundry II - recover the lens velocity dispersion (sigma_v) for each system
from the on-disk FastSpecFit DR1 (Iron) v3.0 sigmav shards, matched by TARGETID.

Foundry II Table-2 sigma_v comes from FastSpecFit (paper sec 4.4): the VDISP of
the LENS spectrum.  We:
  1. collect the TARGETID of every DR1 fiber within 1.5" of each system (the
     lens-centred fibers),
  2. look those TARGETIDs up across all fastspec-iron-*.sigmav.parquet shards,
  3. keep VDISP where VDISP_IVAR>0 (IVAR==0 -> VDISP pinned at the 250 km/s
     default = not fit),
  4. compare recovered VDISP to the published sigma_v.

Note: FastSpecFit reports VDISP per (survey, program, healpix) spectrum; the
same target observed on several tiles can have several VDISP values.  We report
the one with the highest VDISP_IVAR (best-measured), and also the one matching
the EDR/SV3 spectrum where available.
"""
import os, csv, glob
import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = os.path.dirname(__file__)
CAT = "/raid/benson/git/agentic-lensing/reproductions/hsu-2025/data/zall-pix-iron.fits"
FSF = "/raid/benson/git/agentic-lensing/reproductions/hsu-2025/data/fastspecfit/"
TAB = os.path.join(HERE, "data/foundry_ii_table2.csv")
LENS_RADIUS = 1.5 * u.arcsec


def load_systems():
    rows = list(csv.DictReader(open(TAB)))
    for r in rows:
        r["ra_deg"] = float(r["ra_deg"]); r["dec_deg"] = float(r["dec_deg"])
        r["sigma_v_pub"] = int(r["sigma_v_pub"]) if r["sigma_v_pub"] not in ("", "None") else None
        r["sigma_v_err_pub"] = int(r["sigma_v_err_pub"]) if r["sigma_v_err_pub"] not in ("", "None") else None
    return rows


def lens_targetids(systems):
    """Return dict name -> set of TARGETIDs of fibers within 1.5" (lens fibers)."""
    sys_ra = np.array([s["ra_deg"] for s in systems])
    sys_dec = np.array([s["dec_deg"] for s in systems])
    with fits.open(CAT, memmap=True) as f:
        h = f[1]
        ra = np.asarray(h.data["TARGET_RA"], float)
        dec = np.asarray(h.data["TARGET_DEC"], float)
        pad = 0.001
        good = np.zeros(len(ra), bool)
        for ra0, dec0 in zip(sys_ra, sys_dec):
            padra = pad / max(np.cos(np.deg2rad(dec0)), 1e-3)
            good |= (np.abs(dec - dec0) < pad) & (np.abs(((ra - ra0 + 180) % 360) - 180) < padra)
        idx = np.where(good)[0]
        sub = h.data[idx]
        cra = np.asarray(sub["TARGET_RA"], float)
        cdec = np.asarray(sub["TARGET_DEC"], float)
        ctid = np.asarray(sub["TARGETID"])
    cc = SkyCoord(cra * u.deg, cdec * u.deg)
    out = {}
    for s in systems:
        sc = SkyCoord(s["ra_deg"] * u.deg, s["dec_deg"] * u.deg)
        sep = sc.separation(cc)
        m = sep <= LENS_RADIUS
        out[s["name"]] = set(int(t) for t in ctid[m])
    return out


def load_sigmav():
    """Load every sigmav shard into one TARGETID -> best (VDISP,IVAR) map.
    Keep the row with the highest VDISP_IVAR per TARGETID (best measurement)."""
    best = {}  # tid -> (vdisp, ivar)
    shards = sorted(glob.glob(FSF + "*.sigmav.parquet"))
    print(f"Loading {len(shards)} sigmav shards...")
    for sh in shards:
        t = pq.read_table(sh, columns=["TARGETID", "VDISP", "VDISP_IVAR"])
        tid = t.column("TARGETID").to_numpy()
        vd = t.column("VDISP").to_numpy()
        iv = t.column("VDISP_IVAR").to_numpy()
        good = iv > 0
        for ti, v, i in zip(tid[good], vd[good], iv[good]):
            ti = int(ti)
            if ti not in best or i > best[ti][1]:
                best[ti] = (float(v), float(i))
    print(f"  {len(best)} TARGETIDs with VDISP_IVAR>0")
    return best


def main():
    systems = load_systems()
    tids = lens_targetids(systems)
    sigmav = load_sigmav()

    out_rows = []
    nrec = 0
    for s in systems:
        cand = []
        for ti in tids.get(s["name"], ()):
            if ti in sigmav:
                v, iv = sigmav[ti]
                cand.append((v, iv, ti))
        rec_vdisp = rec_err = rec_tid = None
        if cand:
            # best-measured (highest IVAR)
            v, iv, ti = max(cand, key=lambda c: c[1])
            rec_vdisp = round(v, 1)
            rec_err = round(1.0 / np.sqrt(iv), 1) if iv > 0 else None
            rec_tid = ti
            nrec += 1
        dv = (rec_vdisp - s["sigma_v_pub"]) if (rec_vdisp is not None and s["sigma_v_pub"] is not None) else None
        out_rows.append({
            "name": s["name"], "section": s["section"],
            "sigma_v_pub": s["sigma_v_pub"], "sigma_v_err_pub": s["sigma_v_err_pub"],
            "vdisp_dr1_fsf": rec_vdisp, "vdisp_err_dr1_fsf": rec_err,
            "vdisp_targetid": rec_tid, "n_lens_fibers_with_vdisp": len(cand),
            "delta_sigma": round(dv, 1) if dv is not None else None,
        })

    out = os.path.join(HERE, "data/foundry_ii_sigmav.csv")
    cols = list(out_rows[0].keys())
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    npub = sum(1 for r in out_rows if r["sigma_v_pub"] is not None)
    print(f"\n=== sigma_v (FastSpecFit DR1) summary ===")
    print(f"systems with published sigma_v : {npub}")
    print(f"recovered VDISP (IVAR>0) in DR1 : {nrec}")
    dvs = np.array([r["delta_sigma"] for r in out_rows if r["delta_sigma"] is not None])
    if len(dvs):
        print(f"  |delta sigma| median={np.median(np.abs(dvs)):.1f} km/s  max={np.abs(dvs).max():.1f}")
        print(f"  within 10 km/s : {int((np.abs(dvs)<=10).sum())}/{len(dvs)}")
        print(f"  within 30 km/s : {int((np.abs(dvs)<=30).sum())}/{len(dvs)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
