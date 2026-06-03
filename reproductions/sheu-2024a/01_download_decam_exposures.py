#!/usr/bin/env python3
"""
01_download_decam_exposures.py  --  Sheu+2024a (Paper II) §2 data collection

Same per-exposure DECam access path as Paper I (Sheu+2023) -- this is a direct
adaptation of reproductions/sheu-2023/01_download_decam_exposures.py.  The crux,
solved there, is that the variability search of Paper II also runs on *individual*
DECam InstCal exposures (not Legacy Survey coadds), pulled from the NOIRLab Astro
Data Archive CCD-level SIA service (`/api/sia/vohdu`).

Difference from Paper I
-----------------------
Paper II targets *lensed quasars* (persistent, stochastically variable point
sources at the lensed-image positions) rather than one-off lensed supernovae.
The target here is a *known/confirmed lensed quasar candidate* with multi-epoch
DECam coverage, drawn from the paper's recovered list (Table 2).  Default:

    DESI-038.0655-24.4942  --  Grade-A double, image sep 1.54", DES footprint,
    Dawes+2023 (D22) / He+2023 (H23) lensed-quasar candidate; Sheu+2024a Table 2
    reports <sigma> = 0.25 mag, r_band = 18.39.

We also default to a larger --max-per-band so the ~5-yr DECam light curve is
sampled like the paper (their systems have ~20-50 passes/band).

Why we download whole frames (unchanged from Paper I)
-----------------------------------------------------
The archive exposes no positional cutout service and ignores byte-range
requests, and the SIA `?hdus=` covering-CCD hint is unreliable.  The robust path
is: retrieve the full funpacked InstCal frame, find the covering CCD
deterministically by testing every CCD's WCS for the target pixel, slice an
801x801 @ 0.262"/pix stamp, discard the frame.  Net on-disk footprint is a few
MB per epoch.

Usage
-----
  python 01_download_decam_exposures.py --bands g r z --max-per-band 40
  python 01_download_decam_exposures.py --ra 38.0655 --dec -24.4942 --bands g r i z

Outputs (identical schema to Paper I)
-------
  data/sia_manifest.csv          full SIA discovery table (all ooi products)
  data/exposures/<stem>_img.fits 801x801 science cutout (nanomaggie, +WCS)
  data/exposures/<stem>_wt.fits  inverse-variance cutout (from oow product)
  data/exposures/<stem>_dq.fits  data-quality cutout (from ood product)
  data/exposure_manifest.csv     per-cutout metadata
"""
from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP = DATA / "exposures"
EXP.mkdir(parents=True, exist_ok=True)

# Sheu+2024a Table 2 (sec 4.2): Grade-A double lensed-quasar candidate
# (Dawes+2023 / He+2023), DES footprint; <sigma>=0.25 mag, image sep 1.54".
CAND_RA = 38.0655
CAND_DEC = -24.4942
CAND_NAME = "DESI-038.0655-24.4942"

SIA = "https://astroarchive.noirlab.edu/api/sia/vohdu"
RETRIEVE = "https://astroarchive.noirlab.edu/api/retrieve/{fid}/"

PIXSCALE = 0.262
CUTOUT_PIX = 801


def discover(ra: float, dec: float, size_deg: float = 0.02) -> pd.DataFrame:
    params = {"POS": f"{ra},{dec}", "SIZE": f"{size_deg}"}
    r = requests.get(SIA, params=params, timeout=180)
    r.raise_for_status()
    m = re.search(r"(\d+) matches found", r.text)
    print(f"[sia] {m.group(1) if m else '?'} HDU matches at {ra},{dec}")
    tab = parse_single_table(io.BytesIO(r.content)).to_table()
    rows = []
    for i in range(len(tab)):
        fn = str(tab["archive_filename"][i])
        base = fn.split("/")[-1]
        if "c4d_" not in base or "_ooi_" not in base:
            continue
        parts = base.split("_")  # c4d YYMMDD HHMMSS ooi <filt> <ver>.fits.fz
        if len(parts) < 6:
            continue
        url = str(tab["url"][i])
        fid_m = re.search(r"([0-9a-f]{32})", url)
        rows.append(
            dict(
                base=base,
                fid=fid_m.group(1) if fid_m else None,
                band=parts[4],
                expid=f"{parts[1]}_{parts[2]}",
                version=parts[5].split(".")[0],
                dateobs=str(tab["file_dateobs"][i]),
            )
        )
    df = pd.DataFrame(rows).drop_duplicates("base").reset_index(drop=True)

    # dedup by (expid, band): prefer v1 / ls9 / ls10 reductions over vx
    def rank(v: str) -> int:
        return {"v1": 0, "ls9": 1, "ls10": 1}.get(v, 5)

    df["rk"] = df.version.map(rank)
    df = (df.sort_values(["expid", "band", "rk"])
            .drop_duplicates(["expid", "band"])
            .drop(columns="rk").reset_index(drop=True))
    return df


def find_covering_ccd(hdul, sc: SkyCoord):
    """Return (hdu_index, WCS, x, y) of the CCD whose array contains sc, else None."""
    for j in range(1, len(hdul)):
        h = hdul[j].header
        if h.get("NAXIS") != 2 or "CRVAL1" not in h:
            continue
        try:
            w = WCS(h)
            x, y = w.world_to_pixel(sc)
        except Exception:
            continue
        if 0 <= x < h["NAXIS1"] and 0 <= y < h["NAXIS2"]:
            return j, w, float(x), float(y)
    return None


def fetch_frame(fid: str, timeout: int = 200) -> fits.HDUList | None:
    url = RETRIEVE.format(fid=fid)
    for attempt in range(1, 4):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(20)
                continue
            r.raise_for_status()
            return fits.open(io.BytesIO(r.content))
        except Exception as e:
            if attempt == 3:
                print(f"   [frame fail] {fid}: {e}")
                return None
            time.sleep(4 * attempt)
    return None


def sibling_fid(base: str, df_all: pd.DataFrame, token: str) -> str | None:
    """Find the oow/ood product matching an ooi base (same expid, swap token)."""
    sib = base.replace("_ooi_", f"_{token}_")
    hit = df_all[df_all.base == sib]
    return hit.iloc[0].fid if len(hit) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, default=CAND_RA)
    ap.add_argument("--dec", type=float, default=CAND_DEC)
    ap.add_argument("--bands", nargs="+", default=["g", "r", "z"])
    ap.add_argument("--max-per-band", type=int, default=40)
    ap.add_argument("--size-deg", type=float, default=0.02)
    args = ap.parse_args()

    sc = SkyCoord(args.ra * u.deg, args.dec * u.deg)

    # discover EVERYTHING (incl. oow/ood) so we can find weight/dq siblings
    params = {"POS": f"{args.ra},{args.dec}", "SIZE": f"{args.size_deg}"}
    r = requests.get(SIA, params=params, timeout=180)
    tab = parse_single_table(io.BytesIO(r.content)).to_table()
    all_rows = []
    for i in range(len(tab)):
        fn = str(tab["archive_filename"][i])
        base = fn.split("/")[-1]
        if "c4d_" not in base:
            continue
        url = str(tab["url"][i])
        fid_m = re.search(r"([0-9a-f]{32})", url)
        all_rows.append(dict(base=base, fid=fid_m.group(1) if fid_m else None))
    df_all = pd.DataFrame(all_rows).drop_duplicates("base")
    df_all.to_csv(DATA / "sia_manifest.csv", index=False)

    df = discover(args.ra, args.dec, args.size_deg)
    df = df[df.band.isin(args.bands)].copy()
    print(f"[plan] {len(df)} unique ooi products in bands {args.bands}: "
          f"{df.band.value_counts().to_dict()}")

    keep = []
    for b in args.bands:
        keep.append(df[df.band == b].sort_values("dateobs").head(args.max_per_band))
    df = pd.concat(keep).reset_index(drop=True)

    meta_rows = []
    for _, row in df.iterrows():
        if row.fid is None:
            continue
        stem = row.base.replace(".fits.fz", "")
        img_out = EXP / f"{stem}_img.fits"
        if img_out.exists() and img_out.stat().st_size > 2880:
            print(f"   [skip] {stem} (cached)")
        else:
            t0 = time.time()
            hd = fetch_frame(row.fid)
            if hd is None:
                continue
            cov = find_covering_ccd(hd, sc)
            if cov is None:
                print(f"   [off-array] {stem} ({round(time.time()-t0,1)}s) "
                      f"-- point not on any CCD; skipping")
                hd.close()
                continue
            j, w, x, y = cov
            ph = hd[0].header
            img_hdu = hd[j]
            cut = Cutout2D(img_hdu.data.astype("float32"), sc,
                           (CUTOUT_PIX, CUTOUT_PIX), wcs=w,
                           mode="partial", fill_value=np.nan)
            out_hdr = cut.wcs.to_header()
            for k in ("FILTER", "MJD-OBS", "EXPTIME", "EXPREQ", "MAGZERO",
                      "MAGZPT", "SEEING", "FWHM", "OBJECT", "EXPNUM"):
                if k in ph:
                    out_hdr[k] = ph[k]
                elif k in img_hdu.header:
                    out_hdr[k] = img_hdu.header[k]
            out_hdr["EXTNAME"] = img_hdu.header.get("EXTNAME", "")
            out_hdr["TGT_X"] = (cut.input_position_cutout[0], "target x in cutout")
            out_hdr["TGT_Y"] = (cut.input_position_cutout[1], "target y in cutout")
            fits.PrimaryHDU(cut.data, out_hdr).writeto(img_out, overwrite=True)

            # weight (inverse variance) and DQ siblings -> same covering HDU index
            for token, suff in (("oow", "wt"), ("ood", "dq")):
                sfid = sibling_fid(row.base, df_all, token)
                if sfid is None:
                    continue
                shd = fetch_frame(sfid)
                if shd is None:
                    continue
                ext = img_hdu.header.get("EXTNAME")
                sj = None
                for k in range(1, len(shd)):
                    if shd[k].header.get("EXTNAME") == ext:
                        sj = k
                        break
                if sj is None:
                    shd.close()
                    continue
                sw = WCS(shd[sj].header)
                scut = Cutout2D(shd[sj].data.astype("float32"), sc,
                                (CUTOUT_PIX, CUTOUT_PIX), wcs=sw,
                                mode="partial", fill_value=0.0)
                fits.PrimaryHDU(scut.data, scut.wcs.to_header()).writeto(
                    EXP / f"{stem}_{suff}.fits", overwrite=True)
                shd.close()
            print(f"   [ok] {stem} CCD={img_hdu.header.get('EXTNAME')} "
                  f"({round(time.time()-t0,1)}s)")
            hd.close()

        with fits.open(img_out) as hd:
            h = hd[0].header
            meta_rows.append(dict(
                stem=stem, band=row.band, expid=row.expid, version=row.version,
                dateobs=row.dateobs,
                mjd=h.get("MJD-OBS", np.nan),
                exptime=h.get("EXPTIME", h.get("EXPREQ", np.nan)),
                magzero=h.get("MAGZERO", h.get("MAGZPT", np.nan)),
                seeing=h.get("SEEING", h.get("FWHM", np.nan)),
                ccd=h.get("EXTNAME", ""),
                has_wt=(EXP / f"{stem}_wt.fits").exists(),
                has_dq=(EXP / f"{stem}_dq.fits").exists(),
                img=str(img_out),
            ))

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(DATA / "exposure_manifest.csv", index=False)
    print(f"\n[done] {len(meta)} exposure cutouts -> {DATA/'exposure_manifest.csv'}")
    if len(meta):
        print(meta.groupby("band").size().to_dict())


if __name__ == "__main__":
    main()
