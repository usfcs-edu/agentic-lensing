"""
08 -- Build the DESI x DD-Payne stellar set (task 2).

Cross-matches the Zhang+2024 DD-Payne DESI-EDR stellar-label catalogue
(DESI_EDR_DDPAYNE.fits; TARGET_ID, TEFF, LOGG, FEH, VMIC) to the DESI spectra in
MultimodalUniverse/desi (edr_sv3) by object_id == TARGET_ID (exact integer key,
same join as the PROVABGS spectra). Keeps clean labels (CHISQ_FLAG==0, finite),
saves the DESI spectrum arrays and the aligned 4 stellar targets.

Outputs (data/raw/ddpayne/): targets.npy (N,4 = Teff,logg,FeH,vmic),
spec_flux/ivar/mask/wave.npy (N,L), target_id.npy, ra.npy, dec.npy.

Run: HF_HOME=... python 08_fetch_ddpayne_desi.py
"""

import json

import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits

import _config as C
import _data_mmu as D

REPO = "MultimodalUniverse/desi"
OUT = C.RAW / "ddpayne"
TARGET_NAMES = ["Teff", "logg", "FeH", "vmic"]


def _struct_field(tab, struct, field):
    arr = tab.column(struct)
    try:
        return arr.field(field)
    except Exception:
        return arr.combine_chunks().field(field)


def main():
    with fits.open(OUT / "DESI_EDR_DDPAYNE.fits") as h:
        d = h[1].data
        tid = np.asarray(d["TARGET_ID"]).astype(np.int64)
        teff = np.asarray(d["TEFF"], float); logg = np.asarray(d["LOGG"], float)
        feh = np.asarray(d["FEH"], float); vmic = np.asarray(d["VMIC"], float)
        ra = np.asarray(d["TARGET_RA"], float); dec = np.asarray(d["TARGET_DEC"], float)
        chisq_flag = np.asarray(d["CHISQ_FLAG"], float)
    # clean labels: DD-Payne uses -999 as a fill value, [Fe/H] is legitimately
    # negative, and CHISQ_FLAG is a continuous fit-quality metric (not a 0/1 flag;
    # a handful are +inf for failed fits). Keep stars with real labels and finite
    # chi-square.
    def _valid(a):
        return np.isfinite(a) & (a > -990)
    good = _valid(teff) & _valid(logg) & _valid(feh) & _valid(vmic) & \
        np.isfinite(chisq_flag) & (teff > 0)
    lab = {int(t): k for k, t in enumerate(tid)}  # tid -> row in catalog
    print(f"DD-Payne stars: {len(tid)} ({int(good.sum())} clean labels)")

    files = [f for f in D.list_parquet(REPO) if "edr_sv3" in f] or D.list_parquet(REPO)
    cat_rows, flux, ivar, mask, wave = [], [], [], [], []
    for fi, fn in enumerate(files):
        path = D.download_parquet(REPO, fn)
        t = pq.read_table(path, columns=["object_id", "spectrum"])
        oids = t.column("object_id").to_pylist()
        keep = []
        for j, o in enumerate(oids):
            r = lab.get(int(o))
            if r is not None and good[r]:
                keep.append((j, r))
        if not keep:
            print(f"  shard {fi}: 0 / {len(oids)} match"); continue
        jj = [k[0] for k in keep]
        sub = t.take(jj)
        flux.append(np.asarray(_struct_field(sub, "spectrum", "flux").to_pylist(), np.float32))
        ivar.append(np.asarray(_struct_field(sub, "spectrum", "ivar").to_pylist(), np.float32))
        mask.append(np.asarray(_struct_field(sub, "spectrum", "mask").to_pylist(), bool))
        wave.append(np.asarray(_struct_field(sub, "spectrum", "lambda").to_pylist(), np.float32))
        cat_rows.extend(k[1] for k in keep)
        print(f"  shard {fi}: matched {len(keep)} / {len(oids)} (cum {sum(len(x) for x in flux)})",
              flush=True)

    if not cat_rows:
        print("NO OVERLAP DD-Payne x DESI sv3"); return
    cat_rows = np.array(cat_rows)
    flux = np.concatenate(flux); ivar = np.concatenate(ivar)
    mask = np.concatenate(mask); wave = np.concatenate(wave)
    targets = np.stack([teff[cat_rows], logg[cat_rows], feh[cat_rows], vmic[cat_rows]], 1).astype(np.float32)
    np.save(OUT / "targets.npy", targets)
    np.save(OUT / "spec_flux.npy", flux); np.save(OUT / "spec_ivar.npy", ivar)
    np.save(OUT / "spec_mask.npy", mask); np.save(OUT / "spec_wave.npy", wave)
    np.save(OUT / "target_id.npy", tid[cat_rows]); np.save(OUT / "ra.npy", ra[cat_rows])
    np.save(OUT / "dec.npy", dec[cat_rows])
    (OUT / "meta.json").write_text(json.dumps(
        {"n": int(len(cat_rows)), "spec_len": int(flux.shape[1]),
         "targets": TARGET_NAMES}, indent=2))
    print(f"DDPAYNE_DESI_OK matched {len(cat_rows)} stars, spec_len {flux.shape[1]}")


if __name__ == "__main__":
    main()
