"""
03 -- Fetch the PROVABGS galaxy-property dataset (task 1 labels + LS photometry).

Downloads the single MultimodalUniverse/desi_provabgs parquet (robust
hf_hub_download, not streaming), prints its schema, maps the 5 AION Table-2
targets and the Legacy Survey g,r,i,z photometry, and saves field arrays to
data/raw/provabgs/. RA/Dec/TARGETID are kept so later steps can add the DESI
spectrum (key join on TARGETID) and the LS image (cutout by RA/Dec).

5 targets (paper): redshift z, log stellar mass, mass-weighted age,
log stellar metallicity, log sSFR.

Run: HF_HOME=... /home2/benson/.venvs/aion/bin/python 03_fetch_provabgs.py
"""

import json

import numpy as np

import _config as C
import _data_mmu as D

REPO = "MultimodalUniverse/desi_provabgs"
OUT = C.RAW / "provabgs"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    pq_files = D.list_parquet(REPO)
    print("parquet files:", pq_files)
    table = D.read_table(REPO, pq_files)
    cols = table.column_names
    print(f"\n{table.num_rows} rows, {len(cols)} columns:")
    for c in cols:
        print("   ", c, table.schema.field(c).type)

    fp = lambda names: D.first_present(names, cols)
    cid = fp(["object_id", "TARGETID"])
    cra, cdec = fp(["ra", "RA"]), fp(["dec", "DEC"])
    cz = fp(["Z_HP", "Z", "redshift", "zred"])
    cmass = fp(["LOG_MSTAR", "logmstar", "PROVABGS_LOGMSTAR"])
    cage = fp(["TAGE_MW", "PROVABGS_TAGE_MW", "age_mw"])
    cmet = fp(["Z_MW", "PROVABGS_Z_MW", "logzmw"])
    csfr = fp(["AVG_SFR", "PROVABGS_AVG_SFR", "logsfr"])
    # provabgs stores Legacy Survey magnitudes (no i-band); convert to flux for AION.
    phot_bands = ["G", "R", "Z", "W1"]
    cmag = {b: fp([f"MAG_{b}", f"mag_{b}"]) for b in phot_bands}
    mapping = dict(id=cid, ra=cra, dec=cdec, z=cz, logmass=cmass, age=cage,
                   metallicity=cmet, sfr=csfr, mag=cmag, phot_bands=phot_bands)
    print("\ncolumn mapping:", json.dumps(mapping, indent=2))
    assert all([cid, cz, cmass, cage, cmet, csfr]) and all(cmag.values()), \
        "missing required provabgs columns -- inspect schema above"

    z = D.col(table, cz).astype(np.float64)
    logmass = D.col(table, cmass).astype(np.float64)
    age = D.col(table, cage).astype(np.float64)
    zmw = D.col(table, cmet).astype(np.float64)
    sfr = D.col(table, csfr).astype(np.float64)
    # log metallicity and log specific-SFR (paper reports "log Z" and "sSFR")
    logZ = np.log10(np.clip(zmw, 1e-6, None))
    logsSFR = np.log10(np.clip(sfr, 1e-12, None)) - logmass
    targets = np.stack([z, logmass, age, logZ, logsSFR], axis=1)
    target_names = ["z", "logmass", "age", "logZ", "logsSFR"]

    # mag -> flux in nanomaggies: f = 10^((22.5 - mag)/2.5)
    mags = np.stack([D.col(table, cmag[b]).astype(np.float64) for b in phot_bands], axis=1)
    flux = (10.0 ** ((22.5 - mags) / 2.5)).astype(np.float32)  # (N,4) for G,R,Z,W1
    ra = D.col(table, cra).astype(np.float64)
    dec = D.col(table, cdec).astype(np.float64)
    tid = np.array([str(x) for x in D.col(table, cid)])

    # drop rows with non-finite targets/photometry
    good = np.isfinite(targets).all(1) & np.isfinite(flux).all(1) & (flux > 0).all(1)
    print(f"\nkept {good.sum()}/{len(good)} finite rows")
    for name, arr in [("targets", targets), ("flux", flux), ("ra", ra), ("dec", dec)]:
        np.save(OUT / f"{name}.npy", arr[good])
    np.save(OUT / "targetid.npy", tid[good])
    for j, n in enumerate(target_names):
        col_ = targets[good, j]
        print(f"  {n:9s} range [{np.nanmin(col_):.3f}, {np.nanmax(col_):.3f}] median {np.nanmedian(col_):.3f}")
    (OUT / "meta.json").write_text(json.dumps(
        {"n": int(good.sum()), "target_names": target_names, "mapping": mapping,
         "flux_bands": phot_bands}, indent=2))
    print("\nPROVABGS_FETCH_OK ->", OUT)


if __name__ == "__main__":
    main()
