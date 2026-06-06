"""
07 -- Cross-match Gaia XP (MMU) with APOGEE labels (task 3).

The Flatiron convenience file gaia_apogee.fits (used by StellarTutorial) is no
longer served (403), so we rebuild the APOGEE x GaiaXP set from primary sources:
  - Gaia XP continuous coefficients (110 = 55 BP + 55 RP) + source_id from the
    MultimodalUniverse/gaia dataset (data/raw/gaia/*.npy, written by an earlier step).
  - APOGEE DR17 ASPCAP stellar labels (TEFF, LOGG, M_H) keyed by
    GAIAEDR3_SOURCE_ID, from allStarLite-dr17-synspec_rev1.fits.

We inner-join on Gaia source_id, keep stars with finite labels and ASPCAP star
flag clean, and save aligned arrays. Targets mirror the tutorial's
teff50/logg50/met50 (here the ASPCAP point estimates).

Outputs (data/raw/apogee/): xp_bp.npy (N,55), xp_rp.npy (N,55),
targets.npy (N,3 = teff,logg,m_h), source_id.npy.

Run: python 07_xmatch_gaia_apogee.py
"""

import numpy as np
from astropy.table import Table
from astropy.io import fits

import _config as C

GAIA = C.RAW / "gaia"
APO = C.RAW / "apogee"


def main():
    C.seed_everything()
    sid = np.load(GAIA / "source_id.npy")          # (100000,)
    coeff = np.load(GAIA / "xp_coeff.npy")          # (100000,110)
    gaia_by_sid = {int(s): i for i, s in enumerate(sid)}

    # Read only the columns we need from the big APOGEE FITS.
    with fits.open(APO / "allStarLite-dr17.fits", memmap=True) as hdul:
        data = hdul[1].data
        cols = {c.upper(): c for c in data.columns.names}
        def getc(*cands):
            for c in cands:
                if c.upper() in cols:
                    return np.asarray(data[cols[c.upper()]])
            raise KeyError(cands)
        a_sid = getc("GAIAEDR3_SOURCE_ID", "GAIA_SOURCE_ID", "GAIADR2_SOURCE_ID")
        teff = getc("TEFF")
        logg = getc("LOGG")
        m_h = getc("M_H", "FE_H")
        flag = getc("ASPCAPFLAG") if "ASPCAPFLAG" in cols else np.zeros(len(a_sid), np.int64)

    # ASPCAP STAR_BAD bit = 23 (per DR17 bitmask); drop those.
    star_bad = (np.asarray(flag, dtype=np.int64) & (1 << 23)) != 0
    rows_g, rows_a = [], []
    for j in range(len(a_sid)):
        s = int(a_sid[j])
        i = gaia_by_sid.get(s)
        if i is None:
            continue
        if star_bad[j]:
            continue
        if not (np.isfinite(teff[j]) and np.isfinite(logg[j]) and np.isfinite(m_h[j])):
            continue
        if teff[j] < -1000 or logg[j] < -1000 or m_h[j] < -1000:  # APOGEE fill -9999
            continue
        rows_g.append(i); rows_a.append(j)

    rows_g = np.array(rows_g); rows_a = np.array(rows_a)
    c = coeff[rows_g]
    bp, rp = c[:, :55].astype(np.float32), c[:, 55:110].astype(np.float32)
    targets = np.stack([teff[rows_a], logg[rows_a], m_h[rows_a]], axis=1).astype(np.float32)
    np.save(APO / "xp_bp.npy", bp)
    np.save(APO / "xp_rp.npy", rp)
    np.save(APO / "targets.npy", targets)
    np.save(APO / "source_id.npy", sid[rows_g].astype(np.int64))
    print(f"XMATCH_GAIA_APOGEE_OK matched {len(rows_g)} stars "
          f"(teff {targets[:,0].min():.0f}-{targets[:,0].max():.0f}K)")


if __name__ == "__main__":
    main()
