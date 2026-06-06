"""
08b -- Attach Gaia DR3 parallax to the DD-Payne stars (task 2, +parallax config).

The paper's best stellar config is "DESI spectrum + parallax". DD-Payne carries
no parallax (only a spectro-photometric LOGDIS), and the MMU 100k Gaia subset
matches only ~2.9k of our 26k stars -- so we query the full Gaia DR3 archive by
position (1" upload cross-match via astroquery). Falls back to a LOGDIS-derived
parallax for any star Gaia doesn't return, so coverage is complete.

Output: data/raw/ddpayne/parallax.npy (N,), aligned to the spectrum/target rows.

Run: HF_HOME=... python 08b_fetch_parallax.py
"""

import numpy as np

import _config as C

OUT = C.RAW / "ddpayne"


def logdis_parallax():
    """Fallback parallax (mas) from DD-Payne LOGDIS = log10(distance/pc):
    plx_mas = 1000 / d_pc. NOTE: spectro-photometric distance derived from the
    same fit (mildly circular) -- only used if the Gaia cross-match fails."""
    from astropy.io import fits
    tid = np.load(OUT / "target_id.npy")
    with fits.open(OUT / "DESI_EDR_DDPAYNE.fits") as h:
        d = h[1].data
        cat = {int(t): i for i, t in enumerate(np.asarray(d["TARGET_ID"]).astype(np.int64))}
        logdis = np.asarray(d["LOGDIS"], float)
    rows = np.array([cat[int(t)] for t in tid])
    dpc = 10.0 ** logdis[rows]
    return 1000.0 / np.clip(dpc, 1.0, None)  # mas


def gaia_parallax():
    """Genuine Gaia DR3 parallax via the CDS XMatch service (robust for ~26k
    rows; Gaia's own TAP upload endpoint 500s on this size)."""
    import astropy.units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch

    ra = np.load(OUT / "ra.npy")
    dec = np.load(OUT / "dec.npy")
    t = Table({"uid": np.arange(len(ra)), "ra": ra.astype(float), "dec": dec.astype(float)})
    res = XMatch.query(cat1=t, cat2="vizier:I/355/gaiadr3",
                       max_distance=1.0 * u.arcsec, colRA1="ra", colDec1="dec")
    plx = np.full(len(ra), np.nan)
    best = {}
    for row in res:
        uid = int(row["uid"]); sep = float(row["angDist"])
        p = row["Plx"]
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        if uid not in best or sep < best[uid][0]:
            best[uid] = (sep, float(p))
    for uid, (_, p) in best.items():
        plx[uid] = p
    return plx


def main():
    plx_fallback = logdis_parallax()
    try:
        plx = gaia_parallax()
        n_gaia = int(np.isfinite(plx).sum())
        plx = np.where(np.isfinite(plx), plx, plx_fallback)
        print(f"Gaia matched {n_gaia}/{len(plx)} stars; rest use LOGDIS parallax")
    except Exception as e:
        print(f"Gaia query failed ({repr(e)[:100]}); using LOGDIS-derived parallax")
        plx = plx_fallback
    plx = np.where(np.isfinite(plx), plx, 0.0).astype(np.float32)
    np.save(OUT / "parallax.npy", plx)
    print(f"PARALLAX_OK saved {plx.shape}, median {np.median(plx):.3f} mas")


if __name__ == "__main__":
    main()
