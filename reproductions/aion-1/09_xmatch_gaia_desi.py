"""
09 -- Cross-match Gaia XP with DESI spectra of the same stars (task 11).

The Flatiron gaia_desi.fits convenience file is gone (403), so we build the set
from primary data: positional match (1 arcsec) between the MMU Gaia XP stars
(coefficients) and the DD-Payne DESI stellar set (08_fetch_ddpayne_desi.py, which
carries the DESI spectra). Matched stars give a Gaia-XP input and a ground-truth
DESI spectrum for the spectral super-resolution / cross-modal generation task.

Outputs (data/raw/gaia_desi/): xp_bp.npy (M,55), xp_rp.npy (M,55),
spec_flux/ivar/mask/wave.npy (M,L).

Run: HF_HOME=... python 09_xmatch_gaia_desi.py
"""

import numpy as np
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u

import _config as C
import _data_mmu as D

OUT = C.RAW / "gaia_desi"
DDP = C.RAW / "ddpayne"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C.seed_everything()
    # Gaia XP stars: re-read ra/dec + coeffs from the MMU parquet
    f = D.download_parquet("MultimodalUniverse/gaia", "gaia_dr3/train-00000-of-00001.parquet")
    t = pq.read_table(f, columns=["ra", "dec", "spectral_coefficients"])
    gra = np.asarray(t["ra"], float); gdec = np.asarray(t["dec"], float)
    coeff = np.array([s["coeff"] for s in t["spectral_coefficients"].to_pylist()], np.float32)

    dra = np.load(DDP / "ra.npy"); ddec = np.load(DDP / "dec.npy")
    gcat = SkyCoord(gra * u.deg, gdec * u.deg)
    dcat = SkyCoord(dra * u.deg, ddec * u.deg)
    # for each DESI star, nearest Gaia XP star
    idx, sep, _ = match_coordinates_sky(dcat, gcat)
    keep = sep.arcsec < 1.0
    d_sel = np.where(keep)[0]
    g_sel = idx[keep]
    print(f"matched {len(d_sel)} Gaia-XP × DESI stars (<1\")")
    if len(d_sel) == 0:
        print("NO GAIA×DESI OVERLAP"); return

    bp = coeff[g_sel, :55]; rp = coeff[g_sel, 55:110]
    flux = np.load(DDP / "spec_flux.npy", mmap_mode="r")[d_sel]
    ivar = np.load(DDP / "spec_ivar.npy", mmap_mode="r")[d_sel]
    mask = np.load(DDP / "spec_mask.npy", mmap_mode="r")[d_sel]
    wave = np.load(DDP / "spec_wave.npy", mmap_mode="r")[d_sel]
    np.save(OUT / "xp_bp.npy", bp.astype(np.float32))
    np.save(OUT / "xp_rp.npy", rp.astype(np.float32))
    np.save(OUT / "spec_flux.npy", np.asarray(flux, np.float32))
    np.save(OUT / "spec_ivar.npy", np.asarray(ivar, np.float32))
    np.save(OUT / "spec_mask.npy", np.asarray(mask, bool))
    np.save(OUT / "spec_wave.npy", np.asarray(wave, np.float32))
    print(f"XMATCH_GAIA_DESI_OK saved {len(d_sel)} matched stars")


if __name__ == "__main__":
    main()
