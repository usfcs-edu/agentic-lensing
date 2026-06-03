"""06 - Guided source-redshift demonstration (honest scope on the hard half).

The fully-unguided automated source finder in 04 is interloper-prone: across the 60"
MUSE FoV there are often field emission-line galaxies brighter than the faint lensed
arc, so the (z, position) that globally maximizes annular line flux can land on an
interloper rather than the arc (see data/measured_redshifts.csv: the unguided picks
miss the paper's source z for the faint arcs). This is exactly why Lin et al. did the
source IDs by hand.

This script demonstrates that the *extraction + line-ID machinery is correct*: when we
GUIDE the narrow-band line map to the [OII] 3727 line near the published source
redshift (i.e. supply the human prior "look for [OII] around this z"), the finder
locates the arc spaxel and the 1D emission engine recovers the source redshift. This
isolates the remaining gap to the automated *interloper disambiguation* problem, not
the spectroscopy.

Output: data/guided_source_redshifts.csv and figs/06_guided_source.png
"""
from pathlib import Path
import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
import astropy.units as u

import _zfinder as zf

warnings.filterwarnings("ignore")
REPRO = Path(__file__).parent
DATA = REPRO / "data"
CUBES = DATA / "cubes"
FIGS = REPRO / "figs"

# cube file, lens RA/Dec, published source z, name
CASES = [
    ("ADP.2022-07-10T06:32:06.934", 3.6745, -13.5042, 0.908, "Lens16/DESI J003.6745-13.5042"),
    ("ADP.2024-09-07T07:25:54.419", 60.5238, -22.0990, 0.821, "Lens22/DESI J060.5238-22.0990"),
    ("ADP.2023-08-24T10:45:16.017", 65.6453, -28.0646, 1.175, "Lens24/DESI J065.6453-28.0646"),
]
OII = 3727.4


def guided_source(cube_path, ra, dec, zpub):
    cube = Cube(cube_path)
    data = np.asarray(cube.data.filled(np.nan))
    ny, nx = data.shape[1:]
    w = cube.wave.coord()
    yc, xc = cube.wcs.sky2pix([[dec, ra]], unit=u.deg)[0]
    ps = abs(cube.wcs.get_step(unit=u.arcsec)[0])
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - yc, xx - xc) * ps
    ann = (r > 0.8) & (r < 6.0)
    # narrow-band [OII] map at the published z (the human prior)
    lam = OII * (1 + zpub)
    on = (w > lam - 6) & (w < lam + 6)
    off = ((w > lam - 40) & (w < lam - 15)) | ((w > lam + 15) & (w < lam + 40))
    nb = np.nanmean(data[on], axis=0) - np.nanmean(data[off], axis=0)
    mm = nb.copy(); mm[~ann] = -np.inf
    iy, ix = np.unravel_index(np.nanargmax(mm), mm.shape)
    sep = float(np.hypot(iy - yc, ix - xc) * ps)
    sky = cube.wcs.pix2sky([[iy, ix]], unit=u.deg)[0]
    sp = cube.aperture((sky[0], sky[1]), 0.8, unit_center=u.deg, unit_radius=u.arcsec)
    ws = np.asarray(sp.wave.coord(), float); fs = np.asarray(sp.data, float)
    vs = np.asarray(sp.var, float)
    z, score, zg, det = zf.find_z_emission(ws, fs, var=vs, zmin=zpub - 0.06, zmax=zpub + 0.06)
    return z, sep, ws, fs, score, zg, nb, (yc, xc), (iy, ix), ps


def main():
    rows = []
    fig, axes = plt.subplots(len(CASES), 2, figsize=(13, 3.2 * len(CASES)))
    for i, (dpid, ra, dec, zpub, name) in enumerate(CASES):
        path = CUBES / f"{dpid}.fits"
        if not path.exists():
            print(f"missing cube for {name}, skipping")
            continue
        z, sep, ws, fs, score, zg, nb, lpix, spix, ps = guided_source(str(path), ra, dec, zpub)
        dz = z - zpub
        print(f"{name:38} guided [OII] arc sep={sep:4.1f}\"  z={z:.4f} (pub {zpub})  dz={dz:+.4f}")
        rows.append(dict(name=name, z_source_pub=zpub, z_source_guided=round(float(z), 4),
                         dz=round(float(dz), 4), arc_sep_arcsec=round(sep, 1)))
        axes[i, 0].plot(ws, fs, lw=0.5, color="k")
        for nm, l0 in zf.EMIS_LINES.items():
            lam = l0 * (1 + z)
            if ws[0] < lam < ws[-1]:
                axes[i, 0].axvline(lam, color="C0", ls=":", lw=0.7)
        axes[i, 0].set_title(f"{name.split('/')[0]} guided arc spectrum (z={z:.4f}, pub {zpub})", fontsize=9)
        axes[i, 0].set_xlabel("obs wavelength [A]")
        axes[i, 1].plot(zg, score, lw=0.8)
        axes[i, 1].axvline(zpub, color="g", ls="--", label=f"pub {zpub}")
        axes[i, 1].axvline(z, color="r", ls=":", label=f"auto {z:.4f}")
        axes[i, 1].legend(fontsize=8); axes[i, 1].set_xlabel("trial z")
    fig.tight_layout()
    out = FIGS / "06_guided_source.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    if rows:
        f = DATA / "guided_source_redshifts.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {f}")


if __name__ == "__main__":
    main()
