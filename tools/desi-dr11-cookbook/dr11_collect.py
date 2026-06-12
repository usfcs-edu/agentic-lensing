"""
dr11_collect.py -- core library for deriving "One_percent"-style HDF5 data from DR11.

Reproduces, bit-for-bit, the data product under
    /global/cfs/projectdirs/cosmo/work/users/xhuang/dr11/One_percent
from the official Legacy Survey DR11 archive under
    /global/cfs/cdirs/cosmo/data/legacysurvey/dr11

For each brick that has >=1 surviving object it writes two files:

  <out>/<region>/tractor/<RRR>/<brick>.h5
      a pandas DataFrame (PyTables 'fixed' format, key='tractor') with EXACTLY 14
      columns in the canonical order below, one row per surviving object.

  <out>/<region>/coadd/<RRR>/<brick>.h5
      one PyTables CArray per surviving object, named 'o<objid>', float32
      shape (101, 101, 3) = a 101x101 grz image cutout (channel order g, r, z),
      raw nanomaggies copied verbatim from the coadd image, stored TRANSPOSED.

All numeric conventions (column order, dtypes, str object columns, int64-truncated
bx/by, the cutout transpose, raw-nanomaggy pixels) were verified bit-exact against
the existing product on golden brick north/2400p345.  See README.md.
"""

import os
import numpy as np
import pandas as pd
import fitsio
import tables

# --- spec constants (verified against the golden product) -------------------

BANDS = ("g", "r", "z")                      # coadd channel order: 0=g, 1=r, 2=z
SEL_TYPES = ("SER", "EXP", "DEV", "REX")     # ObjectFilters: morphological types kept
HALF = 50                                    # cutout half-size; full window = 2*HALF+1 = 101
IMG_SIZE = 3600                              # DR11 coadd image is 3600 x 3600 px
NOBS_MIN = 3
MAG_Z_MAX = 20.0

# Canonical output column order (differs from the <MetaData> order in config.log):
COLS = ["brickid", "brickname", "objid", "type", "ra", "dec", "bx", "by",
        "nobs_g", "nobs_r", "nobs_z", "mag_g", "mag_r", "mag_z"]

# Only these tractor columns are read from the (186-column) FITS catalog:
NEEDED_TRACTOR_COLS = ["brickid", "brickname", "objid", "type", "ra", "dec",
                       "bx", "by", "nobs_g", "nobs_r", "nobs_z",
                       "flux_g", "flux_r", "flux_z", "brick_primary"]


# --- small helpers ----------------------------------------------------------

def flux2mag(flux):
    """AB magnitude from Legacy Survey flux (nanomaggies), float32 throughout.

    mag = 22.5 - 2.5*log10(flux).  Non-positive flux -> nan (the flux>0 filter
    removes those rows anyway).  Computed in float32 throughout (the standard,
    closest-matching form).

    NOTE: this matches the stored One_percent mag_* to float32 precision (<=1 ULP,
    ~1.9e-6 mag), NOT bit-for-bit -- log10's last-bit rounding differs across
    numpy/libm builds, so the original run's exact bits aren't recoverable. The
    difference is sub-micromag (~4000x below photometric precision).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return (22.5 - 2.5 * np.log10(flux)).astype(np.float32)


def brick_rrr(brick):
    """3-digit RA-strip subdir (== BatchID) for a brick name, e.g. '2400p345' -> '240'."""
    return f"{int(float(brick[:4]) / 10.0):03d}"


def bricks_for_batch(data_root, batchid):
    """Sorted brick names present in tractor/<RRR>/ for a given BatchID (0..359)."""
    d = os.path.join(data_root, "tractor", f"{int(batchid):03d}")
    return sorted(b[len("tractor-"):-len(".fits")]
                  for b in os.listdir(d)
                  if b.startswith("tractor-") and b.endswith(".fits"))


# --- selection + dataframe --------------------------------------------------

def select_mask(t):
    """Boolean mask of objects passing the 5 ObjectFilters (logical AND)."""
    typ = np.char.strip(t["type"].astype("U"))
    m = t["brick_primary"].astype(bool).copy()
    m &= (t["flux_g"] > 0) & (t["flux_r"] > 0) & (t["flux_z"] > 0)
    m &= (t["nobs_g"] >= NOBS_MIN) & (t["nobs_r"] >= NOBS_MIN) & (t["nobs_z"] >= NOBS_MIN)
    m &= flux2mag(t["flux_z"]) < MAG_Z_MAX
    m &= np.isin(typ, SEL_TYPES)
    return m


def build_dataframe(t, mask):
    """Build the 14-column output DataFrame (exact order + dtypes) for masked rows."""
    s = t[mask]
    df = pd.DataFrame({
        "brickid":   s["brickid"].astype(np.int32),
        # brickname/type as Python str objects (stripped), matching the stored product
        "brickname": [str(x) for x in np.char.strip(s["brickname"].astype("U"))],
        "objid":     s["objid"].astype(np.int32),
        "type":      [str(x) for x in np.char.strip(s["type"].astype("U"))],
        "ra":        s["ra"].astype(np.float64),
        "dec":       s["dec"].astype(np.float64),
        "bx":        s["bx"].astype(np.int64),     # truncation toward zero (3123.86 -> 3123)
        "by":        s["by"].astype(np.int64),
        "nobs_g":    s["nobs_g"].astype(np.int16),
        "nobs_r":    s["nobs_r"].astype(np.int16),
        "nobs_z":    s["nobs_z"].astype(np.int16),
        "mag_g":     flux2mag(s["flux_g"]),
        "mag_r":     flux2mag(s["flux_r"]),
        "mag_z":     flux2mag(s["flux_z"]),
    })[COLS]
    return df


# --- image cutouts ----------------------------------------------------------

def load_coadd_images(data_root, brick):
    """Read the g/r/z coadd images (float32, [y, x]) for a brick into a dict."""
    rrr = brick_rrr(brick)
    d = os.path.join(data_root, "coadd", rrr, brick)
    return {b: fitsio.read(os.path.join(d, f"legacysurvey-{brick}-image-{b}.fits.fz"))
            for b in BANDS}


def make_cube(images, bx, by, half=HALF):
    """101x101x3 float32 cutout centered on (bx, by), channels g,r,z, TRANSPOSED.

    cube[:, :, k] = image_k[by-half:by+half+1, bx-half:bx+half+1].T
    so that cube[half, half, k] == image_k[by, bx].
    """
    n = 2 * half + 1
    cube = np.empty((n, n, 3), dtype=np.float32)
    for k, b in enumerate(BANDS):
        win = images[b][by - half:by + half + 1, bx - half:bx + half + 1]
        cube[:, :, k] = win.T
    return cube


def write_tractor_h5(df, path):
    """Write the catalog as a pandas 'fixed'-format HDF5 frame (key='tractor')."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_hdf(path, key="tractor", mode="w", format="fixed")


def write_coadd_h5(images, df, path):
    """Write one CArray 'o<objid>' (101,101,3) float32 per object, no compression."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    objid = df["objid"].to_numpy()
    bx = df["bx"].to_numpy()
    by = df["by"].to_numpy()
    with tables.open_file(path, "w") as h5:
        for oid, x, y in zip(objid, bx, by):
            cube = make_cube(images, int(x), int(y))
            h5.create_carray("/", f"o{int(oid)}", obj=cube)


# --- per-brick driver -------------------------------------------------------

def process_brick(brick, data_root, out_root):
    """Process one brick. Returns the number of surviving objects (0 => no files written)."""
    rrr = brick_rrr(brick)
    tpath = os.path.join(data_root, "tractor", rrr, f"tractor-{brick}.fits")
    t = fitsio.read(tpath, columns=NEEDED_TRACTOR_COLS)

    mask = select_mask(t)
    n = int(mask.sum())
    if n == 0:
        return 0                                   # zero survivors -> emit nothing

    df = build_dataframe(t, mask)

    # brick_primary keeps objects in the central 0.25deg footprint, inset ~82px from the
    # 3600px image edge, so the 101px window always fits. Assert rather than pad/clip.
    lo, hi = HALF, IMG_SIZE - 1 - HALF
    bx, by = df["bx"].to_numpy(), df["by"].to_numpy()
    if not (bx.min() >= lo and by.min() >= lo and bx.max() <= hi and by.max() <= hi):
        raise ValueError(
            f"{brick}: object within {HALF}px of brick edge "
            f"(bx[{bx.min()},{bx.max()}] by[{by.min()},{by.max()}]); "
            f"edge handling is undefined for this product")

    write_tractor_h5(df, os.path.join(out_root, "tractor", rrr, f"{brick}.h5"))
    images = load_coadd_images(data_root, brick)
    write_coadd_h5(images, df, os.path.join(out_root, "coadd", rrr, f"{brick}.h5"))
    return n
