# DR11 → `One_percent` cookbook

Reproducible recipe + reference code to derive `One_percent`-style HDF5 data from the official
Legacy Survey **DR11** archive, in the exact format of
`/global/cfs/projectdirs/cosmo/work/users/xhuang/dr11/One_percent`.

- **Input** (read-only): `/global/cfs/cdirs/cosmo/data/legacysurvey/dr11/<region>` — FITS tractor
  catalogs + coadd images (`<region>` ∈ `north`, `south`).
- **Output**: one HDF5 file per brick, mirroring the input layout, ML-ready (filtered catalog +
  per-object grz image cutouts).

The original "Collection" pipeline that produced `One_percent` is not on the readable filesystem, so
this is a clean reimplementation. Every output convention below was **reverse-engineered and verified
bit-for-bit** against the existing product (see [Conformance](#conformance)).

---

## What it produces

```
<out>/<region>/tractor/<RRR>/<brick>.h5     # RRR = 3-digit RA strip = floor(RA); e.g. 240 for 2400p345
<out>/<region>/coadd/<RRR>/<brick>.h5
<out>/<region>/log/{config.log, logging.log}   # written by run_mpi.py
```

**`tractor/<RRR>/<brick>.h5`** — a pandas DataFrame in PyTables *fixed* format, key `tractor`, with
**exactly 14 columns in this order** (one row per surviving object):

| column | dtype | notes |
|---|---|---|
| `brickid` | int32 | |
| `brickname` | object (Python `str`) | whitespace-stripped |
| `objid` | int32 | |
| `type` | object (Python `str`) | one of SER/EXP/DEV/REX |
| `ra`, `dec` | float64 | degrees |
| `bx`, `by` | int64 | brick pixel coords, **truncated toward zero** (`3123.86 → 3123`) |
| `nobs_g/r/z` | int16 | |
| `mag_g/r/z` | float32 | derived: `22.5 − 2.5·log10(flux_*)` |

**`coadd/<RRR>/<brick>.h5`** — one PyTables `CArray` per object, named `o<objid>`, `float32`
shape `(101, 101, 3)`:
- channel order **(g, r, z)**;
- raw **nanomaggies** copied verbatim from `legacysurvey-<brick>-image-{g,r,z}.fits.fz` (HDU1) — no
  normalization/arcsinh/clip;
- stored **transposed**: `cube[:, :, k] = image_k[by-50:by+51, bx-50:bx+51].T`, so the object sits at
  `cube[50, 50, k] == image_k[by, bx]`.

> ⚠️ The transpose is an x/y convention quirk of the original product. It is reproduced exactly here for
> drop-in compatibility. If you consume these cutouts, remember each `(101,101)` plane is the transpose
> of the sky image (swap the first two axes to get north-up/east-left orientation).

### Object selection (5 filters, logical AND) — from the original `config.log`
1. `brick_primary == True`
2. `flux_g > 0` **and** `flux_r > 0` **and** `flux_z > 0`
3. `nobs_g ≥ 3` **and** `nobs_r ≥ 3` **and** `nobs_z ≥ 3`
4. `mag_z < 20`
5. `type ∈ {SER, EXP, DEV, REX}`

A brick with **zero survivors writes no file** (this is why e.g. north strip 240 yields 397 files from
446 bricks). Object↔cutout correspondence is exactly 1:1.

### RA strips / BatchIDs
`BatchID = floor(RA)` = the 3-digit `<RRR>` subdirectory (`0..359`). The original "One percent" sample
used strips `{0, 120, 240, 359}`. Enumerate bricks for a strip by listing `tractor/<RRR>/`.

---

## Files in this cookbook

| file | purpose |
|---|---|
| `dr11_collect.py` | core library: enumeration, filters, `flux2mag`, dataframe build, cutouts, HDF5 writers, `process_brick` |
| `run_brick.py` | CLI: process a single brick (smoke test / debugging) |
| `run_mpi.py` | `mpi4py` driver: fan brick list across ranks; rank 0 writes logs |
| `conformance.py` | prove a derived brick is array-exact vs the existing `One_percent` |
| `submit_strip.sh` | `sbatch` + `srun` template for full strips on Perlmutter CPU nodes |
| `config.xml` | spec mirror of the original config (documentation; copied to `<out>/log/config.log`) |

---

## Environment

Needs `numpy, pandas, h5py, fitsio, tables` (PyTables is **required** — both outputs are
PyTables-authored), plus `mpi4py` for the MPI driver. The recipe used here layers the two missing
packages on the NERSC python module:

```bash
module load python/3.13-26.1.0                 # provides numpy, pandas, h5py
python -m venv --system-site-packages $SCRATCH/dr11cook
source $SCRATCH/dr11cook/bin/activate
pip install fitsio tables                       # fitsio (FITS I/O) + PyTables
```

For the **MPI** run, add `mpi4py` built against Cray MPICH (binary wheels won't use the high-speed
network):

```bash
module load python/3.13-26.1.0 PrgEnv-gnu cray-mpich
source $SCRATCH/dr11cook/bin/activate
MPICC=cc pip install --no-cache-dir --no-binary mpi4py mpi4py
```

(Alternatively, source the DESI stack — `source /global/common/software/desi/desi_environment.sh 24.11`
— which provides fitsio/astropy/h5py/pandas and a Cray `mpi4py`, then just `pip install tables` into a
`--system-site-packages` venv on top of it.)

---

## Usage

### 1. Single brick (interactive smoke test)
```bash
source $SCRATCH/dr11cook/bin/activate
python run_brick.py --region north --brick 2400p345 \
  --data /global/cfs/cdirs/cosmo/data/legacysurvey/dr11/north \
  --out  $SCRATCH/dr11_test/north
```

### 2. Conformance check (vs the existing product)
```bash
python conformance.py --brick 2400p345 --region north \
  --ref /global/cfs/projectdirs/cosmo/work/users/xhuang/dr11/One_percent \
  --new $SCRATCH/dr11_test
```

### 3. Full strips (MPI batch job)
Edit `submit_strip.sh` (`REGION`, `BATCHIDS`, `-A <repo>`), then:
```bash
sbatch submit_strip.sh        # north strips 0,120,240,359 by default; repeat for south
```
One output file per brick → no parallel/collective HDF5; MPI is pure task fan-out
(`bricks[rank::size]`). ~256 ranks ≈ 2 Perlmutter CPU nodes. The job is **I/O-bound** (each brick reads
three 3600² images); scale ranks for throughput. See [performance note](#performance--io) below.

To produce a different sample, change `--batchids` (any subset of `0..359`) and/or `--region`. The code
is fully parameterized — the filters/columns live in `dr11_collect.py` (and are mirrored in
`config.xml`); edit both together if you change the selection.

---

## Conformance

`conformance.py` re-derives a brick and asserts equality against the existing `One_percent` file.
Verified on golden brick **north/2400p345** (183 objects):

- **Strictly bit-exact**: all 11 identity/position/integer columns (`brickid, brickname, objid, type,
  ra, dec, bx, by, nobs_g, nobs_r, nobs_z`) **and all 183 image cutouts** (every pixel, all 3 bands).
- **Float32-precision (≤ 1 ULP)**: the 3 derived `mag_*` columns. `log10`'s last-bit rounding differs
  between the original run's numpy/libm build and any re-run, so the original bits aren't recoverable
  from any formula (verified: 8 formula variants, best 78% bit-exact, every remaining mismatch exactly
  1 ULP). 1 ULP at mag≈21 is ~1.9e-6 mag — sub-micromag, ~4000× below photometric precision. The test
  asserts each `mag_*` is within ±1 ULP of the stored value and reports the bit-exact fraction.

This makes newly generated bricks **drop-in compatible** with the existing dataset: identical object
identities, positions, and pixels; magnitudes identical to float32 precision.

---

## Performance / I/O

DR11 lives on the Community File System (CFS). The **first** read of a given file can stall for a few
minutes on cold metadata/cache (observed: a tractor catalog took ~210 s cold, then 0.2 s warm; images
~0.2 s warm). This is filesystem latency, not compute. For large runs this averages out across many
bricks/ranks; for a single interactive brick, expect the first read to be slow.
