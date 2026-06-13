# DESI DR11 Cutouts

A reproducible recipe that re-derives the DESI Legacy Survey **DR11
`One_percent`** machine-learning product — a filtered photometric catalog plus
per-object grz image cutouts, one HDF5 file per brick — directly from the
official Legacy Survey archive. The original "Collection" pipeline that produced
`One_percent` is not on the readable filesystem, so this is a clean
reimplementation whose every output convention was **reverse-engineered and
verified array-exact** against the existing product: identical object identities,
positions, and pixels, with magnitudes matching to float32 precision.

[:material-github: View on GitHub](https://github.com/usfcs-edu/agentic-lensing/tree/main/tools/desi-dr11-cookbook){ .md-button }

## Overview

The cookbook reads the read-only DR11 archive and, for every surviving object,
emits a 101×101×3 (g, r, z) cutout plus a row in a 14-column tractor catalog.

- **Input** (read-only):
  `/global/cfs/cdirs/cosmo/data/legacysurvey/dr11/<region>` — FITS tractor
  catalogs and coadd images, where `<region>` ∈ `north`, `south`.
- **Output**: one HDF5 file per brick, mirroring the input layout, ML-ready.

Objects are kept only if they pass **five filters** (logical AND), taken from the
original `config.log`:

1. `brick_primary == True`
2. `flux_g > 0` **and** `flux_r > 0` **and** `flux_z > 0`
3. `nobs_g ≥ 3` **and** `nobs_r ≥ 3` **and** `nobs_z ≥ 3`
4. `mag_z < 20`
5. `type ∈ {SER, EXP, DEV, REX}`

A brick with zero survivors writes no file (this is why, e.g., north strip 240
yields 397 files from 446 bricks). The object↔cutout correspondence is exactly
1:1.

Work is organized by **RA strip**: `BatchID = floor(RA)` is the 3-digit `<RRR>`
subdirectory (`000..359`). The original "One percent" sample used strips
`{0, 120, 240, 359}`; enumerate the bricks for a strip by listing
`tractor/<RRR>/`.

The cookbook is a small set of scripts around one core library:

| file | purpose |
|---|---|
| `dr11_collect.py` | core library: brick enumeration, filters, `flux2mag`, dataframe build, cutouts, HDF5 writers, `process_brick` |
| `run_brick.py` | CLI: process a single brick (smoke test / debugging) |
| `run_mpi.py` | `mpi4py` driver: fan the brick list across ranks; rank 0 writes logs |
| `conformance.py` | prove a derived brick is array-exact vs the existing `One_percent` |
| `submit_strip.sh` | `sbatch` + `srun` template for full strips on Perlmutter CPU nodes |
| `config.xml` | spec mirror of the original config (documentation; copied to `<out>/log/config.log`) |

## Environment setup

You need `numpy, pandas, h5py, fitsio, tables` (PyTables is **required** — both
outputs are PyTables-authored), plus `mpi4py` for the batch driver. The recipe
layers the two missing packages on the NERSC python module:

```bash
module load python/3.13-26.1.0                 # provides numpy, pandas, h5py
python -m venv --system-site-packages $SCRATCH/dr11cook
source $SCRATCH/dr11cook/bin/activate
pip install fitsio tables                       # fitsio (FITS I/O) + PyTables
```

For the MPI run, add `mpi4py` built against Cray MPICH so it uses the high-speed
network (binary wheels won't):

```bash
module load python/3.13-26.1.0 PrgEnv-gnu cray-mpich
source $SCRATCH/dr11cook/bin/activate
MPICC=cc pip install --no-cache-dir --no-binary mpi4py mpi4py
```

!!! note "Alternative: the DESI stack"
    You can instead `source /global/common/software/desi/desi_environment.sh 24.11`
    — which provides fitsio/astropy/h5py/pandas and a Cray `mpi4py` — then just
    `pip install tables` into a `--system-site-packages` venv on top of it.

## Test on a single brick

Start with one brick interactively to confirm your environment and paths. This
runs `process_brick` from the core library and writes the tractor + coadd HDF5
files if at least one object survives:

```bash
source $SCRATCH/dr11cook/bin/activate
python run_brick.py --region north --brick 2400p345 \
  --data /global/cfs/cdirs/cosmo/data/legacysurvey/dr11/north \
  --out  $SCRATCH/dr11_test/north
```

!!! warning "First read is slow"
    DR11 lives on the Community File System (CFS). The first read of a given file
    can stall for a few minutes on cold metadata/cache — see
    [Performance notes](#performance-notes). For a single interactive brick,
    expect the first read to be slow, then sub-second when warm.

## Verify against the reference (conformance)

`conformance.py` re-derives a brick and asserts equality against the existing
`One_percent` product. It checks the tractor catalog (column order, dtypes, row
count; strict equality on all identity/position/integer columns) and every coadd
cutout (`o<objid>` key set, `(101, 101, 3)` shape, `float32` dtype, pixel-exact).

```bash
python conformance.py --brick 2400p345 --region north \
  --ref /global/cfs/projectdirs/cosmo/work/users/xhuang/dr11/One_percent \
  --new $SCRATCH/dr11_test
```

On the golden brick **north/2400p345** (183 objects), all 11
identity/position/integer columns and all 183 cutouts (every pixel, all three
bands) are **strictly bit-exact**. Only the three derived `mag_*` columns differ,
and only within **±1 float32 ULP**: `log10`'s last-bit rounding varies between
the original numpy/libm build and any re-run, so the original bits aren't
recoverable from any formula. One ULP at mag ≈ 21 is ~1.9e-6 mag — roughly 4000×
below photometric precision. The test asserts each `mag_*` is within ±1 ULP and
reports the bit-exact fraction; exit code 0 = pass, 1 = fail.

## Run full strips (MPI / SLURM)

For full RA strips, submit the SLURM template. Edit the variables at the top of
`submit_strip.sh` (`REGION`, `BATCHIDS`, and the NERSC allocation `-A <repo>`),
then:

```bash
sbatch submit_strip.sh        # north strips 0,120,240,359 by default; repeat for south
```

Under the hood the job launches the MPI driver:

```bash
srun -n 256 python run_mpi.py --region north --batchids 0,120,240,359 \
  --data /global/cfs/cdirs/cosmo/data/legacysurvey/dr11/north \
  --out  $SCRATCH/dr11_onepercent/north
```

Rank 0 enumerates all bricks for the requested batchids and broadcasts the list;
each rank then processes `bricks[rank::size]` (round-robin). Because there is one
output file per brick, there is **no parallel/collective HDF5** — MPI is pure
task fan-out, and each brick writes its own files independently. Rank 0 collects
statistics and writes `<out>/log/logging.log` (summary + errors) and
`<out>/log/config.log` (a copy of `config.xml`).

~256 ranks ≈ 2 Perlmutter CPU nodes. The job is I/O-bound (each brick reads three
3600² images), so scale ranks for throughput. To produce a different sample,
change `--batchids` (any subset of `0..359`) and/or `--region`.

!!! note "Changing the selection"
    The filters and columns live in `dr11_collect.py` and are mirrored in
    `config.xml` for documentation. If you change the selection criteria, edit
    both together so the spec stays in sync with the code.

## Outputs & schema

Each non-empty brick produces two HDF5 files (plus per-run logs):

```
<out>/<region>/tractor/<RRR>/<brick>.h5     # RRR = floor(RA); e.g. 240 for 2400p345
<out>/<region>/coadd/<RRR>/<brick>.h5
<out>/<region>/log/{config.log, logging.log}   # written by run_mpi.py
```

**`tractor/<RRR>/<brick>.h5`** is a pandas DataFrame in PyTables *fixed* format
under key `tractor`, with exactly 14 columns in this order (one row per surviving
object):

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

**`coadd/<RRR>/<brick>.h5`** holds one PyTables `CArray` per object, named
`o<objid>`, `float32`, shape `(101, 101, 3)`:

- channel order **(g, r, z)**;
- raw **nanomaggies** copied verbatim from
  `legacysurvey-<brick>-image-{g,r,z}.fits.fz` (HDU1) — no
  normalization/arcsinh/clip;
- stored **transposed**: `cube[:, :, k] = image_k[by-50:by+51, bx-50:bx+51].T`,
  so the object sits at `cube[50, 50, k] == image_k[by, bx]`.

!!! warning "Cutouts are transposed"
    The transpose is an x/y convention quirk of the original product, reproduced
    exactly for drop-in compatibility. Each `(101, 101)` plane is the transpose
    of the sky image — swap the first two axes to get north-up / east-left
    orientation.

Reading the outputs back is straightforward:

```python
import pandas as pd, h5py

cat = pd.read_hdf("2400p345.h5", key="tractor")          # 14-column catalog
with h5py.File("2400p345.h5", "r") as f:                 # coadd file
    cube = f[f"o{cat.objid.iloc[0]}"][:]                 # (101, 101, 3) float32
```

## Performance notes

DR11 lives on the Community File System (CFS). The **first** read of a given file
can stall for a few minutes on cold metadata/cache — a tractor catalog was
observed at ~210 s cold, then ~0.2 s warm; images are ~0.2 s warm. This is
filesystem latency, not compute. For large runs it averages out across many
bricks and ranks; for a single interactive brick, expect the first read to be
slow. Empty bricks write no files, so output counts are smaller than brick
counts.
