#!/bin/bash
# submit_strip.sh -- regenerate full RA strips with MPI fan-out on Perlmutter CPU nodes.
#
#   sbatch submit_strip.sh
#
# Edit REGION / BATCHIDS / DATA / OUT below, set your -A <repo>, and ensure the venv
# (see README.md) has mpi4py built against cray-mpich.
#
#SBATCH -J dr11_collect
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks 256
#SBATCH -t 04:00:00
#SBATCH -A m0000              # <-- set your NERSC repo/allocation
#SBATCH -o dr11_collect_%j.out
#SBATCH -e dr11_collect_%j.err

set -euo pipefail

REGION=north
BATCHIDS=0,120,240,359
DATA=/global/cfs/cdirs/cosmo/data/legacysurvey/dr11/${REGION}
OUT=${SCRATCH}/dr11_onepercent/${REGION}

# --- environment: NERSC python module + the venv from README.md ---
module load python/3.13-26.1.0
source ${SCRATCH}/dr11cook/bin/activate

# fail fast if mpi4py is missing (needed only for the MPI driver):
python -c "import mpi4py" 2>/dev/null || {
    echo "mpi4py not found in venv. Install it against cray-mpich, e.g.:"
    echo "  module load python/3.13-26.1.0 PrgEnv-gnu cray-mpich"
    echo "  source \$SCRATCH/dr11cook/bin/activate"
    echo "  MPICC=cc pip install --no-cache-dir --no-binary mpi4py mpi4py"
    exit 1
}

cd "$(dirname "$0")"

srun -n ${SLURM_NTASKS} python run_mpi.py \
    --region "${REGION}" --batchids "${BATCHIDS}" \
    --data "${DATA}" --out "${OUT}"

# Repeat for --region south with the south DATA/OUT paths (separate submission).
