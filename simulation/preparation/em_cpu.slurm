#!/bin/bash -l
#SBATCH -J em
#SBATCH -o em.%j.out
#SBATCH -e em.%j.err
#SBATCH -D ./
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16          # 16 MPI ranks
#SBATCH --cpus-per-task=4             # 4 OpenMP threads per rank (16*4 = 64 cores)
#SBATCH --mail-type=NONE

# Load modules (adjust to your environment)
module purge
# Prefer the newer build you used interactively (example name below)
module load gromacs/2025.2

# Sanity: ensure em.tpr exists
if [ ! -f em.tpr ]; then
  echo "ERROR: em.tpr not found. Run grompp first." >&2
  exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Run EM
srun gmx_mpi mdrun -deffnm em -v -pin on -pme auto

# Optional quick check at end
grep -i "Steepest Descents" em.log || tail -n 5 em.log
