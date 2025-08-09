#!/bin/bash -l
#SBATCH -J nvt
#SBATCH -o nvt.%j.out
#SBATCH -e nvt.%j.err
#SBATCH -D ./
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32      # 32 MPI ranks (good for ~200k atoms)
#SBATCH --cpus-per-task=2         # 2 OpenMP threads each (total 64 cores if node has them)
#SBATCH --mail-type=NONE

module purge
module load gromacs/2025.2

if [ ! -f nvt.tpr ]; then echo "Missing nvt.tpr"; exit 1; fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun gmx_mpi mdrun -deffnm nvt -pin on -pme auto -v

# Extract quick temperature summary
echo Temperature | gmx energy -f nvt.edr -o nvt_temp.xvg >/dev/null 2>&1
grep "Performance:" nvt.log || true
