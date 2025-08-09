#!/bin/bash -l
#SBATCH -J npt
#SBATCH -o npt.%j.out
#SBATCH -e npt.%j.err
#SBATCH -D ./
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
module purge
module load gromacs/2025.2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
if [ ! -f npt.tpr ]; then echo "npt.tpr missing"; exit 1; fi
srun gmx_mpi mdrun -deffnm npt -pin on -pme auto -v
echo Density | gmx energy -f npt.edr -o npt_density.xvg >/dev/null 2>&1
echo Pressure | gmx energy -f npt.edr -o npt_pressure.xvg >/dev/null 2>&1
grep "Performance:" npt.log || true
