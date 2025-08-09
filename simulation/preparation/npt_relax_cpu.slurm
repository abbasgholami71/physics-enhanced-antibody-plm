#!/bin/bash -l
#SBATCH -J npt2
#SBATCH -o npt2.%j.out
#SBATCH -e npt2.%j.err
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
if [ ! -f npt_relax.tpr ]; then echo "npt_relax.tpr missing"; exit 1; fi
srun gmx_mpi mdrun -deffnm npt_relax -pin on -pme auto -v
echo Density | gmx energy -f npt_relax.edr -o npt_relax_density.xvg >/dev/null 2>&1
echo Pressure | gmx energy -f npt_relax.edr -o npt_relax_pressure.xvg >/dev/null 2>&1
grep "Performance:" npt_relax.log || true
