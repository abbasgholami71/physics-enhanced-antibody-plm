#!/bin/bash -l
#SBATCH -J prod0_rerun
#SBATCH -o prod0_rerun.%j.out
#SBATCH -e prod0_rerun.%j.err
#SBATCH -D ./
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=NONE
#SBATCH --mem=4G

module purge
module load gromacs/2025.2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Run GROMACS rerun for energy calculation
srun gmx_mpi mdrun -s rerun.tpr -rerun md.xtc -deffnm rerun -pin on -pme auto -v
