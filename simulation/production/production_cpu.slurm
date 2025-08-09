#!/bin/bash -l
#SBATCH -J prod50ns
#SBATCH -o prod50ns.%j.out
#SBATCH -e prod50ns.%j.err
#SBATCH -D ./
#SBATCH --time=24:00:00          # adjust to cover chosen segment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32     # MPI ranks
#SBATCH --cpus-per-task=2        # OpenMP threads per rank
#SBATCH --mail-type=NONE

module purge
module load gromacs/2025.2       # adjust module name

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Build TPR if not already (comment out if pre-built)
if [ ! -f md.tpr ]; then
  echo "Building md.tpr"
  gmx grompp -f md.mdp -c npt_relax.gro -p topol.top -o md.tpr || exit 1
fi

# Run / continue
srun gmx_mpi mdrun -deffnm md -cpi md.cpt -append -pin on -pme auto -v

# Basic post-run extracts
echo Temperature | gmx energy -f md.edr -o md_temp.xvg >/dev/null 2>&1
echo Density     | gmx energy -f md.edr -o md_density.xvg >/dev/null 2>&1
grep "Performance:" md.log || true
