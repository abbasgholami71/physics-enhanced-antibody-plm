#!/bin/bash -l
#SBATCH -J prod0
#SBATCH -o prod0.%j.out
#SBATCH -e prod0.%j.err
#SBATCH -D ./
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=NONE

module purge
module load gromacs/2025.2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Build TPR once if missing
if [ ! -f md.tpr ]; then
  echo "[INFO] Building md.tpr (fresh start)"
  gmx grompp -f md.mdp -c npt_relax.gro -p topol.top -o md.tpr || exit 1
fi

# Conditional run: if checkpoint exists, continue with -cpi/-append; else fresh run
if [ -f md.cpt ]; then
  echo "[INFO] Checkpoint md.cpt found -> continuing (append)."
  srun gmx_mpi mdrun -deffnm md -cpi md.cpt -append -pin on -pme auto -v
else
  echo "[INFO] No checkpoint -> starting fresh."
  srun gmx_mpi mdrun -deffnm md -pin on -pme auto -v
fi

# Lightweight post-run summary (optional)
grep "Performance:" md.log || true
