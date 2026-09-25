#!/bin/bash
#SBATCH --job-name=CFL3D_batch
#SBATCH --chdir=.
#SBATCH --output=out.out
#SBATCH --error=error.err
#SBATCH --qos=gp_debug
#SBATCH --account=bsc21
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

module purge
module load intel/2024.0
module load impi/2021.11

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=none

ulimit -s unlimited

# Run CFL3D using the four allocated MPI tasks.
srun --exclusive \
     --ntasks="${SLURM_NTASKS}" \
     --cpu-bind=none \
     cfl3d_mpi < cfl3d.inp