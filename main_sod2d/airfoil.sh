#!/bin/bash
#SBATCH --job-name=sod2d
#SBATCH -D .
#SBATCH --output=out.o
#SBATCH --error=error.e

### ntasks-per-node × cpus-per-task = 80
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

#SBATCH --account=bsc21
##SBATCH --qos=acc_bsccase
##SBATCH --time=2-00:00:00

## Debug configuration
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00

set -euo pipefail

module purge
module load nvidia-hpc-sdk/24.3
module load hdf5/1.14.1-2-nvidia-nvhpcx

export LD_LIBRARY_PATH="/home/bsc/bsc545758/projects/sod2d_gitlab/build_gpu/external/json-fortran/lib:${LD_LIBRARY_PATH:-}"

### Required by NVIDIA HPC-X when launching through mpirun
export SLURM_CPU_BIND=none

### Work around the UCX mlx5 DevX segmentation fault
export UCX_IB_MLX5_DEVX=n

### This is a single-node job: use shared memory and CUDA IPC
export UCX_TLS=self,sm,cuda_copy,cuda_ipc

mpirun \
    -np "${SLURM_NTASKS}" \
    --map-by ppr:4:node \
    --bind-to none \
    ./mn5_bind.sh \
    ./sod2d_p2 \
    BluffBodySolverIncomp