#!/bin/bash
### Job name on queue
#SBATCH --job-name=sod2d

### Output and error files directory
#SBATCH -D .

### Output and error files
#SBATCH --output=out.o
#SBATCH --error=error.e

### Run configuration
### Rule: {ntasks-per-node} \times {cpus-per-task} = 80
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

### Queue and account
#SBATCH --account=bsc21
##SBATCH --qos=acc_bsccase
##SBATCH --time=1-00:30:00
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00

### MN% modules
module purge
module load nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx
export LD_LIBRARY_PATH=/home/bsc/bsc545758/projects/sod2d_gitlab/build_gpu/external/json-fortran/lib:$LD_LIBRARY_PATH

mpirun -np 4 --map-by ppr:4:node:PE=20 --report-bindings ./mn5_bind.sh ./sod2d_p2 BluffBodySolverIncomp
