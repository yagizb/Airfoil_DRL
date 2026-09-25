#!/bin/bash
#SBATCH --job-name=SAC_MaxCLCD_Re3
#SBATCH --chdir=.
#SBATCH --output=LOGX_RE3.out
#SBATCH --error=ERRX_RE3.err

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
##SBATCH --qos=acc_debug
##SBATCH --time=02:00:00
#SBATCH --qos=acc_bsccase
#SBATCH --time=2-00:00:00
#SBATCH --account=bsc21

set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1

source "$HOME/venvs/airfoil_drl/bin/activate"

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "MPI tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"

python3 -u Main_Eval_wSAC.py > rlsod2d.log 2>&1