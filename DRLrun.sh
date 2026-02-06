#!/bin/bash
#SBATCH --job-name=SAC_MaxCL_Re3_AoA00
#SBATCH --chdir=.

### Output and error files directory
#SBATCH -D .

### Output and error files
#SBATCH -o LOGX_RE3.out
#SBATCH -e ERRX_RE3.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # or 16, depending on node
##SBATCH --exclusive              # full node = all RAM

####SBATCH --constraint=highmem

### Queue and account
#SBATCH --qos=gp_bsccase
#SBATCH --time=1-00:00:00

#SBATCH --account=bsc21     

### Load MN% modules + DRL libraries
unset PYTHONPATH
export PYTHONNOUSERSITE=1
## source /home/bsc/bsc545758/scratch/031_DRLwCFL3D_MaxCLCD/.venv/bin/activate
source ~/venvs/airfoil_drl/bin/activate

##srun -n 1 python3 Main_Train.py > rlsod2d.log 2>&1
python3 Main_Train_wSAC.py > rlsod2d.log 2>&1
