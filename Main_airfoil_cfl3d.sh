#!/bin/bash
#SBATCH --job-name=Optuna_16
#SBATCH --chdir=/home/bsc/bsc545758/scratch/00_MyStudy/00_Airfoil_Shape_Opt/032_Train_wXFOIL_Re3_CL_wZoo

#SBATCH --output=out.out
#SBATCH --error=error.err

#SBATCH --qos=gp_bsccase
#SBATCH --account=bsc21
#SBATCH --time=00:45:00

### 16 cases × 4 MPI ranks each = 64 tasks
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1      # pure MPI

module purge
module load intel/2024.0
module load impi/2021.11

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=none
ulimit -s unlimited

echo "===================================="
echo "  CFL3D batch run started: $(date)"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
echo "===================================="

# Collect READY envs
ready_envs=()
for env_dir in runs/env_*; do
  [[ -d "$env_dir" ]] || continue
  [[ -f "${env_dir}/cfl3d_ready.flag" ]] && ready_envs+=("$env_dir")
done

if (( ${#ready_envs[@]} == 0 )); then
  echo "No env has cfl3d_ready.flag → nothing to run. Exiting."
  exit 0
fi

echo "Envs ready for CFL3D: ${#ready_envs[@]}"
echo "READY list: ${ready_envs[*]}"

# Launch CFL3D for READY envs
for env_dir in "${ready_envs[@]}"; do
  (
    env_name=$(basename "$env_dir")
    cd "$env_dir" || exit 1

    echo "[${env_name}] starting CFL3D..."

    if [[ ! -f "cfl3d.inp" ]]; then
      echo "[${env_name}] WARNING: cfl3d.inp not found → skipping."
      echo "FAILED_NO_INP" > cfl3d_failed.flag
      exit 0
    fi

    # Clean only runtime leftovers
    rm -f cfl3d_failed.flag

    # Run. Let CFL3D write cfl3d.out itself (your python parses it).
    # Capture srun exit code; if srun fails, mark this env as failed.
    sleep 1
    srun --exclusive --ntasks=4 --cpu-bind=none cfl3d_mpi < cfl3d.inp
    rc=$?

    if [[ $rc -ne 0 ]]; then
      echo "[${env_name}] srun failed rc=$rc"
      echo "FAILED_SRUN_RC $rc" > cfl3d_failed.flag
      exit 0
    fi

    echo "[${env_name}] finished CFL3D (srun rc=0)."
  ) &
done

wait

echo "===================================="
echo "  CFL3D batch run finished: $(date)"
echo "===================================="