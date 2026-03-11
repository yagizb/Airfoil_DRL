#!/bin/bash
#SBATCH --job-name=CFL3D_batch
#SBATCH --chdir=.
#SBATCH --output=out.out
#SBATCH --error=error.err
#SBATCH --qos=gp_bsccase
#SBATCH --account=bsc21
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

module purge
module load intel/2024.0
module load impi/2021.11

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=none
ulimit -s unlimited

echo "===================================="
echo "CFL3D batch start: $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE"
echo "ROOT=runs"
echo "MPI_PER_ENV=4"
echo "===================================="

ROOT="runs"
MPI_PER_ENV=4

if [[ -z "${SLURM_NTASKS:-}" ]]; then
  echo "ERROR: SLURM_NTASKS is not set. Are you running inside SLURM?"
  exit 1
fi

MAX_ENVS=$(( SLURM_NTASKS / MPI_PER_ENV ))
if (( MAX_ENVS < 1 )); then
  echo "ERROR: Need at least ${MPI_PER_ENV} tasks for 1 env, but SLURM_NTASKS=${SLURM_NTASKS}"
  exit 1
fi

# Collect READY envs (sorted for deterministic behavior)
mapfile -t ready_envs < <(find "${ROOT}" -maxdepth 1 -type d -name "env_*" | sort)

run_list=()
for env_dir in "${ready_envs[@]}"; do
  [[ -f "${env_dir}/cfl3d_ready.flag" ]] || continue
  [[ -f "${env_dir}/cfl3d_failed.flag" ]] && continue
  [[ -f "${env_dir}/cfl3d_conv.flag"   ]] && continue
  run_list+=("${env_dir}")
done

if (( ${#run_list[@]} == 0 )); then
  echo "No READY env found -> nothing to run. Exiting."
  exit 0
fi

# Truncate to what fits in allocation
if (( ${#run_list[@]} > MAX_ENVS )); then
  echo "WARNING: ${#run_list[@]} envs READY but only ${MAX_ENVS} fit (SLURM_NTASKS=${SLURM_NTASKS})."
  echo "         Running first ${MAX_ENVS}; others remain READY for next batch."
  run_list=( "${run_list[@]:0:${MAX_ENVS}}" )
fi

echo "Envs READY total: ${#run_list[@]} (capacity=${MAX_ENVS})"
echo "RUN list: ${run_list[*]}"

# Launch CFL3D for each env in parallel, 4 MPI ranks each
for env_dir in "${run_list[@]}"; do
  (
    env_name=$(basename "${env_dir}")
    cd "${env_dir}" || exit 1

    echo "[${env_name}] starting CFL3D in $(pwd)"

    if [[ ! -f "cfl3d.inp" ]]; then
      echo "[${env_name}] ERROR: cfl3d.inp not found -> mark failed."
      echo "FAILED_NO_INP" > cfl3d_failed.flag
      exit 0
    fi

    # Clear only runtime flag from previous attempt
    rm -f cfl3d_failed.flag

    # Run solver
    srun --exclusive --ntasks=${MPI_PER_ENV} --cpu-bind=none \
     --output="srun_%x_%j_${env_name}.out" \
     --error="srun_%x_%j_${env_name}.err" \
     cfl3d_mpi < cfl3d.inp
    rc=$?

    if [[ $rc -ne 0 ]]; then
      echo "[${env_name}] srun failed rc=$rc"
      echo "FAILED_SRUN_RC $rc" > cfl3d_failed.flag
      exit 0
    fi

    echo "[${env_name}] finished CFL3D (rc=0)"
  ) &
done

wait

echo "===================================="
echo "CFL3D batch finished: $(date)"
echo "===================================="