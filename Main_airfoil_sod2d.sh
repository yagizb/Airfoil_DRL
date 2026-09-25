#!/bin/bash
#SBATCH --job-name=sod2d_batch
#SBATCH -D .
#SBATCH --output=out.o
#SBATCH --error=error.e
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
##SBATCH --qos=acc_bsccase
##SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

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

ROOT="runs"
MPI_PER_ENV=4

echo "===================================="
echo "SOD2D batch start: $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "ROOT=$ROOT"
echo "MPI_PER_ENV=$MPI_PER_ENV"
echo "===================================="

mapfile -t ready_envs < <(
    find "${ROOT}" -maxdepth 1 -type d -name "env_*" | sort
)

run_list=()

for env_dir in "${ready_envs[@]}"; do
    [[ -f "${env_dir}/sod2d_ready.flag" ]] || continue
    [[ -f "${env_dir}/sod2d_failed.flag" ]] && continue
    [[ -f "${env_dir}/sod2d_done.flag"   ]] && continue
    [[ -f "${env_dir}/sod2d_conv.flag"   ]] && continue

    run_list+=("${env_dir}")
done

if (( ${#run_list[@]} == 0 )); then
    echo "No READY env found -> exiting."
    exit 0
fi

MAX_ENVS=${SLURM_JOB_NUM_NODES}

if (( ${#run_list[@]} > MAX_ENVS )); then
    echo "WARNING:"
    echo "  READY envs : ${#run_list[@]}"
    echo "  Capacity   : ${MAX_ENVS}"
    echo "  Running first ${MAX_ENVS} envs only."
    run_list=( "${run_list[@]:0:${MAX_ENVS}}" )
fi

echo "READY envs to run: ${#run_list[@]}"
echo "RUN list:"
printf '  %s\n' "${run_list[@]}"

for env_dir in "${run_list[@]}"; do

    env_name=$(basename "${env_dir}")

    echo "[${env_name}] launching"
    cd "${env_dir}" || exit 1

    echo "[${env_name}] cwd=$(pwd)"
    echo "[${env_name}] SLURM_NTASKS=${SLURM_NTASKS}"
    echo "[${env_name}] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

    if [[ ! -x ./sod2d_p2 ]]; then
        echo "[${env_name}] ERROR: sod2d_p2 missing"
        echo FAILED_NO_SOLVER > sod2d_failed.flag
        cd - >/dev/null || exit 1
        continue
    fi

    if [[ ! -f BluffBodySolverIncomp.json ]]; then
        echo "[${env_name}] ERROR: BluffBodySolverIncomp.json missing"
        echo FAILED_NO_INPUT > sod2d_failed.flag
        cd - >/dev/null || exit 1
        continue
    fi

    rm -f sod2d_failed.flag

    echo "[${env_name}] starting SOD2D"

    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    export SLURM_CPU_BIND=none

    mpirun \
        -np "${MPI_PER_ENV}" \
        --map-by ppr:4:node \
        --bind-to none \
        ./mn5_bind.sh \
        ./sod2d_p2 \
        BluffBodySolverIncomp

    rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "[${env_name}] FAILED rc=${rc}"
        echo FAILED_SOD2D_RC ${rc} > sod2d_failed.flag
        cd - >/dev/null || exit 1
        continue
    fi

    echo "[${env_name}] finished successfully"
    touch sod2d_done.flag

    cd - >/dev/null || exit 1

done

echo "===================================="
echo "SOD2D batch finished: $(date)"
echo "===================================="