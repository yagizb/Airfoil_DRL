#!/bin/bash
set -euo pipefail

local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:?OMPI local rank is not set}"

case "${local_rank}" in
    0)
        gpu=0
        numa=0
        ;;
    1)
        gpu=1
        numa=1
        ;;
    2)
        gpu=2
        numa=2
        ;;
    3)
        gpu=3
        numa=3
        ;;
    *)
        echo "Unsupported local MPI rank: ${local_rank}" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="${gpu}"

exec numactl \
    --cpunodebind="${numa}" \
    --membind="${numa}" \
    "$@"