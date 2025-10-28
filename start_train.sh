#!/bin/bash

# Usage: DEBUG=1 bash start_train.sh [args]
#   In debug mode: runs single GPU CUDA_VISIBLE_DEVICES=0 python ...
# Usage: bash start_train.sh [args]
#   In normal mode: runs torchrun ...

set -euo pipefail

CONFIG_PATH="config/demo_multiscene.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
MASTER_PORT=16669

if [[ "${DEBUG:-}" == "1" ]]; then
    echo "Launching in DEBUG (single GPU)..."
    CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc_per_node=1 \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      "$@"
else
    echo "Launching distributed training ..."
    torchrun \
      --nproc_per_node=$NPROC_PER_NODE \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      "$@"
fi
