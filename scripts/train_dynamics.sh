#!/usr/bin/env bash
# Dynamics training with torchrun. Single node: `bash scripts/train_dynamics.sh name=my_run ...`.
# Multi-node: run the same command on every node with NUM_NODES, NODE_RANK and MASTER_ADDR set.
# Every extra argument is a Hydra override (see configs/).
set -euo pipefail
cd "$(dirname "$0")/.."
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS=${GPUS:-8}
torchrun --nnodes="$NUM_NODES" --node_rank="$NODE_RANK" --nproc_per_node="$GPUS" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m dreamfar.train.train algorithm=dynamics dataset=latents \
  experiment.num_nodes="$NUM_NODES" experiment.num_devices="$GPUS" "$@"
