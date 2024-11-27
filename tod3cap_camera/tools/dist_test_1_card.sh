#!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-22419}

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=3\
CUDA_LAUNCH_BLOCKING=1 \
PYTHONPATH="./" \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=61236 \
    tools/test.py projects/configs/bevformer/bevformer_tiny_test.py work_dirs/bevformer_tiny_24_10_26/epoch_10.pth --launcher pytorch --eval bbox 