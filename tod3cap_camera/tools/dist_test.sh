#!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-22419}

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
CUDA_LAUNCH_BLOCKING=1 \
PYTHONPATH="./" \

python -m torch.distributed.launch --nproc_per_node=8 --master_port=51249 \
    test.py projects/configs/bevformer/bevfusion_tiny_test.py work_dirs/bevfusion_tiny_stage3/epoch_5.pth --launcher pytorch --eval bbox