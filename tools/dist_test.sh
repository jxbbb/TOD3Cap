#!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-22419}

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_LAUNCH_BLOCKING=1 \
PYTHONPATH="./" \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=61236 \
    tools/test.py projects/configs/bevformer/bevformer_small.py work_dirs/bevformer_small/epoch_9.pth --launcher pytorch --eval bbox
