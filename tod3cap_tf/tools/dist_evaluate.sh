#!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-22419}

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
CUDA_LAUNCH_BLOCKING=1 \
PYTHONPATH="./" \
python -m torch.distributed.launch --nproc_per_node=6 --master_port=61231 \
    tools/evaluate_caption.py projects/configs/bevformer/bevformer_tiny.py work_dirs/bevformer_tiny_all_train_from_zero/epoch_24.pth --launcher pytorch --eval bbox
