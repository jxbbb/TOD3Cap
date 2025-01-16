#!/usr/bin/env bash

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH="." \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=34149 \
    train.py projects/configs/bevformer/bevfusion_tiny_stage1.py --launcher pytorch --deterministic --work-dir work_dirs/bevfusion_tiny_stage1
