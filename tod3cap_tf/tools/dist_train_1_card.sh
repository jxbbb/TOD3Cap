#!/usr/bin/env bash

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="." \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=34149 \
    tools/train.py projects/configs/bevformer/bevformer_tiny.py --launcher pytorch --deterministic 

# python -m pdb \
#     $(dirname "$0")/train.py projects/configs/bevformer/bevformer_tiny.py --resume-from
