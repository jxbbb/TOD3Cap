#!/usr/bin/env bash

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=6 --master_port=34149 \
    $(dirname "$0")/train.py projects/configs/bevformer/bevformer_tiny.py --launcher pytorch --deterministic 

# python -m pdb \
#     $(dirname "$0")/train.py projects/configs/bevformer/bevformer_tiny.py --resume-from
