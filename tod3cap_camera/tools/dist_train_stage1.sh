#!/usr/bin/env bash

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH="." \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=34149 \
    train.py projects/configs/bevformer/bevformer_tiny_fusion_stage1.py --launcher pytorch --deterministic --work-dir work_dirs/bevformer_tiny_fusion_satge1
# python -m pdb \
#     $(dirname "$0")/train.py projects/configs/bevformer/bevformer_tiny.py --resume-from



# NCCL_P2P_DISABLE=1 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# CUDA_LAUNCH_BLOCKING=1 \
# PYTHONPATH="./" \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=61236 \
#     tools/test.py projects/configs/bevformer/bevformer_tiny.py work_dirs/bevformer_tiny_24_10_26/epoch_10.pth --launcher pytorch --eval bbox
