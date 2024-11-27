NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=2,3\
PYTHONPATH=./ \
torchpack dist-run -np 2 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --model.stage 2
