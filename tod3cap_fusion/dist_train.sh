NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=3,4,5,7 \
PYTHONPATH=./ \
torchpack dist-run -np 4 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --checkpoint runs/run-122a835f/epoch_6.pth \
    --model.stage 3
