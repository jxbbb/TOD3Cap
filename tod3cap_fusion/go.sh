# NCCL_P2P_DISABLE=1 \
# CUDA_VISIBLE_DEVICES=3 \
# PYTHONPATH=./ \
# torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swin_tiny_patch4_window7_224.pth

NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=./ \
torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/run-7bbc7080_1/epoch_6.pth --eval bbox