# TOD3Cap

## Enviroment Setup

a. Create a conda virtual environment and activate it.
```bash
cd tod3cap_camera
conda create -n todc python=3.8 -y 
conda activate todc
```
b. Install PyTorch and torchvision following the official instructions.
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Recommended torch>=1.9
```

c. Install gcc>=5 in conda env (optional).
```bash
conda install -c omgarcia gcc-6 # gcc-6.2
```

d. Install mmcv-full.
```bash
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

e. Install mmdet and mmseg.
```bash
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

f. Install mmdet3d from source code.
```bash
cd mmdetection3d
python setup.py install
cd ..
```

g. Install Detectron2 and Timm.
```bash
pip install einops fvcore seaborn iopath==0.1.9 timm  typing-extensions==4.5.0 pylint ipython==8.12  matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

h. Install other dependencies.
```bash
pip install -r requirements.txt
```

i. Install the evaluation utils
```bash
git clone https://gitclone.com/github.com/Maluuba/nlg-eval.git
cd nlg-eval
pip install -r requirements.txt
python setup.py install
nlg-eval --setup
```

## Train & Test
a. Train with fusion mode.
```bash
sh tools/dist_train_fusion.sh

# Pretrained weights of BEVFusion (implemented by ourself) and LLaMA-Adapter are required.

```
b. Train with camera-only mode.
```bash
sh tools/dist_train_camera.sh

# Pretrained weights of BEVFormer (implemented by ourself) and LLaMA-Adapter are required.
```

c. Train stage1 to obtain the pre-trained weights of BEVFormer or BEVFusion. (optional)
```bash
sh tools/dist_train_camera1.sh
sh tools/dist_train_fusion1.sh
# Train stage1 optional, we provide the pretrained weights of stage1.
```

## Models

```bash
mkdir checkpoints
# Download pretrained models from [Huggingface](https://huggingface.co/jxbbbb/tod3cap):
# fusion_stage1.pth
# 7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth
# former_stage2_previous_version.pth
```

* We release our best-performing checkpoints. You can download these models at [Huggingface](https://huggingface.co/jxbbbb/tod3cap) and place them under `checkpoints` directory. If the directory does not exist, you can create one.

## Dataset Preparation

You can download the TOD3Cap dataset in [Google Drive](https://drive.google.com/drive/folders/1LGimA3Il-9J1eJ3FYSzASnfD2Gcls_W1).
