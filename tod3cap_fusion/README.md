# TOD3Cap: Lidar-based and fusion-based implementations

## Getting Started

a. Create a conda virtual environment and activate it.
```bash
cd tod3cap_fusion
conda create -n todf python=3.8 -y 
conda activate todf
```
b. Install PyTorch and torchvision following the official instructions.
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Recommended torch>=1.9
```

c. Install mpi4py for torchpack.
```bash
pip install mpi4py==3.0.3
```

d. Install mmcv-full.
```bash
pip install -U openmim
mim install mmcv-full==1.4.0
```

e. Install mmdet and mmseg.
```bash
pip install mmdet==2.20.0
```


h. Install other dependencies.
```bash
pip install -r requirements.txt
```

f. install the codebase.
```bash
python setup.py develop
```

i. Install the evaluation utils
```bash
git clone https://gitclone.com/github.com/Maluuba/nlg-eval.git
cd nlg-eval
pip install -r requirements.txt
python setup.py install
nlg-eval --setup
```

## Three-stage Training

a. In the first stage, we utilize the pre-trained BEV-based detector on object detection task;
Just download the [pretrained detector weights](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1).



b. Then we freeze the detector weights and train the caption generation module; 
```bash
. tod3cap_fusion/dist_train_stage2.sh
```


c. Finally, the entire model is finetuned with a smaller learning rate.
```bash
. tod3cap_fusion/dist_train.sh
```




