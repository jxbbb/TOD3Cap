import torch
# file_path = 'work_dirs/bevformer_tiny_all_train_e24/epoch_24.pth'
# file_path = 'work_dirs/bevformer_small/epoch_9.pth'
file_path = 'work_dirs/bevformer_base_all_train_e10/epoch_10.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
all_tuned = 0
for key in list(model['state_dict'].keys()):
    if "llama_adapter.llama" not in key:
        all_tuned += model['state_dict'][key].nelement()
    all += model['state_dict'][key].nelement()
print(all)
print(all_tuned)
# tiny: all: 6834425274 tuned: 94910906
# small: all: 6860488108 tuned: 120973740
# base: all: 6870059436 tuned: 130545068