import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import sys
sys.path.append('../../')
import torch_integral
from torch_integral.permutation import RandomPermutation
from torch_integral.utils import get_parent_module
from fvcore.nn import FlopCountAnalysis
from torch_integral.utils import base_continuous_dims


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth')

# cont_dims = {}
# lst_names = [f'conv{i}' for i in range(2, 5)]
# 
# for name, param in model.model.named_parameters():
#     last_part = name.split('.')[-2:]
#     
#     if last_part[0] in lst_names:
#         if last_part[1] == 'weight':
#             cont_dims[name] = [0, 1]
#         elif last_part[1] == 'bias':
#             cont_dims[name] = [0]
#             
#     elif last_part[0] == 'conv5' and last_part[1] == 'weight':
#         cont_dims[name] = [0]
#         
#     elif last_part[0] == 'conv1':
#         cont_dims[name] = [0]

cont_dims = base_continuous_dims(model.model)
cont_dims.update({
    'conv_first.weight': [0],
    'conv_first.bias': [0],
    'conv_body.weight': [0, 1],
    'conv_body.bias': [0],
    'conv_up1.weight': [1],
    'conv_up1.bias': [],
    'conv_up2.weight': [],
    'conv_up2.bias': [],
    'conv_hr.weight': [],
    'conv_hr.bias': [],
    'conv_last.weight': [],
    'conv_last.bias': []
})

model.model = torch_integral.IntegralWrapper(
    init_from_discrete=True, fuse_bn=True, verbose=True,
    optimize_iters=0, permutation_iters=20,
    permutation_config={'class': RandomPermutation}
).wrap_model(model.model, [1, 3, 32, 32], cont_dims)

model.model.eval()

# for group in model.model.groups:
#     group.resize(24)

print('compressed: ', model.model.calculate_compression())

model.model = model.model.transform_to_discrete()

path_to_image = '../text2image/astronaut_rides_horse.png'
image = Image.open(path_to_image).convert('RGB')
sr_image = model.predict(image)
sr_image.save('./realesrgan_result_x4.png')


# DISTILLATION
# def save_feature_map_hook(module, input, output):
#     module.feature_map = output
# 
# 
# def register_hook(model, hook):
#     for name, param in model.model.named_parameters():
#         parent = get_parent_module(model.model, name)
#         parent.register_forward_hook(save_feature_map_hook)


# # DATA
# from super_image import Trainer, TrainingArguments
# from super_image.data import EvalDataset, TrainDataset, augment_five_crop
# from datasets import load_dataset
# 
# augmented_dataset = load_dataset(
#     'eugenesiow/Div2k', 'bicubic_x4', split='train'
# ).map(augment_five_crop, batched=True, desc="Augmenting Dataset")
# train_dataset = TrainDataset(augmented_dataset)
# eval_dataset = EvalDataset(
#     load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation')
# )
# 
# # TRAIN
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=2,
#     per_device_train_batch_size=16,
# )
# 
# trainer = Trainer(
#     model=model.model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset
# )
# 
# with torch_integral.grid_tuning(model):
#     trainer.train()
