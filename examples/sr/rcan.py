import torch
from super_image import RcanModel, ImageLoader
from super_image.models.rcan.configuration_rcan import RcanConfig
from PIL import Image
import requests
import sys
sys.path.append('../../')
import torch_integral
from torch_integral.permutation import RandomPermutation


scale = 4
config = RcanConfig(**{
    "scale": scale,
    "bam": True,
    "data_parallel": False,
    "model_type": "RCAN",
    "n_colors": 3,
    "n_feats": 64,
    "n_resblocks": 20,
    "n_resgroups": 10,
    "reduction": 16,
    "res_scale": 1,
    "rgb_mean": [0.4488, 0.4371, 0.404],
    "rgb_range": 255,
    "rgb_std": [1.0, 1.0, 1.0]
})

# 'https://huggingface.co/eugenesiow/rcan-bam/resolve/main/pytorch_model_4x.pt'
model = RcanModel(config)
model.load_state_dict(torch.load('./pytorch_model_4x.pt'))
model.cuda()

cont_dims = {}
cont_dims.update({
    'head.0.weight': [0],
    'tail.0.0.weight': [1],
    'tail.0.0.bias': [],
    'tail.1.weight': [],
    'tail.1.bias': []
})

model = torch_integral.IntegralWrapper(
    init_from_discrete=True, fuse_bn=True,
    optimize_iters=0, start_lr=1e-2, permutation_iters=2,
    permutation_config={'class': RandomPermutation}
).wrap_model(model, [1, 3, 32, 32], cont_dims)

for group in model.groups[:-1]:
    size = model.grids()[0].size()
    if size > 1:
        new_size = int(0.95 * size)
        print(size, new_size)
        group.resize(new_size)

url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = ImageLoader.load_image(image).cuda()
preds = model(inputs)
ImageLoader.save_image(preds, './scaled_4x.png')
ImageLoader.save_compare(inputs, preds, './scaled_4x_compare.png')
