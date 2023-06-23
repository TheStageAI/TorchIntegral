import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
import torch
from super_image import EdsrModel, ImageLoader
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
from super_image import Trainer, TrainingArguments
from datasets import load_dataset
from PIL import Image
import requests
import sys
sys.path.append('../../')
import torch_integral
from torch_integral.permutation import NOptOutFiltersPermutation
from torch_integral.utils import standard_continuous_dims


scale = 4
batch_size = 64
epochs = 1000

# DATA
augmented_dataset = load_dataset(
    'eugenesiow/Div2k', f'bicubic_x{scale}', split='train'
).map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset)
eval_dataset = EvalDataset(
    load_dataset('eugenesiow/Div2k', f'bicubic_x{scale}', split='validation')
)

# MODEL
model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=scale).cuda()
continuous_dims = standard_continuous_dims(model)
continuous_dims.update({
    'sub_mean.weight': [],
    'sub_mean.bias': [],
    'add_mean.weight': [],
    'add_mean.bias': [],
    'head.0.weight': [0],
    'tail.0.0.weight': [1],
    'tail.0.0.bias': [],
    'tail.0.2.weight': [],
    'tail.0.2.bias': [],
    'tail.1.weight': [],
    'tail.1.bias': [],
})
example_input = [1, 3, 32, 32]
model = torch_integral.IntegralWrapper(
    init_from_discrete=True, optimize_iters=300,
    start_lr=1e-2, permutation_iters=300,
    permutation_config={'class': NOptOutFiltersPermutation}
)(model, example_input, continuous_dims).cuda()

# RESAMPLE
for i, group in enumerate(model.groups[:]):
    if 'operator' not in group.operations:
        size = 100 if i > 3 else 256
    else:
        size = 200

    group.reset_grid(torch_integral.TrainableGrid1D(size))

print('Compression: ', model.eval().calculate_compression())

# TRAIN
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    learning_rate=1e-4,
    dataloader_num_workers=48,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

with torch_integral.grid_tuning(model, False, True, False):
    trainer.train()

# EVAL
trainer.eval(1)

url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = ImageLoader.load_image(image).cuda()
preds = model(inputs)
ImageLoader.save_image(preds, './scaled_4x.png')
ImageLoader.save_compare(inputs, preds, './scaled_4x_compare.png')
