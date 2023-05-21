from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests
from super_image import Trainer, TrainingArguments, EdsrModel, EdsrConfig
from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
import sys
sys.path.append('../')
import torch_integral
from torch_integral.permutation import RandomPermutation


# DATA
augmented_dataset = load_dataset(
    'eugenesiow/Div2k', 'bicubic_x4', split='train'
).map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset)
eval_dataset = EvalDataset(
    load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation')
)

# MODEL
model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=4).cuda()
continuous_dims = {}
continuous_dims.update({
    'head.0.weight': [0],
    'tail.0.0.weight': [1],
    'tail.0.0.bias': [],
    'tail.1.weight': [],
    'tail.1.bias': []
})
model = torch_integral.IntegralWrapper(
    init_from_discrete=True, optimize_iters=0,
    start_lr=1e-2, permutation_iters=2,
    permutation_config={'class': RandomPermutation}
).wrap_model(model, [1, 3, 32, 32], continuous_dims)

# TRAIN
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

print([size for size in model.group_sizes()])
model.groups[-1].resize(100)

with torch_integral.grid_tuning(model):
    trainer.train()

url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = ImageLoader.load_image(image).cuda()
preds = model(inputs)
ImageLoader.save_image(preds, './scaled_4x.png')
ImageLoader.save_compare(inputs, preds, './scaled_4x_compare.png')
