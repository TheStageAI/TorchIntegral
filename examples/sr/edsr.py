import argparse
import torch
from super_image import EdsrModel, ImageLoader
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
from super_image import Trainer, TrainingArguments
from datasets import load_dataset
import torch_integral as inn
from torch_integral.permutation import NOptOutFiltersPermutation
from torch_integral.utils import standard_continuous_dims
from PIL import Image


parser = argparse.ArgumentParser(description="INN EDSR")
parser.add_argument(
    "--checkpoint", default=None, help="path to model checkpoint (default: None)"
)
parser.add_argument("--results", default="results", help="save checkpoint directory")
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--integral", action="store_true", help="use integral neural network"
)
parser.add_argument(
    "--grid-tuning",
    action="store_true",
    help="tune only grid of integral neural network",
)
parser.add_argument(
    "--resample", action="store_true", help="prune integral neural network"
)
parser.add_argument(
    "--scale", default=4, type=int, help="super resolution scale (default: 4)"
)
parser.add_argument("--num_images", default=None, type=int, help="Size of dataset")
parser.add_argument(
    "--num_iters", default=4000, type=int, help="number of iterations per epoch"
)
parser.add_argument("-b", "--batch-size", default=32, type=int, metavar="N")
parser.add_argument("-w", "--workers", default=48, type=int)
parser.add_argument(
    "--epochs", default=400, type=int, metavar="N", help="number of total epochs to run"
)
args = parser.parse_args()

# DATA
augmented_dataset = load_dataset(
    "eugenesiow/Div2k", f"bicubic_x{args.scale}", split="train"
).map(augment_five_crop, batched=True, desc="Augmenting Dataset")

if args.num_images is not None:
    from copy import deepcopy

    TrainDataset.__len__ = lambda x: args.num_iters
    base_getitem = deepcopy(TrainDataset.__getitem__)
    TrainDataset.__getitem__ = lambda d, i: base_getitem(d, i % args.num_images)

train_dataset = TrainDataset(augmented_dataset)
eval_dataset = EvalDataset(
    load_dataset("eugenesiow/Div2k", f"bicubic_x{args.scale}", split="validation")
)

# MODEL
model = EdsrModel.from_pretrained("eugenesiow/edsr", scale=args.scale).cuda()

# REPLACE UPSAMPLE MODULE
conv = model.tail[0][0]
model.tail[0][0] = torch.nn.Conv2d(
    conv.in_channels,
    conv.in_channels,
    conv.kernel_size,
    conv.stride,
    conv.padding,
    bias=False,
)
model.tail[0][1] = torch.nn.Upsample(scale_factor=2)
conv = model.tail[0][2]
model.tail[0][2] = torch.nn.Conv2d(
    conv.in_channels,
    conv.in_channels,
    conv.kernel_size,
    conv.stride,
    conv.padding,
    bias=False,
)
model.tail[0][3] = torch.nn.Upsample(scale_factor=2)
model.cuda()

state_dict = {
    key.replace("module.", ""): value
    for key, value in torch.load("./discrete_results/pytorch_model_4x.pt").items()
}
model.load_state_dict(state_dict)

continuous_dims = standard_continuous_dims(model)
discrete_dims = {
    "sub_mean.weight": [0, 1],
    "sub_mean.bias": [0],
    "add_mean.weight": [0, 1],
    "add_mean.bias": [0],
    "head.0.weight": [1],
    "tail.0.0.weight": [0],
    "tail.0.2.weight": [0, 1],
    "tail.1.weight": [0, 1],
}
example_input = (1, 3, 32, 32)

if args.integral:
    model = inn.IntegralWrapper(
        init_from_discrete=(args.checkpoint is None),
        permutation_config={"class": NOptOutFiltersPermutation},
    )(model, example_input, continuous_dims, discrete_dims).cuda()

    # RESAMPLE
    for i, group in enumerate(model.groups[:-1]):
        if "operator" not in group.operations:
            size = 128
        else:
            size = 224

        group.reset_distribution(inn.UniformDistribution(size, 256))

        if args.resample:
            group.resize(size)

        if args.grid_tuning:
            group.reset_grid(inn.TrainableGrid1D(size))

if args.checkpoint is not None:
    state_dict = {
        key.replace("module.", ""): value
        for key, value in torch.load(args.checkpoint).items()
    }
    model.load_state_dict(state_dict)

if args.integral:
    print("Compression: ", model.eval().calculate_compression())

# TRAIN
training_args = TrainingArguments(
    output_dir=args.results,
    num_train_epochs=args.epochs,
    learning_rate=1e-4,
    per_device_train_batch_size=args.batch_size,
    dataloader_num_workers=args.workers,
    dataloader_pin_memory=True,
    gamma=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

if args.integral and args.grid_tuning:
    model.grid_tuning(False, True, False)

eval_len = EvalDataset.__len__
EvalDataset.__len__ = lambda x: 0

if not args.evaluate:
    # with torch.autocast(device_type='cuda'):
    trainer.train()

EvalDataset.__len__ = eval_len

# EVAL
trainer.eval(1)

# image = Image.open("0853x4.png")
# inputs = ImageLoader.load_image(image).cuda()
# preds = model(inputs)
# ImageLoader.save_image(preds, f"{args.results}/scaled_{args.scale}x.png")
# ImageLoader.save_compare(
#     inputs, preds, f"{args.results}/scaled_{args.scale}x_compare.png"
# )
