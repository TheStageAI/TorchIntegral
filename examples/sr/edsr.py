import argparse
import torch
from super_image import EdsrModel, ImageLoader
from super_image.data import EvalDataset, TrainDataset, augment_five_crop
from super_image import Trainer, TrainingArguments
from datasets import load_dataset
import torch_integral as inn
from torch_integral.permutation import NOptOutFiltersPermutation
from torch_integral.utils import standard_continuous_dims


parser = argparse.ArgumentParser(description='INN EDSR')
parser.add_argument('--checkpoint', default=None,
                    help='path to model checkpoint (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--integral', action='store_true',
                    help='use integral neural network')
parser.add_argument('--grid-tuning', action='store_true',
                    help='tune only grid of integral neural network')
parser.add_argument('--resample', action='store_true',
                    help='prune integral neural network')
parser.add_argument('--scale', default=4, type=int,
                    help='super resolution scale (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('-w', '--workers', default=48, type=int)
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()

# DATA
augmented_dataset = load_dataset(
    'eugenesiow/Div2k', f'bicubic_x{args.scale}', split='train'
).map(augment_five_crop, batched=True, desc="Augmenting Dataset")
train_dataset = TrainDataset(augmented_dataset)
eval_dataset = EvalDataset(
    load_dataset('eugenesiow/Div2k', f'bicubic_x{args.scale}', split='validation')
)

# MODEL
model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=args.scale).cuda()

if args.integral:
    continuous_dims = standard_continuous_dims(model)
    discrete_dims = {
        'sub_mean.weight' : [0, 1],
        'add_mean.weight' : [0, 1],
        'head.0.weight' : [1],
        'tail.0.0.weight': [0],
        'tail.0.2.weight': [0, 1],
        'tail.1.weight': [0, 1],
    }
    example_input = [1, 3, 32, 32]
    model = inn.IntegralWrapper(
        init_from_discrete=(args.checkpoint is None), 
        permutation_config={'class': NOptOutFiltersPermutation}
    )(model, example_input, continuous_dims, discrete_dims).cuda()

    # RESAMPLE
    for i, group in enumerate(model.groups):
        if 'operator' not in group.operations:
            size = 100 if i > 1 else 256
        else:
            size = 200

        if args.grid_tuning:
            group.reset_grid(inn.TrainableGrid1D(size))
        elif args.resample:
            group.reset_distribution(inn.UniformDistribution(size, 256))
            group.resize(size)

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))

if args.integral:
    print('Compression: ', model.eval().calculate_compression())

# TRAIN
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=args.epochs,
    learning_rate=1e-4,
    per_device_train_batch_size=args.batch_size,
    dataloader_num_workers=args.workers,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

if args.integral and args.grid_tuning:
    model.grid_tuning(False, True, False)

trainer.train()

# EVAL
trainer.eval(1)
