import os
import torch
import argparse
from torchvision import models
from catalyst import dl
import torchvision.transforms as transforms
from torchvision import datasets
from catalyst.engines import GPUEngine
from catalyst.engines import DataParallelEngine
from torch_integral import UniformDistribution, IntegralWrapper


parser = argparse.ArgumentParser(description="INN IMAGENET")
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default="imagenet",
    help="path to dataset (default: imagenet)",
)
parser.add_argument(
    "--checkpoint", default=None, help="path to model checkpoint (default: None)"
)
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
    "--resample", action="store_true", help="prune integral neural network"
)
parser.add_argument(
    "--data-parallel", action="store_true", help="use data parallel engine"
)
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N")
parser.add_argument("-w", "--workers", default=48, type=int)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
args = parser.parse_args()

# DATA
traindir = os.path.join(args.data, "train")
valdir = os.path.join(args.data, "val")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True,
)
dataloaders = {"train": train_dataloader, "valid": val_dataloader}

# MODEL
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).cuda()
continuous_dims = {}

if args.integral:
    continuous_dims = {
        "layer4.0.conv1.weight": [0, 1],
        "layer4.0.conv2.weight": [0, 1],
        "layer4.0.downsample.0.weight": [0, 1],
        "layer4.1.conv1.weight": [0, 1],
        "layer4.1.conv2.weight": [0, 1],
    }
    discrete_dims = {"fc.weight": [1]}
    wrapper = IntegralWrapper(
        init_from_discrete=(args.checkpoint is None), permutation_iters=1000
    )
    model = wrapper(model, [1, 3, 224, 224], continuous_dims, discrete_dims)
    model.groups[-1].reset_distribution(UniformDistribution(338, 512))
    model.groups[-2].reset_distribution(UniformDistribution(338, 512))

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))

if args.resample:
    model.groups[-1].resize(338)
    model.groups[-2].resize(338)
    print("model compression: ", model.eval().calculate_compression())

# Train
log_dir = "./logs/imagenet/"
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)
callbacks = [
    dl.AccuracyCallback(
        input_key="logits",
        target_key="targets",
        topk=(1,),
        num_classes=1000,
        log_on_batch=True,
    ),
    dl.SchedulerCallback(mode="batch", loader_key="train", metric_key="loss"),
]
if args.data_parallel:
    engine = DataParallelEngine()
else:
    engine = GPUEngine()

if not args.evaluate:
    loggers = []
    cross_entropy = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    epoch_len = len(dataloaders["train"])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        [epoch_len * 10, epoch_len * 20, epoch_len * 30, epoch_len * 40],
        gamma=0.33,
    )
    runner.train(
        model=model,
        criterion=cross_entropy,
        optimizer=opt,
        scheduler=sched,
        loaders=dataloaders,
        num_epochs=args.epochs,
        callbacks=callbacks,
        loggers=loggers,
        logdir=log_dir,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        engine=engine,
        verbose=True,
    )

metrics = runner.evaluate_loader(
    model=model,
    loader=dataloaders["valid"],
    callbacks=callbacks[:1],
    verbose=True,
    engine=engine,
)
