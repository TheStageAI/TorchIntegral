import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets
import torchvision.transforms as transforms
from catalyst import dl
from catalyst.engines import GPUEngine
from catalyst.engines import DataParallelEngine
from catalyst.engines import DistributedDataParallelEngine
import sys
sys.path.append('../../')
from torch_integral.utils import reset_batchnorm
from torch_integral import (
    UniformDistribution, NormalDistribution,
    IntegralWrapper, grid_tuning, TrainableGrid1D
)


# ------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------
batch_size = 256
# Preprocessing transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])
# Augmentation transforms
augmentation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])
train_dataset = datasets.ImageFolder(
    root='/workspace/imagenet/ILSVRC/Data/CLS-LOC/train/',
    transform=augmentation
)
val_dataset = datasets.ImageFolder(
    root='/workspace/imagenet/ILSVRC/Data/CLS-LOC/val/',
    transform=preprocess
)
train_dataloader = DataLoader(
    train_dataset, batch_size, shuffle=True,
    pin_memory=True, num_workers=24
)
val_dataloader = DataLoader(
    val_dataset, batch_size, shuffle=False,
    pin_memory=True, num_workers=24
)
loaders = {'train': train_dataloader, 'valid': val_dataloader}

# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).cuda()
continuous_dims = {}

for name, mod in model.named_modules():
    if 'layer3' in name or 'layer4' in name:
        if not isinstance(mod, torch.nn.BatchNorm2d):
            if hasattr(mod, 'weight'):
                continuous_dims[name + '.weight'] = [0, 1]
            if hasattr(mod, 'bias'):
                continuous_dims[name + '.bias'] = [0]

black_list_dims = {'fc.weight': [1]}
wrapper = IntegralWrapper(
    init_from_discrete=False, fuse_bn=True,
    permutation_iters=10, optimize_iters=0,
    start_lr=1e-2, verbose=True,
)
example_input = [1, 3, 224, 224]
model = wrapper(
    model, example_input,
    continuous_dims=continuous_dims,
    black_list_dims=black_list_dims
).cuda()
# torch.save(model.state_dict(), 'resnet18_converted.pth')
model.load_state_dict(torch.load('./resnet18_converted.pth'))

# ------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------
cross_entropy = torch.nn.CrossEntropyLoss()
log_dir = './logs/imagenet/'
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits",
    target_key="targets", loss_key="loss"
)


class ChangeDistribution(dl.Callback):
    def __init__(self, max_epoch=45):
        super().__init__(order=dl.CallbackOrder.External)
        self.max_epoch = max_epoch

    def on_epoch_start(self, runner: "IRunner") -> None:
        if runner.epoch_step <= self.max_epoch:
            for group in runner.model.module.groups:
                max_val = group.grid_size()
                min_val = int(max_val * (1 - runner.epoch_step/100))
                dist = UniformDistribution(min_val, max_val)
                group.reset_distribution(dist)
                group.resize(min_val)
                print("new dist: ", min_val, max_val)


callbacks = [
    dl.AccuracyCallback(
        input_key="logits", target_key="targets",
        topk=(1,), num_classes=1000, log_on_batch=True
    ),
    ChangeDistribution(),
    dl.SchedulerCallback(
        mode='batch', loader_key='train', metric_key='loss'
    )
]
loggers = []
epochs = 100

for group in model.groups:
    n = group.grid_size()

    if 'operator' not in group.operations:
        new_size = int(n * 0.6)
        group.resize(new_size)
#     group.reset_grid(TrainableGrid1D(new_size))
    else:
        print(group)

print("Compression: ", model.eval().calculate_compression())
# reset_batchnorm(model)

# with grid_tuning(model, True, True, True):
opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-8)
epoch_len = len(train_dataloader)
engine = DataParallelEngine()
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, [epoch_len*10, epoch_len*20, epoch_len*30, epoch_len*40], gamma=0.33
)
runner.train(
    model=model,
    criterion=cross_entropy,
    optimizer=opt,
    scheduler=sched,
    loaders=loaders,
    num_epochs=epochs,
    callbacks=callbacks,
    loggers=loggers,
    logdir=log_dir,
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    engine=engine,
    verbose=True,
)

# ------------------------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------------------------
metrics = runner.evaluate_loader(
    model=model,
    loader=loaders["valid"],
    callbacks=callbacks[:1],
    verbose=True,
    engine=engine
)
