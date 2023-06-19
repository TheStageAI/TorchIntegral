import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from catalyst import dl
from pytorchcv.model_provider import get_model
import sys
import os
sys.path.append('../../')
from torch_integral import (
    IntegralWrapper, NormalDistribution,
    UniformDistribution, standard_continuous_dims,
    grid_tuning, TrainableGrid1D
)


def nin_cifar10(pretrained=True):
    net = get_model('nin_cifar10', pretrained=pretrained)
    net.features.stage2.dropout2 = torch.nn.Identity()
    net.features.stage3.dropout3 = torch.nn.Identity()

    return net

# DATA
batch_size = 128

augmentation = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

root = os.path.expanduser('~') + '/datasets/'
train_dataset = torchvision.datasets.CIFAR10(
    root=root, train=True,
    download=False, transform=augmentation
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle=True
)

val_dataset = torchvision.datasets.CIFAR10(
    root=root, train=False,
    download=False, transform=preprocess
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size, shuffle=False
)
loaders = {'train': train_dataloader, 'valid': val_dataloader}

# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
model = nin_cifar10().cuda()
continuous_dims = {}

for name, mod in model.named_modules():
    if 'stage3' in name:
        if not isinstance(mod, torch.nn.BatchNorm2d):
            if hasattr(mod, 'weight'):
                continuous_dims[name + '.weight'] = [0, 1]
            if hasattr(mod, 'bias'):
                continuous_dims[name + '.bias'] = [0]

model = IntegralWrapper(
    init_from_discrete=True, fuse_bn=True,
    permutation_iters=3000, optimize_iters=0,
    start_lr=1e-3, verbose=True
)(model, [1, 3, 32, 32], continuous_dims)

# ------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------
cross_entropy = nn.CrossEntropyLoss()
log_dir = './logs/cifar'
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits",
    target_key="targets", loss_key="loss"
)
callbacks = [
    dl.AccuracyCallback(
        input_key="logits", target_key="targets",
        topk=(1,), num_classes=10
    ),
    dl.SchedulerCallback(
        mode='batch', loader_key='train', metric_key='loss'
    )
]
loggers = []
epochs = 10

for group in model.groups:
    if 'operator' not in group.operations:
        n = group.size
        new_size = int(float(n) * 0.4)
        group.reset_grid(TrainableGrid1D(new_size))

print('compression: ', model.eval().calculate_compression())

with grid_tuning(model, False, True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    epoch_len = len(train_dataloader)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [epoch_len*2, epoch_len*5, epoch_len*6, epoch_len*8],
        gamma=0.33
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
        verbose=True,
    )

# ------------------------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------------------------
metrics = runner.evaluate_loader(
    model=model,
    loader=loaders["valid"],
    callbacks=callbacks[:1]
)
