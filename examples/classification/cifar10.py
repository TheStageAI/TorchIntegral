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
    UniformDistribution, base_continuous_dims,
    grid_tuning, TrainableGrid1D
)


def nin_cifar10(pretrained=True):
    net = get_model('nin_cifar10', pretrained=pretrained)
    net.features.stage2.dropout2 = torch.nn.Identity()
    net.features.stage3.dropout3 = torch.nn.Identity()

    return net


def resnet20(pretrained=True):
    return get_model('resnet20_cifar10', pretrained=pretrained)


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
model = resnet20().cuda()

continuous_dims = base_continuous_dims(model)
continuous_dims.update({
    'features.init_block.conv.weight': [0],
    'output.weight': [1],
    'output.bias': []
})

model = IntegralWrapper(
    init_from_discrete=True, fuse_bn=True,
    permutation_iters=1000, optimize_iters=0,
    start_lr=1e-2, verbose=True
)(model, [1, 3, 32, 32], continuous_dims=continuous_dims)


class ChangeDistribution(dl.Callback):
    def __init__(self, max_epoch=64):
        super().__init__(order=dl.CallbackOrder.External)
        self.max_epoch = max_epoch

    def on_epoch_end(self, runner: "IRunner") -> None:
        if runner.epoch_step < self.max_epoch:
            if runner.epoch_step % 2 == 0:
                for group in model.groups:
                    if len(group.params) < 5 and group.grid_size() > 32:
                        min_val = 64 - runner.epoch_step//2
                        dist = UniformDistribution(min_val, 64)
                        group.reset_distribution(dist)
                        group.resize(min_val)
                        print("new dist: ", min_val, 64)


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
    ChangeDistribution(),
    dl.SchedulerCallback(
        mode='batch', loader_key='train', metric_key='loss'
    )
]
loggers = []
epochs = 100

# for group in model.groups:
#     if len(group.params) < 5 and group.grid_size() > 32:
#         new_size = int(float(group.grid_size()) * 0.5)
#         group.reset_distribution(UniformDistribution(new_size, 64))
#         group.resize(new_size)
#         group.reset_grid(TrainableGrid1D(new_size))

print('compression: ', model.eval().calculate_compression())


# with grid_tuning(model, False, True):
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
epoch_len = len(train_dataloader)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, [epoch_len*40, epoch_len*70, epoch_len*80, epoch_len*90],
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
    minimize_valid_metric=True,
    cpu=False,
    verbose=True,
    fp16=False
)

# ------------------------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------------------------
metrics = runner.evaluate_loader(
    model=model,
    loader=loaders["valid"],
    callbacks=callbacks[:1]
)
