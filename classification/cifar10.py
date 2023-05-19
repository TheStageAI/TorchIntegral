import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from catalyst import dl
from pytorchcv.model_provider import get_model
import sys
sys.path.append('../')

from torch_integral import IntegralWrapper
from torch_integral import NormalDistribution
from torch_integral import base_continuous_dims
from torch_integral import grid_tuning


def nin_cifar10(pretrained=True):
    net = get_model('nin_cifar10', pretrained=pretrained)
    net.features.stage2.dropout2 = torch.nn.Identity()
    net.features.stage3.dropout3 = torch.nn.Identity()

    return net


def resnet20(pretrained=True):
    return get_model('resnet20_cifar10', pretrained=pretrained)


# DATA
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='/home/azim/datasets/', train=True, 
    download=False, transform=transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle=True
)

val_dataset = torchvision.datasets.CIFAR10(
    root='/home/azim/datasets/', train=False, 
    download=False, transform=transform
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size, shuffle=False
)
loaders = {'train': train_dataloader, 'valid': val_dataloader}

# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
model = resnet20().cuda()

continuous_dims = {}
# continuous_dims = base_continuous_dims(model)
continuous_dims.update({
    'features.init_block.conv.weight': [0],
    'output.weight': [1],
    'output.bias': []
})

model = IntegralWrapper(
    init_from_discrete=True, fuse_bn=True,
    optimize_iters=10, start_lr=1e-2, verbose=False
).wrap_model(model, [1, 3, 32, 32], continuous_dims=continuous_dims)


class ChangeDistribution(dl.Callback):
    def __init__(self, max_epoch=64):
        super().__init__(order=dl.CallbackOrder.External)
        self.max_epoch = max_epoch

    def on_epoch_end(self, runner: "IRunner") -> None:
        if runner.is_train_loader and runner.epoch_step < self.max_epoch:
            if runner.epoch_step % 2 == 1:
                for i in range(1, 2):
                    min_val = 64 - runner.epoch_step//2
                    dist = NormalDistribution(min_val, 64)
                    runner.model.groups()[-i].reset_distribution(dist)


# ------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------
opt = torch.optim.Adam(
    model.parameters(), lr=1e-3, #weight_decay=1e-4
)
epoch_len = len(train_dataloader)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, [epoch_len*10, epoch_len*20, epoch_len*30, epoch_len*50], 
    gamma=0.33
)
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

# for group in model.groups[-1:]:
#     new_size = int(group.grid().size() * 0.8)
#     group.resize(new_size)

# with grid_tuning(model, False):

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
model.groups()[-1].resize(33)
metrics = runner.evaluate_loader(
    model=model,
    loader=loaders["valid"],
    callbacks=callbacks[:1]
)


