import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from catalyst import dl
import sys
import os
sys.path.append('../../')
from torch_integral import IntegralWrapper
from torch_integral import UniformDistribution
from torch_integral import base_continuous_dims


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            1, 16, 3, padding=1, bias=True,
            padding_mode='replicate'
        )
        self.conv_2 = nn.Conv2d(
            16, 32, 5, padding=2, bias=True,
            padding_mode='replicate'
        )
        self.conv_3 = nn.Conv2d(
            32, 64, 5, padding=2, bias=True,
            padding_mode='replicate'
        )
        self.f_1 = nn.ReLU()
        self.f_2 = nn.ReLU()
        self.f_3 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.f_1(self.conv_1(x))
        x = self.pool(x)
        x = self.f_2(self.conv_2(x))
        x = self.pool(x)
        x = self.f_3(self.conv_3(x))
        x = self.pool(x)
        x = self.linear(x[:, :, 0, 0])

        return x


# ------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

root = os.path.expanduser('~')
train_dataset = torchvision.datasets.MNIST(
    root=root, train=True,
    download=True, transform=transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle=True
)

val_dataset = torchvision.datasets.MNIST(
    root=root, train=False,
    download=True, transform=transform
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size, shuffle=False
)
loaders = {'train': train_dataloader, 'valid': val_dataloader}

# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
model = MnistNet().cuda()
# model.load_state_dict(
#     torch.load('./logs/mnist/checkpoints/discrete_model.pth')
# )

continuous_dims = base_continuous_dims(model)
continuous_dims.update({
    'linear.weight': [1],
    'linear.bias': [],
    'conv_1.weight': [0]
})

wrapper = IntegralWrapper(
    init_from_discrete=True, fuse_bn=True,
    optimize_iters=0, start_lr=1e-2
)
model = wrapper(
    model, [1, 1, 28, 28], continuous_dims
)
ranges = [[16, 16], [16, 32], [32, 64]]
model.reset_distributions([
    UniformDistribution(*r) for r in ranges
])

# ------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------
opt = torch.optim.Adam(
    model.parameters(), lr=1e-2,
)
loader_len = len(train_dataloader)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, [loader_len*3, loader_len*5, loader_len*7, loader_len*9], 
    gamma=0.5
)
cross_entropy = nn.CrossEntropyLoss()

log_dir = './logs/mnist'

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
model.resize([16, 16, 32])
model = model.transform_to_discrete()

metrics = runner.evaluate_loader(
    model=model,
    loader=loaders["valid"],
    callbacks=callbacks[:-1]
)
