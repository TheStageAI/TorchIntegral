from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from catalyst import dl
# from pytorchcv.model_provider import get_model
import sys
sys.path.append('../../')
from torch_integral import IntegralWrapper
from torch_integral import NormalDistribution
from torch_integral import base_continuous_dims
from torch_integral import grid_tuning


# DATA
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = torchvision.datasets.ImageNet(
    root='/home/azim/datasets/', train=True, 
    download=False, transform=transform
)
train_dataloader = DataLoader(
    train_dataset, batch_size, shuffle=True
)

val_dataset = torchvision.datasets.ImageNet(
    root='/home/azim/datasets/', train=False, 
    download=False, transform=transform
)
val_dataloader = DataLoader(
    val_dataset, batch_size, shuffle=False
)
loaders = {'train': train_dataloader, 'valid': val_dataloader}


# MODEL
model = torchvision.models.resnet18(True)