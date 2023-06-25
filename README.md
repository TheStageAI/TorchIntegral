# TorchIntegral

## Table of contents
- [TorchIntegral](#torchintegral)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage examples](#usage-examples)
- [Frequently asked questiens](#frequently-asked-questiens)
- [TODO](#todo)
- [Further research](#further-research)
- [References](#references)

This library is official implementation of ["Integral Neural Networks"][paper_link] paper in Pytorch.

![Tux, the Linux mascot](Pipeline.png)

## Requirements
- pytorch 2.0+
- torchvision
- numpy
- scipy
- Cython
- catalyst
- pytorchcv

## Installation

Latest stable version:
```
pip install torchintegral
```
Latest on GitHub:
```
pip install git+https://github.com/TheStageAI/TorchIntegral.git
```

## Usage examples
Convert your model to integral model:
```
import torch
import torch_integral as inn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3, 32, 3)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        return x


model = Model()
wrapper = inn.IntegralWrapper(init_from_discrete=True)
continuous_dims = {'conv_1.weight': [1]}
inn_model = wrapper(model, example_input=(3, 28, 28))
```

Set distribution for number of integration points:
```
inn_model.groups[0].reset_distribution(inn.UniformDistribution(16, 48))
```
Train integral model using vanilla training methods, for example gradient descent. 
Ones the model is trained resample it to arbitrary size in range [16, 48] with:
```
inn_model.groups[0].resize(16)
```


More examples can be found in [`examples`](./examples) directory.

## Frequently asked questiens
See [FAQ](FAQ.md) for frequently asked questions.

## TODO
- Add more examples
- Add more tests
- Add more documentation
- Add more models

# Further research
Here is some ideas for community to continue this research:

## References
If this work was useful for you, please cite it with:
```
@InProceedings{Solodskikh_2023_CVPR,
    author    = {Solodskikh, Kirill and Kurbanov, Azim and Aydarkhanov, Ruslan and Zhelavskaya, Irina and Parfenov, Yury and Song, Dehua and Lefkimmiatis, Stamatios},
    title     = {Integral Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {16113-16122}
}
```
and
```
@misc{TorchIntegral,
	author={Solodskikh K., Kurbanov A.},
	title={TorchIntegral},
	year={2023},
	url={https://github.com/TheStageAI/TorchIntegral},
}
```

[paper_link]: https://openaccess.thecvf.com/content/CVPR2023/papers/Solodskikh_Integral_Neural_Networks_CVPR_2023_paper.pdf