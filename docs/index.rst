.. TorchIntegral documentation master file, created by
   sphinx-quickstart You can adapt this file completely to your liking, 
   but it should at least contain the root `toctree` directive.

TorchIntegral
===========================================

TorchIntegral is official implementation of `Integral Neural Networks <https://openaccess.thecvf.com/content/CVPR2023/papers/Solodskikh_Integral_Neural_Networks_CVPR_2023_paper.pdf>`_ paper.

Quick Start
---------------------
Convert discrete neural network to integral.

.. code-block:: python
    
    import torch
    import torch_integral as inn

    class MnistNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv_2 = nn.Conv2d(16, 32, 5, padding=2)
            self.conv_3 = nn.Conv2d(32, 64, 5, padding=2)
            self.relu = nn.ReLU()
            self.pool = nn.AvgPool2d(2, 2)
            self.linear = nn.Linear(64, 10)

        def forward(self, x):
            x = self.relu(self.conv_1(x))
            x = self.pool(x)
            x = self.relu(self.conv_2(x))
            x = self.pool(x)
            x = self.relu(self.conv_3(x))
            x = self.pool(x)
            x = self.linear(x[:, :, 0, 0])
            return x


    model = MnistNet()
    wrapper = inn.IntegralWrapper(init_from_discrete=True)

    # Specify continuous dimensions which you want to prune
    continuous_dims = {'conv_1.weight': [0], 'conv_2.weight': [0]}

    # Convert to integral model
    inn_model = wrapper(model, example_input=(1, 1, 28, 28))

    # Reset distribution of number of random channels
    inn_model.groups[0].reset_distribution(inn.UniformDistribution(12, 16))
    inn_model.groups[1].reset_distribution(inn.UniformDistribution(18, 32))

    # Train model with usual training methods
    ...

Integral model will have 2 integral layers: conv_1 and conv_2. First one is continuous along output channels, 
second one is continuous along both input and output channels.

Resample (prune) integral model.
---------------------

.. code-block:: python

    # Resample integral model to arbitrary number of channels in range [12, 16] and [18, 32]
    inn_model.groups[0].resample(12)
    inn_model.groups[1].resample(20)

    # Evaluate model as usual discrete model
    ...


Grid tuning
---------------------
Train only integration partitions of INN.

.. code-block:: python

    # Specify continuous dimensions which you want to prune
    continuous_dims = {'conv_1.weight': [0], 'conv_2.weight': [0]}

    # Convert to integral model
    inn_model = wrapper(model, example_input=(1, 3, 28, 28))

    # Resample model to desired shape
    inn_model.groups[0].resample(8)
    inn_model.groups[1].resample(16)    

    with inn.grid_tuning(use_all_grids=True):
        optimizer = torch.optim.Adam(inn_model.parameters(), lr=1e-3)
    #    Train model with usual training methods
        ...

Here `use_all_grids` means that all grids will be replaced with `TrainableGrid1d`.
You can specify trainable grids manually:

.. code-block:: python

    inn.groups[1].reset_grid(inn.TrainableGrid(16))

    with inn.grid_tuning(use_all_grids=False):
        optimizer = torch.optim.Adam(inn_model.parameters(), lr=1e-3)
    #    Train model with usual training methods
        ...


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
    :caption: Quick Start
    :maxdepth: 2
    :hidden:

    self
    Examples <https://github.com/TheStageAI/TorchIntegral/tree/main/examples>

.. toctree::
    :caption: Tutorials
    :maxdepth: 2
    :hidden:

    Tutorial 1 <tutorial_1.rst>
    .. Tutorial 2 <tutorial_2.rst>
    .. Tutorial 3 <tutorial_3.rst>
    .. Tutorial 4 <tutorial_4.rst>

.. toctree::
    :caption: Core

    Model <model.rst>
    Graph <torch_integral.graph.rst>
    Quadrature <quadrature.rst>
    Grid <grid.rst>
    Parametrizations <torch_integral.parametrizations.rst>
    Permutation <permutatioin.rst>
