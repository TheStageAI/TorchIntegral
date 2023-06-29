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
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)
    wrapper = inn.IntegralWrapper(init_from_discrete=True)

    # Specify continuous dimensions which you want to prune
    continuous_dims = {
        "layer4.0.conv1.weight": [0],
        "layer4.1.conv1.weight": [0, 1]
    }

    # Convert to integral model
    inn_model = wrapper(model, (1, 3, 224, 224), continuous_dims)

    # Reset distribution of number of random channels
    inn_model.groups[0].reset_distribution(inn.UniformDistribution(200, 512))
    inn_model.groups[1].reset_distribution(inn.UniformDistribution(300, 512))

    # Train model with usual training methods
    ...

Integral model will have 2 integral layers: layer4.0.conv1 and layer4.1.conv1. 
First one is continuous along output channels and second is continuous along both input and output channels.

Resample (prune) integral model.
---------------------

.. code-block:: python

    # Resample integral model to arbitrary number of channels in range [200, 512] and [300, 512]
    inn_model.groups[0].resample(200)
    inn_model.groups[1].resample(300)

    # Evaluate model as usual discrete model
    ...


Grid tuning
---------------------
Convert pre-trained DNN to INN and train only integration partitions of the model.

.. code-block:: python

    # Specify continuous dimensions which you want to prune
    continuous_dims = {
        "layer4.0.conv1.weight": [0],
        "layer4.1.conv1.weight": [0, 1]
    }

    # Convert to integral model
    inn_model = wrapper(model, (1, 3, 224, 224), continuous_dims)

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

    inn.groups[1].reset_grid(inn.TrainableGrid1D(256))

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

    .. Tutorial 1 <tutorial_1.rst>
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
    Permutation <permutation.rst>
