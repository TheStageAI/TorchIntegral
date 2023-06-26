import torch
from .operations import replace_operations
from .integral_group import IntegralGroup
from ..utils import remove_all_hooks


class Tracer:
    """Class for building dependency graph of the neural network.
    Parameters
    ----------
    model: torch.nn.Module.
    example_input: List[int].
    continuous_dims: Dict[str, List[int]].
        Dictionary which contains names of the model's parameters
        and it's continuous dimension indices.

    black_list_dims: Dict[str, List[int]].
        Dictionary which contains names of the model's parameters
        and dimensions that can not be continuous. 
        If there is the same element in black_list_dims and continuous_dims, then
        the element will be removed from continuous_dims.

    For example, if we have a model with two convolutional layers
    and we want to make continuous only first convolutional layer's 
    output dimension then we can write:

    import torch
    from torch_integral.graph import Tracer

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv_1 = torch.nn.Conv2d(3, 16, 3)
            self.conv_2 = torch.nn.Conv2d(16, 32, 3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv_1(x)
            x = self.relu(x)
            x = self.conv_2(x)
            x = self.relu(x)
            return x

    model = Model()
    example_input = torch.randn(1, 3, 32, 32)
    continuous_dims = {'conv_1.weight': [0], 'conv_1.bias': [0], 'conv_2.weight': [1]}
    tracer = Tracer(model, example_input, continuous_dims)

    Here  first dimension of the conv_1.weight, conv_1.bias and second dim
    of the conv_2.weight are belong to the same IntegralGroup, 
    because it's sizes should be equal.
    Note that in example above it is not necessary to list all of the layers.
    It is enough to list only one tensor of the group and all other tensors will be
    added automatically.
    """

    def __init__(self, model,
                 example_input,
                 continuous_dims,
                 black_list_dims=None):

        if black_list_dims is not None:
            self.black_list_dims = black_list_dims
        else:
            self.black_list_dims = {}

        self.continuous_dims = continuous_dims
        self.example_input = example_input
        self.model = model
        self.groups = None

    def _preprocess_parameters(self):
        """Creates IntegralGroup for each dimension of each parameter of the model."""
        self.groups = []

        for name, p in self.model.named_parameters():
            p.grids = [None] * p.ndim

            if name in self.continuous_dims:
                dims = self.continuous_dims[name]
            else:
                dims = list(range(p.ndim))

            for d in dims:
                size = p.shape[d]
                group = IntegralGroup(size)
                group.append_param(name, p, d)
                p.grids[d] = group
                self.groups.append(group)

    def _postprocess_groups(self):
        """Removes empty groups and build composite groups list."""
        delete_indices = []

        for i, group in enumerate(self.groups):
            delete_group = True

            for p in group.params:
                if p['name'] in self.continuous_dims and \
                        p['dim'] in self.continuous_dims[p['name']]:
                    delete_group = False

                if p['name'] in self.black_list_dims and \
                        p['dim'] in self.black_list_dims[p['name']]:

                    for d in group.params:
                        if d['name'] in self.continuous_dims:
                            self.continuous_dims[d['name']].remove(d['dim'])

                    delete_group = True
                    break

            if delete_group:
                delete_indices.append(i)
            else:
                for p in group.params:
                    if p['name'] in self.continuous_dims:
                        dims = self.continuous_dims[p['name']]

                        if p['dim'] not in dims:
                            dims.append(p['dim'])
                    else:
                        self.continuous_dims[p['name']] = [p['dim']]

        self.groups = [
            group for i, group in enumerate(self.groups)
            if i not in delete_indices
        ]
        parents = set()

        for group in self.groups:
            self._add_parent_groups(group, parents)
            group.build_operations_set()

        for parent in parents:
            parent.build_operations_set()

        self.groups.extend(list(parents))

    def _add_parent_groups(self, group, parents):
        for parent in group.parents:
            if parent not in parents:
                parents.add(parent)
            self._add_parent_groups(parent, parents)

    def build_groups(self):
        """Builds dependency groups of the neural network.

        Returns
        -------
        self.groups: List[IntegralGroup].
            Base groups which grids is not concatenation of another grids.
        parents: List[IntegralGroups].
            List of composite groups. The grid of each composite group
            is the concatenation of another grids.
        """
        self.model.eval()
        tracing_model = replace_operations(self.model)
        self._preprocess_parameters()
        device = next(iter(self.model.parameters())).device

        if type(self.example_input) == torch.Tensor:
            x = self.example_input.to(device)
        else:
            x = torch.rand(self.example_input).to(device)

        tracing_model(x)
        remove_all_hooks(tracing_model)
        del tracing_model
        self.groups = [
            group for group in self.groups if len(group.params) != 0
        ]
        self._postprocess_groups()

        return self.groups
