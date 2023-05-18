import torch
import torch.nn as nn
from .grid import UniformDistribution
from .grid import RandomUniformGrid1D
from .grid import GridND
from .graph import build_groups
from .parametrizations import WeightsParameterization
from .parametrizations import InterpolationWeights1D
from .parametrizations import InterpolationWeights2D
from .permutation import NOptPermutation
from .utils import get_parent_name
from .utils import get_parent_module
from .utils import fuse_batchnorm
from .utils import optimize_parameters
from .utils import get_continuous_parameters
from .quadrature import TrapezoidalQuadrature
from torch.nn.utils import parametrize


class IntegralGroup(torch.nn.Module):
    def __init__(self, grid_1d, parameterizations):
        super(IntegralGroup, self).__init__()
        self.grid_1d = grid_1d
        self.parameterizations = parameterizations
        self.reset_grid(grid_1d)

    def grid(self):
        return self.grid_1d

    def reset_grid(self, grid_1d):
        self.grid_1d = grid_1d

        for obj, dim in self.parameterizations:
            obj.grid.reset_grid(dim, grid_1d)
            obj.clear()

    def resize(self, new_size):
        if hasattr(self.grid_1d, 'resize'):
            self.grid_1d.resize(new_size)

            for obj, _ in self.parameterizations:
                obj.clear()

    def reset_distribution(self, distribution):
        if hasattr(self.grid_1d, 'distribution'):
            self.grid_1d.distribution = distribution

    def count_elements(self):
        num_el = 0

        for p, dim in self.parameterizations:
            flag = False

            if p.training:
                p.eval()
                flag = True

            weight = p(None)
            num_el += weight.numel()

            if flag:
                p.train()

        return num_el


class IntegralModel(torch.nn.Module):
    def __init__(self, model, groups):
        super(IntegralModel, self).__init__()
        self.model = model
        groups.sort(key=lambda x: x.count_elements())
        self.groups = torch.nn.ModuleList(groups)

    def forward(self, x):
        for group in self.groups:
            group.grid().generate_grid()

        return self.model(x)

    def resize(self, sizes):
        for group, size in zip(self.groups, sizes):
            group.resize(size)

    def reset_grids(self, grids_1d):
        for group, grid_1d in zip(self.groups, grids_1d):
            group.reset_grid(grid_1d)

    def reset_distributions(self, distributions):
        for group, dist in zip(self.groups, distributions):
            group.reset_distribution(dist)

    def group_sizes(self):
        return [
            group.count_elements() for group in self.groups
        ]

    def grids(self):
        return [
            group.grid() for group in self.groups
        ]

    def transform_to_discrete(self):
        for name, param in self.model.named_parameters():
            parent = get_parent_module(self.model, name)
            parent_name, attr_name = get_parent_name(name)

            if parametrize.is_parametrized(parent, attr_name):
                weight_tensor = getattr(parent, attr_name)
                parent.parametrizations.pop(attr_name)
                setattr(parent, attr_name, weight_tensor)

        return self.model


class IntegralWrapper:
    def __init__(self, fuse_bn=True, init_from_discrete=True,
                 optimize_iters=100, start_lr=1e-2,
                 permutation_config=None, build_functions=None,
                 permutation_iters=100, verbose=True):

        self.init_from_discrete = init_from_discrete
        self.fuse_bn = fuse_bn
        self.optimize_iters = optimize_iters
        self.start_lr = start_lr
        self.build_functions = build_functions
        self.verbose = verbose

        if permutation_config is not None:
            permutation_class = permutation_config.pop('class')
            self.rearranger = permutation_class(**permutation_config)

        elif self.init_from_discrete:
            self.rearranger = NOptPermutation(permutation_iters)

    def wrap_model(self, model, example_input, cont_parameters=None):

        if self.fuse_bn:
            model.eval()
            model = fuse_batchnorm(model)

        cont_parameters = get_continuous_parameters(
            model, cont_parameters
        )
        groups = build_groups(
            model, example_input, cont_parameters
        )

        if self.init_from_discrete and self.rearranger is not None:
            for i, group in enumerate(groups):
                print(f'Rearranging of group {i}')
                self.rearranger.permute(
                    group['params'], group['size']
                )

        integral_groups = []

        for group in groups:
            distrib = UniformDistribution(group['size'], group['size'])
            grid_1d = RandomUniformGrid1D(distrib)
            group['grid'] = grid_1d

        for group in groups:
            parameterizations = []

            for p in group['params']:
                parent_name, name = get_parent_name(p['name'])
                parent = get_parent_module(model, p['name'])

                if not parametrize.is_parametrized(parent, name):

                    if isinstance(parent, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                        build_function = build_base_parameterization
                    else:
                        build_function = self.build_functions[type(parent)]

                    dims = cont_parameters[p['name']][1]
                    w_func, quadrature = build_function(parent, name, dims)
                    g_dict = {
                        str(i): g['grid']
                        for i, g in enumerate(p['value'].grids)
                        if g is not None
                    }
                    delattr(p['value'], 'grids')
                    grid = GridND(g_dict)

                    parameterization = WeightsParameterization(
                        w_func, grid, quadrature
                    ).to(p['value'].device)

                    target = torch.clone(p['value'])

                    parametrize.register_parametrization(
                        parent, name, parameterization, unsafe=True
                    )

                    if self.init_from_discrete and self.optimize_iters > 0:
                        optimize_parameters(
                            parent, p['name'], target, self.start_lr,
                            self.optimize_iters, self.verbose
                        )

                else:
                    parameterization = parent.parametrizations[name][0]

                parameterizations.append([parameterization, p['dim']])

            integral_groups.append(
                IntegralGroup(group['grid'], parameterizations)
            )

        integral_model = IntegralModel(model, integral_groups)

        return integral_model


def build_base_parameterization(module, name, dims):
    quadrature = None
    func = None

    if 'weight' in name:
        weight = getattr(module, name)
        cont_shape = [
            weight.shape[d] for d in dims
        ]

        if weight.ndim > len(cont_shape):
            discrete_shape = [
                weight.shape[d] for d in range(weight.ndim)
                if d not in dims
            ]
        else:
            discrete_shape = None

        if len(cont_shape) == 2:
            func = InterpolationWeights2D(
                cont_shape, discrete_shape
            )
        elif len(cont_shape) == 1:
            func = InterpolationWeights1D(
                cont_shape[0], discrete_shape, dims[0]
            )

        if 1 in dims and weight.shape[1] > 3:
            grid_indx = 0 if len(cont_shape) == 1 else 1
            quadrature = TrapezoidalQuadrature([1], [grid_indx])

    elif 'bias' in name:
        bias = getattr(module, name)
        cont_shape = bias.shape[0]
        func = InterpolationWeights1D(cont_shape)

    return func, quadrature
