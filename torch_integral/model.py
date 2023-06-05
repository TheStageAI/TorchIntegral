import copy
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from .grid import UniformDistribution
from .grid import RandomUniformGrid1D
from .grid import CompositeGrid1D
from .grid import GridND
from .graph import Tracer
from .parametrizations import WeightsParameterization
from .parametrizations import InterpolationWeights1D
from .parametrizations import InterpolationWeights2D
from .permutation import NOptPermutation
from .permutation import VarianceOptimizer
from .utils import get_parent_name
from .utils import get_parent_module
from .utils import fuse_batchnorm
from .quadrature import TrapezoidalQuadrature


class IntegralModel(nn.Module):
    def __init__(self, model, groups):
        super(IntegralModel, self).__init__()
        self.model = model
        # groups.sort(key=lambda x: x.count_elements())
        self.groups = nn.ModuleList(groups)
        self.orignal_size = 1.
        self.orignal_size = self.calculate_compression()

    def forward(self, x):
        for group in self.groups:
            group.grid.generate_grid()

        return self.model(x)

    def calculate_compression(self):
        out = 0

        for group in self.groups:
            group.grid.generate_grid()

        for name, param in self.model.named_parameters():
            name_parts = name.split('.')

            if len(name_parts) >= 3 and \
                    name_parts[-3] == 'parametrizations' and \
                    name_parts[-1] == 'original':

                attr_path = '.'.join(
                    name_parts[:-3] + [name_parts[-2]]
                )
                parent = get_parent_module(self.model, attr_path)
                param = getattr(parent, name_parts[-2])
                out += param.numel()

            elif len(name_parts) < 3 or \
                    name_parts[-3] != 'parametrizations':
                out += param.numel()

        return out / self.orignal_size

    def resize(self, sizes):
        for group, size in zip(self.groups, sizes):
            group.resize(size)

    def reset_grids(self, grids_1d):
        for group, grid_1d in zip(self.groups, grids_1d):
            group.reset_grid(grid_1d)

    def reset_distributions(self, distributions):
        for group, dist in zip(self.groups, distributions):
            group.reset_distribution(dist)

    def grids(self):
        return [
            group.grid for group in self.groups
        ]

    def __getattr__(self, item):
        if item in dir(self):
            out = super().__getattr__(item)
        else:
            out = getattr(self.model, item)

        return out

    def transform_to_discrete(self):
        for group in self.groups:
            group.grid.generate_grid()

        parametrizations = []

        for name, module in self.model.named_modules():
            for attr_name in ('weight', 'bias'):
                if parametrize.is_parametrized(module, attr_name):
                    parametrization = getattr(
                        module.parametrizations, attr_name
                    )[0]
                    parametrizations.append(
                        (module, attr_name, parametrization)
                    )
                    parametrize.remove_parametrizations(
                        module, attr_name, True
                    )

        discrete_model = copy.deepcopy(self.model)

        for p_data in parametrizations:
            module, attr_name, parametrization = p_data
            parametrize.register_parametrization(
                module, attr_name, parametrization, unsafe=True
            )

        return discrete_model


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
            self.rearranger = NOptPermutation(
                permutation_iters, verbose
            )

    def _fuse(self, model, tracer):
        tracer.build_groups()
        continuous_dims = tracer.continuous_dims
        integral_convs = []

        for name, param in model.named_parameters():
            if name in continuous_dims:
                parent = get_parent_module(model, name)
                dims = continuous_dims[name]

                if isinstance(parent, nn.Conv2d) and 0 in dims:
                    integral_convs.append(get_parent_name(name)[0])

        model.eval()
        model = fuse_batchnorm(model, integral_convs)
        tracer.model = model

        return model

    def _rearrange(self, groups):
        for i, group in enumerate(groups):
            tensors = list(group.params)

            if self.verbose:
                print(f'Rearranging of group {i}')

            for parent in group.parents:
                start = 0

                for j, another_group in enumerate(
                        parent.subgroups
                ):
                    if group is not another_group:
                        start += another_group.size
                    else:
                        break

                for p in parent.params:
                    tensors.append({
                        'name': p['name'],
                        'value': p['value'],
                        'dim': p['dim'],
                        'start_index': start,
                    })

            self.rearranger(tensors, group.size)
            VarianceOptimizer()(tensors, group.size)

    def _set_grid(self, group):
        if group.grid is None:
            if group.subgroups is not None:
                for subgroup in group.subgroups:
                    if subgroup.grid is None:
                        self._set_grid(subgroup)

                group.grid = CompositeGrid1D([
                    sub.grid for sub in group.subgroups
                ])
            else:
                distrib = UniformDistribution(group.size, group.size)
                group.grid = RandomUniformGrid1D(distrib)

        for parent in group.parents:
            if parent.grid is None:
                self._set_grid(parent)

    def wrap_model(self, model, example_input, continuous_dims):
        tracer = Tracer(
            model, example_input, continuous_dims,
        )

        if self.fuse_bn:
            model = self._fuse(model, tracer)

        groups, composite_groups = tracer.build_groups()
        continuous_dims = tracer.continuous_dims

        if self.init_from_discrete and self.rearranger is not None:
            self._rearrange(groups)

        for group in groups:
            self._set_grid(group)

        integral_groups = groups + composite_groups

        for group in integral_groups:
            parametrizations = []

            for p in group.params:
                parent_name, name = get_parent_name(p['name'])
                parent = get_parent_module(model, p['name'])

                if not parametrize.is_parametrized(parent, name):

                    if isinstance(parent, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                        build_function = build_base_parameterization
                    else:
                        build_function = self.build_functions[type(parent)]

                    dims = continuous_dims[p['name']]
                    w_func, quadrature = build_function(parent, name, dims)
                    grids_dict = {}

                    for i, g in enumerate(p['value'].grids):
                        if hasattr(g, 'grid') and g.grid is not None:
                            if g in groups + composite_groups:
                                grids_dict[str(i)] = g.grid

                    grid = GridND(grids_dict)
                    delattr(p['value'], 'grids')

                    parametrization = WeightsParameterization(
                        w_func, grid, quadrature
                    ).to(p['value'].device)

                    target = p['value'].detach().clone()
                    target.requires_grad = False

                    parametrize.register_parametrization(
                        parent, name, parametrization, unsafe=True
                    )

                    if self.init_from_discrete:
                        self._optimize_parameters(
                            parent, p['name'], target,
                        )

                else:
                    parametrization = parent.parametrizations[name][0]

                parametrizations.append([
                    p['name'], parametrization, p['dim']
                ])
                # p['function'] = parametrization
            group.parametrizations = parametrizations

        integral_model = IntegralModel(model, integral_groups)

        return integral_model

    def _optimize_parameters(self, module, name, target):
        module.train()
        parent_name, attr = get_parent_name(name)
        criterion = torch.nn.MSELoss()
        opt = torch.optim.Adam(
            module.parameters(), lr=self.start_lr, weight_decay=0.
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.optimize_iters // 5, gamma=0.2
        )

        if self.verbose:
            print(name)
            print(
                'loss before optimization: ',
                float(criterion(getattr(module, attr), target))
            )

        for i in range(self.optimize_iters):
            weight = getattr(module, attr)
            loss = criterion(weight, target)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()

            if i == self.optimize_iters - 1 and self.verbose:
                print('loss after optimization: ', float(loss))


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
