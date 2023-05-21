import torch
import torch.nn as nn
from .grid import UniformDistribution
from .grid import RandomUniformGrid1D
from .grid import CompositeGrid1D
from .grid import GridND
from .graph import Tracer
from .parametrizations import WeightsParameterization
from .parametrizations import InterpolationWeights1D
from .parametrizations import InterpolationWeights2D
from .permutation import NOptPermutation
from .utils import get_parent_name
from .utils import get_parent_module
from .utils import fuse_batchnorm
from .quadrature import TrapezoidalQuadrature
from torch.nn.utils import parametrize


class IntegralGroup(nn.Module):
    def __init__(self, grid_1d, parameterizations):
        super(IntegralGroup, self).__init__()
        self.grid_1d = grid_1d
        self.parameterizations = parameterizations
        self.reset_grid(grid_1d)

    def grid(self):
        return self.grid_1d

    def size(self):
        return self.grid_1d.size()

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


class IntegralModel(nn.Module):
    def __init__(self, model, groups):
        super(IntegralModel, self).__init__()
        self.model = model
        groups.sort(key=lambda x: x.count_elements())
        self.groups = nn.ModuleList(groups)

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
            params = list(group.params)

            if group.parent is not None:
                if self.verbose:
                    print(f'Rearranging of group {i}')

                start = 0

                for j, another_group in enumerate(
                    group.parent.subgroups
                ):
                    if group is not another_group:
                        start += another_group.size
                    else:
                        break

                for p in group.parent.params:
                    params.append({
                        'value': p['value'], 'dim': p['dim'],
                        'start_index': start
                    })

            self.rearranger.permute(params, group.size)

    def wrap_model(self, model, example_input, continuous_dims):
        tracer = Tracer(
            model, example_input, continuous_dims,
        )

        if self.fuse_bn:
            model = self._fuse(model, tracer)

        groups = tracer.build_groups()
        continuous_dims = tracer.continuous_dims

        if self.init_from_discrete and self.rearranger is not None:
            self._rearrange(groups)

        integral_groups = []
        composite_groups = []

        for group in groups:
            if group.subgroups is None:
                distrib = UniformDistribution(group.size, group.size)
                group.grid = RandomUniformGrid1D(distrib)

        for group in groups: # PUT THIS PEACE IN TRACER
            if group.parent is not None:
                if group.parent not in composite_groups:
                    composite_groups.append(group.parent)
                    group.parent.grid = CompositeGrid1D([
                        sub.grid for sub in group.parent.subgroups
                    ])

        for group in groups + composite_groups:
            parameterizations = []

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

                    groups = p['value'].grids
                    grids_dict = {}

                    for i, g in enumerate(groups):
                        if hasattr(g, 'grid') and g.grid is not None:
                            grids_dict[str(i)] = g.grid

                    grid = GridND(grids_dict)
                    delattr(p['value'], 'grids')

                    parameterization = WeightsParameterization(
                        w_func, grid, quadrature
                    ).to(p['value'].device)

                    target = torch.clone(p['value'])

                    parametrize.register_parametrization(
                        parent, name, parameterization, unsafe=True
                    )

                    if self.init_from_discrete:
                        self._optimize_parameters(
                            parent, p['name'], target,
                        )

                else:
                    parameterization = parent.parametrizations[name][0]

                parameterizations.append([parameterization, p['dim']])

            integral_groups.append(
                IntegralGroup(group.grid, parameterizations)
            )

        integral_model = IntegralModel(model, integral_groups)

        return integral_model

    def _optimize_parameters(self, module, name, target):

        module.train()
        parent_name, attr = get_parent_name(name)
        criterion = torch.nn.MSELoss()
        opt = torch.optim.Adam(module.parameters(), lr=self.start_lr)
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
