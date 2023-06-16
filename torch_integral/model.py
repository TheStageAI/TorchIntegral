import copy
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from .grid import UniformDistribution
from .grid import RandomLinspace
from .grid import CompositeGrid1D
from .grid import GridND
from .graph import Tracer
from .graph.operations import replace_operations
from .graph.integral_group import IntegralGroup
from .parametrizations import IntegralParameterization
from .parametrizations import InterpolationWeights1D
from .parametrizations import InterpolationWeights2D
from .permutation import NOptPermutation
from .permutation import VariationOptimizer
from .quadrature import TrapezoidalQuadrature
from .grid import TrainableGrid1D
from .utils import get_parent_name
from .utils import get_parent_module
from .utils import fuse_batchnorm
from .utils import remove_all_hooks


class IntegralModel(nn.Module):
    def __init__(self, model, groups):
        super(IntegralModel, self).__init__()
        self.model = model
        self.groups = nn.ModuleList(groups)
        # Rename groups to integral_groups
        # sort integral_groups by size or importance
        self.orignal_size = 1.
        self.orignal_size = self.calculate_compression()

    def generate_grid(self):
        for group in self.groups:
            group.grid.generate_grid()

    def forward(self, x):
        self.generate_grid()

        return self.model(x)

    def calculate_compression(self):
        out = 0
        self.generate_grid()

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
        self.generate_grid()
        # discrete_model = copy.deepcopy(self.model)
        discrete_model = self.model

        for name, module in discrete_model.named_modules():
            for attr_name in ('weight', 'bias'):
                if parametrize.is_parametrized(module, attr_name):
                    parametrize.remove_parametrizations(module, attr_name, True)
                    # parametrizations = getattr(
                    #     module.parametrizations, attr_name
                    # )
                    # for key in parametrizations:
                    #     if isinstance(parametrizations[key], IntegralParameterization):
                    #         tensor = getattr(module, attr_name)
                    #         parametrizations.pop(key)
                    #         setattr(module, attr_name, tensor)

        return discrete_model

    def grid_tuning(self,
                    train_bn=False,
                    train_bias=False,
                    use_all_grids=False):

        if use_all_grids:
            for group in self.groups:
                if group.subgroups is None:
                    group.reset_grid(
                        TrainableGrid1D(group.grid_size())
                    )

        for name, param in self.named_parameters():
            parent = get_parent_module(self, name)

            if isinstance(parent, TrainableGrid1D) or\
               (isinstance(parent, torch.nn.BatchNorm2d) and train_bn) or\
               ('bias' in name and train_bias):

                param.requires_grad = True
            else:
                param.requires_grad = False


class IntegralWrapper:
    def __init__(self, fuse_bn=True, init_from_discrete=True,
                 optimize_iters=0, start_lr=1e-2,
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
            feature_maps = group.tensors

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
                    params.append({
                        'name': p['name'],
                        'value': p['value'],
                        'dim': p['dim'],
                        'start_index': start,
                    })
            # VariationOptimizer()(params, group.size)
            self.rearranger(params, feature_maps, group.size)

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
                group.grid = RandomLinspace(distrib)

        for parent in group.parents:
            if parent.grid is None:
                self._set_grid(parent)

    def _preprocess_parameters(self):
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
                            self.continuous_dims[d['name']].pop(d['dim'])

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

        return list(parents)

    def _add_parent_groups(self, group, parents):
        for parent in group.parents:
            if parent not in parents:
                parents.add(parent)
            self._add_parent_groups(parent, parents)

    def build_groups(self):
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
        parents = self._postprocess_groups()

        return self.groups, parents

    def __call__(self, model,
                 example_input,
                 continuous_dims,
                 black_list_dims=None):

        tracer = Tracer(
            model, example_input, continuous_dims, black_list_dims
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
            for p in group.params:
                parent_name, name = get_parent_name(p['name'])
                parent = get_parent_module(model, p['name'])

                if not parametrize.is_parametrized(parent, name) or all([
                    not isinstance(obj, IntegralParameterization)
                    for obj in parent.parametrizations[name]
                ]):

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

                    parametrization = IntegralParameterization(
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

                p['function'] = parametrization

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
