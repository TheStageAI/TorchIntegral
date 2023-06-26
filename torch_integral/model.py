import copy
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from .grid import UniformDistribution
from .grid import RandomLinspace
from .grid import CompositeGrid1D
from .grid import GridND
from .graph import Tracer
from .parametrizations import IntegralParameterization
from .parametrizations import InterpolationWeights1D
from .parametrizations import InterpolationWeights2D
from .permutation import NOptPermutation
from .permutation import NOptOutFiltersPermutation
from .quadrature import TrapezoidalQuadrature
from .grid import TrainableGrid1D
from .utils import (
    reset_batchnorm, get_parent_name, fuse_batchnorm,
    get_parent_module, get_attr_by_name
)


class IntegralModel(nn.Module):
    """
    Contains original model with parametrized layers and IntegralGroups list.

    Parameters
    ----------
    model: torch.nn.Module.
    groups: List[torch_integral.graph.IntegralGroup].
    """

    def __init__(self, model, groups):
        super(IntegralModel, self).__init__()
        self.model = model
        groups.sort(key=lambda g: g.count_parameters())
        self.groups = nn.ModuleList(groups)
        # Rename groups to integral_groups
        self.original_size = None
        self.original_size = self.calculate_compression()

    def generate_grid(self):
        """Creates new grids in each group."""
        for group in self.groups:
            group.grid.generate_grid()

    def forward(self, x):
        """
        Performs forward pass of the model.

        Parameters
        ----------
        x: the same as wrapped model's input type.
        """
        self.generate_grid()

        return self.model(x)

    def calculate_compression(self):
        """
        Returns 1 - ratio of the size of the current
        model to the original size of the model.
        """
        out = 0
        self.generate_grid()

        for group in self.groups:
            group.clear()

        for name, param in self.model.named_parameters():
            if 'parametrizations.' not in name:
                out += param.numel()
            elif name.endswith('.original'):
                name = name.replace('.original', '')
                name = name.replace('parametrizations.', '')
                tensor = get_attr_by_name(self.model, name)
                out += tensor.numel()

        if self.original_size is not None:
            out = 1. - out / self.original_size

        return out

    def resize(self, sizes):
        """
        Resizes grids in each group.

        Parameters
        ----------
        sizes: List[int].
        """
        for group, size in zip(self.groups, sizes):
            group.resize(size)

    def reset_grids(self, grids_1d):
        for group, grid_1d in zip(self.groups, grids_1d):
            group.reset_grid(grid_1d)

    def reset_distributions(self, distributions):
        """
        Sets new distributions in each IntegralGroup.grid.

        Parameters
        ----------
        distributions: List[torch_integral.grid.Distribution].
        """
        for group, dist in zip(self.groups, distributions):
            group.reset_distribution(dist)

    def grids(self):
        """Returns list of grids of each integral group."""
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
        """Samples weights, removes parameterizations and returns discrete model."""
        self.generate_grid()
        parametrizations = []

        for name, module in self.model.named_modules():
            for attr_name in ('weight', 'bias'):
                # DELETE ONLY INTEGRAL PARAM
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

    def grid_tuning(self, train_bn=False, train_bias=False, use_all_grids=False):
        """
        Sets requires_grad = False for all parameters except TrainableGrid's parameters,
        biases and BatchNorm parameters (if corresponding flag is True).

        Parameters
        ----------
        train_bn: bool.
        train_bias: bool.
        use_all_grids: bool.
        """
        if use_all_grids:
            for group in self.groups:
                if group.subgroups is None:
                    group.reset_grid(TrainableGrid1D(group.grid_size()))

        for name, param in self.named_parameters():
            parent = get_parent_module(self, name)

            if isinstance(parent, TrainableGrid1D):
                param.requires_grad = True
            else:
                param.requires_grad = False

        if train_bn:
            reset_batchnorm(self)

        if train_bias:
            for group in self.groups:
                for p in group.params:
                    if 'bias' in p['name']:
                        parent = get_parent_module(self.model, p['name'])
                        if parametrize.is_parametrized(parent, 'bias'):
                            parametrize.remove_parametrizations(
                                parent, 'bias', True
                            )
                        getattr(parent, 'bias').requires_grad = True


class IntegralWrapper:
    """
    Wrapper class which allows batch norm fusion,
    permutation of tensor parameters to obtain continuous structure in the tensor
    and convertation of discrete model to integral.

    Parameters
    ----------
    init_from_discrete: bool.
        If set True, then parametrization will be optimized with
        gradient descent to approximate discrete model's weights.

    fuse_bn: bool.
        If True, then convolutions and batchnorms will be fused.

    optimize_iters: int.
        Number of optimization iterations for discerete weight tensor approximation.

    start_lr: float.
        Learning rate when optimizing parametrizations.

    permutation_config: dict.
        Arguments of permutation method.

    build_functions: dict.
        Dictionary with keys

    permutation_iters: int.
        Number of iterations of total variation optimization process.

    verbose: bool.
    """

    def __init__(self, init_from_discrete=True, fuse_bn=True,
                 optimize_iters=0, start_lr=1e-2,
                 permutation_config=None, build_functions=None,
                 permutation_iters=100, verbose=True):

        self.init_from_discrete = init_from_discrete
        self.fuse_bn = fuse_bn
        self.optimize_iters = optimize_iters
        self.start_lr = start_lr
        self.build_functions = build_functions
        self.verbose = verbose
        self.rearranger = None

        if permutation_config is not None:
            permutation_class = permutation_config.pop('class')
            self.rearranger = permutation_class(**permutation_config)

        elif self.init_from_discrete and permutation_iters > 0:
            self.rearranger = NOptPermutation(
                permutation_iters, verbose
            )

    def _fuse(self, model, continuous_dims):
        """
        Fuses batchnorm with convolution layer only if
        output dimension of convolution is continuous.

        Parameters
        ----------
        model: torch.nn.Module.
        continuous_dims: Dict[str, List[int]].
        """
        integral_convs = []

        for name, _ in model.named_parameters():
            if name in continuous_dims:
                parent = get_parent_module(model, name)
                dims = continuous_dims[name]

                if isinstance(parent, nn.Conv2d) and 0 in dims:
                    integral_convs.append(get_parent_name(name)[0])

        fuse_batchnorm(model.eval(), integral_convs)

    def _rearrange(self, groups):
        """
        Rearranges the tensors in each group along continuous
        dimension to obtain continuous structure in tensors.

        Parameters
        ----------
        groups: List[torch_integral.graph.IntegralGroup].
        """
        for i, group in enumerate(groups):
            params = list(group.params)
            feature_maps = group.tensors

            if self.verbose:
                print(f'Rearranging of group {i}')

            for parent in group.parents:
                start = 0

                for another_group in parent.subgroups:
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

            self.rearranger(params, feature_maps, group.size)

    def _set_grid(self, group):
        """
        Sets default RandomLinspace grid in provided ``group``.

        Parameters
        ----------
        group: IntegralGroup.
        """
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

    def preprocess_model(self, model,
                         example_input,
                         continuous_dims,
                         black_list_dims=None):
        """
        Builds dependency graph of the model, fuses BatchNorms
        and permutes tensor parameters along countinuous
        dimension to obtain smooth structure.

        Parameters
        ----------
        model: torch.nn.Module.
        example_input: List[int] or torch.Tensor.
        continuous_dims: Dict[str, List[int]].
        black_list_dims: Dict[str, List[int]].
        """
        tracer = Tracer(
            model, example_input, continuous_dims, black_list_dims
        )
        tracer.build_groups()
        continuous_dims = tracer.continuous_dims

        if self.fuse_bn:
            self._fuse(model, continuous_dims)

        groups = tracer.build_groups()
        continuous_dims = tracer.continuous_dims

        if self.init_from_discrete and self.rearranger is not None:
            self._rearrange(groups)

        return groups, continuous_dims

    def __call__(self, model,
                 example_input,
                 continuous_dims,
                 black_list_dims=None):
        """
        Parametrizes tensor parameters of the model
        and wraps the model into IntegralModel class.

        Parameters
        ----------
        model: torch.nn.Module.
        example_input: List[int] or torch.Tensor.
        continuous_dims: Dict[str, List[int]].
        black_list_dims: Dict[str, List[int]].

        Returns
        -------
        integral_model: IntegralModel.
        """
        integral_groups, continuous_dims = self.preprocess_model(
            model, example_input, continuous_dims, black_list_dims
        )

        groups = [g for g in integral_groups if g.subgroups is None]
        
        for group in groups:
            self._set_grid(group)

        for group in integral_groups:
            for p in group.params:
                _, name = get_parent_name(p['name'])
                parent = get_parent_module(model, p['name'])

                if not parametrize.is_parametrized(parent, name) or all([
                    not isinstance(obj, IntegralParameterization)
                    for obj in parent.parametrizations[name]
                ]):

                    if self.build_functions is not None \
                       and type(parent) in self.build_functions:
                        build_function = self.build_functions[type(parent)]
                    elif isinstance(parent, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                        build_function = build_base_parameterization
                    else:
                        raise AttributeError(
                            f"Provide build function for attribute {name} of {type(parent)}"
                        )

                    dims = continuous_dims[p['name']]
                    w_func, quadrature = build_function(parent, name, dims)
                    grids_list = []

                    for g in p['value'].grids:
                        if hasattr(g, 'grid') and g.grid is not None:
                            if g in integral_groups:
                                grids_list.append(g.grid)

                    grid = GridND(grids_list)
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
                        self._optimize_parameters(parent, p['name'], target)

                else:
                    parametrization = parent.parametrizations[name][0]

                p['function'] = parametrization

        integral_model = IntegralModel(model, integral_groups)

        return integral_model

    def _optimize_parameters(self, module, name, target):
        """
        Optimize parametrization with Adam
        to approximate tensor attribute of given module.

        Parameters
        ----------
        module: torch.nn.Module.
        name: str.
        target: torch.Tensor.
        """
        module.train()
        _, attr = get_parent_name(name)
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


def build_base_parameterization(module, name, dims, scale=1.):
    """
    Builds parametrization and quadrature objects
    for parameters of Conv2d, Conv1d or Linear

    Parameters
    ----------
    module: torhc.nn.Module.
    name: str.
    dims: List[int].
    scale: float.

    Returns
    -------
    func: IntegralParameterization.
    quadrature: BaseIntegrationQuadrature.
    """
    quadrature = None
    func = None

    if name == 'weight':
        weight = getattr(module, name)
        cont_shape = [
            int(scale * weight.shape[d]) for d in dims
        ]

        if weight.ndim > len(dims):
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
        cont_shape = int(scale * bias.shape[0])
        func = InterpolationWeights1D(cont_shape)

    return func, quadrature
