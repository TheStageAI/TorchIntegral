import torch
from .operations import *
from .integral_group import IntegralGroup
from ..utils import remove_all_hooks


class SymbolicFxTracer(torch.fx.Tracer):
    """torch.fx.Tracer which leaf modules are batch norm layers."""

    def is_leaf_module(self, m, qualname):
        return isinstance(
            m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        )


class IntegralTracer(torch.fx.Interpreter):
    """
    Class for building dependency graph of the neural network.
    Builds related groups of parameter tensors.
    Related group is a set of pairs of tensor and dimensioin.
    Two parameters belong to one related group
    if they should have the same size along the corresponding dimension.

    Parameters
    ----------
    model: torch.nn.Module.
    continuous_dims: Dict[str, List[int]].
        Dictionary which contains names of the model's parameters
        and it's continuous dimension indices.
    discrete_dims: Dict[str, List[int]].
        Dictionary which contains names of the model's parameters
        and dimensions that can not be continuous.
        If there is the same element in discrete_dims and continuous_dims, then
        the element will be removed from continuous_dims.
    additional_operations: Dict[Union[str, Callable], Callable].
        Dictionary which contains custom tracing operations for the graph.
    additional_hooks: Dict[torch.nn.Module, Callable].
        Dictionary which contains custom hooks for the graph.

    Examples
    --------
    For example, if we have a model with two convolutional layers
    and we want to make continuous only first convolutional layer's
    output dimension then we can write:

    .. code-block:: python

        import torch
        from torch_integral.graph import IntegralTracer
        from torchvision.models import resnet18

        model = resnet18(pretrained=True)
        example_input = torch.randn(1, 3, 224, 224)
        continuous_dims = {
            "layer4.0.conv1.weight": [0],
            "layer4.0.conv1.bias": [0],
        }
        IntegralTracer = IntegralTracer(model, example_input, continuous_dims)

    Here  first dimension of the `layer4.0.conv1.weight`, `layer4.0.conv1.bias` and second dim
    of the `conv_2.weight` are belong to the same IntegralGroup,
    because it's sizes should be equal.
    Note that it is not necessary to list all parameter names of the related group.
    It is enough to list only one tensor of the group and all other tensors will be
    added automatically. For example, in example above it was enough to write
    `continuous_dims = {layer4.0.conv1.weight: [0]}`.
    """

    def __init__(
        self,
        model,
        continuous_dims,
        discrete_dims=None,
        additional_operations=None,
        additional_hooks=None,
    ):
        graph = SymbolicFxTracer().trace(model.eval())
        gm = torch.fx.GraphModule(model, graph)
        super().__init__(gm, True)
        self.model = model
        self.groups = None
        self.continuous_dims = continuous_dims

        if discrete_dims is not None:
            self.discrete_dims = discrete_dims
        else:
            self.discrete_dims = {}

        self.default_operations = {
            operator.add: operators_decorator(operator.add),
            operator.sub: operators_decorator(operator.sub),
            operator.mul: operators_decorator(operator.mul),
            operator.getitem: getitem,
            torch.permute: permute,
            torch.transpose: transpose,
            torch.matmul: matmul,
            torch.nn.functional.interpolate: interpolate,
            torch.mean: aggregation_decorator(torch.mean),
            torch.sum: aggregation_decorator(torch.sum),
            torch.max: max_min_decorator(torch.max),
            torch.min: max_min_decorator(torch.min),
            torch.cat: concatenate,
            torch.conv1d: conv_linear_decorator(torch.conv1d),
            torch.conv2d: conv_linear_decorator(torch.conv2d),
            torch.conv3d: conv_linear_decorator(torch.conv3d),
            torch._C._nn.linear: conv_linear_decorator(torch._C._nn.linear),
            torch.nn.functional.batch_norm: batch_norm,
            "mean": aggregation_decorator(torch.mean),
            "sum": aggregation_decorator(torch.sum),
            "view": view,
            "reshape": reshape,
            "mul": operators_decorator(operator.mul),
            "add": operators_decorator(operator.add),
        }
        self.default_hooks = {
            torch.nn.BatchNorm1d: neutral_hook,
            torch.nn.BatchNorm2d: neutral_hook,
            torch.nn.BatchNorm3d: neutral_hook,
            torch.nn.Identity: neutral_hook,
        }

        if additional_operations is not None:
            self.default_operations.update(additional_operations)

        if additional_hooks is not None:
            self.default_hooks.update(additional_hooks)

    def build_groups(self, *args, initial_env=None, enable_io_processing=True):
        """
        Builds dependency groups of the neural network.

        Parameters
        ----------
        *args: List[torch.Tensor] or List[List[int]].
            Input tensors of the model or shapes of input tensors.
        initial_env: Dict[str, torch.Tensor].
        enable_io_processing: bool.
            If True, then input and output tensors will be processed.

        Returns
        -------
        self.groups: List[IntegralGroup].
            List of related parameters groups.
        """
        self.groups = []
        self.model.eval()

        for name, param in self.model.named_parameters():
            param.grids = [None] * param.ndim

            if name in self.continuous_dims:
                dims = self.continuous_dims[name]
            else:
                dims = list(range(param.ndim))

            for dim in dims:
                size = param.shape[dim]
                group = IntegralGroup(size)
                group.append_param(name, param, dim)
                param.grids[dim] = group
                self.groups.append(group)

        device = next(iter(self.model.parameters())).device
        args = list(args)

        for i in range(len(args)):
            if type(args[i]) == torch.Tensor:
                args[i] = args[i].to(device)
            else:
                args[i] = torch.rand(args[i]).to(device)

        output = self.run(*args, initial_env, enable_io_processing)
        remove_all_hooks(self.model)
        self.groups = [group for group in self.groups if len(group.params)]
        delete_indices = []

        for i, group in enumerate(self.groups):
            delete_group = True

            for p in group.params:
                if (
                    p["name"] in self.continuous_dims
                    and p["dim"] in self.continuous_dims[p["name"]]
                ):
                    delete_group = False

                if (
                    p["name"] in self.discrete_dims
                    and p["dim"] in self.discrete_dims[p["name"]]
                ):
                    for d in group.params:
                        if (
                            d["name"] in self.continuous_dims
                            and d["dim"] in self.continuous_dims[d["name"]]
                        ):
                            self.continuous_dims[d["name"]].remove(d["dim"])

                            if len(self.continuous_dims[d["name"]]) == 0:
                                self.continuous_dims.pop(d["name"])

                    delete_group = True
                    break

            if delete_group:
                delete_indices.append(i)
            else:
                for p in group.params:
                    if p["name"] in self.continuous_dims:
                        dims = self.continuous_dims[p["name"]]

                        if p["dim"] not in dims:
                            dims.append(p["dim"])
                    else:
                        self.continuous_dims[p["name"]] = [p["dim"]]

        self.groups = [
            group for i, group in enumerate(self.groups) if i not in delete_indices
        ]

        def add_parent_groups(group, parents):
            for parent in group.parents:
                if parent not in parents:
                    parents.add(parent)
                add_parent_groups(parent, parents)

        parents = set()

        for group in self.groups:
            add_parent_groups(group, parents)
            group.build_operations_set()

        for parent in parents:
            parent.build_operations_set()

        self.groups.extend(list(parents))

        return self.groups

    def call_function(self, target, args, kwargs):
        """
        Instead of usual call_function method,
        this method calls decorated function to build dependency graph.

        Parameters
        ----------
        target: Callable.
            Function to call.
        args: List[torch.Tensor].
            Arguments of the function.
        kwargs: Dict[str, torch.Tensor].
            Keyword arguments of the function.

        Returns
        -------
        result: torch.Tensor.
            Result of the function.
        """
        if target in self.default_operations:
            return self.default_operations[target](*args, **kwargs)
        else:
            return neutral_decorator(target)(*args, **kwargs)

    def call_method(self, target, args, kwargs):
        """
        Instead of usual call_method method,
        this method calls decorated function to build dependency graph.

        Parameters
        ----------
        target: Callable.
            Method to call.
        args: List[torch.Tensor].
            Arguments of the method.
        kwargs: Dict[str, torch.Tensor].
            Keyword arguments of the method.

        Returns
        -------
        result: torch.Tensor.
            Result of the method.
        """
        if target in self.default_operations:
            return self.default_operations[target](*args, **kwargs)
        else:
            return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        """
        Registers tracing forward hooks before calling submodules.

        Parameters
        ----------
        target: Callable.
            Submodule to call.
        args: List[torch.Tensor].
            Arguments of the submodule.
        kwargs: Dict[str, torch.Tensor].
            Keyword arguments of the submodule.

        Returns
        -------
        result: torch.Tensor.
            Result of the submodule.
        """
        submod = self.fetch_attr(target)

        if type(submod) in self.default_hooks:
            submod.register_forward_hook(self.default_hooks[type(submod)])

        return submod(*args, **kwargs)
