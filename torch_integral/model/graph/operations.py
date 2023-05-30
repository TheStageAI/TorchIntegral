import operator
import torch
from .integral_group import IntegralGroup
from .integral_group import merge_groups
from ...utils import get_attr_by_name


def transpose(inp, dim0, dim1):
    out = torch.transpose(inp, dim0, dim1)

    if hasattr(inp, 'grids'):
        out.grids = list(inp.grids)
        out.grids[dim0], out.grids[dim1] = \
            out.grids[dim1], out.grids[dim0]

    IntegralGroup.append_to_groups(out)

    return out


def permute(inp, dims):
    out = torch.permute(inp, dims)

    if hasattr(inp, 'grids'):
        out.grids = [None] * inp.ndim

        for i in range(len(dims)):
            out.grids[i] = inp.grids[dims[i]]

    IntegralGroup.append_to_groups(out)

    return out


def getitem(inp, slices):
    out = operator.getitem(inp, slices)
    out.grids = [None] * out.ndim

    if hasattr(inp, 'grids'):
        j = 0

        for i in range(inp.ndim):
            if i < len(slices):  # ADD Ellipsis
                if slices[i] == slice(None):
                    out.grids[j] = inp.grids[i]
                    j += 1

    IntegralGroup.append_to_groups(out)

    return out


def neutral_hook(module, input, output):
    if hasattr(input[0], 'grids'):
        output.grids = input[0].grids
        IntegralGroup.append_to_groups(output)


def neutral_decorator(call_func):
    def wrapper(*args, **kwargs):
        out = call_func(*args, **kwargs)

        if hasattr(args[0], 'grids'):
            out.grids = args[0].grids
            IntegralGroup.append_to_groups(out)

        return out

    return wrapper


def conv_linear_decorator(function):
    def conv_linear(*args):
        x, weight, bias = args[:3]
        out = function(*args)

        if bias is not None:
            merge_groups(bias, 0, weight, 0)

        merge_groups(weight, 1, x, 1)
        merge_groups(out, 1, weight, 0)
        IntegralGroup.append_to_groups(out)

        return out

    return conv_linear


def batch_norm(*args, **kwargs):
    out = torch.nn.functional.batch_norm(*args, **kwargs)
    inp = args[0]
    weight = kwargs['weight']
    bias = kwargs['bias']
    merge_groups(inp, 1, weight, 0)
    merge_groups(bias, 0, weight, 0)
    merge_groups(out, 1, weight, 0)
    IntegralGroup.append_to_groups(out)

    return out


def aggregation_decorator(func):
    def wrapper(inp, *dims, **kwargs):
        out = func(inp, *dims, **kwargs)

        for d in range(out.ndim):
            if d not in dims:
                merge_groups(out, d, inp, d)

        IntegralGroup.append_to_groups(out)

        return out

    return wrapper


def max_min_decorator(func):
    def wrapper(inp, dim, **kwargs):
        out = func(inp, dim, **kwargs)
        values = out.values

        for d in range(values.ndim):
            if d != dim:
                merge_groups(values, d, inp, d)

        IntegralGroup.append_to_groups(values)

        return out

    return wrapper


def reshape(*args, **kwargs):  # FIX
    inp = args[0]
    name = args[-1]

    if name == 'reshape':
        out = inp.reshape(*args[1:-1])
    else:
        out = inp.view(*args[1:-1])

    out.grids = [None] * out.ndim

    if hasattr(inp, 'grids'):
        i = 1

        for g in inp.grids:
            if g is not None:
                while out.shape[i] != g.size:
                    i += 1

                out.grids[i] = g
                i += 1

        IntegralGroup.append_to_groups(out)

    return out


def concatenate(inputs, dim):
    out = torch.cat(inputs, dim)
    out.grids = [None] * out.ndim

    for d in range(out.ndim):
        if d != dim:
            for x in inputs[1:]:
                merge_groups(inputs[0], d, x, d)

            out.grids[d] = inputs[0].grids[d]

        else:
            out.grids[d] = IntegralGroup(out.shape[d])
            out.grids[d].set_subgroups([
                x.grids[d] for x in inputs
            ])

    IntegralGroup.append_to_groups(out)

    return out


def operators_decorator(operator):
    def wrapper(x, y):
        out = operator(x, y)

        if type(x) in (int, float):
            x = torch.tensor(x)

        if type(y) in (int, float):
            y = torch.tensor(y)

        if y.ndim > x.ndim:
            x, y = y, x

        k = x.ndim - y.ndim

        for dim in range(y.ndim):
            if x.shape[k + dim] != 1 and y.shape[dim] != 1:
                merge_groups(x, k + dim, y, dim)

        out.grids = x.grids

        for dim in range(out.ndim):
            if out.grids[dim] is None:
                if dim - k >= 0 and y.shape[dim - k] > 1:
                    out.grids[dim] = y.grids[dim - k]

            if out.shape[dim] == 1:
                out.grids[dim] = None

        IntegralGroup.append_to_groups(out)

        return out

    return wrapper


def matmul(x, y):
    out = x @ y
    out.grids = [None] * out.ndim

    if y.ndim > x.ndim:
        y, x = x, y

    k = x.ndim - y.ndim
    merge_groups(y, y.ndim - 2, x, x.ndim - 1)

    for i in range(y.ndim - 2):
        merge_groups(x, i + k, y, i)

    for d in range(x.ndim - 1):
        out.grids.append(x.grids[d])

    out.grids.append(y.grids[y.ndim - 1])
    IntegralGroup.append_to_groups(out)

    return out


# def einsum(equation, *args):
#     out = torch.einsum(equation, *args)
#     inp_str, out_str = equation.split('->')
#     tensors = inp_str.split(',')
#
#     return out


def interpolate(*args, **kwargs):
    out = torch.nn.functional.interpolate(
        *args, **kwargs
    )
    out.grids = [None] * out.ndim

    if hasattr(args[0], 'grids'):
        for d in range(out.ndim):
            out.grids[d] = args[0].grids[d]

    return out


DEFAULT_OPERATIONS = {
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
    'mean': aggregation_decorator(torch.mean),
    'sum': aggregation_decorator(torch.sum),
    'view': reshape,
    'reshape': reshape,
    'mul': operators_decorator(operator.mul),
    'add': operators_decorator(operator.add),
}

DEFAULT_HOOKS = {
    torch.nn.BatchNorm1d: neutral_hook,
    torch.nn.BatchNorm2d: neutral_hook,
    torch.nn.BatchNorm3d: neutral_hook,
    torch.nn.Identity: neutral_hook,
}


def replace_operations(module: torch.nn.Module,
                       new_operations=None) -> torch.nn.Module:
    fx_model = torch.fx.symbolic_trace(module)
    graph = fx_model.graph
    operations = DEFAULT_OPERATIONS.copy()
    hooks_dict = DEFAULT_HOOKS.copy()
    nodes = list(graph.nodes)

    if new_operations is not None:
        operations.update(new_operations)

    for node in nodes:
        if node.op == 'call_function':
            if node.target in operations:
                node.target = operations[node.target]
            else:
                node.target = neutral_decorator(node.target)

        elif node.op == 'call_method' and node.target in operations:
            func = operations[node.target]
            args = node.args

            if node.target == 'view' or node.target == 'reshape':
                args = tuple([*node.args, node.target])

            elif node.target == 'mean' or node.target == 'sum':
                args = node.args[1:]

            with graph.inserting_after(node):
                new_node = graph.call_function(
                    func, args, node.kwargs
                )
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)

        elif node.op == 'call_module':
            node_module = get_attr_by_name(module, node.target)

            if type(node_module) not in hooks_dict:
                mod_name = node.target.replace('_', '.')
                new_module = replace_operations(
                    node_module, new_operations,
                )
                fx_model.add_submodule(mod_name, new_module)

                with graph.inserting_after(node):
                    new_node = graph.call_module(
                        mod_name, node.args, node.kwargs
                    )
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
            else:
                node_module.register_forward_hook(
                    hooks_dict[type(node_module)]
                )

    graph.lint()
    fx_model.recompile()

    return fx_model
