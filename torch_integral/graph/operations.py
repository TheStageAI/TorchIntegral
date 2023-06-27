import operator
import torch
from .integral_group import IntegralGroup
from .integral_group import merge_groups
from ..utils import get_attr_by_name


def transpose(inp, dim0, dim1):
    out = torch.transpose(inp, dim0, dim1)

    if hasattr(inp, "grids"):
        out.grids = list(inp.grids)
        out.grids[dim0], out.grids[dim1] = out.grids[dim1], out.grids[dim0]

    IntegralGroup.append_to_groups(out, "transpose")

    return out


def permute(inp, dims):
    out = torch.permute(inp, dims)

    if hasattr(inp, "grids"):
        out.grids = [None] * inp.ndim

        for i in range(len(dims)):
            out.grids[i] = inp.grids[dims[i]]

    IntegralGroup.append_to_groups(out, "permute")

    return out


def getitem(inp, slices):
    out = operator.getitem(inp, slices)
    out.grids = [None] * out.ndim

    if hasattr(inp, "grids"):
        j = 0

        for i in range(inp.ndim):
            if i < len(slices):  # ADD Ellipsis
                if slices[i] == slice(None):
                    out.grids[j] = inp.grids[i]
                    j += 1

    IntegralGroup.append_to_groups(out, "getitem")

    return out


def neutral_hook(module, input, output):
    if hasattr(input[0], "grids"):
        output.grids = input[0].grids
        IntegralGroup.append_to_groups(output, "neutral")


def neutral_decorator(call_func):
    def wrapper(*args, **kwargs):
        out = call_func(*args, **kwargs)

        if hasattr(args[0], "grids"):
            out.grids = args[0].grids
            IntegralGroup.append_to_groups(out, "neutral")

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
        IntegralGroup.append_to_groups(out, "conv_linear")

        return out

    return conv_linear


def batch_norm(*args, **kwargs):
    out = torch.nn.functional.batch_norm(*args, **kwargs)
    inp = args[0]
    weight = kwargs["weight"]
    bias = kwargs["bias"]
    merge_groups(inp, 1, weight, 0)
    merge_groups(bias, 0, weight, 0)
    merge_groups(out, 1, weight, 0)
    IntegralGroup.append_to_groups(out, "batch_norm")

    return out


def aggregation_decorator(func):
    def wrapper(inp, *dims, **kwargs):
        out = func(inp, *dims, **kwargs)

        for d in range(out.ndim):
            if d not in dims:
                merge_groups(out, d, inp, d)

        IntegralGroup.append_to_groups(out, "aggregation")

        return out

    return wrapper


def max_min_decorator(func):
    def wrapper(inp, dim, **kwargs):
        out = func(inp, dim, **kwargs)
        values = out.values

        for d in range(values.ndim):
            if d != dim:
                merge_groups(values, d, inp, d)

        IntegralGroup.append_to_groups(values, "min_max")

        return out

    return wrapper


def view(*args, **kwargs):
    inp = args[0]
    out = inp.view(*args[1:])
    out.grids = [None] * out.ndim

    if hasattr(inp, "grids"):
        i = 1

        for g in inp.grids:
            if g is not None:
                while out.shape[i] != g.size:
                    i += 1

                out.grids[i] = g
                i += 1

        IntegralGroup.append_to_groups(out)

    return out


def reshape(*args, **kwargs):
    inp = args[0]
    out = inp.reshape(*args[1:])
    out.grids = [None] * out.ndim

    if hasattr(inp, "grids"):
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
            out.grids[d].set_subgroups([x.grids[d] for x in inputs])

    IntegralGroup.append_to_groups(out, "concat")

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

        IntegralGroup.append_to_groups(out, "operator")

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
    IntegralGroup.append_to_groups(out, "matmul")

    return out


def interpolate(*args, **kwargs):
    out = torch.nn.functional.interpolate(*args, **kwargs)
    out.grids = [None] * out.ndim

    if hasattr(args[0], "grids"):
        for d in range(out.ndim):
            out.grids[d] = args[0].grids[d]

    IntegralGroup.append_to_groups(out, "interpolate")

    return out


# def einsum(equation, *args):
#     out = torch.einsum(equation, *args)
#     inp_str, out_str = equation.split('->')
#     tensors = inp_str.split(',')
#
#     return out
