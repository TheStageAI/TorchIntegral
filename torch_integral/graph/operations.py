import operator
import torch
from .group import RelatedGroup


def merge_groups(x, x_dim, y, y_dim):
    """Merges two groups of tensors ``x`` and `yy`` with indices ``x_dim`` and ``y_dim``."""
    if type(x) in (int, float):
        x = torch.tensor(x)
    if type(y) in (int, float):
        y = torch.tensor(y)
    if not hasattr(x, "related_groups"):
        x.related_groups = [None for _ in range(x.ndim)]
    if not hasattr(y, "related_groups"):
        y.related_groups = [None for _ in range(y.ndim)]
    if y.related_groups[y_dim] is not None:
        x, x_dim, y, y_dim = y, y_dim, x, x_dim

    if x.related_groups[x_dim] is not None:
        if y.related_groups[y_dim] is not None:
            if len(y.related_groups[y_dim].parents) > 0:
                x, x_dim, y, y_dim = y, y_dim, x, x_dim

            if y.related_groups[y_dim].subgroups is not None:
                x, x_dim, y, y_dim = y, y_dim, x, x_dim

            if x.related_groups[x_dim] is not y.related_groups[y_dim]:
                for param in y.related_groups[y_dim].params:
                    dim = param["dim"]
                    t = param["value"]

                    if t is not y:
                        t.related_groups[dim] = x.related_groups[x_dim]

                x.related_groups[x_dim].params.extend(y.related_groups[y_dim].params)
                y.related_groups[y_dim].clear_params()

                for tensor in y.related_groups[y_dim].tensors:
                    dim = tensor["dim"]
                    t = tensor["value"]

                    if t is not y:
                        t.related_groups[dim] = x.related_groups[x_dim]

                x.related_groups[x_dim].tensors.extend(y.related_groups[y_dim].tensors)
                y.related_groups[y_dim].clear_tensors()

        y.related_groups[y_dim] = x.related_groups[x_dim]


def neutral_hook(module, input, output):
    if hasattr(input[0], "related_groups"):
        output.related_groups = input[0].related_groups
        RelatedGroup.append_to_groups(output, "neutral")


def conv_linear_hook(module, input, output):
    weight = module.weight
    bias = module.bias

    if bias is not None:
        merge_groups(bias, 0, weight, 0)

    merge_groups(weight, 1, input[0], 1)
    merge_groups(output, 1, weight, 0)
    RelatedGroup.append_to_groups(output, "conv_linear")


def conv_transposed_hook(module, input, output):
    weight = module.weight
    bias = module.bias

    if bias is not None:
        merge_groups(bias, 0, weight, 1)

    merge_groups(weight, 0, input[0], 1)
    merge_groups(output, 1, weight, 1)
    RelatedGroup.append_to_groups(output, "conv_transposed")


def transpose(inp, dim0, dim1):
    out = torch.transpose(inp, dim0, dim1)

    if hasattr(inp, "related_groups"):
        out.related_groups = list(inp.related_groups)
        out.related_groups[dim0], out.related_groups[dim1] = (
            out.related_groups[dim1],
            out.related_groups[dim0],
        )

    RelatedGroup.append_to_groups(out, "transpose")

    return out


def permute(inp, dims):
    out = torch.permute(inp, dims)

    if hasattr(inp, "related_groups"):
        out.related_groups = [None] * inp.ndim

        for i in range(len(dims)):
            out.related_groups[i] = inp.related_groups[dims[i]]

    RelatedGroup.append_to_groups(out, "permute")

    return out


def getitem(inp, slices):
    out = operator.getitem(inp, slices)

    if hasattr(inp, "related_groups"):
        out.related_groups = [None] * out.ndim
        j = 0

        for i in range(inp.ndim):
            if i < len(slices):  # ADD Ellipsis
                if slices[i] == slice(None):
                    out.related_groups[j] = inp.related_groups[i]
                    j += 1

    RelatedGroup.append_to_groups(out, "getitem")

    return out


def neutral_decorator(call_func):
    def wrapper(*args, **kwargs):
        out = call_func(*args, **kwargs)

        if hasattr(args[0], "related_groups"):
            out.related_groups = args[0].related_groups
            RelatedGroup.append_to_groups(out, "neutral")

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
        RelatedGroup.append_to_groups(out, "conv_linear")

        return out

    return conv_linear


def conv_transposed_decorator(function):
    def conv_transposed(*args):
        x, weight, bias = args[:3]
        out = function(*args)

        if bias is not None:
            merge_groups(bias, 0, weight, 1)

        merge_groups(weight, 0, x, 1)
        merge_groups(out, 1, weight, 1)
        RelatedGroup.append_to_groups(out, "conv_transposed")

        return out

    return conv_transposed


def batch_norm(*args, **kwargs):
    out = torch.nn.functional.batch_norm(*args, **kwargs)
    inp = args[0]
    weight = kwargs["weight"]
    bias = kwargs["bias"]
    merge_groups(inp, 1, weight, 0)
    merge_groups(bias, 0, weight, 0)
    merge_groups(out, 1, weight, 0)
    RelatedGroup.append_to_groups(out, "batch_norm")

    return out


def aggregation_decorator(func):
    def wrapper(inp, *dims, **kwargs):
        out = func(inp, *dims, **kwargs)

        for d in range(out.ndim):
            if d not in dims:
                merge_groups(out, d, inp, d)

        RelatedGroup.append_to_groups(out, "aggregation")

        return out

    return wrapper


def max_min_decorator(func):
    def wrapper(inp, dim, **kwargs):
        out = func(inp, dim, **kwargs)
        values = out.values

        for d in range(values.ndim):
            if d != dim:
                merge_groups(values, d, inp, d)

        RelatedGroup.append_to_groups(values, "min_max")

        return out

    return wrapper


def view(*args, **kwargs):
    inp = args[0]
    out = inp.view(*args[1:])
    out.related_groups = [None] * out.ndim

    if hasattr(inp, "related_groups"):
        i = 1

        for g in inp.related_groups:
            if g is not None:
                while out.shape[i] != g.size:
                    i += 1

                out.related_groups[i] = g
                i += 1

        RelatedGroup.append_to_groups(out)

    return out


def reshape(*args, **kwargs):
    inp = args[0]
    out = inp.reshape(*args[1:])
    out.related_groups = [None] * out.ndim

    if hasattr(inp, "related_groups"):
        i = 1

        for g in inp.related_groups:
            if g is not None:
                while out.shape[i] != g.size:
                    i += 1

                out.related_groups[i] = g
                i += 1

        RelatedGroup.append_to_groups(out)

    return out


def concatenate(inputs, dim):
    out = torch.cat(inputs, dim)
    out.related_groups = [None] * out.ndim

    for d in range(out.ndim):
        if d != dim:
            for x in inputs[1:]:
                merge_groups(inputs[0], d, x, d)

            out.related_groups[d] = inputs[0].related_groups[d]

        else:
            out.related_groups[d] = RelatedGroup(out.shape[d])
            out.related_groups[d].set_subgroups([x.related_groups[d] for x in inputs])

    RelatedGroup.append_to_groups(out, "concat")

    return out


def operators_decorator(operator):
    def wrapper(x, y):
        out = operator(x, y)

        if type(x) not in (int, float, torch.Tensor):
            return out

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

        out.related_groups = x.related_groups

        for dim in range(out.ndim):
            if out.related_groups[dim] is None:
                if dim - k >= 0 and y.shape[dim - k] > 1:
                    out.related_groups[dim] = y.related_groups[dim - k]

            if out.shape[dim] == 1:
                out.related_groups[dim] = None

        RelatedGroup.append_to_groups(out, "operator")

        return out

    return wrapper


def matmul(x, y):
    out = x @ y
    out.related_groups = [None] * out.ndim

    if y.ndim > x.ndim:
        y, x = x, y

    k = x.ndim - y.ndim
    merge_groups(y, y.ndim - 2, x, x.ndim - 1)

    for i in range(y.ndim - 2):
        merge_groups(x, i + k, y, i)

    for d in range(x.ndim - 1):
        out.related_groups.append(x.related_groups[d])

    out.related_groups.append(y.related_groups[y.ndim - 1])
    RelatedGroup.append_to_groups(out, "matmul")

    return out


def interpolate(*args, **kwargs):
    out = torch.nn.functional.interpolate(*args, **kwargs)
    out.related_groups = [None] * out.ndim

    if hasattr(args[0], "related_groups"):
        for d in range(out.ndim):
            out.related_groups[d] = args[0].related_groups[d]

    RelatedGroup.append_to_groups(out, "interpolate")

    return out
