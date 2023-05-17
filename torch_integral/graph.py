import torch
from torch.fx import symbolic_trace
import operator
from .grid import *
from .utils import get_parent_name
from .utils import get_attr_by_name
from .utils import get_parent_module
from .utils import remove_all_hooks


def append_tensor(x):
    if hasattr(x, 'grids'):
        for i, g in enumerate(x.grids):
            if g is not None:
                g['tensors'].append({
                    'value': x, 'dim': i
                })


def neutral_hook(module, input, output):
    if hasattr(input[0], 'grids'):
        output.grids = input[0].grids
        append_tensor(output)


def neutral_decorator(call_func):
    def wrapper(*args, **kwargs):
        out = call_func(*args, **kwargs)

        if hasattr(args[0], 'grids'):
            out.grids = args[0].grids
            append_tensor(out)

        return out

    return wrapper


def conv_linear_decorator(function):
    def conv_linear(*args):
        x, weight, bias = args[:3]
        out = function(*args)

        if bias is not None:
            secure_merge(bias, 0, weight, 0)

        secure_merge(weight, 1, x, 1)
        secure_merge(out, 1, weight, 0)
        append_tensor(out)

        return out

    return conv_linear


def aggregation_decorator(func):
    def wrapper(x, dims, keepdim=True):
        out = func(x, dims, keepdim=keepdim)

        for d in range(out.ndim):
            if d not in dims:
                secure_merge(out, d, x, d)

        append_tensor(out)

        return out

    return wrapper


def operators_decorator(operator):
    def wrapper(x, y):
        out = operator(x, y)

        if y.ndim > x.ndim:
            x, y = y, x

        k = x.ndim - y.ndim

        for dim in range(y.ndim):
            secure_merge(x, k + dim, y, dim)

        out.grids = x.grids
        append_tensor(out)

        return out

    return wrapper


@operators_decorator
def add(x, y):
    return x + y


@operators_decorator
def sub(x, y):
    return x - y


@operators_decorator
def mul(x, y):
    return x * y


def matmul(x, y):
    out = x @ y
    out.grids = [None] * out.ndim

    if y.ndim > x.ndim:
        y, x = x, y

    k = x.ndim - y.ndim
    secure_merge(y, y.ndim - 2, x, x.ndim - 1)

    for i in range(y.ndim - 2):
        secure_merge(x, i + k, y, i)

    for d in range(x.ndim - 1):
        out.grids.append(x.grids[d])

    out.grids.append(y.grids[y.ndim - 1])
    append_tensor(out)

    return out


def einsum(equation, *args):
    out = torch.einsum(equation, *args)
    inp_str, out_str = equation.split('->')
    tensors = inp_str.split(',')

    return out


def secure_merge(x, x_dim, y, y_dim):
    if not hasattr(x, 'grids'):
        x.grids = [None for _ in range(x.ndim)]

    if not hasattr(y, 'grids'):
        y.grids = [None for _ in range(y.ndim)]

    if y.grids[y_dim] is not None:
        x, x_dim, y, y_dim = y, y_dim, x, x_dim

    if x.grids[x_dim] is not None:
        if y.grids[y_dim] is not None:
            if x.grids[x_dim] is not y.grids[y_dim]:
                for key in ['params', 'tensors']:
                    for d in y.grids[y_dim][key]:
                        dim = d['dim']
                        t = d['value']

                        if t is not y:
                            t.grids[dim] = x.grids[x_dim]

                    x.grids[x_dim][key].extend(y.grids[y_dim][key])
                    y.grids[y_dim][key] = []

        y.grids[y_dim] = x.grids[x_dim]


def replace_operations(module: torch.nn.Module,
                       new_operations=None,
                       skip_modules_list=None) -> torch.nn.Module:
    gm = torch.fx.symbolic_trace(module)
    graph = gm.graph
    operations = {
        operator.add: add,
        operator.sub: sub,
        operator.mul: mul,
        torch.matmul: matmul,
        torch.mean: aggregation_decorator(torch.mean),
        torch.sum: aggregation_decorator(torch.sum),
        torch.conv1d: conv_linear_decorator(torch.conv1d),
        torch.conv2d: conv_linear_decorator(torch.conv2d),
        torch.conv3d: conv_linear_decorator(torch.conv3d),
        torch._C._nn.linear: conv_linear_decorator(torch._C._nn.linear)
    }

    if new_operations is not None:
        operations.update(new_operations)

    if skip_modules_list is not None:
        skip_modules = (torch.nn.BatchNorm2d, *skip_modules_list)
    else:
        skip_modules = (torch.nn.BatchNorm2d,)

    nodes = list(graph.nodes)

    for node in nodes:
        if node.op == 'call_function':
            if node.target in operations:
                node.target = operations[node.target]
            else:
                node.target = neutral_decorator(node.target)

        elif node.op == 'call_module':
            node_module = get_attr_by_name(module, node.target)

            if isinstance(node_module, skip_modules):
                node_module.register_forward_hook(neutral_hook)
            else:
                mod_name = node.target.replace('_', '.')
                new_module = replace_operations(
                    node_module, new_operations, skip_modules_list
                )
                gm.add_submodule(mod_name, new_module)

                with graph.inserting_after(node):
                    new_node = graph.call_module(
                        mod_name, node.args, node.kwargs
                    )
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)

    graph.lint()
    gm.recompile()

    return gm


def prepare_parameters(cont_parameters):
    all_grids = []

    for name in cont_parameters:
        p, dims = cont_parameters[name]
        p.grids = [None for _ in range(p.ndim)]

        for d in dims:
            size = p.shape[d]
            p.grids[d] = {
                'size': size,
                'params': [{'value': p, 'dim': d, 'name': name}],
                'tensors': []
            }
            all_grids.append(p.grids[d])

    return all_grids


def build_groups(model, sample_shape, cont_parameters=None):
    tracing_model = replace_operations(model)
    base_cont_params = {}

    for name, param in model.named_parameters():
        parent_name, attr_name = get_parent_name(name)
        parent = get_parent_module(model, name)

        if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)):
            if 'weight' in attr_name:
                base_cont_params[name] = [param, [0, 1]]

            elif 'bias' in name:
                base_cont_params[name] = [param, [0]]

    if cont_parameters is not None:
        for k, v in cont_parameters.items():
            base_cont_params[k] = [get_attr_by_name(model, k), v]

    all_grids = prepare_parameters(base_cont_params)
    device = next(iter(model.parameters())).device
    x = torch.rand(sample_shape).to(device)
    tracing_model(x)
    remove_all_hooks(tracing_model)
    del tracing_model

    grids = [
        g for g in all_grids if len(g['params']) != 0
    ]

    return grids, base_cont_params


if __name__ == '__main__':
    from torchvision.models import resnet18

    model = resnet18()
    cont_params = {}
    grids = build_groups(model, [1, 3, 100, 100])

    for g in grids:
        print(len(g['params']))
