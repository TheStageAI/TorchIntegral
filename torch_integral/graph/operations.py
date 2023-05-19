import torch


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


def batch_norm(*args, **kwargs):
    out = torch.nn.functional.batch_norm(*args, **kwargs)
    inp = args[0]
    weight = kwargs['weight']
    bias = kwargs['bias']
    secure_merge(inp, 1, weight, 0)
    secure_merge(bias, 0, weight, 0)
    secure_merge(out, 1, weight, 0)
    append_tensor(out)

    return out


def aggregation_decorator(func):
    def wrapper(inp, *dims, **kwargs):
        out = func(inp, *dims, **kwargs)

        for d in range(out.ndim):
            if d not in dims:
                secure_merge(out, d, inp, d)

        append_tensor(out)

        return out

    return wrapper


def max_min_decorator(func):
    def wrapper(inp, dim, **kwargs):
        out = func(inp, dim, **kwargs)
        values = out.values

        for d in range(values.ndim):
            if d != dim:
                secure_merge(values, d, inp, d)

        append_tensor(values)

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
                while out.shape[i] != g['size']:
                    i += 1

                out.grids[i] = g
                i += 1

        append_tensor(out)

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
            if x.shape[k+dim] != 1 and y.shape[dim] != 1:
                secure_merge(x, k + dim, y, dim)

        out.grids = x.grids  # choose grids which is not None from x and y!
        append_tensor(out)

        return out

    return wrapper


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
    if type(x) in (int, float):
        x = torch.tensor(x)
    if type(y) in (int, float):
        y = torch.tensor(y)

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
        