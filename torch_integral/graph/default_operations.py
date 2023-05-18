import operator
from .operations import *


DEFAULT_OPERATIONS = {
    operator.add: operators_decorator(operator.add),
    operator.sub: operators_decorator(operator.sub),
    operator.mul: operators_decorator(operator.mul),
    torch.matmul: matmul,
    torch.mean: aggregation_decorator(torch.mean),
    torch.sum: aggregation_decorator(torch.sum),
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
}
