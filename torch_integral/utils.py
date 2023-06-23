import torch
from typing import Tuple, Dict, Any, List
import torch.fx as fx
import copy
import torch.nn as nn
from collections import OrderedDict
from .grid import TrainableGrid1D
from contextlib import contextmanager
from torch.nn.utils import parametrize


def get_attr_by_name(module, name):
    """
    """
    for s in name.split('.'):
        module = getattr(module, s)

    return module


def get_parent_name(qualname: str) -> Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = qualname.rsplit('.', 1)
    return parent[0] if parent else '', name


def get_parent_module(module, attr_path):
    """
    Returns parent module of module.attr_path.

    Parameters
    ----------
    module: torch.nn.Module.
    attr_path: str.
    """
    parent_name, attr_name = get_parent_name(attr_path)

    if parent_name != '':
        parent = get_attr_by_name(module, parent_name)
    else:
        parent = module

    return parent


def remove_all_hooks(model: torch.nn.Module) -> None:
    """
    """
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_hooks(child)


def fuse_batchnorm(model, convs):
    """
    Fuse conv and bn only if conv is in convs argument.

    Parameters
    ----------
    model: torch.nn.Module.
    convs: List[torch.nn.ConvNd].
    """
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue
        if type(modules[node.target]) is nn.BatchNorm2d \
                and type(modules[node.args[0].target]) is nn.Conv2d:
            if node.args[0].target in convs:
                if len(node.args[0].users) > 1:
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                inplace_conv_bn_fusion(conv, bn)
                parent_name, attr_name = get_parent_name(node.target)
                parent = get_parent_module(model, node.target)
                setattr(parent, attr_name, torch.nn.Identity())


def inplace_conv_bn_fusion(conv, bn):
    """
    """
    assert (not (conv.training or bn.training)), "Fusion only for eval!"
    conv.weight.data, bias = fuse_conv_bn_weights(
        conv.weight, conv.bias, bn.running_mean,
        bn.running_var, bn.eps, bn.weight, bn.bias
    )

    if conv.bias is None:
        conv.bias = torch.nn.Parameter(bias).to(conv.weight.device)
    else:
        conv.bias.data = bias


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """
    """
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return conv_w, conv_b


def reset_batchnorm(model):
    """
    Set new BatchNorm2d in place of fused batch norm layers.

    Parameters
    ----------
    model: torch.nn.Module.
    """
    fx_model = torch.fx.symbolic_trace(model)
    modules = dict(model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue

        if type(modules[node.target]) is nn.Identity:
            conv = modules[node.args[0].target]
            size = conv.weight.shape[0]
            bn = nn.BatchNorm2d(size)
            _, attr_name = get_parent_name(node.target)
            parent = get_parent_module(model, node.target)
            setattr(parent, attr_name, bn)


def standard_continuous_dims(model):
    """
    Returns dict containing names of all Conv2d and Linear layer's parameters as keys
    and [0, 1] / [0] as values for weight / bias.

    Parameters
    ----------
    model: torch.nn.Module.
    """
    continuous_dims = {}

    for name, param in model.named_parameters():
        parent_name, attr_name = get_parent_name(name)
        parent = get_parent_module(model, name)

        if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)):
            if 'weight' in attr_name:
                continuous_dims[name] = [0, 1]

            elif 'bias' in name:
                continuous_dims[name] = [0]

    return continuous_dims


@contextmanager
def grid_tuning(integral_model,
                train_bn=False,
                train_bias=False,
                use_all_grids=False):
    """
    Context manager sets requires_grad=True only for TrainableGrid parameters
    and batch norm and bias parameters if corresponding flag is set True.

    Parameters
    ----------
    train_bn: bool.
    train_bias: bool.
    use_all_grids: bool.
    """
    if use_all_grids:
        for group in integral_model.groups:
            if group.subgroups is None:
                group.reset_grid(TrainableGrid1D(group.grid_size()))

    for name, param in integral_model.named_parameters():
        parent = get_parent_module(integral_model, name)

        if isinstance(parent, TrainableGrid1D):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if train_bn:
        reset_batchnorm(integral_model)

    if train_bias:
        for group in integral_model.groups:
            for p in group.params:
                if 'bias' in p['name']:
                    parent = get_parent_module(integral_model, p['name'])
                    if parametrize.is_parametrized(parent, 'bias'):
                        parametrize.remove_parametrizations(parent, 'bias', True)
                    getattr(parent, 'bias').requires_grad = True
    try:
        yield None

    finally:
        for name, param in integral_model.named_parameters():
            parent = get_parent_module(integral_model, name)

            if isinstance(parent, TrainableGrid1D):
                param.requires_grad = False
            else:
                param.requires_grad = True
