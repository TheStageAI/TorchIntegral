import torch
from typing import Tuple, Dict, Any, List
import torch.fx as fx
import copy
import torch.nn as nn
from torch.nn.utils import fuse_conv_bn_eval
from collections import OrderedDict
from .grid import TrainableGrid1D
from contextlib import contextmanager


def get_attr_by_name(module, name):
    for s in name.split('.'):
        module = getattr(module, s)

    return module


def get_parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def get_parent_module(module, attr_path):
    parent_name, attr_name = get_parent_name(attr_path)

    if parent_name != '':
        parent = get_attr_by_name(module, parent_name)
    else:
        parent = module

    return parent


def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_hooks(child)


def replace_node_module(node: fx.Node,
                        modules: Dict[str, Any],
                        new_module: torch.nn.Module):
    assert (isinstance(node.target, str))
    parent_name, name = get_parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse_batchnorm(model: torch.nn.Module,
                   convs: List[str]) -> torch.nn.Module:
    model = copy.deepcopy(model)
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
                fused_conv = fuse_conv_bn_eval(conv, bn)
                parent_name, attr_name = get_parent_name(node.target)
                parent = get_parent_module(model, node.target)
                setattr(parent, attr_name, torch.nn.Identity())
                parent_name, attr_name = get_parent_name(
                    node.args[0].target
                )
                parent = get_parent_module(model, node.args[0].target)
                setattr(parent, attr_name, fused_conv)

    return model


def reset_batchnorm(model):
    fx_model = torch.fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue

        if type(modules[node.target]) is nn.Identity \
                and type(modules[node.args[0].target]) is nn.Conv2d:
            conv = modules[node.args[0].target]
            size = conv.weight.shape[0]
            bn = nn.BatchNorm2d(size)
            _, attr_name = get_parent_name(node.target)
            parent = get_parent_module(model, node.target)
            setattr(parent, attr_name, bn)


def base_continuous_dims(model):
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
def grid_tuning(integral_model, train_bn=False):
    grids = [
        TrainableGrid1D(g.size())
        for g in integral_model.grids()
    ]
    integral_model.reset_grids(grids)

    for name, param in integral_model.named_parameters():
        parent = get_parent_module(integral_model, name)

        if isinstance(parent, TrainableGrid1D):
            param.requires_grad = True
        elif isinstance(parent, torch.nn.BatchNorm2d) and train_bn:
            param.requires_grad = True
    try:
        yield None

    finally:
        for name, param in integral_model.named_parameters():
            parent = get_parent_module(integral_model, name)

            if isinstance(parent, TrainableGrid1D):
                param.requires_grad = False
            else:
                param.requires_grad = True
