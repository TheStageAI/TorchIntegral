import torch
from typing import Tuple, Dict, Any
import torch.fx as fx
import copy
import torch.nn as nn
from torch.nn.utils import fuse_conv_bn_eval
from collections import OrderedDict


def get_module_by_name(module, name):
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


def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_hooks(child)


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert (isinstance(node.target, str))
    parent_name, name = get_parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse_batchnorm(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            node.replace_all_uses_with(node.args[0])
            fx_model.graph.erase_node(node)

    fx_model.graph.lint()
    fx_model.recompile()

    return fx_model
