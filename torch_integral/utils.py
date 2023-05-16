import torch
from typing import Tuple, Dict, Any
import torch.fx as fx
import copy
import torch.nn as nn
from torch.nn.utils import fuse_conv_bn_eval
from collections import OrderedDict
from .grid import TrainableGrid1D
from contextlib import contextmanager


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


def get_parent_module(module, attr_path):  # use it in other files
    parent_name, attr_name = get_parent_name(attr_path)

    if parent_name != '':
        parent = get_module_by_name(module, parent_name)
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


def fuse_batchnorm(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue
        if type(modules[node.target]) is nn.BatchNorm2d \
           and type(modules[node.args[0].target]) is nn.Conv2d:

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


def optimize_parameters(module, attr, target,
                        start_lr=1e-2, iterations=100):

    module.train()
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(module.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=iterations // 5, gamma=0.2
    )
    print(
        'loss before optimization: ',
        float(criterion(getattr(module, attr), target))
    )

    for i in range(iterations):
        weight = getattr(module, attr)
        loss = criterion(weight, target)
        loss.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()

        if i == iterations - 1:
            print('loss after optimization: ', float(loss))


@contextmanager
def grid_tuning(integral_model):
    grids = [
        TrainableGrid1D(g.size())
        for g in integral_model.get_grids()
    ]
    integral_model.reset_grids(grids)

    for name, param in integral_model.named_parameters():
        parent = get_parent_module(integral_model, name)

        if isinstance(parent, TrainableGrid1D):
            param.requires_grad = True
        else:
            param.requires_grad = False
    try:
        yield None

    finally:
        for name, param in integral_model.named_parameters():
            parent = get_parent_module(integral_model, name)

            if isinstance(parent, TrainableGrid1D):
                param.requires_grad = False
            else:
                param.requires_grad = True
