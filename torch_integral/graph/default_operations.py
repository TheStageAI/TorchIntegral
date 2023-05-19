import operator
from .operations import *
from .operations import neutral_decorator
from ..utils import get_attr_by_name


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
    torch.nn.Identity: neutral_hook,
}


def replace_operations(module: torch.nn.Module,
                       new_operations=None) -> torch.nn.Module:

    fx_model = torch.fx.symbolic_trace(module)
    modules = dict(fx_model.named_modules())
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
            # node_module = modules[node.target]
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