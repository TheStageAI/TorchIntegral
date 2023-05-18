import torch
from .default_operations import DEFAULT_OPERATIONS
from .default_operations import DEFAULT_HOOKS
from .operations import neutral_decorator
from ..utils import remove_all_hooks


def replace_operations(module: torch.nn.Module,
                       new_operations=None,
                       skip_modules_list=None) -> torch.nn.Module:
    
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
            node_module = modules[node.target]

            if type(node_module) not in hooks_dict:
                mod_name = node.target.replace('_', '.')
                new_module = replace_operations(
                    node_module, new_operations, skip_modules_list
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


def build_groups(model, sample_shape, cont_parameters):
    tracing_model = replace_operations(model)
    all_groups = prepare_parameters(cont_parameters)
    device = next(iter(model.parameters())).device
    x = torch.rand(sample_shape).to(device)
    tracing_model(x)
    remove_all_hooks(tracing_model)
    del tracing_model
    groups = [
        group for group in all_groups
        if len(group['params']) != 0
    ]

    return groups


if __name__ == '__main__':
    from torchvision.models import resnet18

    model = resnet18()
    cont_params = {}
    grids = build_groups(model, [1, 3, 100, 100])

    for g in grids:
        print(len(g['params']))
        