import torch
from .default_operations import DEFAULT_OPERATIONS
from .operations import neutral_decorator
from .operations import neutral_hook
from ..utils import get_attr_by_name
from ..utils import remove_all_hooks


def replace_operations(module: torch.nn.Module,
                       new_operations=None,
                       skip_modules_list=None) -> torch.nn.Module:
    
    gm = torch.fx.symbolic_trace(module)
    graph = gm.graph
    operations = DEFAULT_OPERATIONS

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


def build_groups(model, sample_shape, cont_parameters):

    tracing_model = replace_operations(model)
    all_grids = prepare_parameters(cont_parameters)
    device = next(iter(model.parameters())).device
    x = torch.rand(sample_shape).to(device)
    tracing_model(x)
    remove_all_hooks(tracing_model)
    del tracing_model
    grids = [
        g for g in all_grids if len(g['params']) != 0
    ]

    return grids


if __name__ == '__main__':
    from torchvision.models import resnet18

    model = resnet18()
    cont_params = {}
    grids = build_groups(model, [1, 3, 100, 100])

    for g in grids:
        print(len(g['params']))
        