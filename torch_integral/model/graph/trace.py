import torch
from .operations import replace_operations
from .integral_group import IntegralGroup
# from .integral_group import GroupList
from ...utils import remove_all_hooks
    
    
class Tracer:
    def __init__(self, model,
                 sample_shape,
                 continuous_dims):

        self.continuous_dims = continuous_dims
        self.sample_shape = sample_shape
        self.model = model
        self.groups = None

    def _preprocess_parameters(self):
        self.groups = []

        for name, p in self.model.named_parameters():
            p.grids = [None] * p.ndim

            if name in self.continuous_dims:
                dims = self.continuous_dims[name]
            else:
                dims = list(range(p.ndim))

            for d in dims:
                size = p.shape[d]
                group = IntegralGroup(size)
                group.append_param(name, p, d)
                p.grids[d] = group
                self.groups.append(group)

    def _postprocess_groups(self):
        delete_indices = []

        for i, group in enumerate(self.groups):
            delete_group = True

            for p in group.params:
                if p['name'] in self.continuous_dims:
                    if p['dim'] in self.continuous_dims[p['name']]:
                        delete_group = False

            if delete_group:
                delete_indices.append(i)
            else:
                for p in group.params:
                    if p['name'] in self.continuous_dims:
                        dims = self.continuous_dims[p['name']]

                        if p['dim'] not in dims:
                            dims.append(p['dim'])
                    else:
                        self.continuous_dims[p['name']] = [p['dim']]

        self.groups = [
            group for i, group in enumerate(self.groups)
            if i not in delete_indices
        ]

        parents = set()

        for group in self.groups:
            self._add_parent_groups(group, parents)

        return list(parents)

    def _add_parent_groups(self, group, parents):
        for parent in group.parents:
            if parent not in parents:
                parents.add(parent)
            self._add_parent_groups(parent, parents)

    def build_groups(self):
        tracing_model = replace_operations(self.model)
        self._preprocess_parameters()
        device = next(iter(self.model.parameters())).device
        x = torch.rand(self.sample_shape).to(device)
        tracing_model(x)
        remove_all_hooks(tracing_model)
        del tracing_model
        self.groups = [
            group for group in self.groups
            if len(group.params) != 0
        ]
        parents = self._postprocess_groups()

        return self.groups, parents