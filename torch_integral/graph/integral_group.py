import torch


class IntegralGroup(torch.nn.Module):
    def __init__(self, size):
        super(IntegralGroup, self).__init__()
        self.size = size
        self.subgroups = None
        self.parents = []
        self.grid = None
        self.params = []
        self.tensors = []
        self.parametrizations = None

    def append_param(self, name, value, dim, operation=None):
        self.params.append({
            'value': value,
            'name': name,
            'dim': dim,
            'operation': operation
        })

    def append_tensor(self, value, dim, operation=None):
        self.tensors.append({
            'value': value,
            'dim': dim,
            'operation': operation
        })

    def clear_params(self):
        self.params = []

    def clear_tensors(self):
        self.tensors = []

    def set_subgroups(self, groups):
        self.subgroups = groups

        for subgroup in self.subgroups:
            subgroup.parents.append(self)
    
    def grid_size(self):
        return self.grid.size()

    def clear(self):
        for _, obj, dim in self.parametrizations:
            obj.grid.reset_grid(dim, self.grid)
            obj.clear()
            
    def reset_grid(self, grid):
        self.grid = grid
        self.clear()
            
        for parent in self.parents:
            parent.reset_child_grid(self, grid)
            
    def reset_child_grid(self, child, new_grid):
        i = 0
        
        for i, subgroup in enumerate(self.subgroups):
            if child is subgroup:
                break
                
        self.grid.grids[i] = new_grid
        self.clear()
        
    def resize(self, new_size):
        if hasattr(self.grid, 'resize'):
            self.grid.resize(new_size)

        self.clear()
        
        for parent in self.parents:
            parent.clear()

    def reset_distribution(self, distribution):
        if hasattr(self.grid, 'distribution'):
            self.grid.distribution = distribution
            
    @staticmethod
    def append_to_groups(tensor, operation=None, attr_name='grids'):
        if hasattr(tensor, attr_name):
            for i, g in enumerate(getattr(tensor, attr_name)):
                if g is not None:
                    g.append_tensor(tensor, i, operation)


# class GroupList:
#     def __init__(self, groups):
#         self.groups = groups
#
#     def __contains__(self, obj):
#         return obj in self.groups
#
#     def __getitem__(self, item):
#         return self.groups[item]


def merge_groups(x, x_dim, y, y_dim):
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

            if len(y.grids[y_dim].parents) > 0:
                x, x_dim, y, y_dim = y, y_dim, x, x_dim

            if y.grids[y_dim].subgroups is not None:
                x, x_dim, y, y_dim = y, y_dim, x, x_dim

            if x.grids[x_dim] is not y.grids[y_dim]:
                for param in y.grids[y_dim].params:
                    dim = param['dim']
                    t = param['value']

                    if t is not y:
                        t.grids[dim] = x.grids[x_dim]

                x.grids[x_dim].params.extend(y.grids[y_dim].params)
                y.grids[y_dim].clear_params()

                for tensor in y.grids[y_dim].tensors:
                    dim = tensor['dim']
                    t = tensor['value']

                    if t is not y:
                        t.grids[dim] = x.grids[x_dim]

                x.grids[x_dim].tensors.extend(y.grids[y_dim].tensors)
                y.grids[y_dim].clear_tensors()

        y.grids[y_dim] = x.grids[x_dim]