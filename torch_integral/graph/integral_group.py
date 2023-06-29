import torch
from ..grid import RandomLinspace, UniformDistribution, CompositeGrid1D


class IntegralGroup(torch.nn.Module):
    """
    Class for grouping tensors and parameters.
    Group is a collection of paris of tensor and it's dimension.
    Two parameter tensors are considered to be in the same group
    if they should have the same integration grid.
    Group can contain subgroups. This means that parent group's grid is a con
    catenation of subgroups grids.

    Parameters
    ----------
    size: int.
        Each tensor in the group should have the same size along certain dimension.
    """

    def __init__(self, size):
        super(IntegralGroup, self).__init__()
        self.size = size
        self.subgroups = None
        self.parents = []
        self.grid = None
        self.params = []
        self.tensors = []
        self.operations = []

    def append_param(self, name, value, dim, operation=None):
        """
        Adds parameter tensor to the group.

        Parameters
        ----------
        name: str.
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.params.append(
            {"value": value, "name": name, "dim": dim, "operation": operation}
        )

    def append_tensor(self, value, dim, operation=None):
        """
        Adds tensor to the group.

        Parameters
        ----------
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.tensors.append({"value": value, "dim": dim, "operation": operation})

    def clear_params(self):
        self.params = []

    def clear_tensors(self):
        self.tensors = []

    def set_subgroups(self, groups):
        self.subgroups = groups

        for subgroup in self.subgroups:
            subgroup.parents.append(self)

    def build_operations_set(self):
        """Builds set of operations in the group."""
        self.operations = set([t["operation"] for t in self.tensors])

    @staticmethod
    def append_to_groups(tensor, operation=None, attr_name="grids"):
        if hasattr(tensor, attr_name):
            for i, g in enumerate(getattr(tensor, attr_name)):
                if g is not None:
                    g.append_tensor(tensor, i, operation)

    def grid_size(self):
        """Returns size of the grid."""
        return self.grid.size()

    def clear(self, new_grid=None):
        """Resets grid and removes cached values."""
        for param_dict in self.params:
            function = param_dict["function"]
            dim = list(function.grid).index(self.grid)
            grid = new_grid if new_grid is not None else self.grid
            function.grid.reset_grid(dim, grid)
            function.clear()

    def initialize_grids(self):
        """Sets default RandomLinspace grid."""
        if self.grid is None:
            if self.subgroups is not None:
                for subgroup in self.subgroups:
                    if subgroup.grid is None:
                        subgroup.initialize_grids()

                self.grid = CompositeGrid1D([sub.grid for sub in self.subgroups])
            else:
                distrib = UniformDistribution(self.size, self.size)
                self.grid = RandomLinspace(distrib)

        for parent in self.parents:
            if parent.grid is None:
                parent.initialize_grids()

    def reset_grid(self, new_grid):
        """
        Set new integration grid for the group.

        Parameters
        ----------
        new_grid: IntegralGrid.
        """
        self.clear(new_grid)

        for parent in self.parents:
            parent.reset_child_grid(self, new_grid)

        self.grid = new_grid

    def reset_child_grid(self, child, new_grid):
        """Sets new integration grid for given child of the group."""
        i = self.subgroups.index(child)
        self.grid.reset_grid(i, new_grid)
        self.clear()

    def resize(self, new_size):
        """If grid supports resizing, resizes it."""
        if hasattr(self.grid, "resize"):
            self.grid.resize(new_size)

        self.clear()

        for parent in self.parents:
            parent.clear()

    def reset_distribution(self, distribution):
        """Sets new distribution for the group."""
        if hasattr(self.grid, "distribution"):
            self.grid.distribution = distribution

    def __str__(self):
        result = ""

        for p in self.params:
            result += p["name"] + ": " + str(p["dim"]) + "\n"

        return result

    def count_parameters(self):
        ans = 0

        for p in self.params:
            ans += p["value"].numel()

        return ans


def merge_groups(x, x_dim, y, y_dim):
    """Merges two groups of tensors ``x`` and `yy`` with indices ``x_dim`` and ``y_dim``."""
    if type(x) in (int, float):
        x = torch.tensor(x)
    if type(y) in (int, float):
        y = torch.tensor(y)
    if not hasattr(x, "grids"):
        x.grids = [None for _ in range(x.ndim)]
    if not hasattr(y, "grids"):
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
                    dim = param["dim"]
                    t = param["value"]

                    if t is not y:
                        t.grids[dim] = x.grids[x_dim]

                x.grids[x_dim].params.extend(y.grids[y_dim].params)
                y.grids[y_dim].clear_params()

                for tensor in y.grids[y_dim].tensors:
                    dim = tensor["dim"]
                    t = tensor["value"]

                    if t is not y:
                        t.grids[dim] = x.grids[x_dim]

                x.grids[x_dim].tensors.extend(y.grids[y_dim].tensors)
                y.grids[y_dim].clear_tensors()

        y.grids[y_dim] = x.grids[x_dim]
