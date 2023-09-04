import torch
from .grid import RandomLinspace, UniformDistribution, CompositeGrid1D
from .graph import RelatedGroup


class IntegralGroup(RelatedGroup):
    """ """

    def __init__(self, size):
        super(RelatedGroup, self).__init__()
        self.size = size
        self.subgroups = None
        self.parents = []
        self.grid = None
        self.params = []
        self.tensors = []
        self.operations = []

    def forward(self):
        self.grid.generate_grid()

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
