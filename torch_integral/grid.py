import torch
import random
from scipy.special import roots_legendre


class Distribution(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def sample(self):
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class UniformDistribution(Distribution):
    def __init__(self, min_val, max_val):
        super().__init__(min_val, max_val)

    def sample(self):
        return random.randint(self.min_val, self.max_val)


class NormalDistribution(Distribution):
    def __init__(self, min_val, max_val):
        super(NormalDistribution, self).__init__(min_val, max_val)

    def sample(self):
        out = random.normalvariate(
            0, 0.5 * (self.max_val - self.min_val)
        )
        out = self.max_val - int(abs(out))

        if out < self.min_val:
            out = random.randint(self.min_val, self.max_val)

        return out


class IGrid(torch.nn.Module):
    """
    Base Grid class.
    """
    def __init__(self):
        super(IGrid, self).__init__()
        self.curr_grid = None
        self.eval_size = None

    def forward(self):
        """
        Performs forward pass. Generates new grid
        if last generated grid is not saved.
        """
        if self.curr_grid is None:
            out = self.generate_grid()
        else:
            out = self.curr_grid

        return out

    def ndim(self):
        return 1

    def size(self):
        return self.eval_size

    def generate_grid(self):
        """
        """
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class ConstantGrid1D(IGrid):
    """
    Class implements IGrid interface for fixed grid.

    Parameters
    ----------
    init_value: torch.Tensor.
    """
    def __init__(self, init_value):
        super(ConstantGrid1D, self).__init__()
        self.curr_grid = init_value

    def generate_grid(self):
        """
        """
        return self.curr_grid


class TrainableGrid1D(IGrid):
    """
    """
    def __init__(self, size, init_value=None):
        super(TrainableGrid1D, self).__init__()
        self.eval_size = size
        self.curr_grid = torch.nn.Parameter(
            torch.linspace(-1, 1, size)
        )
        if init_value is not None:
            assert size == init_value.shape[0]
            self.curr_grid.data = init_value

    def generate_grid(self):
        return self.curr_grid


class RandomLinspace(IGrid):
    def __init__(self, size_distribution, noise_std=0):
        super(RandomLinspace, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        self.noise_std = noise_std
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size

        self.curr_grid = torch.linspace(-1, 1, size)

        if self.noise_std > 0:
            noise = torch.normal(
                torch.zeros(size), self.noise_std * torch.ones(size)
            )
            self.curr_grid = self.curr_grid + noise

        return self.curr_grid

    def resize(self, new_size):
        self.eval_size = new_size
        self.generate_grid()


class RandomLegendreGrid(RandomLinspace):
    def __init__(self, size_distribution):
        super(RandomLinspace, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size

        self.curr_grid, _ = roots_legendre(size)
        self.curr_grid = torch.tensor(self.curr_grid, dtype=torch.float32)

        return self.curr_grid


class TrainableRandomGrids(RandomLinspace):
    def __init__(self, size_distribution):
        super(RandomLinspace, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        min_val, max_val = self.distribution.min_val, self.distribution.max_val + 1
        self.grids = torch.nn.ParameterDict({
            str(size): torch.nn.Parameter(torch.linspace(-1, 1, size))
            for size in range(min_val, max_val)
        })
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size

        if not str(size) in self.grids:
            self.grids[str(size)] = torch.nn.Parameter(
                torch.linspace(-1, 1, size)
            )

        self.curr_grid = self.grids[str(size)]

        return self.curr_grid


class CompositeGrid1D(IGrid):
    def __init__(self, grids):
        super(CompositeGrid1D, self).__init__()
        self.grids = torch.nn.ModuleList(grids)
        size = self.size()
        self.proportions = [
            (grid.size() - 1) / (size - 1) for grid in grids
        ]
        self.generate_grid()

    def generate_grid(self):
        g_list = []
        start = 0.
        h = 1 / (self.size() - 1)
        device = None

        for i, grid in enumerate(self.grids):
            g = grid.generate_grid()
            device = g.device if device is None else device
            g = (g + 1.) / 2.
            g = start + g * self.proportions[i]
            g_list.append(g.to(device))
            start += self.proportions[i] + h

        self.curr_grid = 2. * torch.cat(g_list) - 1.

        return self.curr_grid

    def size(self):
        return sum([g.size() for g in self.grids])


class GridND(IGrid):
    def __init__(self, grid_objects_dict):
        super(GridND, self).__init__()
        self.grid_objects = torch.nn.ModuleDict(grid_objects_dict)
        self.generate_grid()

    def ndim(self):
        return sum([
            grid.ndim() for _, grid in self.grid_objects.items()
        ])

    def reset_grid(self, dim, new_grid):
        self.grid_objects[str(dim)] = new_grid
        self.generate_grid()

    def generate_grid(self):
        self.curr_grid = [
            self.grid_objects[dim].generate_grid()
            for dim in self.grid_objects
        ]

        return self.curr_grid

    def forward(self):
        self.curr_grid = [
            self.grid_objects[dim]()
            for dim in self.grid_objects
        ]

        return self.curr_grid
