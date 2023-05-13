import torch
import random


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
        out = random.normalvariate(0, 0.5 * (self.max_val - self.min_val))
        out = max(1, self.max_val - int(abs(out)))
        
        return out


class IGrid(torch.nn.Module):
    def __init__(self):
        super(IGrid, self).__init__()
        self.curr_grid = None

    def forward(self):
        if self.curr_grid is None:
            out = self.generate_grid()
        else:
            out = self.curr_grid

        return out

    def ndim(self):
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def generate_grid(self):
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class TrainableGrid1D(IGrid):
    def __int__(self, size, init_value=None):
        super(TrainableGrid1D, self).__int__()

        if init_value is None:
            self._deltas = torch.nn.Parameter(torch.ones(size - 1))
        else:
            self._deltas = torch.nn.Parameter(init_value)

    def ndim(self):
        return 1

    def generate_grid(self):
        device = self.deltas.device
        grid = self.deltas.abs() + 1e-8
        grid = grid / grid.sum()
        grid = grid.cumsum(0)
        self.curr_grid = torch.cat(
            [torch.tensor([0.], device=device), grid]
        )

        return self.curr_grid


class RandomUniformGrid1D(IGrid):
    def __init__(self, distribution):
        super(RandomUniformGrid1D, self).__init__()
        self._distribution = distribution
        self.eval_size = distribution.max_val

    def ndim(self):
        return 1

    def generate_grid(self):
        if self.training:
            size = self._distribution.sample()
        else:
            size = self.eval_size
            
        self.curr_grid = torch.linspace(-1, 1, size)

        return self.curr_grid
    
    def resize(self, new_size):
        self.eval_size = new_size


class GridND(IGrid):
    def __init__(self, *grid_objects):
        super(GridND, self).__init__()
        self.grid_objects = torch.nn.ModuleList(grid_objects)

    def ndim(self):
        return sum([
            grid.ndim() for grid in self.grid_objects
        ])

    def reset_grid(self, dim, new_grid):
        self.grid_objects[dim] = new_grid

    def generate_grid(self):
        self.curr_grid = [
            grid_obj() for grid_obj in self.grid_objects
        ]

        return self.curr_grid
