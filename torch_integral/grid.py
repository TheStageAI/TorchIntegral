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
        out = random.normalvariate(
            0, 0.5 * (self.max_val - self.min_val)
        )
        out = max(1, self.max_val - int(abs(out)))
        
        return out


class IGrid(torch.nn.Module):
    def __init__(self):
        super(IGrid, self).__init__()
        self.curr_grid = None
        self.eval_size = None

    def forward(self):
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
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class ConstantGrid1D(IGrid):
    def __init__(self, init_value):
        super(ConstantGrid1D, self).__init__()
        self.curr_grid = init_value

    def generate_grid(self):
        return self.curr_grid


class TrainableGrid1D(IGrid):
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


class RandomUniformGrid1D(IGrid):  # RENAME TO RANDOMLINSPACE
    def __init__(self, size_distribution):  #  NOISE ?
        super(RandomUniformGrid1D, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size
            
        self.curr_grid = torch.linspace(-1, 1, size)

        return self.curr_grid

    def resize(self, new_size):
        self.eval_size = new_size
        self.generate_grid()


class CompositeGrid1D(IGrid):
    def __init__(self, grids):
        super(CompositeGrid1D, self).__init__()
        self.grids = torch.nn.ModuleList(grids)
        size = self.size()
        self.proportions = [
            grid.size()/size for grid in grids
        ]
        self.generate_grid()

    def generate_grid(self):
        g_list = []
        start = 0.

        for i, grid in enumerate(self.grids):
            g = grid.generate_grid()
            g = (g + 1.)/2.

            if i != len(self.grids) - 1:
                g = g * (g.shape[0]/(g.shape[0] + 1))

            g = start + g * self.proportions[i]
            g_list.append(g)
            start += self.proportions[i]

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
