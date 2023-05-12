import torch


class BaseIntegrationQuadrature(torch.nn.Module):
    def __init__(self, integration_dims):
        super().__init__()
        self.integration_dims = integration_dims

    def discretize(self, function, grid):
        if callable(function):
            discretization = function(grid)
        else:
            discretization = function

        return discretization

    def multiply_coefficients(self, discretization, grid):
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def forward(self, function, grid):
        discretization = self.discretize(function, grid)
        discretization = self.multiply_coefficients(discretization, grid)

        return discretization


class TrapezoidalQuadrature(BaseIntegrationQuadrature):
    def __init__(self, integration_dims):
        super(TrapezoidalQuadrature, self).__init__(integration_dims)

    def multiply_coefficients(self, discretization, grid):
        for dim in self.integration_dims:
            x = grid[dim].to(discretization.device)
            h = torch.zeros_like(x)
            h[1:-1] = x[2:] - x[:-2]
            h[0] = x[1] - x[0]
            h[-1] = x[-1] - x[-2]
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * (h * 0.5)

        return discretization
