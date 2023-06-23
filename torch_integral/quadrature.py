import torch
from scipy.special import roots_legendre


class BaseIntegrationQuadrature(torch.nn.Module):
    """
    Base quadrature class

    Parameters
    ----------
    integration_dims: List[int]. Numbers of dimensions along which we multiply by the quadrature weights
    grid_indices: List[int]. Indices of corresponding grids.
    """
    def __init__(self, integration_dims, grid_indices=None):
        super().__init__()
        self.integration_dims = integration_dims

        if grid_indices is None:
            self.grid_indices = integration_dims
        else:
            self.grid_indices = grid_indices
            assert len(grid_indices) == len(integration_dims)

    def discretize(self, function, grid):
        if callable(function):
            discretization = function(grid)
        else:
            discretization = function

        return discretization

    def multiply_coefficients(self, discretization, grid):
        """
        Multiply discretization tensor by quadrature weights along integration_dims.

        Parameters
        ----------
        discretization: torch.Tensor.
        grid: List[torch.Tensor].
        """
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def forward(self, function, grid):
        discretization = self.discretize(function, grid)
        discretization = self.multiply_coefficients(
            discretization, grid
        )

        return discretization


class TrapezoidalQuadrature(BaseIntegrationQuadrature):
    """
    Class for integration with trapezoidal rule.
    """
    def multiply_coefficients(self, discretization, grid):
        """
        """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i].to(discretization.device)
            h = torch.zeros_like(x)
            h[1:-1] = x[2:] - x[:-2]
            h[0] = x[1] - x[0]
            h[-1] = x[-1] - x[-2]
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * (h * 0.5)

        return discretization


class RiemannQuadrature(BaseIntegrationQuadrature):
    """
    Rectangular integration rule.
    """
    def multiply_coefficients(self, discretization, grid):
        """
        """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i].to(discretization.device)
            h = x[1:] - x[:-1]
            h = torch.cat([0.5*h[0], 0.5*(h[:-1] + h[1:]), 0.5*h[-1]])
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * h

        return discretization


class SimpsonQuadrature(BaseIntegrationQuadrature):
    """
    Integratioin of the function in propositioin
    that function is quadratic between sampling points.
    """
    def multiply_coefficients(self, discretization, grid):
        """
        """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i].to(discretization.device)
            # assert x.shape[0] % 2 == 1
            step = x[1] - x[0]
            h = torch.ones_like(x)
            h[1::2] *= 4.
            h[2:-1:2] *= 2.
            h *= step / 3.
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * h

        return discretization


class LegendreQuadrature(BaseIntegrationQuadrature):
    """
    """
    def multiply_coefficients(self, discretization, grid):
        """
        """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i].to(discretization.device)
            _, weights = roots_legendre(x.shape[0])
            h = torch.tensor(
                weights, dtype=torch.float32, device=discretization.device
            )
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * h

        return discretization
