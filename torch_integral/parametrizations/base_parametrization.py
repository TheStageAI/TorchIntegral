import torch


class IWeights(torch.nn.Module):
    """
    Class for weights parametrization. Can be registereg as parametrization
    with torch.nn.utils.parametrize.register_parametrization

    Parameters
    ----------
    weight_function: torch.nn.Module.
    grid: torch_integral.grid.IGrid.
    quadrature: torch_integral.quadrature.BaseIntegrationQuadrature.
    """

    def __init__(self, grid, quadrature):
        super().__init__()
        self.quadrature = quadrature
        self.grid = grid
        self.last_value = None
        self.train_volume = None

    def reset_quadrature(self, quadrature):
        """Replaces quadrature object."""
        weight = self.sample_weights(None)
        self.quadrature = quadrature
        self.right_inverse(weight)

    def clear(self):
        self.last_value = None

    def right_inverse(self, x):
        """Initialization method which is used when setattr of parametrized tensor called."""
        train_volume = 1.

        if self.quadrature is not None:
            ones = torch.ones_like(x, device=x.device)
            q_coeffs = self.quadrature.multiply_coefficients(ones, self.grid())
            x = x / q_coeffs

            for dim in self.quadrature.integration_dims:
                train_volume *= x.shape[dim] - 1

            train_volume *= 0.5

            if self.train_volume is None:
                self.train_volume = train_volume

            x = x / self.train_volume

        x = self.init_values(x)

        return x

    def init_values(self, weight):
        """ """
        return weight

    def forward(self, w):
        """
        Performs forward pass. Samples new weights on grid
        if training or last sampled tensor is not cached.

        Parameters
        ----------
        w: torch.Tensor.
        """
        # if self.training or self.last_value is None:
        #     weight = self.sample_weights(w)
        #
        #     if self.training:
        #         self.clear()
        #     else:
        #         self.last_value = weight
        #
        # else:
        #     weight = self.last_value
        weight = self.sample_weights(w)

        return weight.to(w.device)

    def sample_weights(self, w):
        """
        Evaluate pparametrization function on grid.

        Parameters
        ----------
        w: torch.Tensor.

        Returns
        -------
        torch.Tensor.
            Sampled weight function on grid.
        """
        x = self.grid()
        weight = self.evaluate_function(x, w)

        if self.quadrature is not None:
            weight = self.quadrature(weight, x) * self.train_volume

        return weight

    def evaluate_function(self, grid, weight):
        """ """
        raise NotImplementedError("Implement this method in derived class.")
