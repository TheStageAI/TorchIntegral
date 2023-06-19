import torch


class IntegralParameterization(torch.nn.Module):
    def __init__(self, weight_function, grid, quadrature):
        super().__init__()
        self.weight_function = weight_function
        self.quadrature = quadrature
        self.grid = grid
        self.last_value = None
        self.train_volume = 1.

    def sample_weights(self, w):
        x = self.grid()
        weight = self.weight_function(x)

        if self.quadrature is not None:
            weight = self.quadrature(weight, x) * self.train_volume

        return weight

    def clear(self):
        self.last_value = None

    def forward(self, w):
        if self.training or self.last_value is None:
            weight = self.sample_weights(w)

            if self.training:
                self.clear()
            else:
                self.last_value = weight

        else:
            weight = self.last_value

        return weight.to(w.device)

    def right_inverse(self, x):
        if hasattr(self.weight_function, 'init_values'):
            if self.quadrature is not None:
                ones = torch.ones_like(x, device=x.device)
                q_coeffs = self.quadrature.multiply_coefficients(
                    ones, self.grid()
                )
                x = x / q_coeffs

                for dim in self.quadrature.integration_dims:
                    self.train_volume *= x.shape[dim] - 1

                self.train_volume *= 0.5
                x = x / self.train_volume

            self.weight_function.init_values(x)

        return x


if __name__ == '__main__':
    import torch
    import sys

    sys.path.append('../../')
    from interpolation_weights import InterpolationWeights1D
    from interpolation_weights import InterpolationWeights2D
    from torch_integral.grid import RandomUniformGrid1D
    from torch_integral.grid import ConstantGrid1D
    from torch_integral.grid import GridND
    from torch_integral.grid import UniformDistribution
    from torch.nn.utils import parametrize
    from torch_integral.quadrature import TrapezoidalQuadrature
    from torch_integral import IntegralWrapper

    N = 64
    func = InterpolationWeights2D([64, 64], [5, 5]).cuda()
    conv = torch.nn.Conv2d(64, 64, 5).cuda()
    target = conv.weight.data.clone()
    grid = GridND({
        # '0': ConstantGrid1D((torch.rand(64)*2-1).sort().values),
        '0': RandomUniformGrid1D(UniformDistribution(64, 64)),
        '1': RandomUniformGrid1D(UniformDistribution(64, 64))
    })
    quadrature = TrapezoidalQuadrature([1])
    param = IntegralParameterization(
        func, grid, quadrature,
    )
    parametrize.register_parametrization(
        conv, 'weight', param, unsafe=True
    )
    wrapper = IntegralWrapper(optimize_iters=3000, start_lr=1e-2)
    wrapper._optimize_parameters(conv, 'weight', target, param.parameters())
