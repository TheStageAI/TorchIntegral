import torch
from functools import reduce
from torch.nn.functional import grid_sample


class IWeights(torch.nn.Module):
    def __init__(self, discrete_shape):
        super().__init__()
        self._discrete_shape = discrete_shape

    def init_values(self):
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def forward(self, grid):
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class InterpolationWeightsBase(IWeights):
    def __init__(self, cont_size, discrete_shape=None,
                 interpolate_mode='bicubic',
                 padding_mode='border',
                 align_corners=True):

        super(InterpolationWeightsBase, self).__init__(discrete_shape)

        self.iterpolate_mode = interpolate_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        if discrete_shape is not None:
            self.planes_num = int(
                reduce(lambda a, b: a * b, discrete_shape)
            )
        else:
            self.planes_num = 1

        self.values = torch.nn.Parameter(
            torch.rand(1, self.planes_num, *cont_size)
        )

    def preprocess_grid(self, grid):
        device = self.values.device

        for i in range(len(grid)):
            grid[i] = grid[i].to(device)

        if len(grid) == 1:
            grid.append(torch.tensor(0., device=device))

        grid = torch.stack(
            torch.meshgrid(grid[::-1], indexing='ij'), dim=len(grid),
        ).unsqueeze(0)

        return grid

    def postprocess_output(self, out):
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def forward(self, grid):
        grid = self.preprocess_grid(grid)
        out = grid_sample(
            self.values, grid, mode=self.iterpolate_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )
        return self.postprocess_output(out)


class InterpolationWeights1D(InterpolationWeightsBase):
    def __init__(self, cont_size, discrete_shape=None,
                 cont_dim=0, interpolate_mode='bicubic',
                 padding_mode='border', align_corners=True):

        super(InterpolationWeights1D, self).__init__(
            [cont_size, 1], discrete_shape, interpolate_mode,
            padding_mode, align_corners
        )
        self.cont_dim = cont_dim

    def init_values(self, x):
        if x.ndim == 1:
            x = x[None, None, :, None]
        else:
            permutation = [
                i for i in range(x.ndim) if i != self.cont_dim
            ]
            x = x.permute(*permutation, self.cont_dim)
            x = x.reshape(1, -1, x.shape[-1], 1)

        self.values.data = x

    def postprocess_output(self, out):
        discrete_shape = self._discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        shape = out.shape[-1:]
        out = out.view(*discrete_shape, *shape)
        permutation = list(range(out.ndim))
        permutation[self.cont_dim] = out.ndim - 1
        j = 0

        for i in range(len(permutation)):
            if i != self.cont_dim:
                permutation[i] = j
                j += 1

        out = out.permute(*permutation).contiguous()

        return out


class InterpolationWeights2D(InterpolationWeightsBase):
    def __init__(self, cont_size, discrete_shape=None,
                 interpolate_mode='bicubic',
                 padding_mode='border',
                 align_corners=True):

        super(InterpolationWeights2D, self).__init__(
            cont_size, discrete_shape, interpolate_mode,
            padding_mode, align_corners
        )

    def init_values(self, x):
        if x.ndim == 2:
            x = x[None, None, :, :]
        else:
            permutation = list(range(2, x.ndim)) #  [::-1]  # ????
            shape = x.shape[:2]
            x = x.permute(*permutation, 0, 1)
            x = x.reshape(1, -1, *shape)

        self.values.data = x

    def postprocess_output(self, out):
        discrete_shape = self._discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        shape = out.shape[-2:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 2)
        out = out.permute(out.ndim - 1, out.ndim - 2, *dims)

        return out.contiguous()


class WeightsParameterization(torch.nn.Module):
    def __init__(self, weight_function, grid, quadrature):
        super().__init__()
        self.weight_function = weight_function
        self.quadrature = quadrature
        self.grid = grid
        self.last_value = None

    def sample_weights(self, w):
        x = self.grid()
        weight = self.weight_function(x)

        if self.quadrature is not None:
            weight = self.quadrature(weight, x)

        return weight

    def clear(self):
        self.last_value = None

    def forward(self, w):
        if self.training or self.last_value is None:
            weight = self.sample_weights(w)

            if self.training:
                self.last_value = None
            else:
                self.last_value = weight

        else:
            weight = self.last_value

        return weight

    def right_inverse(self, x):
        if hasattr(self.weight_function, 'init_values'):
            if self.quadrature is not None:
                ones = torch.ones_like(x, device=x.device)
                q_coeffs = self.quadrature.multiply_coefficients(
                    ones, self.grid()
                )
                x = x / q_coeffs

            self.weight_function.init_values(x)

        return x


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from torch_integral.utils import optimize_parameters
    from torch_integral.grid import RandomUniformGrid1D
    from torch_integral.grid import GridND
    from torch_integral.grid import UniformDistribution
    from torch.nn.utils import parametrize
    from torch_integral.quadrature import TrapezoidalQuadrature

    N = 64
    func = InterpolationWeights2D([64, 64], [5, 5]).cuda()
    conv = torch.nn.Conv2d(64, 64, 5).cuda()
    target = conv.weight.data.clone()
    grid = GridND({
        '0': RandomUniformGrid1D(UniformDistribution(64, 64)),
        '1': RandomUniformGrid1D(UniformDistribution(64, 64))
    })
    quadrature = TrapezoidalQuadrature([1])
    param = WeightsParameterization(
        func, grid, quadrature,
    )
    parametrize.register_parametrization(
        conv, 'weight', param, unsafe=True
    )
    optimize_parameters(conv, 'weight', target, 1e-2, 0)
