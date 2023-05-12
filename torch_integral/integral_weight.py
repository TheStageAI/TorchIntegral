import torch
from functools import reduce
from torch.nn.functional import grid_sample


class IWeights(torch.nn.Module):
    def __init__(self, discrete_shape):
        super().__init__()
        self._discrete_shape = discrete_shape

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
            self.planes_num = reduce(lambda a, b: a * b, discrete_shape)
        else:
            self.planes_num = 1

        self.values = torch.nn.Parameter(
            torch.rand(self.planes_num, 1, *cont_size) - 0.5
        )

    def init_from_tensor(self, x):
        pass

    def preprocess_grid(self, grid):
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def postprocess_output(self, out):
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def forward(self, grid):
        grid = self.preprocess_grid(grid)
        out = grid_sample(
            self.values, grid, mode=self.iterpolate_mode,
            padding_mode=self.padding_mode, align_corners=self.align_corners
        )
        return self.postprocess_output(out)


class InterpolationWeights1D(InterpolationWeightsBase):
    def __init__(self, cont_size, discrete_shape=None,
                 interpolate_mode='bicubic',
                 padding_mode='border',
                 align_corners=True):
        super(InterpolationWeights1D, self).__init__(
            [cont_size, 1], discrete_shape, interpolate_mode,
            padding_mode, align_corners
        )

    def preprocess_grid(self, grid):
        device = self.values.device
        grid[0] = grid[0].to(device)
        grid.append(torch.tensor(0., device=device))
        grid = torch.stack(
            torch.meshgrid([grid[1], grid[0]]), dim=2
        )
        grid = grid.unsqueeze(0).repeat(self.planes_num, *([1] * grid.ndim))

        return grid

    def postprocess_output(self, out):
        discrete_shape = self._discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        shape = out.shape[-1:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 1)
        out = out.permute(out.ndim - 1, *dims).contiguous()

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

    def preprocess_grid(self, grid):
        device = self.values.device
        grid[0] = grid[0].to(device)
        grid[1] = grid[1].to(device)
        grid = torch.stack(
            torch.meshgrid([grid[1], grid[0]]), dim=2
        )
        grid = grid.unsqueeze(0).repeat(self.planes_num, *([1] * grid.ndim))

        return grid

    def postprocess_output(self, out):
        discrete_shape = self._discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        shape = out.shape[-2:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 2)
        out = out.permute(out.ndim - 1, out.ndim - 2, *dims).contiguous()

        return out


class WeightsParameterization(torch.nn.Module):
    def __init__(self, weight_function, grid, quadrature):
        super().__init__()
        self.weight_function = weight_function
        self.quadrature = quadrature
        self.grid = grid
        self.last_value = None

    def forward(self, weight):
        if self.training or self.last_value is None:
            x = self.grid()
            weight = self.weight_function(x)

            if self.quadrature is not None:
                out = self.quadrature(weight, x)
            else:
                out = weight

            self.last_value = out
        else:
            out = self.last_value

        return out

    def right_inverse(self, A):
        return self.forward(A)


if __name__ == '__main__':

    import sys

    sys.path.append('../../')
    from torch_integral.grid import GridND, RandomUniformGrid1D, UniformDistribution
    from torch_integral.quadrature import TrapezoidalQuadrature

    w_func = InterpolationWeights2D([32, 64], [3, 3])
    grid = [
        torch.linspace(-1, 1, 12),
        torch.linspace(-1, 1, 11)
    ]
    print(w_func(grid).shape)
    conv = torch.nn.Conv2d(16, 32, 3, bias=True)
    dist = UniformDistribution(16, 16)
    grid_2d = GridND(
        RandomUniformGrid1D(dist), RandomUniformGrid1D(dist)
    )
    quadrature = TrapezoidalQuadrature([1])
    w_parameterization = WeightsParameterization(
        w_func, grid_2d, quadrature
    )
    torch.nn.utils.parametrize.register_parametrization(
        conv, "weight", w_parameterization, unsafe=True
    )

    grid_2d = GridND(
        RandomUniformGrid1D(UniformDistribution(16, 16)),
        RandomUniformGrid1D(UniformDistribution(1, 1))
    )
    b_func = InterpolationWeights1D(16, None)
    bias = b_func([torch.linspace(-1, 1, 13), torch.linspace(0, 1, 1)])
    print(bias.shape)
    b_parameterization = WeightsParameterization(
        b_func, grid_2d, None
    )
    torch.nn.utils.parametrize.register_parametrization(
        conv, "bias", b_parameterization, unsafe=True
    )

    for key, param in conv.parametrizations.items():
        print(key, param)
    print(conv(torch.rand(1, 16, 28, 28)).shape)

# class MLPWeights(IWeights):
#     def __init__(self, plane_size, discrete_shape,
#                  layer_sizes, activation='prelu'):
# 
#         super(MLPWeights, self).__init__(discrete_shape)
#         planes_num = reduce(lambda a,b: a*b, discrete_shape)
#         activation = {
#             'prelu': torch.nn.PReLU,
#             'relu' : torch.nn.ReLU,
#             # 'sin' : Sin
#         }[activation]
# 
#         layer_sizes = layer_sizes + [planes_num]
#         net = [torch.nn.Linear(len(plane_size), layer_sizes[0])]
# 
#         for i in range(1, len(layer_sizes)):
#             net.append(activation())
#             net.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
# 
#         self.net = torch.nn.Sequential(net)
# 
# 
#     def forward(self, grid):
#         inp = grid.get_grid()
#         inp = inp.view(-1, len(grid.shape))
#         out = self.net(inp)
#         out = out.view(*grid.shape, *self._discrete_shape)
# 
#         return out
