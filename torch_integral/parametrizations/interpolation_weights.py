import torch
from functools import reduce
import torch.nn.functional as F
from .base_parametrization import IWeights


class GridSampleWeightsBase(IWeights):
    """
    Base class for parametrization based on torch.nn.functional.grid_sample.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        Same modes as in torch.nn.functional.grid_sample.
    padding_mode: str.
    align_corners: bool.
    """

    def __init__(
        self,
        grid,
        quadrature,
        cont_size,
        discrete_shape=None,
        interpolate_mode="bicubic",
        padding_mode="border",
        align_corners=True,
    ):
        super(GridSampleWeightsBase, self).__init__(grid, quadrature)
        self.iterpolate_mode = interpolate_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.cont_size = cont_size
        self.discrete_shape = discrete_shape

        if discrete_shape is not None:
            self.planes_num = int(reduce(lambda a, b: a * b, discrete_shape))
        else:
            self.planes_num = 1

    def _postprocess_output(self, out):
        """ """
        raise NotImplementedError("Implement this method in derived class.")

    def evaluate_function(self, grid, weight):
        """
        Performs forward pass

        Parameters
        ----------
        grid: List[torch.Tensor].
            List of discretization grids along each dimension.

        Returns
        -------
        torch.Tensor.
            Sampled ``self.values`` on grid.

        """
        for i in range(len(grid)):
            grid[i] = grid[i].to(weight.device)

        if len(grid) == 1:
            grid.append(torch.tensor(0.0, device=weight.device))

        grid = torch.stack(torch.meshgrid(grid[::-1], indexing="ij"), dim=-1)
        grid = grid.unsqueeze(0)
        out = F.grid_sample(
            weight,
            grid,
            mode=self.iterpolate_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return self._postprocess_output(out)


class GridSampleWeights1D(GridSampleWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with one continuous dimension.

    Parameters
    ----------
    cont_size: int.
        Size of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    cont_dim: int.
        Index of continuous dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def __init__(
        self,
        grid,
        quadrature,
        cont_size,
        discrete_shape=None,
        cont_dim=0,
        interpolate_mode="bicubic",
        padding_mode="border",
        align_corners=True,
    ):
        super(GridSampleWeights1D, self).__init__(
            grid,
            quadrature,
            (cont_size, 1),
            discrete_shape,
            interpolate_mode,
            padding_mode,
            align_corners,
        )
        self.cont_dim = cont_dim

    def init_values(self, x):
        """ """
        weight = x

        if x.ndim == 1:
            x = x[None, None, :, None]
        else:
            permutation = [i for i in range(x.ndim) if i != self.cont_dim]
            x = x.permute(*permutation, self.cont_dim)
            x = x.reshape(1, -1, x.shape[-1], 1)

        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(
                x, self.cont_size, mode=self.iterpolate_mode
            ).contiguous()

        return weight

    def _postprocess_output(self, out):
        """ """
        discrete_shape = self.discrete_shape

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


class GridSampleWeights2D(GridSampleWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with two continuous dimensions.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def init_values(self, x):
        """ """
        weight = x

        if x.ndim == 2:
            x = x[None, None, :, :]
        else:
            permutation = list(range(2, x.ndim))
            shape = x.shape[:2]
            x = x.permute(*permutation, 0, 1)
            x = x.reshape(1, -1, *shape)

        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(
                x, self.cont_size, mode=self.iterpolate_mode
            ).contiguous()

        return weight

    def _postprocess_output(self, out):
        discrete_shape = self.discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        shape = out.shape[-2:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 2)
        out = out.permute(out.ndim - 1, out.ndim - 2, *dims)

        return out.contiguous()


def get_kernel_params(downsample_factor_x, downsample_factor_y):
    base_kernel_size = 3
    base_sigma = 1. / 3.

    if downsample_factor_x > 1:
        kernel_size_x = base_kernel_size * downsample_factor_x
        kernel_size_x = (
            int(kernel_size_x) + 1 if kernel_size_x % 2 == 0 else int(kernel_size_x)
        )
    else:
        kernel_size_x = 1

    if downsample_factor_y > 1:
        kernel_size_y = base_kernel_size * downsample_factor_y
        kernel_size_y = (
            int(kernel_size_y) + 1 if kernel_size_y % 2 == 0 else int(kernel_size_y)
        )
    else:
        kernel_size_y = 1
    
    sigma_x = base_sigma
    sigma_y = base_sigma

    return kernel_size_x, kernel_size_y, sigma_x, sigma_y


def gaussian_kernel_2d(size_x: int, size_y: int, sigma_x: float, sigma_y: float):
    coords_x = torch.linspace(-1, 1, size_x)
    coords_y = torch.linspace(-1, 1, size_y)
    g_x = torch.exp(-(coords_x**2) / (2 * sigma_x**2))
    g_y = torch.exp(-(coords_y**2) / (2 * sigma_y**2))
    g_x /= g_x.sum()
    g_y /= g_y.sum()
    g = g_y[:, None] * g_x[None, :]

    return g


def filter_image(image: torch.Tensor, kernel: torch.Tensor):
    kernel = kernel[None, None, ...]
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)
    padding_x = kernel.shape[-1] // 2
    padding_y = kernel.shape[-2] // 2
    out = F.conv2d(
        image,
        kernel,
        bias=None,
        padding=(padding_y, padding_x),
        groups=image.shape[1],
    )

    return out


def antialiasing_filter(shape, weight):
    """ """
    downsample_factor_x = weight.shape[-1] // shape[-1]
    downsample_factor_y = weight.shape[-2] // shape[-2]

    if downsample_factor_x > 1 or downsample_factor_y > 1:
        kernel_size_x, kernel_size_y, sigma_x, sigma_y = get_kernel_params(
            downsample_factor_x, downsample_factor_y
        )
        kernel = gaussian_kernel_2d(
            kernel_size_x, kernel_size_y, sigma_x, sigma_y
        ).to(weight.device)
        weight = filter_image(weight, kernel)

    return weight
    

class InterpolationWeightsBase(IWeights):
    """
    Base class for parametrization based on torch.nn.functional.grid_sample.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        Same modes as in torch.nn.functional.grid_sample.
    padding_mode: str.
    align_corners: bool.
    """

    def __init__(
        self,
        grid,
        quadrature,
        cont_size,
        discrete_shape=None,
        interpolate_mode="bicubic",
        padding_mode="border",
        align_corners=True,
    ):
        super(InterpolationWeightsBase, self).__init__(grid, quadrature)
        self.iterpolate_mode = interpolate_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.cont_size = cont_size
        self.discrete_shape = discrete_shape

        if discrete_shape is not None:
            self.planes_num = int(reduce(lambda a, b: a * b, discrete_shape))
        else:
            self.planes_num = 1

    def _postprocess_output(self, out):
        """ """
        raise NotImplementedError("Implement this method in derived class.")

    def evaluate_function(self, grid, weight):
        """
        Performs forward pass

        Parameters
        ----------
        grid: List[torch.Tensor].
            List of discretization grids along each dimension.

        Returns
        -------
        torch.Tensor.
            Sampled ``self.values`` on grid.

        """
        shape = [g.shape[0] for g in grid]

        if len(shape) == 1:
            shape.append(1)

        # weight = antialiasing_filter(shape, weight)

        out = F.interpolate(
            weight,
            size=shape,
            mode=self.iterpolate_mode,
        )

        return self._postprocess_output(out)


class InterpolationWeights1D(InterpolationWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with one continuous dimension.

    Parameters
    ----------
    cont_size: int.
        Size of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    cont_dim: int.
        Index of continuous dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def __init__(
        self,
        grid,
        quadrature,
        cont_size,
        discrete_shape=None,
        cont_dim=0,
        interpolate_mode="bicubic",
        padding_mode="border",
        align_corners=True,
    ):
        super(InterpolationWeights1D, self).__init__(
            grid,
            quadrature,
            (cont_size, 1),
            discrete_shape,
            interpolate_mode,
            padding_mode,
            align_corners,
        )
        self.cont_dim = cont_dim

    def init_values(self, x):
        """ """
        weight = x

        if x.ndim == 1:
            x = x[None, None, :, None]
        else:
            permutation = [i for i in range(x.ndim) if i != self.cont_dim]
            x = x.permute(*permutation, self.cont_dim)
            x = x.reshape(1, -1, x.shape[-1], 1)

        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(
                x, self.cont_size, mode=self.iterpolate_mode
            ).contiguous()

        return weight

    def _postprocess_output(self, out):
        """ """
        discrete_shape = self.discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        out = out.view(*discrete_shape, out.shape[-2])
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
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with two continuous dimensions.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def init_values(self, x):
        """ """
        weight = x

        if x.ndim == 2:
            x = x[None, None, :, :]
        else:
            permutation = list(range(2, x.ndim))
            shape = x.shape[:2]
            x = x.permute(*permutation, 0, 1)
            x = x.reshape(1, -1, *shape)

        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(
                x, self.cont_size, mode=self.iterpolate_mode
            ).contiguous()

        return weight

    def _postprocess_output(self, out):
        discrete_shape = self.discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        shape = out.shape[-2:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 2)
        out = out.permute(out.ndim - 2, out.ndim - 1, *dims)

        return out.contiguous()
