import torch
from functools import reduce
from torch.nn.functional import grid_sample, interpolate


class IWeights(torch.nn.Module):
    """
    Base weight parametrization class.

    Parameters
    ----------
    discrete_shape: List[int]. Sizes of parametrized tensor along discrete dimension.
    """
    # ---------------------------------------------------------------------------------
    def __init__(self, discrete_shape):
        super().__init__()
        self._discrete_shape = discrete_shape

    def init_values(self):
        """
        """
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def forward(self, grid):
        """
        Performs forward pass

        Parameters
        ----------
        grid: List[torch.Tensor].
        """
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class InterpolationWeightsBase(IWeights):
    """
    Base class for parametrization based on torch.nn.functional.grid_sample.

    Parameters
    ----------
    cont_size: List[int]. Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int]. Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str. Same modes as in torch.nn.functional.grid_sample.
    padding_mode: str.
    align_corners: bool.
    """
    # ---------------------------------------------------------------------------------

    def __init__(self, cont_size, discrete_shape=None,
                 interpolate_mode='bicubic',
                 padding_mode='border',
                 align_corners=True):

        super(InterpolationWeightsBase, self).__init__(discrete_shape)
        self.iterpolate_mode = interpolate_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        if discrete_shape is not None:
            self.planes_num = int(reduce(lambda a, b: a * b, discrete_shape))
        else:
            self.planes_num = 1

        self.values = torch.nn.Parameter(
            torch.rand(1, self.planes_num, *cont_size)
        )

    def _preprocess_grid(self, grid):
        """
        """
        device = self.values.device

        for i in range(len(grid)):
            grid[i] = grid[i].to(device)

        if len(grid) == 1:
            grid.append(torch.tensor(0., device=device))

        grid = torch.stack(
            torch.meshgrid(grid[::-1], indexing='ij'), dim=len(grid),
        ).unsqueeze(0)

        return grid

    def _postprocess_output(self, out):
        """
        """
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def forward(self, grid):
        """
        Performs forward pass

        Parameters
        ----------
        grid: List[torch.Tensor].
        """
        grid = self._preprocess_grid(grid)
        out = grid_sample(
            self.values, grid, mode=self.iterpolate_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )
        return self._postprocess_output(out)


class InterpolationWeights1D(InterpolationWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with one continuous dimension.

    Parameters
    ----------
    cont_size: List[int].
    discrete_shape: List[int].
    cont_dim: int.
    interpolate_mode: str.
    padding_mode: str.
    align_corners: bool.
    # ---------------------------------------------------------------------------------
    """
    def __init__(self, cont_size, discrete_shape=None,
                 cont_dim=0, interpolate_mode='bicubic',
                 padding_mode='border', align_corners=True):

        super(InterpolationWeights1D, self).__init__(
            [cont_size, 1], discrete_shape, interpolate_mode,
            padding_mode, align_corners
        )
        self.cont_dim = cont_dim

    def init_values(self, x):
        """
        """
        if x.ndim == 1:
            x = x[None, None, :, None]
        else:
            permutation = [
                i for i in range(x.ndim) if i != self.cont_dim
            ]
            x = x.permute(*permutation, self.cont_dim)
            x = x.reshape(1, -1, x.shape[-1], 1)

        if x.shape[-2:] == self.values.shape[-2:]:
            self.values.data = x
        else:
            self.values.data = interpolate(
                x, self.values.shape[-2:], mode=self.iterpolate_mode
            )

    def _postprocess_output(self, out):
        """
        """
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
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with two continuous dimensions.

    Parameters
    ----------
    cont_size: List[int].
    discrete_shape: List[int].
    cont_dim: int.
    interpolate_mode: str.
    padding_mode: str.
    align_corners: bool.
    # ---------------------------------------------------------------------------------
    """
    def init_values(self, x):
        """
        """
        if x.ndim == 2:
            x = x[None, None, :, :]
        else:
            permutation = list(range(2, x.ndim))
            shape = x.shape[:2]
            x = x.permute(*permutation, 0, 1)
            x = x.reshape(1, -1, *shape)

        if x.shape[-2:] == self.values.shape[-2:]:
            self.values.data = x
        else:
            self.values.data = interpolate(
                x, self.values.shape[-2:], mode=self.iterpolate_mode
            )

    def _postprocess_output(self, out):
        discrete_shape = self._discrete_shape

        if discrete_shape is None:
            discrete_shape = []

        shape = out.shape[-2:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 2)
        out = out.permute(out.ndim - 1, out.ndim - 2, *dims)

        return out.contiguous()
