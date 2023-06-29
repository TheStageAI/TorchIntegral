import torch
from .tsp_solver import two_opt_find_permutation


def total_variance(tensors):
    """
    Calculates total variation of tensors along given dimension.

    Parameters
    ----------
    tensors: List[Dict[str, obj]].
        List of dicts with keys 'value' and 'dim'.

    Returns
    -------
    total_var: float.
        Estimated total variation.
    """
    total_var = 0.0

    for t in tensors:
        tensor = t["value"]
        dim = t["dim"]
        tensor = tensor.transpose(dim, 0)
        diff = (tensor[1:] - tensor[:-1]).abs().mean()
        total_var = total_var + diff

    return total_var


class BasePermutation:
    """Base class for tensors permutaiton."""

    def __call__(self, params, feature_maps, size):
        """
        Performs permutation of weight tensors along given dimension.

        Parameters
        ----------
        params: List[Dict[str, obj]].
            List of dicts with keys 'value', 'dim', 'name'.
            Value is a parameter tensor.
        feature_maps: List[Dict[str, obj]].
            List of dicts with keys 'value', 'dim', 'name'.
            Value is a feature map tensor.
        size: int.
            Size of tensor dimension along which permutation should be performed.
        """
        permutation = self.find_permutation(params, feature_maps, size)

        for t in params:
            dim = t["dim"]
            tensor = t["value"]

            if "start_index" not in t:
                start = 0
            else:
                start = t["start_index"]

            permuted = torch.index_select(tensor, dim, permutation + start)
            tensor.data = torch.slice_scatter(
                tensor, permuted, dim, start, start + size
            )

    def find_permutation(self, params, feature_maps, size):
        """Method should return list of indices."""
        raise NotImplementedError("Implement this method in derived class.")


class RandomPermutation(BasePermutation):
    def find_permutation(self, params, feature_maps, size):
        """Returns random permutation of given size."""
        return torch.randperm(size, device=params[0]["value"].device)


class NOptPermutation(BasePermutation):
    """
    Class for total variation optimization using py2opt algorithm.

    Parameters
    ----------
    iters: int.
    threshold: float.
    verbose: bool.
    """

    def __init__(self, iters=100, threshold=0.001, verbose=True):
        super(NOptPermutation, self).__init__()
        self.iters = iters
        self.verbose = verbose
        self.threshold = threshold

    def find_permutation(self, params, feature_maps, size):
        """Uses py2opt algorithm to find permutation of given tensors."""
        optimize_tensors = self._select_tensors(params, feature_maps)
        indices = two_opt_find_permutation(
            optimize_tensors, size, self.iters, self.threshold
        )
        device = params[0]["value"].device
        indices = indices.type(torch.long).to(device)

        return indices

    def _select_tensors(self, params, feature_maps):
        """Returns list of tensors which total variation should be optimized."""
        return params


class NOptOutFiltersPermutation(NOptPermutation):
    """
    Class implements NOptPermutation
    interface for optimzation of out filters total variation.
    """

    def __init__(self, iters=100, verbose=True):
        super(NOptOutFiltersPermutation, self).__init__(iters, verbose)

    def _select_tensors(self, params, feature_maps):
        tensors = [t for t in params if "bias" not in t["name"] and t["dim"] == 0]

        if len(tensors) == 0:
            tensors = params

        return tensors


class NOoptFeatureMapPermutation(NOptPermutation):
    """
    Class implements NOptPermutation interface
    for optimzation of feature maps total variation.
    """

    def _select_tensors(self, params, feature_maps):
        """ """
        out = []

        for f in feature_maps:
            if f["operation"] == "conv_linear":
                out.append(f)

        if len(out) == 0:
            out = feature_maps

        return out
