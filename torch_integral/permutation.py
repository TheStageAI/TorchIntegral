import torch
from py2opt.routefinder import RouteFinder
from py2opt.solver import Solver


def l1_dist_function(x, y):
    """
    """
    return (x - y).abs().sum()


def distance_matrix(tensors, size, dist_fn=l1_dist_function):
    """
    """
    mat = []

    for i in range(size):
        n = list()
        mat.append(n)

        for j in range(size):
            dist = 0.0

            for t in tensors:
                tensor = t['value']
                dim = t['dim']
                x_i = torch.select(tensor, dim, i)
                x_j = torch.select(tensor, dim, j)
                dist += float(dist_fn(x_i, x_j))

            mat[i].append(dist)

    return mat


def total_variance(tensors):
    total_var = 0.

    for t in tensors:
        tensor = t['value']
        dim = t['dim']
        tensor = tensor.transpose(dim, 0)
        diff = (tensor[1:] - tensor[:-1]).abs().sum()
        total_var = total_var + diff

    return total_var


class BasePermutation:
    def __init__(self):
        pass

    def __call__(self, params, feature_maps, size):
        permutation = self.find_permutation(params, feature_maps, size)

        for t in params:
            dim = t['dim']
            tensor = t['value']

            if 'start_index' not in t:
                start = 0
            else:
                start = t['start_index']

            permuted = torch.index_select(
                tensor, dim, permutation + start
            )
            tensor.data = torch.slice_scatter(
                tensor, permuted, dim, start, start + size
            )

    def find_permutation(self, params, feature_maps, size):
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class RandomPermutation(BasePermutation):
    def find_permutation(self, params, feature_maps, size):
        return torch.randperm(size, device=params[0]['value'].device)


class NOptPermutation(BasePermutation):
    def __init__(self, iters=100, verbose=True):
        super(NOptPermutation, self).__init__()
        self.iters = iters
        self.verbose = verbose
        self.feature_maps = None
        self.params = None
        self.size = None

    def find_permutation(self, params, feature_maps, size):
        self.feature_maps = feature_maps
        self.params = params
        self.size = size
        cities_names = [i for i in range(size)]
        dist_mat = self.distance_matrix()
        path_distance = Solver.calculate_path_dist(
            dist_mat, torch.arange(size)
        )
        route_finder = RouteFinder(
            dist_mat, cities_names,
            iterations=self.iters, verbose=False
        )
        best_distance, indices = route_finder.solve()

        if self.verbose:
            print(f'variation before permutation:', path_distance)
            print(f'variation after permutation:', best_distance)

        device = params[0]['value'].device
        indices = torch.tensor(indices).to(device)

        return indices

    def distance_matrix(self):
        return distance_matrix(self.params, self.size)


class NOptOutFiltersPermutation(NOptPermutation):
    def __init__(self, iters=100, verbose=True):
        super(NOptOutFiltersPermutation, self).__init__(iters, verbose)

    def distance_matrix(self):
        tensors = [
            t for t in self.params
            if 'bias' not in t['name'] and t['dim'] == 0
        ]

        if len(tensors) == 0:
            tensors = self.params

        return distance_matrix(tensors, self.size)


class NOoptFeatureMapPermutation(NOptPermutation):
    def distance_matrix(self):
        return distance_matrix(self.feature_maps, self.size)


class VariationOptimizer:
    def __init__(self, num_iters=100,
                 learning_rate=1e-3,
                 verbose=True):

        self.num_iters = num_iters
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.eps = 1e-8

    def choose_tensors(self, tensors):
        out = [
            t for t in tensors if 'bias' not in t['name']
        ]

        if len(out) == 0:
            out = tensors

        return out

    def find_scale_vector(self, tensors, size):
        tensors = self.choose_tensors(tensors)
        device = tensors[0]['value'].device
        # scale_vect = 2. * (torch.rand(size) > 0.5) - 1.
        scale_vect = torch.ones(size)
        scale_vect = torch.nn.Parameter(scale_vect.to(device))
        opt = torch.optim.Adam(
            [scale_vect], lr=self.learning_rate, weight_decay=0.
        )

        if self.verbose:
            print('before scaling: ', total_variance(tensors))

        for i in range(self.num_iters):
            scaled_tensors = []

            for t in tensors:
                tensor = t['value']
                dim = t['dim']
                shape = [1] * tensor.ndim
                shape[dim] = scale_vect.shape[0]
                scale = scale_vect.view(shape)

                if dim == 0:
                    scaled_tensor = tensor * scale
                else:
                    scaled_tensor = tensor / scale

                scaled_tensors.append({
                    'dim': dim, 'value': scaled_tensor
                })

            total_var = total_variance(scaled_tensors)
            total_var.backward()
            opt.step()
            opt.zero_grad()

        if self.verbose:
            print('after scaling: ', total_variance(scaled_tensors))

        return scale_vect

    def __call__(self, tensors, size):
        scale_vector = self.find_scale_vector(tensors, size)

        for t in tensors:
            dim = t['dim']
            tensor = t['value']
            shape = [1] * tensor.ndim
            shape[dim] = scale_vector.shape[0]
            scale = scale_vector.view(shape)

            if dim == 0:
                tensor.data = tensor.data * scale
            else:
                tensor.data = tensor.data / scale
