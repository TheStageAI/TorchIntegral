import torch
from py2opt.routefinder import RouteFinder
from py2opt.solver import Solver


class BasePermutation:
    def __init__(self):
        pass

    def permute(self, tensors, size):
        permutation = self.find_permutation(tensors, size)

        for t in tensors:
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

    def find_permutation(self, tensors, size):
        raise NotImplementedError(
            "Implement this method in derived class."
        )

    def dist_function(self, x, y):
        raise NotImplementedError(
            "Implement this method in derived class."
        )


class RandomPermutation(BasePermutation):
    def find_permutation(self, tensors, size):
        return torch.randperm(size, device=tensors[0]['value'].device)


class NOptPermutation(BasePermutation):
    def __init__(self, iters=100, verbose=True):
        super(NOptPermutation, self).__init__()
        self.iters = iters
        self.verbose = verbose

    def find_permutation(self, tensors, size):
        cities_names = [i for i in range(size)]
        dist_mat = self.distance_matrix(tensors, size)
        path_distance = Solver.calculate_path_dist(
            dist_mat, torch.arange(size)
        )
        route_finder = RouteFinder(
            dist_mat, cities_names, iterations=self.iters
        )
        best_distance, indices = route_finder.solve()

        if self.verbose:
            print(f'variation before permutation:', path_distance)
            print(f'variation after permutation:', best_distance)

        device = tensors[0]['value'].device
        indices = torch.tensor(indices).to(device)

        return indices

    def dist_function(self, x, y):
        """
        """
        return (x - y).abs().mean()

    def distance_matrix(self, tensors, size):
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
                    dist += float(self.dist_function(x_i, x_j))

                mat[i].append(dist)

        return mat
