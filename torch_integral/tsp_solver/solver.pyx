import torch
import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
cnp.import_array()


#----------------------------------------------------------------------------------------
# Simple symmetric matrix implementation
#----------------------------------------------------------------------------------------
cdef class SymmMatrix:
    """
    Cython class to keep float symmetric matrix.

    Parameters
    ----------
    n: size_t. Size of matrix.
    """
    cdef size_t _size
    cdef cnp.ndarray _matrix
#----------------------------------------------------------------------------------------
    def __init__(self, size_t n):

        self._size = n
        self._matrix = np.zeros( ((n + 1)*n//2, ), dtype=np.float32 )


    cpdef size_t size(self):
        """
        """
        return self._size


    cpdef cnp.float32_t get_item(self, size_t raw, size_t col):

        if col > raw:
            return self.get_item(col, raw)

        return self._matrix[(raw + 1)*raw//2 + col]


    cpdef cnp.float32_t set_item(self, size_t raw, size_t col, cnp.float32_t value):

        if col > raw:
            return self.set_item(col, raw, value)

        self._matrix[(raw + 1)*raw//2 + col] = value


#----------------------------------------------------------------------------------------
# 2opt Solver: TSP return structure
#----------------------------------------------------------------------------------------
cdef class TSPResult:
    """
    """
    cdef cnp.float32_t distance
    cdef cnp.ndarray route


    def __cinit__(self, best_route, best_distance):

        self.route = best_route
        self.distance = best_distance


# TSP Solver: 2opt
#----------------------------------------------------------------------------------------
cpdef TSPResult two_opt(
    SymmMatrix distance_matrix, cnp.ndarray initial_route, cnp.float32_t threshold=0.01
):
    """
    Performs 2opt search from the given initial path.

    Parameters
    ----------
    distance_matrix: SymmMatrix.
    initial_route: np.array.
    threshold: float. Relative threshold.
    """
    cdef size_t swap_first, swap_last
    cdef size_t before_start, start
    cdef size_t end, after_end
    cdef cnp.float32_t previous_best
    cdef cnp.float32_t before, after
    cdef cnp.float32_t improvement_factor = 1.0
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] best_route = initial_route
    cdef cnp.float32_t best_distance = _calculate_path_dist(
        distance_matrix, best_route
    )
    
    while improvement_factor > threshold:
        previous_best = best_distance
        
        for swap_first in range(1, distance_matrix.size() - 2):
            for swap_last in range(swap_first + 1, distance_matrix.size() - 1):
                before_start = best_route[swap_first - 1]
                start = best_route[swap_first]
                end = best_route[swap_last]
                after_end = best_route[swap_last+1]

                before = distance_matrix.get_item(before_start, start)\
                    + distance_matrix.get_item(before_start, start)
                after = distance_matrix.get_item(before_start, end)\
                    + distance_matrix.get_item(start, after_end)

                if after < before:
                    best_route = _swap(best_route, swap_first, swap_last)
                    best_distance = _calculate_path_dist(distance_matrix, best_route)

        improvement_factor = 1.0 - best_distance / (previous_best + 1e-8)

    return TSPResult(best_route, best_distance)


cdef cnp.float32_t _calculate_path_dist(
    SymmMatrix distance_matrix, cnp.ndarray path
):

    cdef size_t i
    cdef cnp.float32_t path_distance = 0.0

    for i in range(path.shape[0] - 1):
        path_distance += distance_matrix.get_item(path[i], path[i + 1])

    return path_distance


cdef cnp.ndarray _swap(
    cnp.ndarray path, cnp.uint32_t swap_first, cnp.uint32_t swap_last
):

    return np.concatenate((
        path[0:swap_first], 
        path[swap_last:- path.shape[0] + swap_first - 1:-1],
        path[swap_last + 1:path.shape[0]]
    ))


#----------------------------------------------------------------------------------------
# 2opt BruteForce with L1 metric
#----------------------------------------------------------------------------------------
cpdef SymmMatrix get_distance_matrix(list tensors, size_t size):
    """
    Computes L1 distance matrix for group of connected tensors.

    Parameters
    ----------
    tensors: list.
    size: size_t.
    """
    cdef size_t i
    cdef size_t j
    cdef size_t k
    cdef cnp.float32_t new_distance
    cdef cnp.ndarray[cnp.float32_t, ndim=2] tensor_np
    cdef SymmMatrix dist_mat = SymmMatrix(size)

    for k in range( len(tensors) ):
        tensor = tensors[k]
        tensor = tensor['value'].transpose(0, tensor['dim'])
        tensor_np = tensor.cpu().reshape(
            tensor.shape[0], -1
        ).detach().numpy()

        for i in range(size):
            for j in range(i + 1):
                new_distance = np.mean( 
                    np.abs(tensor_np[i] - tensor_np[j])
                )
                dist_mat.set_item(
                    i, j, dist_mat.get_item(i, j) + new_distance
                )

    return dist_mat


cpdef tuple brute_force_2opt(
    SymmMatrix distance_matrix, size_t iterations, cnp.float32_t threshold=0.01
):
    """
    Runs 2opt TSP solver multiple times and selects best route.

    Parameters
    ----------
    distance_matrix: SymmMatrix.
    iterations: size_t.

    Returns
    -------
    Tuple containing best route and route length.
    """
    cdef size_t i
    cdef cnp.float32_t best_distance = np.inf
    cdef cnp.ndarray best_route
    cdef cnp.ndarray initial_route
    cdef TSPResult result
    cdef size_t num_cities = distance_matrix.size()

    for i in range(iterations):
        initial_route = np.random.permutation(num_cities).astype(np.uint32)
        result = two_opt(distance_matrix, initial_route, threshold)

        if result.distance < best_distance:
            best_distance = result.distance
            best_route = result.route

    return best_distance, best_route


#----------------------------------------------------------------------------------------
# 2opt BruteForce for INN interface
#----------------------------------------------------------------------------------------
cpdef object two_opt_find_permutation(
    object tensors, size_t size, size_t iters, cnp.float32_t threshold=0.01
):
    """
    Finds permutation using 2opt for TSP.

    Parameters 
    ----------
    tesnors: list[dict{'value': torch.Tensor, 'dim': size_t}].
    iters: size_t.
    threshold: float.
    """
    cdef vector[size_t] cities_names = [i for i in range(size)]
    cdef SymmMatrix dist_mat = get_distance_matrix(tensors, size)
    cdef cnp.float32_t path_distance = _calculate_path_dist(
        dist_mat, np.arange(0, size, 1, dtype=np.uint32)
    )
    print(f'variation before permutation:', path_distance)
    best_distance, indices = brute_force_2opt(dist_mat, iters, threshold)
    print(f'variation after permutation:', best_distance)
    device = tensors[0]['value'].device
    indices = torch.tensor(indices.astype(np.int32)).to(device)

    return indices
