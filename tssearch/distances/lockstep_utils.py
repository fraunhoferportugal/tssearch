import numpy as np
from numba import njit, prange, double


@njit(parallel=True, fastmath=True)
def _lnorm_multidimensional(x, y, weight, p=2):

    l1 = x.shape[0]
    l3 = x.shape[1]

    distance = np.zeros_like(x, dtype=float)
    for i in prange(l1):
        dist = 0.0
        for di in range(l3):
            diff = x[i, di] - y[i, di]
            dist += weight[i, di] * (diff ** p)
        distance[i] = dist ** (1 / p)

    return distance


def _lnorm_unidimensional(x, y, weight, p=2):

    distance = (weight * ((x - y) ** p)) ** (1 / p)

    return distance
