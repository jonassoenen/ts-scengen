from numba import jit, float64
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

@jit(float64(float64[:], float64[:]), nogil = True, nopython = True)
def euc_dist_missing(a1, a2):
    return np.nanmean((a1-a2)**2)

euc_distance_matrix_missing = lambda x: pairwise_distances(x, metric = euc_dist_missing, force_all_finite = 'allow-nan')
euc_distance_matrix = lambda x: euclidean_distances(x)
