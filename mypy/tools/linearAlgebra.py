import numpy as np

def vec_length(vec):
    return np.sqrt(np.sum(vec[:]**2))

def normalize_vec(vec):
    vec[:] /= np.sqrt(np.sum(vec[:]**2))
    return vec
