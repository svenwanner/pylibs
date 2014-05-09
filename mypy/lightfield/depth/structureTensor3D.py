import vigra
import numpy as np
from numpy import linalg as LA


def structure_tensor3d(lf, inner_scale, outer_scale):
    """
    computes the 3D structure tensor an a 3D light field and returns the
    eigen values and the eigen vectors.

    :param lf: numpy array of structure [num_of_cams,height,width,channels]
    :param inner_scale: float inner scale of the structure tensor
    :param outer_scale: float outer scale of the structure tensor
    :return evals, evecs: numpy array of eigenvalues [height,width,3]
                          and eigenvectors [height,width,3,3]
    """

    assert isinstance(lf, np.ndarray)
    assert isinstance(inner_scale, float)
    assert isinstance(outer_scale, float)

    st = vigra.filters.structureTensor(lf, inner_scale, outer_scale)
    evals = np.zeros((lf.shape[1], lf.shape[2], 3), dtype=np.float32)
    evecs = np.zeros((lf.shape[1], lf.shape[2], 3, 3), dtype=np.float32)

    for y in xrange(lf.shape[1]):
        for x in xrange(lf.shape[2]):
            mat = np.zeros((3, 3), dtype=np.float64)
            mat[0, 0] = st[lf.shape[0] / 2, y, x, 0]
            mat[0, 1] = st[lf.shape[0] / 2, y, x, 1]
            mat[0, 2] = st[lf.shape[0] / 2, y, x, 2]
            mat[1, 0] = st[lf.shape[0] / 2, y, x, 1]
            mat[1, 1] = st[lf.shape[0] / 2, y, x, 3]
            mat[1, 2] = st[lf.shape[0] / 2, y, x, 4]
            mat[2, 0] = st[lf.shape[0] / 2, y, x, 3]
            mat[2, 1] = st[lf.shape[0] / 2, y, x, 4]
            mat[2, 2] = st[lf.shape[0] / 2, y, x, 5]

            evals[y, x, :], evecs[y, x, :, :] = LA.eigh(mat)

    return evals, evecs


def structure_tensor3d_conditioner(evals, evecs):
    """
    checks the 3D structure tensor conditions, by deciding if the
    eigenvalues describe a sphere (0), a disc (1) or a cigar shape (2).

    :param evals: ndarray of pixelwise eigenvalues [height,width,3]
    :param evecs: ndarray of pixelwise eigenvectors [height,width,3,3]
    :return disparity: ndarray center view disparity
    """

    assert isinstance(evals, np.ndarray)
    assert isinstance(evecs, np.ndarray)

    disparity = np.zeros((evals.shape[0], evals.shape[1]))
    for y in xrange(evals.shape[0]):
        for x in xrange(evals.shape[1]):
            axis = np.array([evals[y, x, 0], evals[y, x, 1], evals[y, x, 2]])
            stype, order = shapeEstimator(axis)

            #Todo handle eigenvectors depending stype and order
            if (stype == 0):
                disparity[y, x] = 0
            elif (stype == 1):
                disparity[y, x] = 1
            elif (stype == 2):
                disparity[y, x] = 2
            elif (stype == 3):
                disparity[y, x] = 3

    return disparity


def shapeEstimator(axis, dev=0.2):
    """
    checks the 3D structure tensor conditions, by deciding if the
    eigenvalues describe a sphere (0), a disc (1) or a cigar shape (2).
    returns the type found and the order of the eigenvalues from small to big.

    :param axis: ndarray of three eigenvalues
    :param dev: float describing the difference between the eigenvalue lengths
    as deviation from a sphere shape in percent
    :return type, order: int defining the type sphere (0), a disc (1) or a cigar shape (2),
                         ndarray containing the order of the eigenvalue sizes from small to big
    """

    assert isinstance(axis, np.ndarray)
    assert isinstance(dev, float)

    order = axis.argsort()
    axis[order[0]] = axis[order[0]] / axis[order[2]]
    axis[order[1]] = axis[order[1]] / axis[order[2]]
    axis[order[2]] = 1

    high = axis[order[2]]
    mid = axis[order[1]]
    low = axis[order[0]]

    if high - mid < dev and high - low < dev:
        return 0, order
    elif (high - mid < dev and high - low >= dev) or (high - mid >= dev and high - low < dev):
        return 1, order
    elif (high - mid >= dev and high - low >= dev):
        return 2, order
    else:
        return 3, order