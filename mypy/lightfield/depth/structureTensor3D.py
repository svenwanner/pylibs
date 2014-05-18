import vigra
import numpy as np
from numpy import linalg as LA
from scipy import linalg
from mypy.visualization.imshow import imoverlay, imshow


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
            # mat[0, 0] = st[lf.shape[0] / 2, y, x, 0]
            # mat[0, 1] = st[lf.shape[0] / 2, y, x, 1]
            # mat[0, 2] = st[lf.shape[0] / 2, y, x, 2]
            # mat[1, 0] = st[lf.shape[0] / 2, y, x, 1]
            # mat[1, 1] = st[lf.shape[0] / 2, y, x, 3]
            # mat[1, 2] = st[lf.shape[0] / 2, y, x, 4]
            # mat[2, 0] = st[lf.shape[0] / 2, y, x, 3]
            # mat[2, 1] = st[lf.shape[0] / 2, y, x, 4]
            # mat[2, 2] = st[lf.shape[0] / 2, y, x, 5]
            mat[0,0] = st[lf.shape[0] / 2, y, x, 0]
            mat[1,0] = st[lf.shape[0] / 2, y, x, 1]
            mat[2,0] = st[lf.shape[0] / 2, y, x, 2]
            mat[0,1] = st[lf.shape[0] / 2, y, x, 1]
            mat[1,1] = st[lf.shape[0] / 2, y, x, 3]
            mat[2,1] = st[lf.shape[0] / 2, y, x, 4]
            mat[0,2] = st[lf.shape[0] / 2, y, x, 3]
            mat[1,2] = st[lf.shape[0] / 2, y, x, 4]
            mat[2,2] = st[lf.shape[0] / 2, y, x, 5]

            tmp_evals, tmp_evecs = linalg.eigh(mat)
            #rearanged = sorted(zip(tmp_evals, tmp_evecs.T), key=lambda v: v[0].real, reverse=True)

            evals[y, x, :] = tmp_evals[:]
            evecs[y, x, :, :] = tmp_evecs[:, :]

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
    coherence = np.zeros((evals.shape[0], evals.shape[1]))

    mean_vec1 = np.zeros((3,3))
    mean_eval1 = np.zeros((3))
    mean_vec2 = np.zeros((3,3))
    mean_eval2 = np.zeros((3))

    region = np.zeros((evals.shape[0], evals.shape[1]))

    for y in xrange(evals.shape[0]):
        for x in xrange(evals.shape[1]):

            if evecs[y, x, 0, 0] < 0:
                evecs[y, x, 0, 0] *= -1

            if x > 10 and x < 21 and y > 10 and y < 21:
                region[y,x] = 1
                for i in range(3):
                    for j in range(3):
                        mean_vec1[i,j] += evecs[y, x, i, j]
                    mean_eval1[i] += evals[y,x,i]

            if x > 74 and x < 85 and y > 10 and y < 21:
                region[y,x] = 1
                for i in range(3):
                    for j in range(3):
                        mean_vec2[i,j] += evecs[y, x, i, j]
                    mean_eval2[i] += evals[y,x,i]

            if x == 20 and y == 20:
                mean_vec1[:]/=10.0
                mean_eval1[:]/=10.0
                print "mean_vec1",mean_vec1
                print "mean_val1",mean_eval1
                #print "eval 1",evals[y,x,0] ," evec 1: ", evecs[y, x, 0, 0],",",evecs[y, x, 1, 0],",",evecs[y, x, 2, 0]
                # print "eval 2",evals[y,x,1] ," evec 2: ", evecs[y, x, 0, 1],",",evecs[y, x, 1, 1],",",evecs[y, x, 2, 1]
                # print "eval 3",evals[y,x,2] ," evec 3: ", evecs[y, x, 0, 2],",",evecs[y, x, 1, 2],",",evecs[y, x, 2, 2]

            if x == 84 and y == 20:
                mean_vec2[:]/=10.0
                mean_eval2[:]/=10.0
                print "\nmean_vec2",mean_vec2
                print "mean_val2",mean_eval2


            disparity[y, x] = evecs[y, x, 0, 2]/evecs[y, x, 0, 1]
            if disparity[y, x] > 0.4 or disparity[y, x] < 0.26:
                disparity[y, x] = -1
            else:
                disparity[y, x] = 0




            # axis = np.array([evals[y, x, 0], evals[y, x, 1], evals[y, x, 2]])
            # stype, coh, order = shape_estimator(axis)
            #
            # coherence[y, x] = coh
            #
            # if (stype == 0):
            #     disparity[y, x] = 0
            # elif (stype == 1):
            #     vec = evecs[x, y, :, order[0]]
            #     #Todo: must be adapted if vertical lf is input vec[2] should then be vec[1]
            #     disparity[y, x] = np.arctan2(vec[2], vec[0])
            #     if disparity[y, x] >= np.pi/2.0:
            #         disparity[y, x] -= np.pi/2.0
            # elif (stype == 2):
            #     vec = evecs[x, y, :, order[2]]
            #     #Todo: must be adapted if vertical lf is input vec[2] should then be vec[1]
            #     disparity[y, x] = np.arctan2(vec[2], vec[0])
            #     disparity[y, x] -= np.pi/2.0
            #     if disparity[y, x] >= np.pi/2.0:
            #         disparity[y, x] -= np.pi/2.0
            # elif (stype == 3):
            #     disparity[y, x] = 0

    imshow(region)
    return disparity, coherence


def shape_estimator(axis):
    """
    checks the 3D structure tensor conditions, by deciding if the
    eigenvalues describe a sphere (0), a cigar (1) or a disc shape (2).
    returns the type found and the order of the eigenvalues from small to big.

    :param axis: ndarray of three eigenvalues
    :return type, order: int defining the type sphere (0), a disc (1) or a cigar shape (2),
                         ndarray containing the order of the eigenvalue sizes from small to big
    """

    assert isinstance(axis, np.ndarray)

    formfac1 = 0.85
    formfac2 = 0.3

    order = axis.argsort()

    if axis[order[0]] < 1e-6:
        axis[order[0]] = 1e-6
    if axis[order[1]] < 1e-6:
        axis[order[1]] = 1e-6

    mid = axis[order[1]] / axis[order[2]]
    low = axis[order[0]] / axis[order[2]]

    if mid >= formfac1 <= low:   #case sphere
        return 0, 0.0, order
    elif mid >= formfac1 > low:  #case disc
        return 1, 1.0-low, order
    elif mid < formfac1 > low:   #case cigar
        return 2, 1.0-(low+mid), order
    else:
        print "unintended case happend, check the shape check properties!"
        return 3, 0.0, order