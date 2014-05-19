from __future__ import division

import vigra
import numpy as np

from mypy.visualization.imshow import imshow

def structureTensor2D(lf3d, inner_scale=0.6, outer_scale=1.3, direction='h'):

    st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)

    if direction == 'h':
        for y in xrange(lf3d.shape[1]):
            epis = []
            for c in range(lf3d.shape[3]):
                epis.append(lf3d[:, y, :, c])
            for n, epi in enumerate(epis):
                epis[n] = vigra.filters.structureTensor(epi, inner_scale, outer_scale)
            st = np.zeros_like(epis[0])
            for epi in epis:
                st[:, :, :] += epi[:, :, :]

            st3d[:, y, :, :] = st[:]

    elif direction == 'v':
        for x in xrange(lf3d.shape[2]):
            epis = []
            for c in range(lf3d.shape[3]):
                epis.append(lf3d[:, :, x, c])

            for n, epi in enumerate(epis):
                epis[n] = vigra.filters.structureTensor(epi, inner_scale, outer_scale)
            st = np.zeros_like(epis[0])
            for epi in epis:
                st[:, :, :] += epi[:, :, :]

            st3d[:, :, x, :] = st[:]

    else:
        assert False, "unknown lightfield direction!"

    c = lf3d.shape[3]
    st3d[:] /= c

    return st3d


def evaluateStructureTensor(tensor):
    assert isinstance(tensor, np.ndarray)
    coherence = np.sqrt((tensor[:, :, :, 2]-tensor[:, :, :, 0])**2+2*tensor[:, :, :, 1]**2)/(tensor[:, :, :, 2]+tensor[:, :, :, 0] + 1e-16)
    orientation = 1/2.0*vigra.numpy.arctan2(2*tensor[:, :, :, 1], tensor[:, :, :, 2]-tensor[:, :, :, 0])
    orientation = vigra.numpy.tan(orientation[:])
    invalid_ubounds = np.where(orientation > 1)
    invalid_lbounds = np.where(orientation < -1)
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    orientation[invalid_ubounds] = -1
    orientation[invalid_lbounds] = -1
    return orientation, coherence


def preDerivation(lf3d, scale=0.1):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    for i in xrange(lf3d.shape[0]):
        for c in xrange(lf3d.shape[3]):
            grad = vigra.filters.gaussianGradient(lf3d[i, :, :, c], scale)
            tmp = vigra.colors.linearRangeMapping(grad[:, :, 0], newRange=(0.0, 1.0))
            lf3d[i, :, :, c] = tmp

    return lf3d

def epiPreDerivation(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    if direction == 'h':
        for y in xrange(lf3d.shape[1]):
            for c in xrange(lf3d.shape[3]):
                grad = vigra.filters.gaussianGradient(lf3d[:, y, :, c], scale)
                try:
                    tmp = vigra.colors.linearRangeMapping(grad[:, :, 0], newRange=(0.0, 1.0))
                except:
                    tmp = grad[:, :, 0]
                lf3d[:, y, :, c] = tmp[:]

    elif direction == 'v':
        for x in xrange(lf3d.shape[2]):
            for c in xrange(lf3d.shape[3]):
                grad = vigra.filters.gaussianGradient(lf3d[:, :, x, c], scale)
                try:
                    tmp = vigra.colors.linearRangeMapping(grad[:, :, 0], newRange=(0.0, 1.0))
                except:
                    tmp = grad[:, :, 0]
                lf3d[:, :, x, c] = tmp[:]
    else:
        assert False, "unknown lightfield direction!"

    return lf3d


def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):

    labels = np.zeros_like(orientation1)
    orientation = np.zeros_like(orientation1)
    coherence = np.zeros_like(orientation1)
    for n in xrange(orientation.shape[0]):
        for y in xrange(orientation.shape[1]):
            for x in xrange(orientation.shape[2]):
                if coherence1[n, y, x] >= coherence2[n, y, x]:
                    orientation[n, y, x] = orientation1[n, y, x]
                    coherence[n, y, x] = coherence1[n, y, x]
                else:
                    orientation[n, y, x] = orientation2[n, y, x]
                    coherence[n, y, x] = coherence2[n, y, x]
                    labels[n, y, x] = 255
    # labels = np.zeros_like(orientation1, dtype=np.uint8)
    # winner = np.where(coherence2 > coherence1)
    # orientation1[winner] = orientation2[winner]
    # coherence1[winner] = coherence2[winner]
    # labels[winner] = 1
    return orientation, coherence, labels