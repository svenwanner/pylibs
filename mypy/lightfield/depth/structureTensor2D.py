from __future__ import division

import vigra
import numpy as np
import scipy.misc as misc

from mypy.visualization.imshow import imshow



class StructureTensor(object):

    def __init__(self):
        self.st = None

    def compute(self, lf3d, params):
        """
        call this method to compute the structure tensor subroutines in the order following:
        - pre_filter
        - inner_smoothing
        - derivations
        - outer_smoothing
        - post_filter
        - post_processing

        overwrite these methods in your structure tensor class derivative to use them. Each
        subroutine not overwritten is ignored
        """
        assert isinstance(lf3d, np.ndarray), "lf3d input is not a numpy array"
        assert len(lf3d.shape) == 4, "lf3d need 4 dimensions"
        assert isinstance(params, type({})), "params needs to be a dictionary"
        assert params.has_key("direction"), "missing parameter direction"

        self.st = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)

        if params["direction"] == 'h':
            for y in xrange(lf3d.shape[1]):
                epis = []
                e = lf3d[:, y, :, :]
                # misc.imsave("EPT_{0}.tiff".format(y),e)
                for c in range(lf3d.shape[3]):
                    epis.append(lf3d[:, y, :, c])
                for n, epi in enumerate(epis):
                    epis[n] = self.pre_filter(epi, params)
                    epis[n] = self.inner_smoothing(epis[n], params)
                    epis[n] = self.derivations(epis[n], params)
                    epis[n] = self.outer_smoothing(epis[n], params)
                    epis[n] = self.post_filter(epis[n], params)

                for epi in epis:
                    self.st[:, y, :, :] += epi[:, :, :]

        elif params["direction"] == 'v':
            for x in xrange(lf3d.shape[2]):
                epis = []
                for c in range(lf3d.shape[3]):
                    epis.append(lf3d[:, :, x, c])
                for n, epi in enumerate(epis):
                    epis[n] = self.pre_filter(epi, params)
                    epis[n] = self.inner_smoothing(epis[n], params)
                    epis[n] = self.derivations(epis[n], params)
                    epis[n] = self.outer_smoothing(epis[n], params)
                    epis[n] = self.post_filter(epis[n], params)

                for epi in epis:
                    self.st[:, :, x, :] += epi[:, :, :]

        self.st = self.post_processing(lf3d, params)

    def post_processing(self, lf3d, params):
        """
        overwrite this method to do some post processing on
        the structure tensor result
        """
        self.st[:] /= lf3d.shape[3]
        return self.st

    def pre_filter(self, epi, params):
        """
        overwrite this method to do some pre filtering on the
        epi channels
        """
        assert isinstance(epi, np.ndarray)
        assert isinstance(params, type({}))
        return epi

    def inner_smoothing(self, epi, params):
        """
        overwrite this method to do the inner smooting on the
        epi channels
        """
        assert isinstance(epi, np.ndarray)
        assert isinstance(params, type({}))
        return epi

    def derivations(self, epi, params):
        """
        overwrite this method to compute the derivatives on the
        epi channels
        """
        assert isinstance(epi, np.ndarray)
        assert isinstance(params, type({}))
        return epi

    def outer_smoothing(self, epi, params):
        """
        overwrite this method to do the outer smooting on the
        epi channels
        """
        assert isinstance(epi, np.ndarray)
        assert isinstance(params, type({}))
        return epi

    def post_filter(self, epi, params):
        """
        overwrite this method to do some post filtering on the
        epi channels
        """
        assert isinstance(epi, np.ndarray)
        assert isinstance(params, type({}))
        return epi

    def get_result(self):
        return self.st


class StructureTensorClassic(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")

        return vigra.filters.structureTensor(epi, params["inner_scale"], params["outer_scale"])


class StructureTensorHourGlass(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")


        # tmp_Grad = vigra.filters.gaussianGradient(epi,params["inner_scale"])
        # print tmp_Grad.shape

        tensor =  vigra.filters.structureTensor(epi, params["inner_scale"], params["outer_scale"])
        # tensor = vigra.filters.vectorToTensor(tmp_Grad)
        # print tensor.shape

        strTen = vigra.filters.hourGlassFilter2D(tensor, params["hour-glass"], 0.4)
        # print strTen.shape

        return strTen











#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################




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
    invalid_ubounds = np.where(orientation > 1.0)
    invalid_lbounds = np.where(orientation < -1.0)
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    orientation[invalid_ubounds] = -1.0
    orientation[invalid_lbounds] = -1.0
    return orientation, coherence


def changeColorSpace(lf3d, cspace=0):
    if lf3d.shape[3] == 3 and cspace > 0:
        for n in range(lf3d.shape[0]):
            if cspace == 1:
                lf3d[n, :, :, :] = vigra.colors.transform_RGB2Lab(lf3d[n, :, :, :])
            if cspace == 2:
                lf3d[n, :, :, :] = vigra.colors.transform_RGB2Luv(lf3d[n, :, :, :])
    return lf3d


def preImgDerivation(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    for i in xrange(lf3d.shape[0]):
        for c in xrange(lf3d.shape[3]):
            grad = vigra.filters.gaussianGradient(lf3d[i, :, :, c], scale)
            if direction == 'h':
                tmp = vigra.colors.linearRangeMapping(grad[:, :, 0], newRange=(0.0, 1.0))
            if direction == 'v':
                tmp = vigra.colors.linearRangeMapping(grad[:, :, 1], newRange=(0.0, 1.0))
            lf3d[i, :, :, c] = tmp

    return lf3d


def preImgLaplace(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    for i in xrange(lf3d.shape[0]):
        for c in xrange(lf3d.shape[3]):
            laplace = vigra.filters.laplacianOfGaussian(lf3d[i, :, :, c], scale)
            lf3d[i, :, :, c] = laplace[:]

    return lf3d


def preEpiDerivation(lf3d, scale=0.1, direction='h'):
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


def preEpiLaplace(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    if direction == 'h':
        for y in xrange(lf3d.shape[1]):
            for c in xrange(lf3d.shape[3]):
                laplace = vigra.filters.laplacianOfGaussian(lf3d[:, y, :, c], scale)
                lf3d[:, y, :, c] = laplace[:]

    elif direction == 'v':
        for x in xrange(lf3d.shape[2]):
            for c in xrange(lf3d.shape[3]):
                laplace = vigra.filters.laplacianOfGaussian(lf3d[:, :, x, c], scale)
                lf3d[:, :, x, c] = laplace[:]
    else:
        assert False, "unknown lightfield direction!"

    return lf3d


def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
    winner = np.where(coherence2 > coherence1)
    orientation1[winner] = orientation2[winner]
    coherence1[winner] = coherence2[winner]
    ### apply memory of coherence, good values get enhanced if they stay longer
    winner = np.where(0.90 > coherence1)
    coherence1[winner] =  coherence1[winner] * 1.05
    winner = np.where(0.98 > coherence1)
    coherence1[winner] =  coherence1[winner] * 1.07

    return orientation1, coherence1