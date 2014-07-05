from __future__ import division

import vigra
import numpy as np

from mypy.visualization.imshow import imshow
from scipy.ndimage import convolve


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


class StructureTensorScharr(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")

        KernelV = np.array([[3,  10,  3],
                            [0,   0,  0],
                            [-3, -10, -3]]) / 32.0
        KernelH = KernelV.T

        scharrh = vigra.filters.Kernel2D()
        scharrh.initExplicitly((-1, -1), (1, 1), KernelH)

        scharrv = vigra.filters.Kernel2D()
        scharrv.initExplicitly((-1, -1), (1, 1), KernelV)

        grad = np.zeros((epi.shape[0], epi.shape[1], 2), dtype=np.float32)

        epi = vigra.filters.gaussianSmoothing(epi, params["inner_scale"])

        epi = vigra.filters.convolve(epi, scharrh)
        grad[:, :, 1] = vigra.filters.convolve(epi, scharrh)
        grad[:, :, 0] = vigra.filters.convolve(epi, scharrv)

        tensor = vigra.filters.vectorToTensor(grad)

        tensor[:, :, 0] = vigra.filters.gaussianSmoothing(tensor[:, :, 0], params["outer_scale"])
        tensor[:, :, 1] = vigra.filters.gaussianSmoothing(tensor[:, :, 1], params["outer_scale"])
        tensor[:, :, 2] = vigra.filters.gaussianSmoothing(tensor[:, :, 2], params["outer_scale"])

        tensor = vigra.filters.hourGlassFilter2D(tensor, params["hour-glass"], 0.4)

        return tensor



class StructureTensorHourGlass(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")

        tensor =  vigra.filters.structureTensor(epi, params["inner_scale"], params["outer_scale"])
        strTen = vigra.filters.hourGlassFilter2D(tensor, params["hour-glass"], 0.4)

        return strTen











#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################


def evaluateStructureTensor(tensor):
    assert isinstance(tensor, np.ndarray)
    print "evaluate structure tensor..."
    coherence = np.sqrt((tensor[:, :, :, 2]-tensor[:, :, :, 0])**2+(2*tensor[:, :, :, 1])**2)/(tensor[:, :, :, 2]+tensor[:, :, :, 0] + 1e-16)
    orientation = 1/2.0*vigra.numpy.arctan2(2*tensor[:, :, :, 1], tensor[:, :, :, 2]-tensor[:, :, :, 0])
    orientation = vigra.numpy.tan(orientation[:])
    invalid_ubounds = np.where(orientation > 1.1)
    invalid_lbounds = np.where(orientation < -1.1)
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    orientation[invalid_ubounds] = -1.1
    orientation[invalid_lbounds] = -1.1
    return orientation, coherence



# def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
#     print "merge orientations wta..."
#     winner = np.where(coherence2 > coherence1)
#     orientation1[winner] = orientation2[winner]
#     coherence1[winner] = coherence2[winner]
#     return orientation1, coherence1


def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
    print "merge orientations wta..."
    winner = np.where(coherence2 > coherence1)
    orientation1[winner] = orientation2[winner]
    coherence1[winner] = coherence2[winner]
    ### apply memory of coherence
    # winner = np.where(0.99 < coherence1)
    # coherence1[winner] =  coherence1[winner] * 1.01
    # winner = np.where(0.9995 < coherence1)
    # coherence1[winner] =  coherence1[winner] * 1.05

    return orientation1, coherence1