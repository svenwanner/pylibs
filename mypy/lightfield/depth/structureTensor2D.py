from __future__ import division

import vigra
import numpy as np


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


class StructureTensorHourGlass(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")

        tensor = vigra.filters.structureTensor(epi, params["inner_scale"], params["outer_scale"])
        strTen = vigra.filters.hourGlassFilter2D(tensor, params["hour-glass"], 0.4)

        return strTen

class StructureTensorScharr(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")

        K = np.array([-1, 0, 1]) / 2.0
        scharr1dim = vigra.filters.Kernel1D()
        scharr1dim.initExplicitly(-1, 1, K)

        K = np.array([3, 10, 3]) / 16.0
        scharr2dim = vigra.filters.Kernel1D()
        scharr2dim.initExplicitly(-1, 1, K)

        epi = vigra.filters.gaussianSmoothing(epi, sigma=params["inner_scale"])

        # print("apply scharr pre-filter along 2rd dimension")
        epi = vigra.filters.convolveOneDimension(epi, 1, scharr1dim)
        epi = vigra.filters.convolveOneDimension(epi, 0, scharr2dim)

        grad = np.zeros((epi.shape[0], epi.shape[1], 2), dtype = np.float32)
        ### Derivative computation ###
        # print("apply scharr filter along 1st dimension")
        grad[:, :, 0] = vigra.filters.convolveOneDimension(epi, 0, scharr1dim)
        grad[:, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, 0], 1, scharr2dim)
        # print("apply scharr filter along 2rd dimension")
        grad[:, :, 1] = vigra.filters.convolveOneDimension(epi, 1, scharr1dim)
        grad[:, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, 1], 0, scharr2dim)

        # Kernel_H = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32.0
        # scharrh = vigra.filters.Kernel2D()
        # scharrh.initExplicitly((-1, -1), (1, 1), Kernel_H)
        #
        # Kernel_V = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]) / 32.0
        # scharrv = vigra.filters.Kernel2D()
        # scharrv.initExplicitly((-1, -1), (1, 1), Kernel_V)
        #
        # epi = vigra.filters.gaussianSmoothing(epi, sigma=params["inner_scale"])
        #
        # epi = vigra.filters.convolve(epi, scharrh)
        # d_2dim = vigra.filters.convolve(epi, scharrh)
        # d_1dim = vigra.filters.convolve(epi, scharrv)
        #
        # grad = np.zeros((d_1dim.shape[0], d_1dim.shape[1], 2), dtype = np.float32)
        # grad[:, :, 0] = d_1dim[:, :]
        # grad[:, :, 1] = d_2dim[:, :]

        tensor = vigra.filters.vectorToTensor(grad)

        tensor[:, :, 0] = vigra.filters.gaussianSmoothing(tensor[:, :, 0], sigma=params["outer_scale"])
        tensor[:, :, 1] = vigra.filters.gaussianSmoothing(tensor[:, :, 1], sigma=params["outer_scale"])
        tensor[:, :, 2] = vigra.filters.gaussianSmoothing(tensor[:, :, 2], sigma=params["outer_scale"])

        # tensor = vigra.filters.hourGlassFilter2D(tensor, params["outer_scale"], 0.4)
        # tensor = vigra.filters.hourGlassFilter2D(tensor, params["hour-glass"], 0.4)

        return tensor


class StructureTensorForward(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")

        Kernel_H = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]) / 2.0
        forwh = vigra.filters.Kernel2D()
        forwh.initExplicitly((-1, -1), (1, 1), Kernel_H)

        Kernel_V = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]]) / 2.0
        forwv = vigra.filters.Kernel2D()
        forwv.initExplicitly((-1, -1), (1, 1), Kernel_V)

        epi = vigra.filters.gaussianSmoothing(epi, sigma=params["inner_scale"])

        # epi = vigra.filters.convolve(epi, forwh)
        d_2dim = vigra.filters.convolve(epi, forwh)
        d_1dim = vigra.filters.convolve(epi, forwv)

        grad = np.zeros((d_1dim.shape[0], d_1dim.shape[1], 2), dtype = np.float32)
        grad[:, :, 0] = d_1dim[:,:]
        grad[:, :, 1] = d_2dim[:,:]

        tensor = vigra.filters.vectorToTensor(grad)

        tensor[:, :, 0] = vigra.filters.gaussianSmoothing(tensor[:, :, 0], sigma=params["outer_scale"])
        tensor[:, :, 1] = vigra.filters.gaussianSmoothing(tensor[:, :, 1], sigma=params["outer_scale"])
        tensor[:, :, 2] = vigra.filters.gaussianSmoothing(tensor[:, :, 2], sigma=params["outer_scale"])

        # tensor = vigra.filters.hourGlassFilter2D(tensor, params["outer_scale"], 0.4)
        # tensor = vigra.filters.hourGlassFilter2D(tensor, params["hour-glass"], 0.4)

        return tensor

class StructureTensorBackward(StructureTensor):

    def __init__(self):
        StructureTensor.__init__(self)

    def derivations(self, epi, params):
        assert isinstance(epi, np.ndarray)
        assert params.has_key("inner_scale")
        assert params.has_key("outer_scale")

        Kernel_H = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]) / 2.0
        forwh = vigra.filters.Kernel2D()
        forwh.initExplicitly((-1, -1), (1, 1), Kernel_H)

        Kernel_V = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]) / 2.0
        forwv = vigra.filters.Kernel2D()
        forwv.initExplicitly((-1, -1), (1, 1), Kernel_V)

        epi = vigra.filters.gaussianSmoothing(epi, sigma=params["inner_scale"])

        # epi = vigra.filters.convolve(epi, forwh)
        d_2dim = vigra.filters.convolve(epi, forwh)
        d_1dim = vigra.filters.convolve(epi, forwv)

        grad = np.zeros((d_1dim.shape[0], d_1dim.shape[1], 2), dtype = np.float32)
        grad[:, :, 0] = d_1dim[:,:]
        grad[:, :, 1] = d_2dim[:,:]

        tensor = vigra.filters.vectorToTensor(grad)

        tensor[:, :, 0] = vigra.filters.gaussianSmoothing(tensor[:, :, 0], sigma=params["outer_scale"])
        tensor[:, :, 1] = vigra.filters.gaussianSmoothing(tensor[:, :, 1], sigma=params["outer_scale"])
        tensor[:, :, 2] = vigra.filters.gaussianSmoothing(tensor[:, :, 2], sigma=params["outer_scale"])

        # tensor = vigra.filters.hourGlassFilter2D(tensor, params["outer_scale"], 0.4)
        # tensor = vigra.filters.hourGlassFilter2D(tensor, params["hour-glass"], 0.4)

        return tensor

#############################################################################################################
############# Computation of disparity and coherence map and merge to global solution
#############################################################################################################


def evaluateStructureTensor(tensor):

    assert isinstance(tensor, np.ndarray)

    ### compute coherence value ###
    up = np.sqrt((tensor[:, :, :, 2]-tensor[:, :, :, 0])**2 + 4*tensor[:, :, :, 1]**2)
    down = (tensor[:, :, :, 2]+tensor[:, :, :, 0] + 1e-25)
    coherence = up / down

    ### compute disparity value ###
    orientation = vigra.numpy.arctan2(2*tensor[:, :, :, 1], tensor[:, :, :, 2]-tensor[:, :, :, 0]) / 2.0
    orientation = vigra.numpy.tan(orientation[:])

    ### mark out of boundary orientation estimation ###
    invalid_ubounds = np.where(orientation > 1.1)
    invalid_lbounds = np.where(orientation < -1.1)

    ### set coherence of invalid values to zero ###
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0


    ### set orientation of invalid values to related maximum/minimum value
    orientation[invalid_ubounds] = 1.1
    orientation[invalid_lbounds] = -1.1

    return orientation, coherence


def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):

    ### replace orientation/coherence values, where coherence 1 > coherence 2 ###
    print "merge orientations wta..."
    winner = np.where(coherence2 > coherence1)

    orientation1[winner] = orientation2[winner]
    coherence1[winner] = coherence2[winner]

    ### apply memory of coherence
    # winner = np.where(0.999 < coherence1)
    # coherence1[winner] =  coherence1[winner] * 1.01
    # winner = np.where(0.99995 < coherence1)
    # coherence1[winner] =  coherence1[winner] * 1.05

    return orientation1, coherence1