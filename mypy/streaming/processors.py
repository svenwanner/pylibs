# -*- coding: utf-8 -*-

import vigra
import numpy as np
from scipy.ndimage import shift
from joblib import Parallel, delayed

from mypy.streaming.globals import DEBUG

########################################################################################################################
########################################################################################################################
######################  processing function needs to be defined globally to be accepted by the  ########################
######################  multiprocessing module. Create a new function with the same name and    ########################
######################  parameter signature and rename the old one to implement a new epi       ########################
######################  processing behaviour of the EpiProcessor class.                         ########################
########################################################################################################################
########################################################################################################################

def process(input):
    """
    this function is the main routine called on each epi through the
    Processor class. The input is a list containing the epi as first
    entry and the parameter dictionary as second. Ensure that the
    number of channels of your output array are correct.
    :param input: <[]> list [epi<ndarray>,parameter<{}>]
    :return: <ndarray> result
    """
    assert isinstance(input, type([]))
    assert isinstance(input[0], np.ndarray)
    assert isinstance(input[1], type({}))
    out = np.zeros((input[0].shape[0], input[0].shape[1], 2), dtype=np.float32)
    epi = input[0]

    if not input[1].has_key("prefilter"):
        input[1]["prefilter"] = True

    if input[1]["prefilter"]:
        epi = vigra.filters.gaussianGradient(epi, 0.4)[:, :, 1]

    gaussianInner = vigra.filters.gaussianKernel(input[1]["inner_scale"])
    gaussianOuter = vigra.filters.gaussianKernel(input[1]["outer_scale"])

    grad = np.zeros((epi.shape[0], epi.shape[1], 2), dtype=np.float32)

    GD = vigra.filters.Kernel1D()
    GD.initGaussianDerivative(input[1]["inner_scale"], 1)
    #set border Treatment
    GD.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    #inner gaussian filter
    epi = vigra.filters.convolveOneDimension(epi, 1, gaussianInner)
    epi = vigra.filters.convolveOneDimension(epi, 0, gaussianInner)


    #derivative computation
    grad[:, :, 0] = vigra.filters.convolveOneDimension(epi, 0, GD)
    grad[:, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, 0], 1, gaussianInner)
    grad[:, :, 1] = vigra.filters.convolveOneDimension(epi, 1, GD)
    grad[:, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, 1], 0, gaussianInner)

    tensor = vigra.filters.vectorToTensor(grad)
    tensor[:, :, 0] = vigra.filters.convolveOneDimension(tensor[:, :, 0], 1, gaussianOuter)
    tensor[:, :, 1] = vigra.filters.convolveOneDimension(tensor[:, :, 1], 1, gaussianOuter)
    tensor[:, :, 2] = vigra.filters.convolveOneDimension(tensor[:, :, 2], 1, gaussianOuter)

    tensor[:, :, 0] = vigra.filters.convolveOneDimension(tensor[:, :, 0], 0, gaussianOuter)
    tensor[:, :, 1] = vigra.filters.convolveOneDimension(tensor[:, :, 1], 0, gaussianOuter)
    tensor[:, :, 2] = vigra.filters.convolveOneDimension(tensor[:, :, 2], 0, gaussianOuter)

    #compute coherence value
    up = np.sqrt((tensor[:, :, 2]-tensor[:, :, 0])**2 + 4*tensor[:, :, 1]**2)
    down = (tensor[:, :, 2]+tensor[:, :, 0] + 1e-25)
    coherence = up / down

    #compute disparity value
    orientation = vigra.numpy.arctan2(2*tensor[:, :, 1], tensor[:, :, 2]-tensor[:, :, 0]) / 2.0
    orientation = vigra.numpy.tan(orientation[:])

    #mask out of boundary orientation estimation
    invalid_ubounds = np.where(orientation > 1.1)
    invalid_lbounds = np.where(orientation < -1.1)
    if not input[1].has_key("min_coherence"):
        input[1]["min_coherence"] = 0.5
    invalid_coh = np.where(coherence < input[1]["min_coherence"])

    #set coherence of invalid values to zero
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    coherence[invalid_coh] = 0

    #set orientation of invalid values to related maximum/minimum value
    orientation[invalid_ubounds] = -1.5
    orientation[invalid_lbounds] = -1.5
    orientation[invalid_coh] = -1.5

    out[:, :, 0] = orientation[:, :]
    out[:, :, 1] = coherence[:, :]
    return out

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################



def refocus(epi, focus):
    tmp = np.zeros_like(epi)
    for h in xrange(epi.shape[0]):
        tmp[h, :] = shift(epi[h, :], (h-epi.shape[0]/2)*focus)
    return tmp


class EpiProcessor(object):
    """
    This class is used to be taking control over the epi iteration calling
    the process method defined globally above multithreaded for each epi.
    It is necessary to set the parameter object containing the number of
    channels the result array should have and all parameter used in the
    process method. The rest ist done by the Engine class.
    """
    def __init__(self, parameter):
        self.data = None
        self.result = None

        self.parameter = parameter

    def setData(self, data):
        """
        sets the data array to be processed.
        :param data: <ndarray> input data
        """
        assert isinstance(data, np.ndarray)
        assert len(data.shape) == 3
        self.data = data

    def getResult(self):
        """
        returns the result array
        """
        return self.result

    def start(self):
        """
        this is the main routine of the class calling process()
        on each epi in parallel.
        """
        assert self.data is not None, "Need data before processing can be started!"

        self.result = np.zeros((self.data.shape[1], self.data.shape[2], 2), dtype=np.float32)

        assert self.result is not None, "No result array is defined!"
        assert self.parameter is not None, "No parameter object is defined!"

        for f in self.parameter.focuses:
            print "process focus", f
            inputs = []
            for n in range(self.data.shape[1]):
                epi = refocus(self.data[:, n, :], f)
                parameter = {"inner_scale" : self.parameter.inner_scale,
                             "outer_scale" : self.parameter.outer_scale,
                             "min_coherence" : self.parameter.min_coherence,
                             "focuses" : self.parameter.focuses,
                             "prefilter" : self.parameter.prefilter}
                inputs.append([epi, parameter])

            result = Parallel(n_jobs=4)(delayed(process)(inputs[i]) for i in range(len(inputs)))
            tmp = np.zeros((self.result.shape[0], self.result.shape[1]), dtype=np.float32)
            for m, res in enumerate(result):
                tmp[m, :] = res[self.data.shape[0]/2, :, 0]+f
            for m, res in enumerate(result):
                winner = np.where(res[self.data.shape[0]/2, :, 1] > self.result[m, :, 1])
                self.result[m, winner, 0] = res[self.data.shape[0]/2, winner, 0]+f
                self.result[m, winner, 1] = res[self.data.shape[0]/2, winner, 1]
