import numpy as np
import vigra
from mypy.lightfield.helpers import enum
import logging
import pylab as plt
from scipy.misc import imsave
import skimage.color as color
#============================================================================================================
#==========================                     prefiltering methods (color space)              ===========================
#============================================================================================================


COLORSPACE = enum(RGB=0, LAB=1, LUV=2, GRAY=3, HSV=4, SGRAY=5)
PREFILTER = enum(NO=0, IMGD=1, EPID=2, IMGD2=3, EPID2=4, SCHARR=5, DOG=6, GAUSS=7)


def changeColorSpace(lf3d, cspace=COLORSPACE.RGB):

    if lf3d.shape[3] == 3:
        if lf3d.dtype == np.uint8:
            lf3d = lf3d.astype(np.float32)
        if np.amax(lf3d) > 1.0:
            lf3d[:] /= 255.0

        if cspace == COLORSPACE.HSV:
            print("Change to HSV")
            for i in range(lf3d.shape[0]):
                lf3d[i, :, :, :] = color.rgb2hsv(lf3d[i, :, :, :])
            return lf3d

        elif cspace == COLORSPACE.LUV:
            print("Change to LUV")
            for i in range(lf3d.shape[0]):
                #lf3d[i, :, :, :] = color.rgb2luv(lf3d[i, :, :, :])
                lf3d[i, :, :, :] = vigra.colors.transform_RGB2Luv(lf3d[i, :, :, :])
            return lf3d

        elif cspace == COLORSPACE.LAB:
            print("Change to LAB")
            for i in range(lf3d.shape[0]):
                #lf3d[i, :, :, :] = color.rgb2lab(lf3d[i, :, :, :])
                lf3d[i, :, :, :] = vigra.colors.transform_RGB2Lab(lf3d[i, :, :, :])
            return lf3d

        elif cspace == COLORSPACE.GRAY:
            print("Change to GRAY")
            tmp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 1), dtype=np.float32)
            weight = [0.298, 0.5870, 0.1140]#RGB convertion like in Matlab
            tmp[:,:,:,0] = weight[0]*lf3d[:, :, :, 0]+weight[1]*lf3d[:, :, :, 1]+weight[2]*lf3d[:, :, :, 2]
            return tmp

        elif cspace == COLORSPACE.SGRAY:
            print("Change to SPECIAL GRAY")
            tmp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 1), dtype=np.float32)
            weight = [0, 256, 59536] #TODO: look which weighting order is better
            tmp[:,:,:,0] = weight[0]*lf3d[:, :, :, 0]+weight[1]*lf3d[:, :, :, 1]+weight[2]*lf3d[:, :, :, 2]
            return tmp

    return lf3d





#============================================================================================================
#==========================                     prefiltering methods (derivatives)             ===========================
#============================================================================================================

def preImgDerivation(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    print("apply image derivative prefilter")
    for i in xrange(lf3d.shape[0]):
        for c in xrange(lf3d.shape[3]):
            grad = vigra.filters.gaussianGradient(lf3d[i, :, :, c], scale)
            if direction == 'h':
                tmp = grad[:, :, 1]
            if direction == 'v':
                tmp = grad[:, :, 0]
            lf3d[i, :, :, c] = tmp

    return lf3d


def preImgLaplace(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    print("apply image laplace prefilter")
    for i in xrange(lf3d.shape[0]):
        for c in xrange(lf3d.shape[3]):
            laplace = vigra.filters.laplacianOfGaussian(lf3d[i, :, :, c], scale)
            lf3d[i, :, :, c] = laplace[:]

    return lf3d


def preEpiDerivation(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    print("apply epi derivative prefilter")
    if direction == 'h':
        for y in xrange(lf3d.shape[1]):
            for c in xrange(lf3d.shape[3]):
                grad = vigra.filters.gaussianGradient(lf3d[:, y, :, c], scale)
                # try:
                #     tmp = vigra.colors.linearRangeMapping(grad[:, :, 0], newRange=(0.0, 1.0))
                # except:
                tmp = grad[:, :, 1]
                lf3d[:, y, :, c] = tmp[:]

    elif direction == 'v':
        for x in xrange(lf3d.shape[2]):
            for c in xrange(lf3d.shape[3]):
                grad = vigra.filters.gaussianGradient(lf3d[:, :, x, c], scale)
                # try:
                #     tmp = vigra.colors.linearRangeMapping(grad[:, :, 0], newRange=(0.0, 1.0))
                # except:
                tmp = grad[:, :, 1]
                lf3d[:, :, x, c] = tmp[:]
    else:
        assert False, "unknown lightfield direction!"

    return lf3d


def preEpiLaplace(lf3d, scale=0.1, direction='h'):
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(scale, float)

    print("apply epi laplace prefilter")
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


def preImgScharr(lf3d, config=None, direction='h'):

    assert isinstance(lf3d, np.ndarray)

    K = np.array([-1, 0, 1]) / 2.0
    scharr1dim = vigra.filters.Kernel1D()
    scharr1dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    ##BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr1dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    K = np.array([3, 10, 3]) / 16.0
    scharr2dim = vigra.filters.Kernel1D()
    scharr2dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    #BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr2dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    print("apply image scharr prefilter")
    if direction == 'h':
        lf3d = vigra.filters.convolveOneDimension(lf3d, 2, scharr1dim)
        lf3d = vigra.filters.convolveOneDimension(lf3d, 1, scharr2dim)
        for t in range(lf3d.shape[0]):
            if config.output_level >3:
                plt.imsave(config.result_path+config.result_label+"Horizontal_Scharr_Image_{0}.png".format(t), np.abs(lf3d[t, :, :, :]))

    elif direction == 'v':

        lf3d = vigra.filters.convolveOneDimension(lf3d, 1, scharr1dim)
        lf3d = vigra.filters.convolveOneDimension(lf3d, 2, scharr2dim)

        for t in range(lf3d.shape[0]):
            if config.output_level >3:
                plt.imsave(config.result_path+config.result_label+"Vertical_Scharr_Image_{0}.png".format(t), np.abs(lf3d[t, :, :, :]))
    else:
        assert False, "unknown lightfield direction!"

    return lf3d


def preDoG(lf3d, config=None):

    assert isinstance(lf3d, np.ndarray)

    if lf3d.shape[3] == 3:
        tmp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 1), dtype=np.float32)
        tmp[:, :, :, 0] = 0.3*lf3d[:, :, :, 0]+0.59*lf3d[:, :, :, 1]+0.11*lf3d[:, :, :, 2]
        lf3d = tmp

    print("apply image DoG prefilter")

    sigmas = [0.4, 0.9, 1.5, 2.0, 2.8, 4.0]
    if config.prefilter_scale > 0.0:
        for i in range(6):
            sigmas[i] = config.prefilter_scale*np.exp(i/2.0)
    for view in range(lf3d.shape[0]):
        for c in range(lf3d.shape[3]):
            tmp = np.copy(lf3d[view, :, :, c])
            lf3d[view, :, :, c] = 0
            for s in sigmas:
                level = vigra.filters.gaussianSmoothing(tmp, s)
                lf3d[view, :, :, c] += tmp - level
                tmp = level
            lf3d[view, :, :, c] /= np.amax(lf3d[view, :, :, c])


    return lf3d