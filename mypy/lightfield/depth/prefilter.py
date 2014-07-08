import numpy as np
import vigra
from mypy.lightfield.helpers import enum
import logging
import pylab as plt

#============================================================================================================
#==========================                     prefiltering methods (color space)              ===========================
#============================================================================================================


COLORSPACE = enum(RGB=0, LAB=1, LUV=2)



def changeColorSpace(lf3d, cspace=0):
    if lf3d.shape[3] == 3 and cspace > 0:
        for n in range(lf3d.shape[0]):
            if cspace == 1:
                lf3d[n, :, :, :] = vigra.colors.transform_RGB2Lab(lf3d[n, :, :, :])
            if cspace == 2:
                lf3d[n, :, :, :] = vigra.colors.transform_RGB2Luv(lf3d[n, :, :, :])
    return lf3d





#============================================================================================================
#==========================                     prefiltering methods (derivatives)             ===========================
#============================================================================================================
PREFILTER = enum(NO=0, IMGD=1, EPID=2, IMGD2=3, EPID2=4, SCHARR=5)

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


def preImgScharr(lf3d, config = None, direction='h'):

    assert isinstance(lf3d, np.ndarray)

    print("apply image scharr prefilter")
    if direction == 'h':
        Kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32.0
        scharr = vigra.filters.Kernel2D()
        scharr.initExplicitly((-1,-1), (1,1), Kernel)
        for t in xrange(lf3d.shape[0]):
            for c in xrange(lf3d.shape[3]):
                lf3d[t, :, :, c] = vigra.filters.gaussianSmoothing(lf3d[t, :, :, c], 0.4)
                lf3d[t, :, :, c] = vigra.filters.convolve(lf3d[t, :, :, c], scharr)
                lf3d[t, :, :, c] = lf3d[t, :, :, c]**2
            if config.output_level >3:
                    plt.imsave(config.result_path+config.result_label+"Horizontal_Scharr_Image_{0}.png".format(t), np.abs(lf3d[t, :, :, :]))

    elif direction == 'v':
        Kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]) / 32.0
        scharr = vigra.filters.Kernel2D()
        scharr.initExplicitly((-1,-1), (1,1), Kernel)
        for t in xrange(lf3d.shape[0]):
            for c in xrange(lf3d.shape[3]):
                lf3d[t, :, :, c] = vigra.filters.gaussianSmoothing(lf3d[t, :, :, c], 0.4)
                lf3d[t, :, :, c] = vigra.filters.convolve(lf3d[t, :, :, c], scharr)
                lf3d[t, :, :, c] = lf3d[t, :, :, c]**2
            if config.output_level >3:
                    plt.imsave(config.result_path+config.result_label+"Vertical_Scharr_Image_{0}.png".format(t), np.abs(lf3d[t, :, :, :]))
    else:
        assert False, "unknown lightfield direction!"

    return lf3d