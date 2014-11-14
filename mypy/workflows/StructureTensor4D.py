import os
import logging
import vigra
import numpy as np
import pylab as plt
import scipy.misc as misc

from mypy.lightfield import io as lfio

import mypy.pointclouds.depthToCloud as dtc
import mypy.lightfield.helpers as lfhelpers
import mypy.lightfield.depth.prefilter as prefilter
from scipy.ndimage import median_filter

#============================================================================================================
#=========                                       LF processing                                   ===========
#============================================================================================================

def Compute(lf3d, config, direction):

    if direction == 'h':
        tensor, temp = compute_horizontal(lf3d, config)
    if direction == 'v':
        tensor, temp = compute_vertical(lf3d, config)

    return tensor, temp

def Compute4D(lf, config):


    gaussianpre = vigra.filters.gaussianKernel(config.prefilter_scale)
    gaussianInner = vigra.filters.gaussianKernel(config.inner_scale)

    K = np.array([-1, 0, 1]) / 2.0
    scharr1dim = vigra.filters.Kernel1D()
    scharr1dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    #BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr1dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    K = np.array([3, 10, 3]) / 16.0
    scharr2dim = vigra.filters.Kernel1D()
    scharr2dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    #BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr2dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    GD = vigra.filters.Kernel1D()
    GD.initGaussianDerivative(config.prefilter_scale, 1)
    #Border Treatments:
    ##BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    GD.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)


    lfh = lf.copy()


    grad = np.zeros((lf.shape[0], lf.shape[1], lf.shape[2], lf.shape[3], 2), dtype=np.float32)
    temph = np.zeros((lf.shape[0], lf.shape[1], lf.shape[2], lf.shape[3], 2, 3), dtype=np.float32)
    # ergh = np.zeros((lf.shape[1], lf.shape[2], lf.shape[3], 2, 3), dtype=np.float32)
    tempv = np.zeros((lf.shape[0], lf.shape[1], lf.shape[2], lf.shape[3], 2, 3), dtype=np.float32)
    # ergv = np.zeros((lf.shape[0], lf.shape[2], lf.shape[3], 2, 3), dtype=np.float32)

    if(config.structure_tensor_type == "scharr"):

        for j in range(lfh.shape[0]):
            for i in range(lfh.shape[4]):

                ### Inner gaussian filter ###
                # print("apply gaussian filter along 3rd dimension")
                lfh[j, :, :, :, i] = vigra.filters.convolveOneDimension(lfh[j, :, :, :, i], 2, gaussianInner)
                # print("apply gaussian filter along 1rd dimension")
                lfh[j, :, :, :, i] = vigra.filters.convolveOneDimension(lfh[j, :, :, :, i], 0, gaussianInner)
                # print("apply gaussian filter along 2rd dimension")
                # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
                if (config.prefilter == "True"):
                    ### EPI prefilter ###
                    # print("apply scharr pre-filter along 3rd dimension")
                    lfh[j, :, :, :, i] = vigra.filters.convolveOneDimension(lfh[j, :, :, :, i], 2, scharr1dim)
                    lfh[j, :, :, :, i] = vigra.filters.convolveOneDimension(lfh[j, :, :, :, i], 0, scharr2dim)

                ### Derivative computation ###
                # print("apply scharr filter along 1st dimension")
                grad[j, :, :, :, 0] = vigra.filters.convolveOneDimension(lfh[j, :, :, :, i], 0, scharr1dim)
                grad[j, :, :, :, 0] = vigra.filters.convolveOneDimension(grad[j, :, :, :, 0], 2, scharr2dim)
                # print("apply scharr filter along 3rd dimension")
                grad[j, :, :, :, 1] = vigra.filters.convolveOneDimension(lfh[j, :, :, :, i], 2, scharr1dim)
                grad[j, :, :, :, 1] = vigra.filters.convolveOneDimension(grad[j, :, :, :, 1], 0, scharr2dim)

                temph[j, :, :, :, 0, i] = grad[j, :, :, :, 0]
                temph[j, :, :, :, 1, i] = grad[j, :, :, :, 1]

        for j in range(lf.shape[1]):
            for i in range(lf.shape[4]):

                ### Inner gaussian filter ###
                # print("apply gaussian filter along 2rd dimension")
                lf[:, j, :, :, i] = vigra.filters.convolveOneDimension(lf[:, j, :, :, i], 1, gaussianInner)
                # print("apply gaussian filter along 1rd dimension")
                lf[:, j, :, :, i] = vigra.filters.convolveOneDimension(lf[:, j, :, :, i], 0, gaussianInner)
                # print("apply gaussian filter along 2rd dimension")
                # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
                if (config.prefilter == "True"):
                    ### EPI prefilter ###
                    # print("apply scharr pre-filter along 2rd dimension")
                    lf[:, j, :, :, i] = vigra.filters.convolveOneDimension(lf[:, j, :, :, i], 1, scharr1dim)
                    lf[:, j, :, :, i] = vigra.filters.convolveOneDimension(lf[:, j, :, :, i], 0, scharr2dim)

                ### Derivative computation ###
                # print("apply scharr filter along 1st dimension")
                grad[:, j, :, :, 0] = vigra.filters.convolveOneDimension(lf[:, j, :, :, i], 0, scharr1dim)
                grad[:, j, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, j, :, :, 0], 1, scharr2dim)
                # print("apply scharr filter along 2rd dimension")
                grad[:, j, :, :, 1] = vigra.filters.convolveOneDimension(lf[:, j, :, :, i], 1, scharr1dim)
                grad[:, j, :, :, 1] = vigra.filters.convolveOneDimension(grad[:, j, :, :, 1], 0, scharr2dim)

                tempv[:, j, :, :, 0, i] = grad[:, j, :, :, 0]
                tempv[:, j, :, :, 1, i] = grad[:, j, :, :, 1]


    return temph, tempv


def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
    winner = np.where(coherence2 > coherence1)
    orientation1[winner] = orientation2[winner]
    coherence1[winner] = coherence2[winner]

    return orientation1, coherence1

def evaluateStructureTensor(tensor):

    assert isinstance(tensor, np.ndarray)

    ### compute coherence value ###
    up = np.sqrt((tensor[:, :, 2]-tensor[:, :, 0])**2 + 4*tensor[:, :, 1]**2)
    down = (tensor[:, :, 2]+tensor[:, :, 0] + 1e-25)
    coherence = up / down

    ### compute disparity value ###
    orientation = vigra.numpy.arctan2(2*tensor[:, :, 1], tensor[:, :, 2]-tensor[:, :, 0]) / 2.0
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

def orientationClassic(strTensorh, strTensorv, config, shift):

        orientationH, coherenceH = evaluateStructureTensor(strTensorh[strTensorh.shape[0]/2,:,:,:])

        if config.output_level >= 4:
            plt.imsave(config.result_path+config.result_label+"orientation_Horizontal_{0}.png".format(shift), orientationH, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"coherence_Horizontal_{0}.png".format(shift), coherenceH, cmap=plt.cm.jet)

        orientationV, coherenceV = evaluateStructureTensor(strTensorv[strTensorv.shape[0]/2,:,:,:])

        if config.output_level >= 4:
            plt.imsave(config.result_path+config.result_label+"orientation_Vertical_{0}.png".format(shift), orientationV, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"coherence_Vertical_{0}.png".format(shift), coherenceV, cmap=plt.cm.jet)

        orientation, coherence = mergeOrientations_wta(orientationH, coherenceH, orientationV, coherenceV)
        orientation[:] += float(shift)

        if config.output_level >= 4:
            plt.imsave(config.result_path+config.result_label+"orientation_merged_local_{0}.png".format(shift), orientation, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"coherence_merged_local_{0}.png".format(shift), coherence, cmap=plt.cm.jet)

        return orientation, coherence, coherenceH, coherenceV

def orientationCompute4D(gradh, gradv, config, shift):

        ten = np.zeros((gradh.shape[0], gradh.shape[1], gradh.shape[2], gradh.shape[3], 3), dtype=np.float32)
        gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)


### this is for the center view components ###
        for i in range(gradh.shape[5]):
            ten[:, :, :, :, 0] += (gradh[:, :, :, :, 0, i] + gradv[:, :, :, :, 0, i])**2
            ten[:, :, :, :, 1] += ((gradh[:, :, :, :, 0, i] + gradv[:, :, :, :, 0, i])*(gradh[:, :, :, :, 1, i] + gradv[:, :, :, :, 1, i]))
            ten[:, :, :, :, 2] += (gradh[:, :, :, :, 1, i] + gradv[:, :, :, :, 1, i])**2

        ten[:, :, :, :, 0] /= gradh.shape[3]
        ten[:, :, :, :, 1] /= gradh.shape[3]
        ten[:, :, :, :, 2] /= gradh.shape[3]

### Copy tensorCenterView in horizontal light field at center view position ###

        for i in range(ten.shape[0]):
            ten[i, :, :, :, 0] = vigra.filters.convolveOneDimension(ten[i, :, :, :, 0], 0, gaussianOuter)
            ten[i, :, :, :, 1] = vigra.filters.convolveOneDimension(ten[i, :, :, :, 1], 0, gaussianOuter)
            ten[i, :, :, :, 2] = vigra.filters.convolveOneDimension(ten[i, :, :, :, 2], 0, gaussianOuter)

            ten[i, :, :, :, 0] = vigra.filters.convolveOneDimension(ten[i, :, :, :, 0], 2, gaussianOuter)
            ten[i, :, :, :, 1] = vigra.filters.convolveOneDimension(ten[i, :, :, :, 1], 2, gaussianOuter)
            ten[i, :, :, :, 2] = vigra.filters.convolveOneDimension(ten[i, :, :, :, 2], 2, gaussianOuter)


        for i in range(ten.shape[1]):
            ten[:, i, :, :, 0] = vigra.filters.convolveOneDimension(ten[:, i, :, :, 0], 0, gaussianOuter)
            ten[:, i, :, :, 1] = vigra.filters.convolveOneDimension(ten[:, i, :, :, 1], 0, gaussianOuter)
            ten[:, i, :, :, 2] = vigra.filters.convolveOneDimension(ten[:, i, :, :, 2], 0, gaussianOuter)

            ten[:, i, :, :, 0] = vigra.filters.convolveOneDimension(ten[:, i, :, :, 0], 1, gaussianOuter)
            ten[:, i, :, :, 1] = vigra.filters.convolveOneDimension(ten[:, i, :, :, 1], 1, gaussianOuter)
            ten[:, i, :, :, 2] = vigra.filters.convolveOneDimension(ten[:, i, :, :, 2], 1, gaussianOuter)

        orientation, coherence = evaluateStructureTensor(ten[ten.shape[0]/2, gradh.shape[1]/2, :, :, :])
        orientation += shift

        return orientation, coherence

#============================================================================================================
#=========                              Horizontal LF computation                                ===========
#============================================================================================================

def compute_horizontal(lf3d, config):

    # print("compute horizontal shift {0}".format(shift))
    # tmp = np.copy(lf3d)
    # tmp = lfhelpers.refocus_3d(tmp, shift, 'h')
    # print("shape of horizontal light field: " + str(tmp.shape))
    #
    # if config.color_space:
    #     lf3d = prefilter.changeColorSpace(tmp, config.color_space)
    # else:
    #     lf3d = tmp

    gaussianpre = vigra.filters.gaussianKernel(config.prefilter_scale)
    gaussianInner = vigra.filters.gaussianKernel(config.inner_scale)

    K = np.array([-1, 0, 1]) / 2.0
    scharr1dim = vigra.filters.Kernel1D()
    scharr1dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    #BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr1dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    K = np.array([3, 10, 3]) / 16.0
    scharr2dim = vigra.filters.Kernel1D()
    scharr2dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    #BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr2dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    GD = vigra.filters.Kernel1D()
    GD.initGaussianDerivative(config.prefilter_scale, 1)
    #Border Treatments:
    ##BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    GD.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    grad = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
    ten = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
    temp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2, 3), dtype=np.float32)

    if(config.structure_tensor_type == "classic"):
        for i in range(lf3d.shape[3]):
            if (config.prefilter == "True"):
                ### Prefilter ###
                # print("apply gaussian derivative prefilter along 3rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, GD)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianpre)

            ### Derivative computation ###
            # print("apply Gaussian derivative filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, GD)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 2, gaussianInner)
            # print("apply Gaussian derivative filter along 3rd dimension")
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, GD)
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, :, 1], 0, gaussianInner)

            ten[:, :, :, 0] += grad[:, :, :, 0]**2
            ten[:, :, :, 1] += grad[:, :, :, 1]*grad[:, :, :, 0]
            ten[:, :, :, 2] += grad[:, :, :, 1]**2

            temp[:, :, :, 0, i] = grad[:, :, :, 0]
            temp[:, :, :, 1, i] = grad[:, :, :, 1]


        ten[:, :, :, 0] /= lf3d.shape[3]
        ten[:, :, :, 1] /= lf3d.shape[3]
        ten[:, :, :, 2] /= lf3d.shape[3]

    if(config.structure_tensor_type == "scharr"):

        for i in range(lf3d.shape[3]):

            ### Inner gaussian filter ###
            # print("apply gaussian filter along 3rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, gaussianInner)
            # print("apply gaussian filter along 1rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner)
            # print("apply gaussian filter along 2rd dimension")
            # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
            if (config.prefilter == "True"):
                ### EPI prefilter ###
                # print("apply scharr pre-filter along 3rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, scharr1dim)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr2dim)

            ### Derivative computation ###
            # print("apply scharr filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr1dim)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 2, scharr2dim)
            # print("apply scharr filter along 3rd dimension")
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, scharr1dim)
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, :, 1], 0, scharr2dim)

            ten[:, :, :, 0] += grad[:, :, :, 0]**2
            ten[:, :, :, 1] += grad[:, :, :, 1]*grad[:, :, :, 0]
            ten[:, :, :, 2] += grad[:, :, :, 1]**2

            temp[:, :, :, 0, i] = grad[:, :, :, 0]
            temp[:, :, :, 1, i] = grad[:, :, :, 1]

        ten[:, :, :, 0] /= lf3d.shape[3]
        ten[:, :, :, 1] /= lf3d.shape[3]
        ten[:, :, :, 2] /= lf3d.shape[3]


    return ten, temp

#============================================================================================================
#=========                                Vertical LF computation                                ===========
#============================================================================================================

def compute_vertical(lf3d, config):

    # print("compute vertical shift {0}".format(shift))
    # tmp = np.copy(lf3d)
    # tmp = lfhelpers.refocus_3d(tmp, shift, 'v')
    # print("shape of vertical light field: " + str(tmp.shape))
    #
    # if config.color_space:
    #     lf3d = prefilter.changeColorSpace(tmp, config.color_space)
    # else:
    #     lf3d = tmp

    gaussianpre = vigra.filters.gaussianKernel(config.prefilter_scale)
    gaussianInner = vigra.filters.gaussianKernel(config.inner_scale)
    gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)

    K = np.array([-1, 0, 1]) / 2.0
    scharr1dim = vigra.filters.Kernel1D()
    scharr1dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    #BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr1dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    K = np.array([3, 10, 3]) / 16.0
    scharr2dim = vigra.filters.Kernel1D()
    scharr2dim.initExplicitly(-1, 1, K)
    #Border Treatments:
    #BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    scharr2dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    GD = vigra.filters.Kernel1D()
    GD.initGaussianDerivative(config.prefilter_scale,1)
    #Border Treatments:
    ##BORDER_TREATMENT_AVOID, BORDER_TREATMENT_REPEAT, BORDER_TREATMENT_REFLECT, BORDER_TREATMENT_ZEROPAD, BORDER_TREATMENT_WARP
    GD.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

    grad = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
    ten = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
    temp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2, 3), dtype=np.float32)

    if(config.structure_tensor_type == "classic"):
        for i in range(lf3d.shape[3]):
            if (config.prefilter == "True"):
                ### Prefilter ###
                # print("apply gaussian derivative prefilter along 2rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, GD)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianpre)

            ### Derivative computation ###
            # print("apply Gaussian derivative filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, GD)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 1, gaussianInner)
            # print("apply Gaussian derivative filter along 2rd dimension")
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, GD)
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, :, 1], 0, gaussianInner)

            ten[:, :, :, 0] += grad[:, :, :, 0]**2
            ten[:, :, :, 1] += grad[:, :, :, 1]*grad[:, :, :, 0]
            ten[:, :, :, 2] += grad[:, :, :, 1]**2

            temp[:, :, :, 0, i] = grad[:, :, :, 0]
            temp[:, :, :, 1, i] = grad[:, :, :, 1]

        ten[:, :, :, 0] /= lf3d.shape[3]
        ten[:, :, :, 1] /= lf3d.shape[3]
        ten[:, :, :, 2] /= lf3d.shape[3]


    if(config.structure_tensor_type == "scharr"):

        for i in range(lf3d.shape[3]):

            ### Inner gaussian filter ###
            # print("apply gaussian filter along 2rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, gaussianInner)
            # print("apply gaussian filter along 1rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner)
            # print("apply gaussian filter along 2rd dimension")
            # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
            if (config.prefilter == "True"):
                ### EPI prefilter ###
                # print("apply scharr pre-filter along 2rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, scharr1dim)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr2dim)

            ### Derivative computation ###
            # print("apply scharr filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr1dim)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 1, scharr2dim)
            # print("apply scharr filter along 2rd dimension")
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, scharr1dim)
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, :, 1], 0, scharr2dim)

            ten[:, :, :, 0] += grad[:, :, :, 0]**2
            ten[:, :, :, 1] += grad[:, :, :, 1]*grad[:, :, :, 0]
            ten[:, :, :, 2] += grad[:, :, :, 1]**2

            temp[:, :, :, 0, i] = grad[:, :, :, 0]
            temp[:, :, :, 1, i] = grad[:, :, :, 1]

        ten[:, :, :, 0] /= lf3d.shape[3]
        ten[:, :, :, 1] /= lf3d.shape[3]
        ten[:, :, :, 2] /= lf3d.shape[3]


    return ten, temp


#============================================================================================================
#=========                           Structure tensor processing chain                            ===========
#============================================================================================================


def structureTensor4D(config):

    if not config.result_path.endswith("/"):
        config.result_path += "/"
    if not config.result_label.endswith("/"):
        config.result_label += "/"
    if not os.path.isdir(config.result_path+config.result_label):
        os.makedirs(config.result_path+config.result_label)

    compute_h = False
    compute_v = False
    lf_shape = None
    lf3dh = None
    lf3dv = None

### Load data into two 3D volumes of the same shape ###

    if not config.path_4D.endswith("/"):
        config.path_4D += "/"
    print('Load 4D light field')
    lf4d_input = lfio.load_4d(config.path_4D, config.horizontalCameras, config.verticalCameras, config, rgb=config.rgb, roi=config.roi)


### Allocate memory for results ###

    orientationTrad = np.zeros((lf4d_input.shape[2], lf4d_input.shape[3]), dtype=np.float32)
    coherenceTrad = np.zeros((lf4d_input.shape[2], lf4d_input.shape[3]), dtype=np.float32)
    orientation4D = np.zeros((lf4d_input.shape[2], lf4d_input.shape[3]), dtype=np.float32)
    coherence4D = np.zeros((lf4d_input.shape[2], lf4d_input.shape[3]), dtype=np.float32)
    orientationCombined = np.zeros((lf4d_input.shape[2], lf4d_input.shape[3]), dtype=np.float32)
    coherenceCombined = np.zeros((lf4d_input.shape[2], lf4d_input.shape[3]), dtype=np.float32)

    logging.debug("Allocated memory!")

### compute both directions independent from each other ###

    for shift in config.global_shifts:
        print('Shift: ' + str(shift))

        tmp = np.copy(lf4d_input)
        tmp = lfhelpers.refocus_4d(tmp, shift, config)
        print("shape of horizontal light field: " + str(tmp.shape))
        lf4d = tmp

        lf3dh = lf4d[4,:,:,:,:].copy()
        lf3dv = lf4d[:,4,:,:,:].copy()
        lf_shape = lf3dh.shape

        gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)

        print("compute horizontal LightField")
        strTensorh, temph = Compute(lf3dh, config, direction='h')
        # print("apply gaussian filter along 3rd dimension")
        strTensorh[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 0], 2, gaussianOuter)
        strTensorh[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 1], 2, gaussianOuter)
        strTensorh[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 2], 2, gaussianOuter)

        # print("apply gaussian filter along 1rd dimension")
        strTensorh[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 0], 0, gaussianOuter)
        strTensorh[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 1], 0, gaussianOuter)
        strTensorh[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 2], 0, gaussianOuter)


        print("compute vertical LightField")
        strTensorv, tempv = Compute(lf3dv, config, direction='v')
        # print("apply gaussian filter along 3rd dimension")
        strTensorv[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 0], 1, gaussianOuter)
        strTensorv[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 1], 1, gaussianOuter)
        strTensorv[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 2], 1, gaussianOuter)

        # print("apply gaussian filter along 1rd dimension")
        strTensorv[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 0], 0, gaussianOuter)
        strTensorv[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 1], 0, gaussianOuter)
        strTensorv[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 2], 0, gaussianOuter)

        orientationL, coherenceL, coherenceH, coherenceV= orientationClassic(strTensorh, strTensorv, config,shift)
        orientationTrad, coherenceTrad = mergeOrientations_wta(orientationTrad, coherenceTrad, orientationL, coherenceL)


        print("compute 4D LightField")
        derivativeX, derivativeY = Compute4D(lf4d, config)

        orientationR, coherenceR = orientationCompute4D(derivativeX, derivativeY, config, shift)
        orientation4D, coherence4D = mergeOrientations_wta(orientation4D, coherence4D, orientationR, coherenceR)


        if config.output_level >= 3:
            plt.imsave(config.result_path+config.result_label+"orientation_global_with_Coherence_Merge_{0}.png".format(shift), orientationTrad, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"coherence_global_with_Coherence_Merge_{0}.png".format(shift), coherenceTrad, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"orientation_global_4D_with_Coherence_Merge_{0}.png".format(shift), orientation4D, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"coherence_global_4D_with_Coherence_Merge_{0}.png".format(shift), coherence4D, cmap=plt.cm.jet)


    orientation4D[:] += config.disparity_offset
    orientationTrad[:] += config.disparity_offset
    orientationCombined[:] += config.disparity_offset

    invalids = np.where(coherence4D < config.coherence_threshold)
    orientation4D[invalids] = 0
    coherence4D[invalids] = 0

    invalids = np.where(coherenceTrad < config.coherence_threshold)
    orientationTrad[invalids] = 0
    coherenceTrad[invalids] = 0

    orientationCombined[:] = orientation4D[:]
    coherenceCombined[:] = coherence4D[:]
    invalids = np.where(coherence4D < 0.95)
    orientationCombined[invalids] = orientationTrad[invalids]
    coherenceCombined[invalids] = coherenceTrad[invalids]



    # if isinstance(config.selective_gaussian, float) and config.selective_gaussian > 0:
    #     print "apply masked gauss..."
    #     mask = coherence4D[:, :]
    #     cv = None
    #     if lf3dh is not None:
    #         if lf_shape[3] == 3:
    #             cv = 0.298*lf3dh[lf_shape[0]/2, :, :, 0]+0.5870*lf3dh[lf_shape[0]/2, :, :, 1]+0.1141*lf3dh[lf_shape[0]/2, :, :, 2]
    #         else:
    #             cv = lf3dh[lf_shape[0]/2, :, :, 0]
    #     elif lf3dv is not None:
    #         if lf_shape[3] == 3:
    #             cv = 0.298*lf3dv[lf_shape[0]/2, :, :, 0]+0.5870*lf3dv[lf_shape[0]/2, :, :, 1]+0.1141*lf3dv[lf_shape[0]/2, :, :, 2]
    #         else:
    #             cv = lf3dv[lf_shape[0]/2, :, :, 0]
    #
    #     borders = vigra.filters.gaussianGradientMagnitude(cv, 1.6)
    #     borders /= np.amax(borders)
    #     mask *= 1.0-borders
    #     mask /= np.amax(mask)
    #     gauss = vigra.filters.Kernel2D()
    #     vigra.filters.Kernel2D.initGaussian(gauss, config.selective_gaussian)
    #     gauss.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_CLIP)
    #     orientation4D = vigra.filters.normalizedConvolveImage(orientation4D, mask, gauss)


    # if isinstance(config.selective_gaussian, float) and config.selective_gaussian > 0:
    #     print "apply masked gauss..."
    #     mask = coherenceTrad[:, :]
    #     cv = None
    #     if lf3dh is not None:
    #         if lf_shape[3] == 3:
    #             cv = 0.298*lf3dh[lf_shape[0]/2, :, :, 0]+0.5870*lf3dh[lf_shape[0]/2, :, :, 1]+0.1141*lf3dh[lf_shape[0]/2, :, :, 2]
    #         else:
    #             cv = lf3dh[lf_shape[0]/2, :, :, 0]
    #     elif lf3dv is not None:
    #         if lf_shape[3] == 3:
    #             cv = 0.298*lf3dv[lf_shape[0]/2, :, :, 0]+0.5870*lf3dv[lf_shape[0]/2, :, :, 1]+0.1141*lf3dv[lf_shape[0]/2, :, :, 2]
    #         else:
    #             cv = lf3dv[lf_shape[0]/2, :, :, 0]
    #
    #     borders = vigra.filters.gaussianGradientMagnitude(cv, 1.6)
    #     borders /= np.amax(borders)
    #     mask *= 1.0-borders
    #     mask /= np.amax(mask)
    #     gauss = vigra.filters.Kernel2D()
    #     vigra.filters.Kernel2D.initGaussian(gauss, config.selective_gaussian)
    #     gauss.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_CLIP)
    #     orientationTrad = vigra.filters.normalizedConvolveImage(orientationTrad, mask, gauss)

    if isinstance(config.median, int) and config.median > 0:
        print "apply median filter ..."
        orientationTrad = median_filter(orientationTrad, config.median)
        orientation4D = median_filter(orientation4D, config.median)
        orientationCombined = median_filter(orientationCombined, config.median)


    # if isinstance(config.selective_gaussian, float) and config.selective_gaussian > 0:
    #     print "apply masked gauss..."
    #     mask = coherenceTrad[:, :]
    #     cv = None
    #     if lf3dh is not None:
    #         if lf_shape[3] == 3:
    #             cv = 0.298*lf3dh[lf_shape[0]/2, :, :, 0]+0.5870*lf3dh[lf_shape[0]/2, :, :, 1]+0.1141*lf3dh[lf_shape[0]/2, :, :, 2]
    #         else:
    #             cv = lf3dh[lf_shape[0]/2, :, :, 0]
    #     elif lf3dv is not None:
    #         if lf_shape[3] == 3:
    #             cv = 0.298*lf3dv[lf_shape[0]/2, :, :, 0]+0.5870*lf3dv[lf_shape[0]/2, :, :, 1]+0.1141*lf3dv[lf_shape[0]/2, :, :, 2]
    #         else:
    #             cv = lf3dv[lf_shape[0]/2, :, :, 0]
    #
    #     borders = vigra.filters.gaussianGradientMagnitude(cv, 1.6)
    #     borders /= np.amax(borders)
    #     mask *= 1.0-borders
    #     mask /= np.amax(mask)
    #     gauss = vigra.filters.Kernel2D()
    #     vigra.filters.Kernel2D.initGaussian(gauss, config.selective_gaussian)
    #     gauss.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_CLIP)
    #     orientationTrad = vigra.filters.normalizedConvolveImage(orientationTrad, mask, gauss)
    #     orientation4D = vigra.filters.normalizedConvolveImage(orientation4D, mask, gauss)

    if config.output_level >= 1:
        plt.imsave(config.result_path+config.result_label+"orientation2D_final.png", orientationTrad[:,:], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence2D_final.png", coherenceTrad[:,:], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"orientation4D_final.png", orientation4D[:,:], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence4D_final.png", coherence4D[:,:], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"orientationCombo_final.png", orientationCombined[:,:], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherenceCombo_final.png", coherenceCombined[:,:], cmap=plt.cm.jet)

    logging.info("Computed final disparity map!")

## Light field computation has to be changed just to compute the core of the disparity and just transfer it here to the disparity map

    depth = dtc.disparity_to_depth(orientationTrad[:, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)
    mask = coherenceTrad[:, :]

    invalids = np.where(mask == 0)
    depth[invalids] = 0

    if config.output_level >= 2:
        plt.imsave(config.result_path+config.result_label+"depth_final2D.png", depth, cmap=plt.cm.jet)

    tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
    tmp[:, :, 0] = orientationTrad[:]
    tmp[:, :, 1] = coherenceTrad[:]
    tmp[:, :, 2] = depth[:]
    vim = vigra.RGBImage(tmp)
    vim.writeImage(config.result_path+config.result_label+"final2D.exr")



    tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
    tmp[:, :, 0] = orientationCombined[:]
    tmp[:, :, 1] = coherenceCombined[:]
    vim = vigra.RGBImage(tmp)
    vim.writeImage(config.result_path+config.result_label+"finalCombo.exr")


## Light field computation has to be changed just to compute the core of the disparity and just transfer it here to the disparity map

    depth = dtc.disparity_to_depth(orientation4D[:, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)
    mask = coherence4D[:, :]

    invalids = np.where(mask == 0)
    depth[invalids] = 0

    if config.output_level >= 2:
        plt.imsave(config.result_path+config.result_label+"depth_final4D.png", depth, cmap=plt.cm.jet)

    tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
    tmp[:, :, 0] = orientation4D[:]
    tmp[:, :, 1] = coherence4D[:]
    tmp[:, :, 2] = depth[:]
    vim = vigra.RGBImage(tmp)
    vim.writeImage(config.result_path+config.result_label+"final4D.exr")
    #
    # tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
    # tmp[:, :, 0] = orientationCombined[:]
    # tmp[:, :, 1] = coherenceCombined[:]
    # vim = vigra.RGBImage(tmp)
    # vim.writeImage(config.result_path+config.result_label+"final_combo.exr")

    # if config.output_level >= 2:
    #     if isinstance(config.centerview_path, str):
    #         color = misc.imread(config.centerview_path)
    #         if isinstance(config.roi, type({})):
    #             sposx = config.roi["pos"][0]
    #             eposx = config.roi["pos"][0] + config.roi["size"][0]
    #             sposy = config.roi["pos"][1]
    #             eposy = config.roi["pos"][1] + config.roi["size"][1]
    #             color = color[sposx:eposx, sposy:eposy, 0:3]
    #
    #     print "make pointcloud...",
    #     if isinstance(color, np.ndarray):
    #         dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, color=color, focal_length=config.focal_length)
    #     else:
    #         dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, focal_length=config.focal_length)
    #     print "ok"


#============================================================================================================
#=========                         Config settings class and log                                  ===========
#============================================================================================================

class Config:
    def __init__(self):

        self.result_path = None                 # path to store the results
        self.result_label = None                # name of the results folder

        self.path_horizontal = None             # path to the horizontal images [optional]
        self.path_vertical = None               # path to the vertical images [optional]

        self.roi = None                         # region of interest

        self.centerview_path = None             # path to the center view image to get color for pointcloud [optional]

        self.structure_tensor_type = "classic"  # type of the structure tensor class to be used
        self.inner_scale = 0.6                  # structure tensor inner scale
        self.outer_scale = 0.9
        self.coherence_threshold = 0.7          # if coherence less than value the disparity is set to invalid
        self.focal_length = 5740.38             # focal length in pixel [default Nikon D800 f=28mm]
        self.global_shifts = [0]                # list of horopter shifts in pixel
        self.base_line = 0.001                  # camera baseline

        self.color_space = prefilter.COLORSPACE.RGB         # colorscape to convert the images into [RGB,LAB,LUV]
        self.prefilter_scale = 0.4                          # scale of the prefilter
        self.prefilter = "False"          # type of the prefilter [NO,IMGD, EPID, IMGD2, EPID2]

        self.median = 5                         # apply median filter on disparity map
        self.nonlinear_diffusion = [0.5, 5]     # apply nonlinear diffusion [0] edge threshold, [1] scale
        self.selective_gaussian = 2.0           # apply a selective gaussian post filter
        self.tv = {"alpha": 1.0, "steps": 1000} # apply total variation to depth map

        self.min_depth = 0.01                   # minimum depth possible
        self.max_depth = 1.0                    # maximum depth possible

        self.rgb = True                         # forces grayscale if False

        self.output_level = 2                   # level of detail for file output possible 1,2,3

    def saveLog(self, filename=None):
        if filename is not None:
            f = open(filename, "w")
        else:
            f = open(self.result_path+self.result_label+"/log.txt", "w")
        f.write("roi : "); f.write(str(self.roi)+"\n")
        f.write("inner_scale : "); f.write(str(self.inner_scale)+"\n")
        f.write("coherence_threshold : "); f.write(str(self.coherence_threshold)+"\n")
        f.write("focal_length : "); f.write(str(self.focal_length)+"\n")
        f.write("global_shifts : "); f.write(str(self.global_shifts)+"\n")
        f.write("base_line : "); f.write(str(self.base_line)+"\n")
        f.write("color_space : "); f.write(str(self.color_space)+"\n")
        f.write("prefilter_scale : "); f.write(str(self.prefilter_scale)+"\n")
        f.write("prefilter : "); f.write(str(self.prefilter)+"\n")
        f.write("median : "); f.write(str(self.median)+"\n")
        f.write("nonlinear_diffusion : "); f.write(str(self.nonlinear_diffusion)+"\n")
        f.write("selective_gaussian : "); f.write(str(self.selective_gaussian)+"\n")
        f.write("total variation : "); f.write(str(self.tv)+"\n")
        f.write("min_depth : "); f.write(str(self.min_depth)+"\n")
        f.write("max_depth : "); f.write(str(self.max_depth)+"\n")
        f.close()




