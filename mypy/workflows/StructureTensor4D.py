import os
import logging
import vigra
import numpy as np
import pylab as plt
import scipy.misc as misc

from mypy.lightfield import io as lfio
import mypy.lightfield.depthToCloud as dtc
import mypy.lightfield.helpers as lfhelpers
import mypy.lightfield.depth.prefilter as prefilter
from scipy.ndimage import median_filter

#============================================================================================================
#=========                                       LF processing                                   ===========
#============================================================================================================

def Compute(lf3d, shift, config, direction):

    if direction == 'h':
        tensor, temp = compute_horizontal(lf3d, shift, config)
    if direction == 'v':
        tensor, temp = compute_vertical(lf3d, shift, config)

    return tensor, temp

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

        print(orientation.shape)

        if config.output_level >= 4:
            plt.imsave(config.result_path+config.result_label+"orientation_merged_local_{0}.png".format(shift), orientation, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"coherence_merged_local_{0}.png".format(shift), coherence, cmap=plt.cm.jet)

        return orientation, coherence, coherenceH, coherenceV

def orientationCompute4D(gradh, gradv, config, shift,coherenceH, coherenceV):

        tenh = np.zeros((gradh.shape[0], gradh.shape[1], gradh.shape[2], 3), dtype=np.float32)
        tenv = np.zeros((gradh.shape[0], gradh.shape[1], gradh.shape[2], 3), dtype=np.float32)
        gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)

        A = np.where(coherenceH > coherenceV )
        B = np.where(coherenceV > coherenceH )
        print("coeherece")
        print (coherenceH.shape)

        for i in range(gradh.shape[3]):

            # tenh[:, :, :, 0] += gradh[:, :, :, 0, i]**2
            # tenh[:, :, :, 1] += gradh[:, :, :, 1, i]*gradh[:, :, :, 0, i]
            # tenh[:, :, :, 2] += gradh[:, :, :, 1, i]**2
            tenh[:, :, :, 0] += (gradh[:, :, :, 0, i] + gradv[:, :, :, 0, i])**2
            tenh[:, :, :, 1] += ((gradh[:, :, :, 0, i] + gradv[:, :, :, 0, i])*(gradh[:, :, :, 1, i] + gradv[:, :, :, 1, i]))
            tenh[:, :, :, 2] += (gradh[:, :, :, 1, i] + gradv[:, :, :, 1, i])**2
            # tenv[:, :, :, 0] += gradv[:, :, :, 0, i]**2
            # tenv[:, :, :, 1] += gradv[:, :, :, 1, i]*gradv[:, :, :, 0, i]
            # tenv[:, :, :, 2] += gradv[:, :, :, 1, i]**2

        tenh[:, :, :, 0] /= gradh.shape[3]
        tenh[:, :, :, 1] /= gradh.shape[3]
        tenh[:, :, :, 2] /= gradh.shape[3]

        # tenv[:, :, :, 0] /= gradh.shape[3]
        # tenv[:, :, :, 1] /= gradh.shape[3]
        # tenv[:, :, :, 2] /= gradh.shape[3]

        tenh[:, :, :, 0] = vigra.filters.convolveOneDimension(tenh[:, :, :, 0], 2, gaussianOuter)
        tenh[:, :, :, 1] = vigra.filters.convolveOneDimension(tenh[:, :, :, 1], 2, gaussianOuter)
        tenh[:, :, :, 2] = vigra.filters.convolveOneDimension(tenh[:, :, :, 2], 2, gaussianOuter)

        tenh[:, :, :, 0] = vigra.filters.convolveOneDimension(tenh[:, :, :, 0], 0, gaussianOuter)
        tenh[:, :, :, 1] = vigra.filters.convolveOneDimension(tenh[:, :, :, 1], 0, gaussianOuter)
        tenh[:, :, :, 2] = vigra.filters.convolveOneDimension(tenh[:, :, :, 2], 0, gaussianOuter)

        tenh[:, :, :, 0] = vigra.filters.convolveOneDimension(tenh[:, :, :, 0], 1, gaussianOuter)
        tenh[:, :, :, 1] = vigra.filters.convolveOneDimension(tenh[:, :, :, 1], 1, gaussianOuter)
        tenh[:, :, :, 2] = vigra.filters.convolveOneDimension(tenh[:, :, :, 2], 1, gaussianOuter)

        # tenv[:, :, :, 0] = vigra.filters.convolveOneDimension(tenv[:, :, :, 0], 2, gaussianOuter)
        # tenv[:, :, :, 1] = vigra.filters.convolveOneDimension(tenv[:, :, :, 1], 2, gaussianOuter)
        # tenv[:, :, :, 2] = vigra.filters.convolveOneDimension(tenv[:, :, :, 2], 2, gaussianOuter)
        #
        # tenv[:, :, :, 0] = vigra.filters.convolveOneDimension(tenv[:, :, :, 0], 0, gaussianOuter)
        # tenv[:, :, :, 1] = vigra.filters.convolveOneDimension(tenv[:, :, :, 1], 0, gaussianOuter)
        # tenv[:, :, :, 2] = vigra.filters.convolveOneDimension(tenv[:, :, :, 2], 0, gaussianOuter)

        # orientation, coherence, coherenceH, coherenceV= orientationClassic(tenh, tenv, config, shift)
        orientation, coherence = evaluateStructureTensor(tenh[tenh.shape[0]/2,:,:,:])
        orientation += shift


        return orientation, coherence

#============================================================================================================
#=========                              Horizontal LF computation                                ===========
#============================================================================================================

def compute_horizontal(lf3dh, shift, config):

    print("compute horizontal shift {0}".format(shift))
    tmp = np.copy(lf3dh)
    tmp = lfhelpers.refocus_3d(tmp, shift, 'h')
    print("shape of horizontal light field: " + str(tmp.shape))

    if config.color_space:
        lf3d = prefilter.changeColorSpace(tmp, config.color_space)
    else:
        lf3d = tmp

    gaussianpre = vigra.filters.gaussianKernel(config.prefilter_scale)
    gaussianInner = vigra.filters.gaussianKernel(config.inner_scale)
    gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)

    K = np.array([-1, 0, 1]) / 2.0
    scharr1dim = vigra.filters.Kernel1D()
    scharr1dim.initExplicitly(-1, 1, K)

    K = np.array([3, 10, 3]) / 16.0
    scharr2dim = vigra.filters.Kernel1D()
    scharr2dim.initExplicitly(-1, 1, K)

    GD = vigra.filters.Kernel1D()
    GD.initGaussianDerivative(config.prefilter_scale, 1)

    grad = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
    ten = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
    temp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2, 3), dtype=np.float32)

    if(config.structure_tensor_type == "classic"):
        for i in range(lf3d.shape[3]):
            if (config.prefilter == "True"):
                ### Prefilter ###
                print("apply gaussian derivative prefilter along 3rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, GD)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianpre)

            ### Derivative computation ###
            print("apply Gaussian derivative filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, GD)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 2, gaussianInner)
            print("apply Gaussian derivative filter along 3rd dimension")
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
            print("apply gaussian filter along 3rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, gaussianInner)
            print("apply gaussian filter along 1rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner)
            # print("apply gaussian filter along 2rd dimension")
            # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
            if (config.prefilter == "True"):
                ### EPI prefilter ###
                print("apply scharr pre-filter along 3rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, scharr1dim)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr2dim)

            ### Derivative computation ###
            print("apply scharr filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr1dim)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 2, scharr2dim)
            print("apply scharr filter along 3rd dimension")
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

def compute_vertical(lf3dv, shift, config):

    print("compute vertical shift {0}".format(shift))
    tmp = np.copy(lf3dv)
    tmp = lfhelpers.refocus_3d(tmp, shift, 'v')
    print("shape of vertical light field: " + str(tmp.shape))

    if config.color_space:
        lf3d = prefilter.changeColorSpace(tmp, config.color_space)
    else:
        lf3d = tmp

    gaussianpre = vigra.filters.gaussianKernel(config.prefilter_scale)
    gaussianInner = vigra.filters.gaussianKernel(config.inner_scale)
    gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)

    K = np.array([-1, 0, 1]) / 2.0
    scharr1dim = vigra.filters.Kernel1D()
    scharr1dim.initExplicitly(-1, 1, K)

    K = np.array([3, 10, 3]) / 16.0
    scharr2dim = vigra.filters.Kernel1D()
    scharr2dim.initExplicitly(-1, 1, K)

    GD = vigra.filters.Kernel1D()
    GD.initGaussianDerivative(config.prefilter_scale,1)

    grad = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
    ten = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
    temp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2, 3), dtype=np.float32)

    if(config.structure_tensor_type == "classic"):
        for i in range(lf3d.shape[3]):
            if (config.prefilter == "True"):
                ### Prefilter ###
                print("apply gaussian derivative prefilter along 2rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, GD)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianpre)

            ### Derivative computation ###
            print("apply Gaussian derivative filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, GD)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 1, gaussianInner)
            print("apply Gaussian derivative filter along 2rd dimension")
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
            print("apply gaussian filter along 2rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, gaussianInner)
            print("apply gaussian filter along 1rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner)
            # print("apply gaussian filter along 2rd dimension")
            # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
            if (config.prefilter == "True"):
                ### EPI prefilter ###
                print("apply scharr pre-filter along 2rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, scharr1dim)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr2dim)

            ### Derivative computation ###
            print("apply scharr filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr1dim)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 1, scharr2dim)
            print("apply scharr filter along 2rd dimension")
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

    try:
        if not config.path_horizontal.endswith("/"):
            config.path_horizontal += "/"
        print('Load horizontal light field')
        lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
        lf_shape = lf3dh.shape
        compute_h = True
    except:
        logging.error("Could not load Data")

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        print('Load vertical light field')
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        lf_shape = lf3dv.shape
        compute_v = True

    except:
        logging.error("Could not load Data")

### Allocate memory for results ###

    orientationTrad = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
    coherenceTrad = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
    orientation4D = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
    coherence4D = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
    logging.debug("Allocated memory!")

### compute both directions independent from each other ###

    for shift in config.global_shifts:
        print('Shift: ' + str(shift))

        gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)

        if compute_h:
            print("compute horizontal LightField")
            strTensorh, temph = Compute(lf3dh, shift, config, direction='h')
            print("apply gaussian filter along 3rd dimension")
            strTensorh[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 0], 2, gaussianOuter)
            strTensorh[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 1], 2, gaussianOuter)
            strTensorh[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 2], 2, gaussianOuter)

            print("apply gaussian filter along 1rd dimension")
            strTensorh[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 0], 0, gaussianOuter)
            strTensorh[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 1], 0, gaussianOuter)
            strTensorh[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorh[:, :, :, 2], 0, gaussianOuter)

        if compute_v:
            print("compute vertical LightField")
            strTensorv, tempv = Compute(lf3dv, shift, config, direction='v')
            print("apply gaussian filter along 3rd dimension")
            strTensorv[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 0], 1, gaussianOuter)
            strTensorv[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 1], 1, gaussianOuter)
            strTensorv[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 2], 1, gaussianOuter)

            print("apply gaussian filter along 1rd dimension")
            strTensorv[:, :, :, 0] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 0], 0, gaussianOuter)
            strTensorv[:, :, :, 1] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 1], 0, gaussianOuter)
            strTensorv[:, :, :, 2] = vigra.filters.convolveOneDimension(strTensorv[:, :, :, 2], 0, gaussianOuter)

        orientationL, coherenceL, coherenceH, coherenceV= orientationClassic(strTensorh, strTensorv, config,shift)
        orientationTrad, coherenceTrad = mergeOrientations_wta(orientationTrad, coherenceTrad, orientationL, coherenceL)

        if config.output_level >= 3:
            plt.imsave(config.result_path+config.result_label+"orientation_global_with_Coherence_Merge_{0}.png".format(shift), orientationTrad, cmap=plt.cm.jet)
            plt.imsave(config.result_path+config.result_label+"coherence_global_with_Coherence_Merge_{0}.png".format(shift), coherenceTrad, cmap=plt.cm.jet)


        #orientationR, coherenceR = orientationClassic(tenh, tenv, config,shift)
        orientationR, coherenceR = orientationCompute4D(temph, tempv, config,shift,coherenceH, coherenceV)
        orientation4D, coherence4D = mergeOrientations_wta(orientation4D, coherence4D, orientationR, coherenceR)




    invalids = np.where(coherence4D < config.coherence_threshold)
    orientation4D[invalids] = 0
    coherence4D[invalids] = 0

    invalids = np.where(coherenceTrad < config.coherence_threshold)
    orientationTrad[invalids] = 0
    coherenceTrad[invalids] = 0

    if isinstance(config.median, int) and config.median > 0:
        print "apply median filter ..."
        orientationTrad = median_filter(orientationTrad, config.median)
        orientation4D = median_filter(orientation4D, config.median)

    if isinstance(config.selective_gaussian, float) and config.selective_gaussian > 0:
        print "apply masked gauss..."
        mask = coherenceTrad[:, :]
        cv = None
        if lf3dh is not None:
            if lf_shape[3] == 3:
                cv = 0.298*lf3dh[lf_shape[0]/2, :, :, 0]+0.5870*lf3dh[lf_shape[0]/2, :, :, 1]+0.1141*lf3dh[lf_shape[0]/2, :, :, 2]
            else:
                cv = lf3dh[lf_shape[0]/2, :, :, 0]
        elif lf3dv is not None:
            if lf_shape[3] == 3:
                cv = 0.298*lf3dv[lf_shape[0]/2, :, :, 0]+0.5870*lf3dv[lf_shape[0]/2, :, :, 1]+0.1141*lf3dv[lf_shape[0]/2, :, :, 2]
            else:
                cv = lf3dv[lf_shape[0]/2, :, :, 0]

        borders = vigra.filters.gaussianGradientMagnitude(cv, 1.6)
        borders /= np.amax(borders)
        mask *= 1.0-borders
        mask /= np.amax(mask)
        gauss = vigra.filters.Kernel2D()
        vigra.filters.Kernel2D.initGaussian(gauss, config.selective_gaussian)
        gauss.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_CLIP)
        orientationTrad = vigra.filters.normalizedConvolveImage(orientationTrad, mask, gauss)
        orientation4D = vigra.filters.normalizedConvolveImage(orientation4D, mask, gauss)

    if config.output_level >= 1:
        plt.imsave(config.result_path+config.result_label+"orientation2D_final.png", orientationTrad[:,:], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence2D_final.png", coherenceTrad[:,:], cmap=plt.cm.jet)

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


    if config.output_level >= 1:
        plt.imsave(config.result_path+config.result_label+"orientation4D_final.png", orientation4D[:,:], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence4D_final.png", coherence4D[:,:], cmap=plt.cm.jet)


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


    if config.output_level >= 2:
        if isinstance(config.centerview_path, str):
            color = misc.imread(config.centerview_path)
            if isinstance(config.roi, type({})):
                sposx = config.roi["pos"][0]
                eposx = config.roi["pos"][0] + config.roi["size"][0]
                sposy = config.roi["pos"][1]
                eposy = config.roi["pos"][1] + config.roi["size"][1]
                color = color[sposx:eposx, sposy:eposy, 0:3]

        print "make pointcloud...",
        if isinstance(color, np.ndarray):
            dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, color=color, focal_length=config.focal_length)
        else:
            dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, focal_length=config.focal_length)
        print "ok"


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


















# import os
# import logging
# import vigra
# import numpy as np
# import pylab as plt
# import scipy.misc as misc
# import mypy.lightfield.depth.prefilter as prefilter
# from mypy.lightfield import io as lfio
#
# import mypy.lightfield.depthToCloud as dtc
# from mypy.lightfield import helpers as lfhelpers
#
#
# def orientationClassic(strTensorh, strTensorv, config, shift):
#
#         orientationH, coherenceH = evaluateStructureTensor(strTensorh)
#
#         if config.output_level >= 4:
#             plt.imsave(config.result_path+config.result_label+"orientation_Horizontal_{0}.png".format(shift), orientationH, cmap=plt.cm.jet)
#             plt.imsave(config.result_path+config.result_label+"coherence_Horizontal_{0}.png".format(shift), coherenceH, cmap=plt.cm.jet)
#
#         orientationV, coherenceV = evaluateStructureTensor(strTensorv)
#
#         if config.output_level >= 4:
#             plt.imsave(config.result_path+config.result_label+"orientation_Vertical_{0}.png".format(shift), orientationV, cmap=plt.cm.jet)
#             plt.imsave(config.result_path+config.result_label+"coherence_Vertical_{0}.png".format(shift), coherenceV, cmap=plt.cm.jet)
#
#         orientation, coherence = mergeOrientations_wta(orientationH, coherenceH, orientationV, coherenceV)
#         orientation[:] += float(shift)
#
#         print(orientation.shape)
#
#         if config.output_level >= 4:
#             plt.imsave(config.result_path+config.result_label+"orientation_merged_local_{0}.png".format(shift), orientation, cmap=plt.cm.jet)
#             plt.imsave(config.result_path+config.result_label+"coherence_merged_local_{0}.png".format(shift), coherence, cmap=plt.cm.jet)
#
#         return orientation, coherence
#
# #============================================================================================================
# #=========                                       LF processing                                   ===========
# #============================================================================================================
#
# def Compute(lf3d, shift, config, direction):
#
#     if direction == 'h':
#         grad = compute_horizontal(lf3d, shift, config)
#     if direction == 'v':
#         grad = compute_vertical(lf3d, shift, config)
#
#     return grad
#
# def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
#     winner = np.where(coherence2 > coherence1)
#     orientation1[winner] = orientation2[winner]
#     coherence1[winner] = coherence2[winner]
#
#     return orientation1, coherence1
#
# def computeCoherence(tensor):
#
#     assert isinstance(tensor, np.ndarray)
#
#     ### compute coherence value ###
#     up = np.sqrt((tensor[:, :, 2]-tensor[:, :, 0])**2 + 4*tensor[:, :, 1]**2)
#     down = (tensor[:, :, 2]+tensor[:, :, 0] + 1e-25)
#     coherence = up / down
#
#     return coherence
#
# def evaluateStructureTensor(tensor):
#
#     assert isinstance(tensor, np.ndarray)
#
#     ### compute coherence value ###
#     up = np.sqrt((tensor[:, :, 2]-tensor[:, :, 0])**2 + 4*tensor[:, :, 1]**2)
#     down = (tensor[:, :, 2]+tensor[:, :, 0] + 1e-25)
#     coherence = up / down
#
#     ### compute disparity value ###
#     orientation = vigra.numpy.arctan2(2*tensor[:, :, 1], tensor[:, :, 2]-tensor[:, :, 0]) / 2.0
#     orientation = vigra.numpy.tan(orientation[:])
#
#     ### mark out of boundary orientation estimation ###
#     invalid_ubounds = np.where(orientation > 1.1)
#     invalid_lbounds = np.where(orientation < -1.1)
#
#     ### set coherence of invalid values to zero ###
#     coherence[invalid_ubounds] = 0
#     coherence[invalid_lbounds] = 0
#
#
#     ### set orientation of invalid values to related maximum/minimum value
#     orientation[invalid_ubounds] = 1.1
#     orientation[invalid_lbounds] = -1.1
#
#     return orientation, coherence
#
# #============================================================================================================
# #=========                              Horizontal LF computation                                ===========
# #============================================================================================================
#
# def compute_horizontal(lf3dh, shift, config):
#
#     print("compute horizontal shift {0}".format(shift))
#     tmp = np.copy(lf3dh)
#     tmp = lfhelpers.refocus_3d(tmp, shift, 'h')
#     print("shape of horizontal light field: " + str(tmp.shape))
#
#     if config.color_space:
#         lf3d = prefilter.changeColorSpace(tmp, config.color_space)
#     else:
#         lf3d = tmp
#
#     gaussianpre = vigra.filters.gaussianKernel(config.prefilter_scale)
#     gaussianInner = vigra.filters.gaussianKernel(config.inner_scale)
#     gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)
#
#     K = np.array([-1, 0, 1]) / 2.0
#     scharr1dim = vigra.filters.Kernel1D()
#     scharr1dim.initExplicitly(-1, 1, K)
#
#     K = np.array([3, 10, 3]) / 16.0
#     scharr2dim = vigra.filters.Kernel1D()
#     scharr2dim.initExplicitly(-1, 1, K)
#
#     GD = vigra.filters.Kernel1D()
#     GD.initGaussianDerivative(config.prefilter_scale, 1)
#
#     gradient = np.zeros((3, lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
#
#     if(config.structure_tensor_type == "classic"):
#         for i in range(lf3d.shape[3]):
#             if (config.prefilter == "True"):
#                 ### Prefilter ###
#                 print("apply gaussian derivative prefilter along 3rd dimension")
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, GD)
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianpre)
#
#             ### Derivative computation ###
#             print("apply Gaussian derivative filter along 1st dimension")
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, GD)
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 0], 2, gaussianInner)
#             print("apply Gaussian derivative filter along 3rd dimension")
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, GD)
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 1], 0, gaussianInner)
#
#
#     if(config.structure_tensor_type == "scharr"):
#
#         for i in range(lf3d.shape[3]):
#
#             ### Inner gaussian filter ###
#             print("apply gaussian filter along 3rd dimension")
#             lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, gaussianInner)
#             print("apply gaussian filter along 1rd dimension")
#             lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner)
#             # print("apply gaussian filter along 2rd dimension")
#             # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
#             if (config.prefilter == "True"):
#                 ### EPI prefilter ###
#                 print("apply scharr pre-filter along 3rd dimension")
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, scharr1dim)
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr2dim)
#
#             ### Derivative computation ###
#             print("apply scharr filter along 1st dimension")
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr1dim)
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 0], 2, scharr2dim)
#             print("apply scharr filter along 3rd dimension")
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, scharr1dim)
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 1], 0, scharr2dim)
#
#
#     return gradient
#
#
# #============================================================================================================
# #=========                                Vertical LF computation                                ===========
# #============================================================================================================
#
# def compute_vertical(lf3dv, shift, config):
#
#     print("compute vertical shift {0}".format(shift))
#     tmp = np.copy(lf3dv)
#     tmp = lfhelpers.refocus_3d(tmp, shift, 'v')
#     print("shape of vertical light field: " + str(tmp.shape))
#
#     if config.color_space:
#         lf3d = prefilter.changeColorSpace(tmp, config.color_space)
#     else:
#         lf3d = tmp
#
#     gaussianpre = vigra.filters.gaussianKernel(config.prefilter_scale)
#     gaussianInner = vigra.filters.gaussianKernel(config.inner_scale)
#     gaussianOuter = vigra.filters.gaussianKernel(config.outer_scale)
#
#     K = np.array([-1, 0, 1]) / 2.0
#     scharr1dim = vigra.filters.Kernel1D()
#     scharr1dim.initExplicitly(-1, 1, K)
#
#     K = np.array([3, 10, 3]) / 16.0
#     scharr2dim = vigra.filters.Kernel1D()
#     scharr2dim.initExplicitly(-1, 1, K)
#
#     GD = vigra.filters.Kernel1D()
#     GD.initGaussianDerivative(config.prefilter_scale,1)
#
#     gradient = np.zeros((3, lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
#
#     if(config.structure_tensor_type == "classic"):
#         for i in range(lf3d.shape[3]):
#             if (config.prefilter == "True"):
#                 ### Prefilter ###
#                 print("apply gaussian derivative prefilter along 2rd dimension")
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, GD)
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianpre)
#
#             ### Derivative computation ###
#             print("apply Gaussian derivative filter along 1st dimension")
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, GD)
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 0], 1, gaussianInner)
#             print("apply Gaussian derivative filter along 2rd dimension")
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, GD)
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 1], 0, gaussianInner)
#
#     if(config.structure_tensor_type == "scharr"):
#
#         for i in range(lf3d.shape[3]):
#
#             ### Inner gaussian filter ###
#             print("apply gaussian filter along 2rd dimension")
#             lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, gaussianInner)
#             print("apply gaussian filter along 1rd dimension")
#             lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner)
#             # print("apply gaussian filter along 2rd dimension")
#             # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
#             if (config.prefilter == "True"):
#                 ### EPI prefilter ###
#                 print("apply scharr pre-filter along 2rd dimension")
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, scharr1dim)
#                 lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr2dim)
#
#             ### Derivative computation ###
#             print("apply scharr filter along 1st dimension")
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr1dim)
#             gradient[i, :, :, :, 0] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 0], 1, scharr2dim)
#             print("apply scharr filter along 2rd dimension")
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, scharr1dim)
#             gradient[i, :, :, :, 1] = vigra.filters.convolveOneDimension(gradient[i, :, :, :, 1], 0, scharr2dim)
#
#     return gradient
#
#
# #============================================================================================================
# #=========                           Structure tensor processing chain                            ===========
# #============================================================================================================
#
#
# def structureTensor4D(config):
#
#     if not config.result_path.endswith("/"):
#         config.result_path += "/"
#     if not config.result_label.endswith("/"):
#         config.result_label += "/"
#     if not os.path.isdir(config.result_path+config.result_label):
#         os.makedirs(config.result_path+config.result_label)
#
#     compute_h = False
#     compute_v = False
#     lf_shape = None
#     lf3dh = None
#     lf3dv = None
#
# ### Load data into two 3D volumes of the same shape ###
#
#     try:
#         if not config.path_horizontal.endswith("/"):
#             config.path_horizontal += "/"
#         print('Load horizontal light field')
#         lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
#         lf_shape = lf3dh.shape
#         compute_h = True
#     except:
#         logging.error("Could not load Data")
#
#     try:
#         if not config.path_vertical.endswith("/"):
#             config.path_vertical += "/"
#         print('Load vertical light field')
#         lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
#         lf_shape = lf3dv.shape
#         compute_v = True
#
#     except:
#         logging.error("Could not load Data")
#
# ### Allocate memory for results ###
#
#     orientation4D = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
#     coherence4D = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
#     orientation2D = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
#     coherence2D = np.zeros((lf_shape[1], lf_shape[2]), dtype=np.float32)
#     logging.debug("Allocated memory!")
#
# ### compute both directions independent from each other ###
#
#     for shift in config.global_shifts:
#         print('Shift: ' + str(shift))
#
#         strTensorh = None
#         strTensorv = None
#
#         if compute_h:
#             print("compute horizontal LightField")
#             gradh = Compute(lf3dh, shift, config, direction='h')
#         if compute_v:
#             print("compute vertical LightField")
#             gradv = Compute(lf3dv, shift, config, direction='v')
#
#         print("4D structure tensor")
#
#         tensorA = np.zeros((gradh.shape[2],gradh.shape[3],3),dtype=np.float32)
#         tensorB = np.zeros((gradh.shape[2],gradh.shape[3],3),dtype=np.float32)
#
#         for i in range(gradh.shape[0]):
#             Ix = gradh[i,gradh.shape[1]/2, :, :, 0]
#             Iy = gradv[i,gradv.shape[1]/2, :, :, 0]
#             Iu = gradh[i,gradh.shape[1]/2, :, :, 1]
#             Iv = gradv[i,gradv.shape[1]/2, :, :, 1]
#
#             tensorA[:,:,0] += Ix**2
#             tensorA[:,:,1] += Ix*Iu
#             tensorA[:,:,2] += Iu**2
#
#             tensorB[:,:,0] += Iy**2
#             tensorB[:,:,1] += Iy*Iv
#             tensorB[:,:,2] += Iv**2
#
#         tensorA[:,:,0] = vigra.filters.gaussianSmoothing(tensorA[:,:,0], config.outer_scale)
#         tensorA[:,:,1] = vigra.filters.gaussianSmoothing(tensorA[:,:,1], config.outer_scale)
#         tensorA[:,:,2] = vigra.filters.gaussianSmoothing(tensorA[:,:,2], config.outer_scale)
#         tensorB[:,:,0] = vigra.filters.gaussianSmoothing(tensorB[:,:,0], config.outer_scale)
#         tensorB[:,:,1] = vigra.filters.gaussianSmoothing(tensorB[:,:,1], config.outer_scale)
#         tensorB[:,:,2] = vigra.filters.gaussianSmoothing(tensorB[:,:,2], config.outer_scale)
#
#         orientation, coherence  = orientationClassic(tensorA, tensorB, config, shift)
#         orientation2D, coherence2D = mergeOrientations_wta(orientation2D, coherence2D, orientation, coherence)
#
#         ### compute coherence value ###
#         # up = np.sqrt((Ixx- Iuu)**2 + 4*(Ixu)**2)
#         # down = (Ixx+Iuu + 1e-25)
#         # cxu = up / down
#         #
#         # up = np.sqrt((Ivv-Iyy)**2 + 4*(Iyv)**2)
#         # down = (Ivv+Iyy + 1e-25)
#         # cyv = up / down
#         #
#         # A = np.where(cxu > cyv)
#         # B = np.where(cxu < cyv)
#
#         Ix_plus_Iy_square = np.zeros((gradh.shape[2],gradh.shape[3]),dtype=np.float32)
#         Iu_plus_Iv_square = np.zeros((gradh.shape[2],gradh.shape[3]),dtype=np.float32)
#         Ix_plus_Iy_mul_Iu_plus_Iv = np.zeros((gradh.shape[2],gradh.shape[3]),dtype=np.float32)
#
#         for i in range(gradh.shape[0]):
#             Ix = gradh[i,gradh.shape[1]/2, :, :, 0]
#             Iy = gradv[i,gradv.shape[1]/2, :, :, 0]
#             Iu = gradh[i,gradh.shape[1]/2, :, :, 1]
#             Iv = gradv[i,gradv.shape[1]/2, :, :, 1]
#
#             # Ix[B] = 0
#             # Iu[B] = 0
#             # Iy[A] = 0
#             # Iv[A] = 0
#
#             Ix_plus_Iy_square += (Ix + Iy)**2
#             Iu_plus_Iv_square += (Iu + Iv)**2
#             Ix_plus_Iy_mul_Iu_plus_Iv += (Ix + Iy)*(Iu + Iv)
#
#
#         Ix_plus_Iy_square = vigra.filters.gaussianSmoothing(Ix_plus_Iy_square, config.outer_scale)
#         Iu_plus_Iv_square = vigra.filters.gaussianSmoothing(Iu_plus_Iv_square, config.outer_scale)
#         Ix_plus_Iy_mul_Iu_plus_Iv = vigra.filters.gaussianSmoothing(Ix_plus_Iy_mul_Iu_plus_Iv, config.outer_scale)
#
#
#         ### compute coherence value ###
#         up = np.sqrt((Iu_plus_Iv_square[ :, :]-Ix_plus_Iy_square[ :, :])**2 + 4*Ix_plus_Iy_mul_Iu_plus_Iv[ :, :]**2)
#         down = (Iu_plus_Iv_square[ :, :]+Ix_plus_Iy_square[ :, :] + 1e-25)
#         coherence = up / down
#
#         ### compute disparity value ###
#         orientation = vigra.numpy.arctan2(2*Ix_plus_Iy_mul_Iu_plus_Iv[ :, :], Iu_plus_Iv_square[ :, :]-Ix_plus_Iy_square[ :, :]) / 2.0
#         orientation = vigra.numpy.tan(orientation[:])
#
#         ### mark out of boundary orientation estimation ###
#         invalid_ubounds = np.where(orientation > 1.1)
#         invalid_lbounds = np.where(orientation < -1.1)
#
#         ### set coherence of invalid values to zero ###
#         coherence[invalid_ubounds] = 0
#         coherence[invalid_lbounds] = 0
#
#         ### set orientation of invalid values to related maximum/minimum value
#         orientation[invalid_ubounds] = 1.1
#         orientation[invalid_lbounds] = -1.1
#
#         orientation += shift
#         # coherence4D = np.copy(coherence)
#
#         winner = np.where(coherence > coherence4D)
#         orientation4D[winner] = orientation[winner]
#         coherence4D[winner] = coherence[winner]
#
#         if config.output_level >= 3:
#             plt.imsave(config.result_path+config.result_label+"orientation_4D_{0}.png".format(shift), orientation4D[:,:], cmap=plt.cm.jet)
#             plt.imsave(config.result_path+config.result_label+"coherence_4D_{0}.png".format(shift), coherence4D[:,:], cmap=plt.cm.jet)
#
#     invalids = np.where(coherence4D < config.coherence_threshold)
#     orientation4D[invalids] = 0
#     coherence4D[invalids] = 0
#
#     invalids = np.where(coherence2D < config.coherence_threshold)
#     orientation2D[invalids] = 0
#     coherence2D[invalids] = 0
#
#
#     if config.output_level >= 1:
#         plt.imsave(config.result_path+config.result_label+"orientation2D_final.png", orientation2D[:,:], cmap=plt.cm.jet)
#         plt.imsave(config.result_path+config.result_label+"coherence2D_final.png", coherence2D[:,:], cmap=plt.cm.jet)
#         plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation4D[:,:], cmap=plt.cm.jet)
#         plt.imsave(config.result_path+config.result_label+"coherence_final.png", coherence4D[:,:], cmap=plt.cm.jet)
#
#     logging.info("Computed final disparity map!")
#
# ## Light field computation has to be changed just to compute the core of the disparity and just transfer it here to the disparity map
#
#     depth = dtc.disparity_to_depth(orientation4D[:, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)
#     mask = coherence[:, :]
#
#     invalids = np.where(mask == 0)
#     depth[invalids] = 0
#
#     if config.output_level >= 2:
#         plt.imsave(config.result_path+config.result_label+"depth4D_final.png", depth, cmap=plt.cm.jet)
#
#     tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
#     tmp[:, :, 0] = orientation4D[:]
#     tmp[:, :, 1] = coherence4D[:]
#     tmp[:, :, 2] = depth[:]
#     vim = vigra.RGBImage(tmp)
#     vim.writeImage(config.result_path+config.result_label+"final4D.exr")
#
#     depth = dtc.disparity_to_depth(orientation2D[:, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)
#     mask = coherence[:, :]
#
#     invalids = np.where(mask == 0)
#     depth[invalids] = 0
#
#     tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
#     tmp[:, :, 0] = orientation2D[:]
#     tmp[:, :, 1] = coherence2D[:]
#     tmp[:, :, 2] = depth[:]
#     vim = vigra.RGBImage(tmp)
#     vim.writeImage(config.result_path+config.result_label+"final2D.exr")
#
#     if config.output_level >= 2:
#         if isinstance(config.centerview_path, str):
#             color = misc.imread(config.centerview_path)
#             if isinstance(config.roi, type({})):
#                 sposx = config.roi["pos"][0]
#                 eposx = config.roi["pos"][0] + config.roi["size"][0]
#                 sposy = config.roi["pos"][1]
#                 eposy = config.roi["pos"][1] + config.roi["size"][1]
#                 color = color[sposx:eposx, sposy:eposy, 0:3]
#
#         print "make pointcloud...",
#         if isinstance(color, np.ndarray):
#             dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, color=color, focal_length=config.focal_length)
#         else:
#             dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, focal_length=config.focal_length)
#
#         print "ok"
#
#
#
# #============================================================================================================
# #=========                         Config settings class and log                                  ===========
# #============================================================================================================
#
# class Config:
#     def __init__(self):
#
#         self.result_path = None                 # path to store the results
#         self.result_label = None                # name of the results folder
#
#         self.path_horizontal = None             # path to the horizontal images [optional]
#         self.path_vertical = None               # path to the vertical images [optional]
#
#         self.roi = None                         # region of interest
#
#         self.centerview_path = None             # path to the center view image to get color for pointcloud [optional]
#
#         self.structure_tensor_type = "classic"  # type of the structure tensor class to be used
#         self.inner_scale = 0.6                  # structure tensor inner scale
#         self.outer_scale = 0.9
#         self.coherence_threshold = 0.7          # if coherence less than value the disparity is set to invalid
#         self.focal_length = 5740.38             # focal length in pixel [default Nikon D800 f=28mm]
#         self.global_shifts = [0]                # list of horopter shifts in pixel
#         self.base_line = 0.001                  # camera baseline
#
#         self.color_space = prefilter.COLORSPACE.RGB         # colorscape to convert the images into [RGB,LAB,LUV]
#         self.prefilter_scale = 0.4                          # scale of the prefilter
#         self.prefilter = prefilter.PREFILTER.NO          # type of the prefilter [NO,IMGD, EPID, IMGD2, EPID2]
#
#         self.median = 5                         # apply median filter on disparity map
#         self.nonlinear_diffusion = [0.5, 5]     # apply nonlinear diffusion [0] edge threshold, [1] scale
#         self.selective_gaussian = 2.0           # apply a selective gaussian post filter
#         self.tv = {"alpha": 1.0, "steps": 1000} # apply total variation to depth map
#
#         self.min_depth = 0.01                   # minimum depth possible
#         self.max_depth = 1.0                    # maximum depth possible
#
#         self.rgb = True                         # forces grayscale if False
#
#         self.output_level = 2                   # level of detail for file output possible 1,2,3
#
#     def saveLog(self, filename=None):
#         if filename is not None:
#             f = open(filename, "w")
#         else:
#             f = open(self.result_path+self.result_label+"/log.txt", "w")
#         f.write("roi : "); f.write(str(self.roi)+"\n")
#         f.write("inner_scale : "); f.write(str(self.inner_scale)+"\n")
#         f.write("outer_scale : "); f.write(str(self.outer_scale)+"\n")
#         f.write("double_tensor : "); f.write(str(self.double_tensor)+"\n")
#         f.write("coherence_threshold : "); f.write(str(self.coherence_threshold)+"\n")
#         f.write("focal_length : "); f.write(str(self.focal_length)+"\n")
#         f.write("global_shifts : "); f.write(str(self.global_shifts)+"\n")
#         f.write("base_line : "); f.write(str(self.base_line)+"\n")
#         f.write("color_space : "); f.write(str(self.color_space)+"\n")
#         f.write("prefilter_scale : "); f.write(str(self.prefilter_scale)+"\n")
#         f.write("prefilter : "); f.write(str(self.prefilter)+"\n")
#         f.write("median : "); f.write(str(self.median)+"\n")
#         f.write("nonlinear_diffusion : "); f.write(str(self.nonlinear_diffusion)+"\n")
#         f.write("selective_gaussian : "); f.write(str(self.selective_gaussian)+"\n")
#         f.write("total variation : "); f.write(str(self.tv)+"\n")
#         f.write("min_depth : "); f.write(str(self.min_depth)+"\n")
#         f.write("max_depth : "); f.write(str(self.max_depth)+"\n")
#         f.close()