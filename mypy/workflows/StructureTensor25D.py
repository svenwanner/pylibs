import os
import logging
import vigra
import numpy as np
import pylab as plt
import scipy.misc as misc
import mypy.lightfield.depth.prefilter as prefilter
from mypy.lightfield import io as lfio
from scipy.ndimage import median_filter

import mypy.pointclouds.depthToCloud as dtc
from mypy.lightfield import helpers as lfhelpers



#============================================================================================================
#=========                                       EPI processing                                   ===========
#============================================================================================================


def Compute(lf3d, shift, config, direction):

    if direction == 'h':
        orientation, coherence = compute_horizontal(lf3d, shift, config)
    if direction == 'v':
        orientation, coherence = compute_vertical(lf3d, shift, config)

    return orientation, coherence

def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
    winner = np.where(coherence2 > coherence1)
    orientation1[winner] = orientation2[winner]
    coherence1[winner] = coherence2[winner]

    return orientation1, coherence1

#============================================================================================================
#=========                              Horizontal EPI computation                                ===========
#============================================================================================================

def compute_horizontal(lf3dh, shift, config):

    logging.info("compute horizontal shift {0}".format(shift))
    tmp = np.copy(lf3dh)
    tmp = lfhelpers.refocus_3d(tmp, shift, 'h')
    logging.debug("New size of lf3dh after shifting to horoptor: " + str(tmp.shape))

    # if config.output_level == 4:
    #     for i in range(lf3d.shape[0]):
    #         misc.imsave(config.result_path+config.result_label+"horizontal_Input_shifted_{0}.png".format(i), lf3d[i ,: ,: ,:])

    if config.color_space:
        lf3d = prefilter.changeColorSpace(tmp, config.color_space)
    else:
        lf3d = tmp

    # if config.output_level == 4:
    #     for i in range(lf3d.shape[0]):
    #         misc.imsave(config.result_path+config.result_label+"horizontal_Input_shifted_color_space_changed_{0}.png".format(i), lf3d[i ,: ,: ,:])

    if(config.structure_tensor_type == "scharr"):

        gaussianInner1 = vigra.filters.gaussianKernel(config.inner_scale[0])
        gaussianInner2 = vigra.filters.gaussianKernel(config.inner_scale[1])
        gaussianInner3 = vigra.filters.gaussianKernel(config.inner_scale[2])
        gaussianOuter1 = vigra.filters.gaussianKernel(config.outer_scale[0])
        gaussianOuter2 = vigra.filters.gaussianKernel(config.outer_scale[1])
        gaussianOuter3 = vigra.filters.gaussianKernel(config.outer_scale[2])

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
        scharr1dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

        grad = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
        st3d_f = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)

###
        # tensor = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
        # st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
###
        tensor = np.zeros((lf3d.shape[0]*2-1, lf3d.shape[1]*2-1, lf3d.shape[2]*2-1, 3), dtype=np.float32)
        st3d = np.zeros((lf3d.shape[0]*2-1, lf3d.shape[1]*2-1, lf3d.shape[2]*2-1, 3), dtype=np.float32)
###

        for i in range(lf3d.shape[3]):

            ### Inner gaussian filter ###
            print("apply gaussian filter along 2rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, gaussianInner3)#Additional Smoothing
            print("apply gaussian filter along 3rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, gaussianInner2)
            print("apply gaussian filter along 1rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner1)
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
            print("apply scharr filter along 2rd dimension")
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, scharr1dim)
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, :, 1], 0, scharr2dim)

###Upsampling###
            pyramid0 = vigra.sampling.resize(grad[:, :, :, 0],shape=(grad.shape[0]*2-1, grad.shape[1]*2-1, grad.shape[2]*2-1), order = 3)
            pyramid1 = vigra.sampling.resize(grad[:, :, :, 1],shape=(grad.shape[0]*2-1, grad.shape[1]*2-1, grad.shape[2]*2-1), order = 3)
            tensor[:, :, :, 0] = pyramid0**2
            tensor[:, :, :, 1] = pyramid1*pyramid0
            tensor[:, :, :, 2] = pyramid1**2
###
            # tensor[:, :, :, 0] = grad[:, :, :, 0]**2
            # tensor[:, :, :, 1] = grad[:, :, :, 1]*grad[:, :, :, 0]
            # tensor[:, :, :, 2] = grad[:, :, :, 1]**2
###
            st3d[:, :, :, 0] += tensor[:, :, :, 0]
            st3d[:, :, :, 1] += tensor[:, :, :, 1]
            st3d[:, :, :, 2] += tensor[:, :, :, 2]

        st3d[:, :, :, 0] /= lf3d.shape[3]
        st3d[:, :, :, 1] /= lf3d.shape[3]
        st3d[:, :, :, 2] /= lf3d.shape[3]

        print("apply gaussian filter along 2rd dimension")
        st3d[:, :, :, 0] = vigra.filters.convolveOneDimension(st3d[:, :, :, 0], 1, gaussianOuter3)#Additional Smoothing
        st3d[:, :, :, 1] = vigra.filters.convolveOneDimension(st3d[:, :, :, 1], 1, gaussianOuter3)#Additional Smoothing
        st3d[:, :, :, 2] = vigra.filters.convolveOneDimension(st3d[:, :, :, 2], 1, gaussianOuter3)#Additional Smoothing

        print("apply gaussian filter along 1rd dimension")
        st3d[:, :, :, 0] = vigra.filters.convolveOneDimension(st3d[:, :, :, 0], 0, gaussianOuter1)
        st3d[:, :, :, 1] = vigra.filters.convolveOneDimension(st3d[:, :, :, 1], 0, gaussianOuter1)
        st3d[:, :, :, 2] = vigra.filters.convolveOneDimension(st3d[:, :, :, 2], 0, gaussianOuter1)

        print("apply gaussian filter along 2rd dimension")
        st3d[:, :, :, 0] = vigra.filters.convolveOneDimension(st3d[:, :, :, 0], 2, gaussianOuter2)
        st3d[:, :, :, 1] = vigra.filters.convolveOneDimension(st3d[:, :, :, 1], 2, gaussianOuter2)
        st3d[:, :, :, 2] = vigra.filters.convolveOneDimension(st3d[:, :, :, 2], 2, gaussianOuter2)

###downsampling###
        # st3d_f[:] = st3d[:]
###
        st3d_f[:, :, :, 0] = vigra.sampling.resize(st3d[:, :, :, 0], shape=(grad.shape[0], grad.shape[1], grad.shape[2]), order = 3)
        st3d_f[:, :, :, 1] = vigra.sampling.resize(st3d[:, :, :, 1], shape=(grad.shape[0], grad.shape[1], grad.shape[2]), order = 3)
        st3d_f[:, :, :, 2] = vigra.sampling.resize(st3d[:, :, :, 2], shape=(grad.shape[0], grad.shape[1], grad.shape[2]), order = 3)
###

    if(config.structure_tensor_type == "classic"):
        if config.prefilter == "True":

            lf3d = prefilter.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='h')

        print "compute 2.5D structure tensor (vigra)"
        st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
        tmp = vigra.filters.structureTensor(lf3d,config.inner_scale,config.outer_scale)
        logging.debug('Size of 3D structure Tensor: ' + str(tmp.shape))

        st3d[:,:,:,0] = tmp[:,:,:,0]
        st3d[:,:,:,1] = tmp[:,:,:,2]
        st3d[:,:,:,2] = tmp[:,:,:,5]

    coherence = np.sqrt((st3d_f[:, :, :, 2]-st3d_f[:, :, :, 0])**2+4*st3d_f[:, :, :, 1]**2)/(st3d_f[:, :, :, 2]+st3d_f[:, :, :, 0] + 1e-16)
    orientation = 1/2.0*vigra.numpy.arctan2(2*st3d_f[:, :, :, 1], st3d_f[:, :, :, 2]-st3d_f[:, :, :, 0])
    orientation = vigra.numpy.tan(orientation[:])
    invalid_ubounds = np.where(orientation > 1)
    invalid_lbounds = np.where(orientation < -1)
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    orientation[invalid_ubounds] = -1
    orientation[invalid_lbounds] = -1

    if config.output_level == 3:
        misc.imsave(config.result_path+config.result_label+"orientation_h_shift_{0}.png".format(shift), orientation[orientation.shape[0]/2, :, :])
    if config.output_level == 3:
        misc.imsave(config.result_path+config.result_label+"coherence_h_{0}.png".format(shift), coherence[orientation.shape[0]/2, :, :])
    print "ok"

    orientation[:] += shift

    return orientation, coherence


#============================================================================================================
#=========                                Vertical EPI computation                                ===========
#============================================================================================================

def compute_vertical(lf3dv, shift, config):

    logging.info("compute vertical shift {0}".format(shift))
    tmp = np.copy(lf3dv)
    tmp = lfhelpers.refocus_3d(tmp, shift, 'v')
    logging.debug("New size of lf3dv after shifting to horoptor: " + str(tmp.shape))

    # if config.output_level == 4:
    #   for i in range(lf3d.shape[0]):
    #         misc.imsave(config.result_path+config.result_label+"vertical_Input_shifted_{0}.png".format(i), lf3d[i ,: ,: ,:])

    if config.color_space:
        lf3d = prefilter.changeColorSpace(tmp, config.color_space)
    else:
        lf3d = tmp

    # if config.output_level == 4:
    #    for i in range(lf3d.shape[0]):
    #         misc.imsave(config.result_path+config.result_label+"vertical_Input_shifted_color_space_changed_{0}.png".format(i), lf3d[i ,: ,: ,:])

    if(config.structure_tensor_type == "scharr"):

        gaussianInner1 = vigra.filters.gaussianKernel(config.inner_scale[0])
        gaussianInner2 = vigra.filters.gaussianKernel(config.inner_scale[1])
        gaussianInner3 = vigra.filters.gaussianKernel(config.inner_scale[2])

        gaussianOuter1 = vigra.filters.gaussianKernel(config.outer_scale[0])
        gaussianOuter2 = vigra.filters.gaussianKernel(config.outer_scale[1])
        gaussianOuter3 = vigra.filters.gaussianKernel(config.outer_scale[2])

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
        scharr1dim.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_AVOID)

        grad = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 2), dtype=np.float32)
        tensor = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
        st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)

        for i in range(lf3d.shape[3]):

            ### Inner gaussian filter ###
            print("apply gaussian filter along 3rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, gaussianInner2)
            print("apply gaussian filter along 2rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 2, gaussianInner3)#Additional Smoothing
            print("apply gaussian filter along 1rd dimension")
            lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, gaussianInner1)
            # print("apply gaussian filter along 2rd dimension")
            # lf3d = vigra.filters.convolveOneDimension(lf3d, 1, gaussianInner)
            if (config.prefilter == "True"):
                ### EPI prefilter ###
                print("apply scharr pre-filter along 3rd dimension")
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, scharr1dim)
                lf3d[:, :, :, i] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr2dim)

            ### Derivative computation ###
            print("apply scharr filter along 1st dimension")
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 0, scharr1dim)
            grad[:, :, :, 0] = vigra.filters.convolveOneDimension(grad[:, :, :, 0], 1, scharr2dim)
            print("apply scharr filter along 3rd dimension")
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(lf3d[:, :, :, i], 1, scharr1dim)
            grad[:, :, :, 1] = vigra.filters.convolveOneDimension(grad[:, :, :, 1], 0, scharr2dim)

            tensor[:, :, :, 0] = grad[:, :, :, 0]**2
            tensor[:, :, :, 1] = grad[:, :, :, 1]*grad[:, :, :, 0]
            tensor[:, :, :, 2] = grad[:, :, :, 1]**2

            st3d[:, :, :, 0] += tensor[:, :, :, 0]
            st3d[:, :, :, 1] += tensor[:, :, :, 1]
            st3d[:, :, :, 2] += tensor[:, :, :, 2]

        st3d[:, :, :, 0] /= lf3d.shape[3]
        st3d[:, :, :, 1] /= lf3d.shape[3]
        st3d[:, :, :, 2] /= lf3d.shape[3]

        print("apply gaussian filter along 3rd dimension")
        st3d[:, :, :, 0] = vigra.filters.convolveOneDimension(st3d[:, :, :, 0], 1, gaussianOuter2)
        st3d[:, :, :, 1] = vigra.filters.convolveOneDimension(st3d[:, :, :, 1], 1, gaussianOuter2)
        st3d[:, :, :, 2] = vigra.filters.convolveOneDimension(st3d[:, :, :, 2], 1, gaussianOuter2)

        print("apply gaussian filter along 1rd dimension")
        st3d[:, :, :, 0] = vigra.filters.convolveOneDimension(st3d[:, :, :, 0], 0, gaussianOuter1)
        st3d[:, :, :, 1] = vigra.filters.convolveOneDimension(st3d[:, :, :, 1], 0, gaussianOuter1)
        st3d[:, :, :, 2] = vigra.filters.convolveOneDimension(st3d[:, :, :, 2], 0, gaussianOuter1)

        print("apply gaussian filter along 2rd dimension")
        st3d[:, :, :, 0] = vigra.filters.convolveOneDimension(st3d[:, :, :, 0], 2, gaussianOuter3)#Additional Smoothing
        st3d[:, :, :, 1] = vigra.filters.convolveOneDimension(st3d[:, :, :, 1], 2, gaussianOuter3)#Additional Smoothing
        st3d[:, :, :, 2] = vigra.filters.convolveOneDimension(st3d[:, :, :, 2], 2, gaussianOuter3)#Additional Smoothing

    if(config.structure_tensor_type == "classic"):

        if config.prefilter == "True":
            lf3d = prefilter.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='v')

        print "compute 2.5D structure tensor (vigra)"
        st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
        tmp = vigra.filters.structureTensor(lf3d,config.inner_scale,config.outer_scale)
        logging.debug('Size of 2.5D structure Tensor: ' + str(tmp.shape))

        st3d[:,:,:,0] = tmp[:,:,:,0]
        st3d[:,:,:,1] = tmp[:,:,:,1]
        st3d[:,:,:,2] = tmp[:,:,:,3]


    coherence = np.sqrt((st3d[:, :, :, 2]-st3d[:, :, :, 0])**2+4*st3d[:, :, :, 1]**2)/(st3d[:, :, :, 2]+st3d[:, :, :, 0] + 1e-16)
    orientation = 1/2.0*vigra.numpy.arctan2(2*st3d[:, :, :, 1], st3d[:, :, :, 2]-st3d[:, :, :, 0])
    orientation = vigra.numpy.tan(orientation[:])
    invalid_ubounds = np.where(orientation > 1)
    invalid_lbounds = np.where(orientation < -1)
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    orientation[invalid_ubounds] = -1
    orientation[invalid_lbounds] = -1

    if config.output_level == 3:
      misc.imsave(config.result_path+config.result_label+"orientation_v_shift_{0}.png".format(shift), orientation[orientation.shape[0]/2, :, :])
    if config.output_level == 3:
      misc.imsave(config.result_path+config.result_label+"coherence_v_{0}.png".format(shift), coherence[orientation.shape[0]/2, :, :])

    print "ok"

    orientation[:] += shift

    return orientation, coherence



#============================================================================================================
#=========                           Structure tensor processing chain                            ===========
#============================================================================================================


def structureTensor25D(config):


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
        logging.debug('Load horizontal light field in folder:' + config.path_horizontal)
        compute_h = True
        lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
        lf_shape = lf3dh.shape
        logging.debug('Size:' + str(lf_shape))
    except:
        logging.error("Could not load Data")

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        logging.debug('Load horizontal light field in folder:' + config.path_horizontal)
        compute_v = True
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        if lf_shape is None:
            lf_shape = lf3dv.shape
        logging.debug('Size:' + str(lf_shape))

    except:
        logging.error("Could not load Data")

### Allocate memory for results ###

    orientation = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    coherence = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    logging.debug("Allocated memory!")

### compute both directions independent from each other ###

    for shift in config.global_shifts:
        logging.info('Shift: ' + str(shift))

        orientation_h = None
        coherence_h = None
        orientation_v = None
        coherence_v = None

        if compute_h:
            logging.debug("compute horizontal LightField")
            [orientation_h, coherence_h] = Compute(lf3dh, shift, config, direction='h')
        if compute_v:
            logging.debug("compute vertical LightField")
            [orientation_v, coherence_v] = Compute(lf3dv, shift, config, direction='v')

        if compute_h and compute_v:
            logging.info("merge vertical and horizontal direction into global result")
            orientation_tmp, coherence_tmp = mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
            orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

            if config.output_level >= 3:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            logging.info("done!")

        else:
            logging.info("merge vertical or horizontal direction into global result")
            if compute_h:
                orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
            if compute_v:
                orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)
            if config.output_level >= 3:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
                plt.imsave(config.result_path+config.result_label+"coherence_merged_shift_{0}.png".format(shift), coherence[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            logging.info("done!")

    invalids = np.where(coherence < config.coherence_threshold)
    orientation[invalids] = 0
    coherence[invalids] = 0

    orientation += config.offsetDisparity

    if isinstance(config.median, int) and config.median > 0:
        print "apply median filter ..."
        orientation = median_filter(orientation, config.median)

    if isinstance(config.selective_gaussian, float) and config.selective_gaussian > 0:
        print "apply masked gauss..."
        mask = coherence[:, :]
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
        orientation = vigra.filters.normalizedConvolveImage(orientation, mask, gauss)


    if config.output_level >= 1:
        plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence_final.png", coherence[lf_shape[0]/2, :, :], cmap=plt.cm.jet)

    logging.info("Computed final disparity map!")

## Light field computation has to be changed just to compute the core of the disparity and just transfer it here to the disparity map

    depth = dtc.disparity_to_depth(orientation[lf_shape[0]/2, :, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)
    mask = coherence[lf_shape[0]/2, :, :]

    invalids = np.where(mask == 0)
    depth[invalids] = 0


    tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
    tmp[:, :, 0] = orientation[lf_shape[0]/2, :, :]
    tmp[:, :, 1] = coherence[lf_shape[0]/2, :, :]
    tmp[:, :, 2] = depth[:]
    vim = vigra.RGBImage(tmp)
    vim.writeImage(config.result_path+config.result_label+"final25D.exr")

    if config.output_level >= 1:
        plt.imsave(config.result_path+config.result_label+"depth_final.png", depth, cmap=plt.cm.jet)

    if config.output_level >= 1:
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
        self.outer_scale = 0.9                  # structure tensor outer scale
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
	self.offsetDisparity = 0

        self.rgb = True                         # forces grayscale if False

        self.output_level = 2                   # level of detail for file output possible 1,2,3

    def saveLog(self, filename=None):
        if filename is not None:
            f = open(filename, "w")
        else:
            f = open(self.result_path+self.result_label+"/log.txt", "w")
        f.write("roi : "); f.write(str(self.roi)+"\n")
        f.write("inner_scale : "); f.write(str(self.inner_scale)+"\n")
        f.write("outer_scale : "); f.write(str(self.outer_scale)+"\n")
        f.write("double_tensor : "); f.write(str(self.double_tensor)+"\n")
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




