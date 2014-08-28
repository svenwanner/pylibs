import os
import logging
import vigra
import numpy as np
import pylab as plt
import scipy.misc as misc
from scipy.ndimage import median_filter
import mypy.lightfield.depth.prefilter as prefilter
from mypy.lightfield import io as lfio

import mypy.pointclouds.depthToCloud as dtc
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor2D as st2d





#============================================================================================================
#=========                                       EPI processing                                   ===========
#============================================================================================================


def Compute(lf3d, shift, config, disableForward, disableBackward, direction):

    if direction == 'h':
        orientation, coherence = compute_horizontal(lf3d, shift, config, disableForward, disableBackward)
    if direction == 'v':
        orientation, coherence = compute_vertical(lf3d, shift, config, disableForward, disableBackward)

    return orientation, coherence

def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
    winner = np.where(coherence2 > coherence1)
    orientation1[winner] = orientation2[winner]
    coherence1[winner] = coherence2[winner]

    return orientation1, coherence1

#============================================================================================================
#=========                              Horizontal EPI computation                                ===========
#============================================================================================================

def compute_horizontal(lf3dh, shift, config, disableForward, disableBackward):

    logging.info("compute horizontal shift {0}".format(shift))
    lf3d = np.copy(lf3dh)
    lf3d = lfhelpers.refocus_3d(lf3d, shift, 'h')
    logging.debug("New size of lf3dh after shifting to horoptor: " + str(lf3d.shape))

    if config.output_level == 4:
        for i in range(lf3d.shape[0]):
            misc.imsave(config.result_path+config.result_label+"horizontal_Input_shifted_{0}.png".format(i), lf3d[i ,: ,: ,:])

    if config.color_space:
        lf3d = prefilter.changeColorSpace(lf3d, config.color_space)

    if config.output_level == 4:
        for i in range(lf3d.shape[0]):
            misc.imsave(config.result_path+config.result_label+"horizontal_Input_shifted_color_space_changed_{0}.png".format(i), lf3d[i ,: ,: ,:])

    logging.debug("Prefilter status: " + str(config.prefilter))
    if config.prefilter > 0:
       if config.prefilter == prefilter.PREFILTER.IMGD:
           lf3d = prefilter.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='h')
       if config.prefilter == prefilter.PREFILTER.EPID:
           lf3d = prefilter.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='h')
       if config.prefilter == prefilter.PREFILTER.IMGD2:
            lf3d = prefilter.preImgLaplace(lf3d, scale=config.prefilter_scale)
       if config.prefilter == prefilter.PREFILTER.EPID2:
            lf3d = prefilter.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='h')

    coherence = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2]), dtype=np.float32)
    orientation = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2]), dtype=np.float32)

    if disableForward == False:
        print("structure tensor forward")

        structureTensor = st2d.StructureTensorForward()

        params = {"direction": 'h', "inner_scale": config.inner_scale, "outer_scale": config.outer_scale, "hour-glass": config.hourglass_scale}
        structureTensor.compute(lf3d, params)
        st3d = structureTensor.get_result()

        orientation_tmp, coherence_tmp = st2d.evaluateStructureTensor(st3d)
        orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

    if disableBackward == False:
        print("structure tensor backward")
        structureTensor = st2d.StructureTensorBackward()
        params = {"direction": 'h', "inner_scale": config.inner_scale, "outer_scale": config.outer_scale, "hour-glass": config.hourglass_scale}
        structureTensor.compute(lf3d, params)
        st3d = structureTensor.get_result()

        orientation_tmp, coherence_tmp = st2d.evaluateStructureTensor(st3d)
        orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

        if config.coherence_threshold > 0.0:
            invalids = np.where(coherence < config.coherence_threshold)
            coherence[invalids] = 0.0

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

def compute_vertical(lf3dv, shift, config, disableForward, disableBackward):

    logging.info("compute vertical shift {0}".format(shift))
    lf3d = np.copy(lf3dv)
    lf3d = lfhelpers.refocus_3d(lf3d, shift, 'v')
    logging.debug("New size of lf3dv after shifting to horoptor: " + str(lf3d.shape))

    if config.output_level == 4:
      for i in range(lf3d.shape[0]):
            misc.imsave(config.result_path+config.result_label+"vertical_Input_shifted_{0}.png".format(i), lf3d[i ,: ,: ,:])

    if config.color_space:
        lf3d = prefilter.changeColorSpace(lf3d, config.color_space)

    if config.output_level == 4:
       for i in range(lf3d.shape[0]):
            misc.imsave(config.result_path+config.result_label+"vertical_Input_shifted_color_space_changed_{0}.png".format(i), lf3d[i ,: ,: ,:])

    if config.prefilter > 0:
       if config.prefilter == prefilter.PREFILTER.IMGD:
           lf3d = prefilter.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='v')
       if config.prefilter == prefilter.PREFILTER.EPID:
           lf3d = prefilter.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='v')
       if config.prefilter == prefilter.PREFILTER.IMGD2:
            lf3d = prefilter.preImgLaplace(lf3d, scale=config.prefilter_scale)
       if config.prefilter == prefilter.PREFILTER.EPID2:
            lf3d = prefilter.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='v')

    coherence = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2]), dtype=np.float32)
    orientation = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2]), dtype=np.float32)

    if disableForward == False:
        print("structure tensor forward")

        structureTensor = st2d.StructureTensorBackward()
        params = {"direction": 'v', "inner_scale": config.inner_scale, "outer_scale": config.outer_scale, "hour-glass": config.hourglass_scale}
        structureTensor.compute(lf3d, params)
        st3d = structureTensor.get_result()

        orientation_tmp, coherence_tmp = st2d.evaluateStructureTensor(st3d)
        orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)
    if disableBackward == False:
        print("structure tensor bachward")

        structureTensor = st2d.StructureTensorForward()
        params = {"direction": 'v', "inner_scale": config.inner_scale, "outer_scale": config.outer_scale, "hour-glass": config.hourglass_scale}
        structureTensor.compute(lf3d, params)
        st3d = structureTensor.get_result()

        orientation_tmp, coherence_tmp = st2d.evaluateStructureTensor(st3d)
        orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

    if config.coherence_threshold > 0.0:
      invalids = np.where(coherence < config.coherence_threshold)
      coherence[invalids] = 0.0

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


def structureTensor2D_forwardBackward(config, disableForward, disableBackward):


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
        print('Load horizontal light field in folder:' + config.path_horizontal)
        compute_h = True
        lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
        lf_shape = lf3dh.shape
        logging.debug('Size:' + str(lf_shape))
    except:
        logging.error("Could not load Data")

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        print('Load vertical light field in folder:' + config.path_horizontal)
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        compute_v = True
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
            [orientation_h, coherence_h] = Compute(lf3dh, shift, config, disableForward, disableBackward, direction='h')
        if compute_v:
            logging.debug("compute vertical LightField")
            [orientation_v, coherence_v] = Compute(lf3dv, shift, config, disableForward, disableBackward, direction='v')

        if compute_h and compute_v:
            logging.info("merge vertical and horizontal direction into global result")
            orientation_tmp, coherence_tmp = mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
            orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            logging.info("done!")

        else:
            logging.info("merge vertical or horizontal direction into global result")
            if compute_h:
                orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
            if compute_v:
                orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)
            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
                plt.imsave(config.result_path+config.result_label+"coherence_merged_shift_{0}.png".format(shift), coherence[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            logging.info("done!")

    invalids = np.where(coherence < config.coherence_threshold)
    orientation[invalids] = 0
    coherence[invalids] = 0

    # if config.output_level >= 2:
    #     plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
    #     plt.imsave(config.result_path+config.result_label+"coherence_final.png", coherence[lf_shape[0]/2, :, :], cmap=plt.cm.gray)

    logging.info("Computed final disparity map!")

## Light field computation has to be changed just to compute the core of the disparity and just transfer it here to the disparity map

    orientation = orientation[lf_shape[0]/2, :, :]
    depth = dtc.disparity_to_depth(orientation, config.base_line, config.focal_length, config.min_depth, config.max_depth)
    mask = np.copy(coherence[lf_shape[0]/2, :, :])

    if isinstance(config.median, int) and config.median > 0:
        print "apply median filter ..."
        depth = median_filter(depth, config.median)
        orientation = median_filter(orientation, config.median)
    if isinstance(config.nonlinear_diffusion, type([])):
        print "apply nonlinear diffusion"
        vigra.filters.nonlinearDiffusion(depth, config.nonlinear_diffusion[0], config.nonlinear_diffusion[1])
        vigra.filters.nonlinearDiffusion(orientation, config.nonlinear_diffusion[0], config.nonlinear_diffusion[1])
    if isinstance(config.tv, type({})):
        print "apply total variation..."
        mask = coherence[lf_shape[0]/2, :, :]
        cv = None
        if lf3dh is not None:
            if lf_shape[3] == 3:
                cv = 0.3*lf3dh[lf_shape[0]/2, :, :, 0]+0.59*lf3dh[lf_shape[0]/2, :, :, 1]+0.11*lf3dh[lf_shape[0]/2, :, :, 2]
            else:
                cv = lf3dh[lf_shape[0]/2, :, :, 0]
        elif lf3dv is not None:
            if lf_shape[3] == 3:
                cv = 0.3*lf3dv[lf_shape[0]/2, :, :, 0]+0.59*lf3dv[lf_shape[0]/2, :, :, 1]+0.11*lf3dv[lf_shape[0]/2, :, :, 2]
            else:
                cv = lf3dv[lf_shape[0]/2, :, :, 0]

        borders = vigra.filters.gaussianGradientMagnitude(cv, 1.6)
        borders /= np.amax(borders)
        mask *= 1.0-borders
        mask /= np.amax(mask)
        assert depth.shape == mask.shape
        drange = config.max_depth-config.min_depth
        drange2 = np.abs(np.amax(orientation) - np.amin(orientation))
        depth = vigra.filters.totalVariationFilter(depth.astype(np.float64), mask.astype(np.float64), 0.01*drange*config.tv["alpha"], config.tv["steps"], 0)
        orientation = vigra.filters.totalVariationFilter(orientation.astype(np.float64), mask.astype(np.float64), 0.01*drange2*config.tv["alpha"], config.tv["steps"], 0)


    if config.output_level >= 2:
        plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation, cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence_final.png", mask, cmap=plt.cm.jet)

    invalids = np.where(mask == 0)
    depth[invalids] = 0

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
        self.prefilter = prefilter.PREFILTER.IMGD2          # type of the prefilter [NO,IMGD, EPID, IMGD2, EPID2]

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




