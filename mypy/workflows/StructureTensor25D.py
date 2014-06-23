import os
import logging
import vigra
import numpy as np
import pylab as plt
import scipy.misc as misc
import mypy.lightfield.depth.prefilter as prefilter
from mypy.lightfield import io as lfio

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

    print "compute 3D structure tensor"
    st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)

    tmp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 6), dtype=np.float32)
    epiVol = []
    for c in range(lf3d.shape[3]):
        epiVol.append(lf3d[:, :, :, c])
    for n, epi in enumerate(epiVol):
        epiVol[n] = vigra.filters.structureTensor(epi,config.inner_scale_h,config.outer_scale_h)
    st = np.zeros_like(epiVol[0])
    for epi in epiVol:
        st[:, :, :, :] += epi[:, :, :, :]

    tmp[:, :, :, :] = st[:]/3

    #tmp = vigra.filters.structureTensor(lf3d,config.inner_scale,config.outer_scale)
    logging.debug('Size of 3D structure Tensor: ' + str(tmp.shape))

    st3d[:,:,:,0] = tmp[:,:,:,0]
    st3d[:,:,:,1] = tmp[:,:,:,2]
    st3d[:,:,:,2] = tmp[:,:,:,5]


    coherence = np.sqrt((st3d[:, :, :, 2]-st3d[:, :, :, 0])**2+2*st3d[:, :, :, 1]**2)/(st3d[:, :, :, 2]+st3d[:, :, :, 0] + 1e-16)
    orientation = 1/2.0*vigra.numpy.arctan2(2*st3d[:, :, :, 1], st3d[:, :, :, 2]-st3d[:, :, :, 0])
    orientation = vigra.numpy.tan(orientation[:])
    invalid_ubounds = np.where(orientation > 1)
    invalid_lbounds = np.where(orientation < -1)
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    orientation[invalid_ubounds] = -1
    orientation[invalid_lbounds] = -1


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

def compute_vertical(lf3dv, shift, config):

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


    print "compute 2.5D structure tensor"
    st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)

    tmp = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 6), dtype=np.float32)
    epiVol = []
    for c in range(lf3d.shape[3]):
        epiVol.append(lf3d[:, :, :, c])
    for n, epi in enumerate(epiVol):
        epiVol[n] = vigra.filters.structureTensor(epi,config.inner_scale_h,config.outer_scale_h)
    st = np.zeros_like(epiVol[0])
    for epi in epiVol:
        st[:, :, :, :] += epi[:, :, :, :]

    tmp[:, :, :, :] = st[:]/3


    #tmp = vigra.filters.structureTensor(lf3d,config.inner_scale,config.outer_scale)
    logging.debug('Size of 2.5D structure Tensor: ' + str(tmp.shape))

    st3d[:,:,:,0] = tmp[:,:,:,0]
    st3d[:,:,:,1] = tmp[:,:,:,1]
    st3d[:,:,:,2] = tmp[:,:,:,3]


    coherence = np.sqrt((st3d[:, :, :, 2]-st3d[:, :, :, 0])**2+2*st3d[:, :, :, 1]**2)/(st3d[:, :, :, 2]+st3d[:, :, :, 0] + 1e-16)
    orientation = 1/2.0*vigra.numpy.arctan2(2*st3d[:, :, :, 1], st3d[:, :, :, 2]-st3d[:, :, :, 0])
    orientation = vigra.numpy.tan(orientation[:])
    invalid_ubounds = np.where(orientation > 1)
    invalid_lbounds = np.where(orientation < -1)
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    orientation[invalid_ubounds] = -1
    orientation[invalid_lbounds] = -1

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

            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
            logging.info("done!")

        else:
            logging.info("merge vertical or horizontal direction into global result")
            if compute_h:
                orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
            if compute_v:
                orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)
            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
                plt.imsave(config.result_path+config.result_label+"coherence_merged_shift_{0}.png".format(shift), coherence[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
            logging.info("done!")

    invalids = np.where(coherence < config.coherence_threshold)
    orientation[invalids] = 0
    coherence[invalids] = 0

    if config.output_level >= 2:
        plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence_final.png", coherence[lf_shape[0]/2, :, :], cmap=plt.cm.jet)

    logging.info("Computed final disparity map!")

## Light field computation has to be changed just to compute the core of the disparity and just transfer it here to the disparity map

    depth = dtc.disparity_to_depth(orientation[lf_shape[0]/2, :, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)
    mask = coherence[lf_shape[0]/2, :, :]

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

        tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
        tmp[:, :, 0] = orientation[lf_shape[0]/2, :, :]
        tmp[:, :, 1] = coherence[lf_shape[0]/2, :, :]
        tmp[:, :, 2] = depth[:]
        vim = vigra.RGBImage(tmp)
        vim.writeImage(config.result_path+config.result_label+"final.exr")

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








# import os
# import logging
# import vigra
# import numpy as np
# import pylab as plt
# import scipy.misc as misc
#
# from mypy.lightfield import io as lfio
# from mypy.lightfield.helpers import enum
# import mypy.pointclouds.depthToCloud as dtc
# from mypy.lightfield import helpers as lfhelpers
# import mypy.lightfield.depth.prefilter as prefilter
#
#
#
# COLORSPACE = enum(RGB=0, LAB=1, LUV=2)
# PREFILTER = enum(NO=0, IMGD=1, EPID=2, IMGD2=3, EPID2=4)
#
#
#
#
#
#
#
# #============================================================================================================
# #=========                                       EPI processing                                   ===========
# #============================================================================================================
#
#
# def Compute(lf3d, shift, config, direction):
#
#     if direction == 'h':
#         orientation, coherence = compute_horizontal(lf3d, shift, config)
#     if direction == 'v':
#         orientation, coherence = compute_vertical(lf3d, shift, config)
#
#     return orientation, coherence
#
#
# def mergeOrientations_wta(orientation1, coherence1, orientation2, coherence2):
#     winner = np.where(coherence2 > coherence1)
#     orientation1[winner] = orientation2[winner]
#     coherence1[winner] = coherence2[winner]
#     return orientation1, coherence1
#
#
# #============================================================================================================
# #=========                              Horizontal EPI computation                                ===========
# #============================================================================================================
#
# def compute_horizontal(lf3dh, shift, config):
#
#     logging.info("compute horizontal shift {0}".format(shift))
#     lf3d = np.copy(lf3dh)
#     lf3d = lfhelpers.refocus_3d(lf3d, shift, 'h')
#     logging.info("New size of lf3dh after shifting to horoptor: " + str(lf3d.shape))
#
#     if config.output_level == 4:
#         for i in range(lf3d.shape[0]):
#             misc.imsave(config.result_path+config.result_label+"horizontal_Input_shifted_{0}.png".format(i), lf3d[i ,: ,: ,:])
#
#     if config.color_space:
#         lf3d = prefilter.changeColorSpace(lf3d, config.color_space)
#
#     if config.output_level == 4:
#         for i in range(lf3d.shape[0]):
#             misc.imsave(config.result_path+config.result_label+"horizontal_Input_shifted_color_space_changed_{0}.png".format(i), lf3d[i ,: ,: ,:])
#
#     lf3d[:,:,:,0] = vigra.filters.gaussianSmoothing(lf3d[:,:,:,0],config.volume_pre_smoothing_scale)
#     lf3d[:,:,:,1] = vigra.filters.gaussianSmoothing(lf3d[:,:,:,1],config.volume_pre_smoothing_scale)
#     lf3d[:,:,:,2] = vigra.filters.gaussianSmoothing(lf3d[:,:,:,2],config.volume_pre_smoothing_scale)
#
#     logging.debug("Prefilter status: " + str(config.prefilter))
#     if config.prefilter > 0:
#        print "use prefilter"
#        if config.prefilter == PREFILTER.IMGD:
#            lf3d = prefilter.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='h')
#        if config.prefilter == PREFILTER.EPID:
#            lf3d = prefilter.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='h')
#        if config.prefilter == PREFILTER.IMGD2:
#             lf3d = prefilter.preImgLaplace(lf3d, scale=config.prefilter_scale)
#        if config.prefilter == PREFILTER.EPID2:
#             lf3d = prefilter.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='h')
#
#     st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
#     for y in xrange(lf3d.shape[1]):
#             epis = []
#             for c in range(lf3d.shape[3]):
#                 epis.append(lf3d[:, y, :, c])
#             if config.output_level == 4:
#                 misc.imsave(config.result_path+config.result_label+"EPIH_{0}.png".format(y), lf3d[:, y, :, :])
#
#             for n, epi in enumerate(epis):
#                 epis[n] = vigra.filters.structureTensor(epi, config.inner_scale, config.outer_scale)
#
#             st = np.zeros_like(epis[0])
#             for epi in epis:
#                 st[:, :, :] += epi[:, :, :]
#
#             st3d[:, y, :, :] = st[:]/3
#             # logging.info("Structure Tensor Resulting Size: " + str(st.shape))
#             # logging.info("Structure Tensor Volume Size: " + str(st3d.shape))
#
#     if config.output_level == 4:
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_H_{0}.png".format(0), st3d[8 ,: ,: ,:])
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_H_{0}.png".format(1), st3d[8 ,: ,: ,0])
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_H_{0}.png".format(2), st3d[8 ,: ,: ,1])
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_H_{0}.png".format(3), st3d[8 ,: ,: ,2])
#
#     st3d[:,:,:,0] = vigra.filters.gaussianSmoothing(st3d[:,:,:,0],config.volume_post_smoothing_scale)
#     st3d[:,:,:,1] = vigra.filters.gaussianSmoothing(st3d[:,:,:,1],config.volume_post_smoothing_scale)
#     st3d[:,:,:,2] = vigra.filters.gaussianSmoothing(st3d[:,:,:,2],config.volume_post_smoothing_scale)
#
#     if config.output_level == 4:
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_H_{0}.png".format(0), st3d[8 ,: ,: ,:])
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_H_{0}.png".format(2), st3d[8 ,: ,: ,0])
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_H_{0}.png".format(1), st3d[8 ,: ,: ,1])
#             misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_H_{0}.png".format(2), st3d[8 ,: ,: ,2])
#
#     coherence = np.sqrt((st3d[:, :, :, 2]-st3d[:, :, :, 0])**2+2*st3d[:, :, :, 1]**2)/(st3d[:, :, :, 2]+st3d[:, :, :, 0] + 1e-16)
#     orientation = 1/2.0*vigra.numpy.arctan2(2*st3d[:, :, :, 1], st3d[:, :, :, 2]-st3d[:, :, :, 0])
#     orientation = vigra.numpy.tan(orientation[:])
#     invalid_ubounds = np.where(orientation > 1)
#     invalid_lbounds = np.where(orientation < -1)
#     coherence[invalid_ubounds] = 0
#     coherence[invalid_lbounds] = 0
#     orientation[invalid_ubounds] = -1
#     orientation[invalid_lbounds] = -1
#
#
#
#     if config.coherence_threshold > 0.0:
#         invalids = np.where(coherence < config.coherence_threshold)
#         coherence[invalids] = 0.0
#
#     if config.output_level == 3:
#         misc.imsave(config.result_path+config.result_label+"orientation_h_shift_{0}.png".format(shift), orientation[orientation.shape[0]/2, :, :])
#     if config.output_level == 3:
#         misc.imsave(config.result_path+config.result_label+"coherence_h_{0}.png".format(shift), coherence[orientation.shape[0]/2, :, :])
#     print "ok"
#
#     orientation[:] += shift
#
#     return orientation, coherence
#
#
# #============================================================================================================
# #=========                                Vertical EPI computation                                ===========
# #============================================================================================================
#
# def compute_vertical(lf3dv, shift, config):
#
#     logging.info("compute vertical shift {0}".format(shift))
#     lf3d = np.copy(lf3dv)
#     lf3d = lfhelpers.refocus_3d(lf3d, shift, 'v')
#     logging.debug("New size of lf3dv after shifting to horoptor: " + str(lf3d.shape))
#
#     if config.output_level == 4:
#       for i in range(lf3d.shape[0]):
#             misc.imsave(config.result_path+config.result_label+"vertical_Input_shifted_{0}.png".format(i), lf3d[i ,: ,: ,:])
#
#     if config.color_space:
#         lf3d = prefilter.changeColorSpace(lf3d, config.color_space)
#
#     if config.output_level == 4:
#        for i in range(lf3d.shape[0]):
#             misc.imsave(config.result_path+config.result_label+"vertical_Input_shifted_color_space_changed_{0}.png".format(i), lf3d[i ,: ,: ,:])
#
#     lf3d[:,:,:,0] = vigra.filters.gaussianSmoothing(lf3d[:,:,:,0],config.volume_pre_smoothing_scale)
#     lf3d[:,:,:,1] = vigra.filters.gaussianSmoothing(lf3d[:,:,:,1],config.volume_pre_smoothing_scale)
#     lf3d[:,:,:,2] = vigra.filters.gaussianSmoothing(lf3d[:,:,:,2],config.volume_pre_smoothing_scale)
#
#     if config.prefilter > 0:
#        if config.prefilter == PREFILTER.IMGD:
#            lf3d = prefilter.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='h')
#        if config.prefilter == PREFILTER.EPID:
#            lf3d = prefilter.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='h')
#        if config.prefilter == PREFILTER.IMGD2:
#             lf3d = prefilter.preImgLaplace(lf3d, scale=config.prefilter_scale)
#        if config.prefilter == PREFILTER.EPID2:
#             lf3d = prefilter.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='h')
#
#
#     st3d = np.zeros((lf3d.shape[0], lf3d.shape[1], lf3d.shape[2], 3), dtype=np.float32)
#     for x in xrange(lf3d.shape[2]):
#         epis = []
#         for c in range(lf3d.shape[3]):
#              epis.append(lf3d[:, :, x, c])
#         if config.output_level == 4:
#              misc.imsave(config.result_path+config.result_label+"EPIV_{0}.png".format(x), lf3d[:, :, x, :])
#
#
#         for n, epi in enumerate(epis):
#             epis[n] = vigra.filters.structureTensor(epi, config.inner_scale, config.outer_scale)
#
#         st = np.zeros_like(epis[0])
#         for epi in epis:
#              st[:, :, :] += epi[:, :, :]
#
#         st3d[:, :, x, :] = st[:]/3
#          # logging.info("Structure Tensor Resulting Size: " + str(st.shape))
#          # logging.info("Structure Tensor Volume Size: " + str(st3d.shape))
#
#     if config.output_level == 4:
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_V_{0}.png".format(0), st3d[8 ,: ,: ,:])
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_V_{0}.png".format(1), st3d[8 ,: ,: ,0])
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_V_{0}.png".format(2), st3d[8 ,: ,: ,1])
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_presmoothing_V_{0}.png".format(3), st3d[8 ,: ,: ,2])
#
#     st3d[:,:,:,0] = vigra.filters.gaussianSmoothing(st3d[:,:,:,0],config.volume_post_smoothing_scale)
#     st3d[:,:,:,1] = vigra.filters.gaussianSmoothing(st3d[:,:,:,1],config.volume_post_smoothing_scale)
#     st3d[:,:,:,2] = vigra.filters.gaussianSmoothing(st3d[:,:,:,2],config.volume_post_smoothing_scale)
#
#     if config.output_level == 4:
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_V_{0}.png".format(0), st3d[8 ,: ,: ,:])
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_V_{0}.png".format(2), st3d[8 ,: ,: ,0])
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_V_{0}.png".format(1), st3d[8 ,: ,: ,1])
#         misc.imsave(config.result_path+config.result_label+"structure_Tensor_postsmoothing_V_{0}.png".format(2), st3d[8 ,: ,: ,2])
#
#     coherence = np.sqrt((st3d[:, :, :, 2]-st3d[:, :, :, 0])**2+2*st3d[:, :, :, 1]**2)/(st3d[:, :, :, 2]+st3d[:, :, :, 0] + 1e-16)
#     orientation = 1/2.0*vigra.numpy.arctan2(2*st3d[:, :, :, 1], st3d[:, :, :, 2]-st3d[:, :, :, 0])
#     orientation = vigra.numpy.tan(orientation[:])
#     invalid_ubounds = np.where(orientation > 1)
#     invalid_lbounds = np.where(orientation < -1)
#     coherence[invalid_ubounds] = 0
#     coherence[invalid_lbounds] = 0
#     orientation[invalid_ubounds] = -1
#     orientation[invalid_lbounds] = -1
#
#     if config.coherence_threshold > 0.0:
#       invalids = np.where(coherence < config.coherence_threshold)
#       coherence[invalids] = 0.0
#
#     if config.output_level == 3:
#       misc.imsave(config.result_path+config.result_label+"orientation_v_shift_{0}.png".format(shift), orientation[orientation.shape[0]/2, :, :])
#     if config.output_level == 3:
#       misc.imsave(config.result_path+config.result_label+"coherence_v_{0}.png".format(shift), coherence[orientation.shape[0]/2, :, :])
#
#     print "ok"
#
#     orientation[:] += shift
#
#     return orientation, coherence
#
#
#
#
#
#
#
#
#
#
#
# #============================================================================================================
# #=========                           Structure tensor processing chain                            ===========
# #============================================================================================================
#
#
# def structureTensor25D(config):
#
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
#         logging.debug('Load horizontal light field in folder:' + config.path_horizontal)
#         compute_h = True
#         lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
#         lf_shape = lf3dh.shape
#         logging.debug('Size:' + str(lf_shape))
#     except:
#         logging.error("Could not load Data")
#
#     try:
#         if not config.path_vertical.endswith("/"):
#             config.path_vertical += "/"
#         logging.debug('Load horizontal light field in folder:' + config.path_horizontal)
#         compute_v = True
#         lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
#         if lf_shape is None:
#             lf_shape = lf3dv.shape
#         logging.debug('Size:' + str(lf_shape))
#
#     except:
#         logging.error("Could not load Data")
#
# ### Allocate memory for results ###
#
#     orientation = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
#     coherence = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
#     logging.debug("Allocated memory!")
#
# ### compute both directions independent from each other ###
#
#     for shift in config.global_shifts:
#         logging.info('Shift: ' + str(shift))
#
#         orientation_h = None
#         coherence_h = None
#         orientation_v = None
#         coherence_v = None
#
#         if compute_h:
#             logging.debug("compute horizontal LightField")
#             [orientation_h, coherence_h] = Compute(lf3dh, shift, config, direction='h')
#         if compute_v:
#             logging.debug("compute vertical LightField")
#             [orientation_v, coherence_v] = Compute(lf3dv, shift, config, direction='v')
#
#         if compute_h and compute_v:
#             logging.info("merge vertical and horizontal direction into global result")
#             orientation_tmp, coherence_tmp = mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
#             orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)
#
#             if config.output_level >= 2:
#                 plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
#             logging.info("done!")
#
#         else:
#             logging.info("merge vertical or horizontal direction into global result")
#             if compute_h:
#                 orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
#             if compute_v:
#                 orientation, coherence = mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)
#             if config.output_level >= 2:
#                 plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
#                 plt.imsave(config.result_path+config.result_label+"coherence_merged_shift_{0}.png".format(shift), coherence[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
#             logging.info("done!")
#
#     invalids = np.where(coherence < config.coherence_threshold)
#     orientation[invalids] = 0
#     coherence[invalids] = 0
#
#     if config.output_level >= 2:
#         plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
#         plt.imsave(config.result_path+config.result_label+"coherence_final.png", coherence[lf_shape[0]/2, :, :], cmap=plt.cm.gray)
#
#     logging.info("Computed final disparity map!")
#
# ## Light field computation has to be changed just to compute the core of the disparity and just transfer it here to the disparity map
#
#     depth = dtc.disparity_to_depth(orientation[lf_shape[0]/2, :, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)
#     mask = coherence[lf_shape[0]/2, :, :]
#
#     invalids = np.where(mask == 0)
#     depth[invalids] = 0
#
#     if config.output_level >= 1:
#         plt.imsave(config.result_path+config.result_label+"depth_final.png", depth, cmap=plt.cm.jet)
#
#     if config.output_level >= 1:
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
#
#
#
#
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
#         self.outer_scale = 0.9                  # structure tensor outer scale
#         self.coherence_threshold = 0.7          # if coherence less than value the disparity is set to invalid
#         self.focal_length = 5740.38             # focal length in pixel [default Nikon D800 f=28mm]
#         self.global_shifts = [0]                # list of horopter shifts in pixel
#         self.base_line = 0.001                  # camera baseline
#
#         self.color_space = COLORSPACE.RGB       # colorscape to convert the images into [RGB,LAB,LUV]
#         self.prefilter_scale = 0.4              # scale of the prefilter
#         self.prefilter = PREFILTER.IMGD2        # type of the prefilter [NO,IMGD, EPID, IMGD2, EPID2]
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
#         ########  filter options for 2.5D structure tensor ##################
#         self.volume_pre_smoothing_scale = 0.3 # Image volume presmoothing
#         self.volume_post_smoothing_scale = 1.3 # Image volume postsmoothing
#         #######################################################################
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



