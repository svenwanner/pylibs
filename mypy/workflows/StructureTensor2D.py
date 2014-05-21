import os
import vigra
import numpy as np
import scipy.misc as misc


from mypy.lightfield import io as lfio
from mypy.lightfield.helpers import enum
import mypy.pointclouds.depthToCloud as dtc
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor2D as st2d
from mypy.visualization.imshow import epishow


COLORSPACE = enum(RGB=0, LAB=1, LUV=2)
PREFILTER = enum(NO=0, IMGD=1, EPID=2, IMGD2=3, EPID2=4)



class Config:
    def __init__(self):

        self.result_path = None             # path to store the results
        self.result_label = None            # name of the results folder

        self.path_horizontal = None         # path to the horizontal images [optional]
        self.path_vertical = None           # path to the vertical images [optional]

        self.roi = None                     # region of interest

        self.centerview_path = None         # path to the center view image to get color for pointcloud [optional]

        self.inner_scale = 0.6              # structure tensor inner scale
        self.outer_scale = 0.9              # structure tensor outer scale
        self.double_tensor = 2.0            # if > 0.0 a second structure tensor with the outerscale specified is applied
        self.coherence_threshold = 0.7      # if coherence less than value the disparity is set to invalid
        self.focal_length = 5740.38         # focal length in pixel [default Nikon D800 f=28mm]
        self.global_shifts = [0]            # list of horopter shifts in pixel
        self.base_line = 0.001              # camera baseline

        self.color_space = COLORSPACE.RGB   # colorscape to convert the images into [RGB,LAB,LUV]
        self.prefilter_scale = 0.4          # scale of the prefilter
        self.prefilter = PREFILTER.IMGD2    # type of the prefilter [NO,IMGD, EPID, IMGD2, EPID2]

        self.min_depth = 0.01               # minimum depth possible
        self.max_depth = 1.0                # maximum depth possible

        self.rgb = True                     # forces grayscale if False

        self.output_level = 2               # level of detail for file output possible 1,2,3


#============================================================================================================
#============================================================================================================
#============================================================================================================

def structureTensor2D(config):
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

    print "load data...",
    try:
        if not config.path_horizontal.endswith("/"):
            config.path_horizontal += "/"
        lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
        compute_h = True
        lf_shape = lf3dh.shape
    except:
        pass

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        compute_v = True
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        if lf_shape is None:
            lf_shape = lf3dv.shape
    except:
        pass

    orientation = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    coherence = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    print "ok"

    for shift in config.global_shifts:

        if compute_h:
            print "compute horizontal shift {0}".format(shift), "...",
            lf3d = np.copy(lf3dh)
            lf3d = lfhelpers.refocus_3d(lf3d, shift, 'h')

            if config.color_space:
                lf3d = st2d.changeColorSpace(lf3d, config.color_space)

            if config.prefilter > 0:
                if config.prefilter == PREFILTER.IMGD:
                    lf3d = st2d.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='h')
                if config.prefilter == PREFILTER.EPID:
                    lf3d = st2d.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='h')
                if config.prefilter == PREFILTER.IMGD2:
                    lf3d = st2d.preImgLaplace(lf3d, scale=config.prefilter_scale)
                if config.prefilter == PREFILTER.EPID2:
                    lf3d = st2d.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='h')


            st3d = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.outer_scale, direction='h')
            if config.double_tensor > 0.0:
                tmp = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.double_tensor, direction='h')
                st3d[:] += tmp[:]
                st3d /= 2.0

            orientation_h, coherence_h = st2d.evaluateStructureTensor(st3d)
            orientation_h[:] += shift

            if config.coherence_threshold > 0.0:
                invalids = np.where(coherence_h < config.coherence_treshold)
                coherence_h[invalids] = 0.0

            if config.output_level == 3:
                misc.imsave(config.result_path+config.result_label+"orientation_h_shift_{0}.png".format(shift), orientation_h[lf_shape[0]/2, :, :])
            if config.output_level == 3:
                misc.imsave(config.result_path+config.result_label+"coherence_h_{0}.png".format(shift), coherence_h[lf_shape[0]/2, :, :])
            print "ok"

        if compute_v:
            print "compute vertical shift {0}".format(shift), "...",
            lf3d = np.copy(lf3dv)
            lf3d = lfhelpers.refocus_3d(lf3d, shift, 'v')

            if config.color_space:
                lf3d = st2d.changeColorSpace(lf3d, config.color_space)

            if config.prefilter > 0:
                if config.prefilter == PREFILTER.IMGD:
                    lf3d = st2d.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='v')
                if config.prefilter == PREFILTER.EPID:
                    lf3d = st2d.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='v')
                if config.prefilter == PREFILTER.IMGD2:
                    lf3d = st2d.preImgLaplace(lf3d, scale=config.prefilter_scale)
                if config.prefilter == PREFILTER.EPID2:
                    lf3d = st2d.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='v')

            st3d = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.outer_scale, direction='v')
            if config.double_tensor > 0.0:
                tmp = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.double_tensor, direction='v')
                st3d[:] += tmp[:]
                st3d /= 2.0

            orientation_v, coherence_v = st2d.evaluateStructureTensor(st3d)
            orientation_v[:] += shift

            if config.coherence_threshold > 0.0:
                invalids = np.where(coherence_v < config.coherence_treshold)
                coherence_v[invalids] = 0.0

            if config.output_level == 3:
                misc.imsave(config.result_path+config.result_label+"orientation_v_shift_{0}.png".format(shift), orientation_v[lf_shape[0]/2, :, :])
            if config.output_level == 3:
                misc.imsave(config.result_path+config.result_label+"coherence_v_{0}.png".format(shift), coherence_v[lf_shape[0]/2, :, :])
            print "ok"

        if compute_h and compute_v:
            print "merge vertical/horizontal ...",
            orientation_tmp, coherence_tmp = st2d.mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
            orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

            if config.output_level >= 2:
                misc.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :])
            print "ok"

        else:
            print "merge shifts"
            if compute_h:
                orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
            if compute_v:
               orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)
            if config.output_level >= 2:
                misc.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :])
            print "ok"


    invalids = np.where(coherence < 0.01)
    orientation[invalids] = 0

    if config.output_level >= 2:
        misc.imsave(config.result_path+config.result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :])
        misc.imsave(config.result_path+config.result_label+"coherence_final.png", coherence[lf_shape[0]/2, :, :])

    depth = dtc.disparity_to_depth(orientation[lf_shape[0]/2, :, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)

    if config.output_level >= 1:
        try:
            color = misc.imread(config.centerview_path)
        except:
            pass

        tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
        tmp[:, :, 0] = orientation[lf_shape[0]/2, :, :]
        tmp[:, :, 1] = coherence[lf_shape[0]/2, :, :]
        tmp[:, :, 2] = depth[:]
        vim = vigra.RGBImage(tmp)
        vim.writeImage(config.result_path+config.result_label+"final.exr")

        print "make pointcloud...",
        try:
            dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, color=color, focal_length=config.focal_length)
        except:
            dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, focal_length=config.focal_length)
        print "ok"
