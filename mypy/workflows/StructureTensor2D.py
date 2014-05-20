import os
import vigra
import numpy as np


from mypy.lightfield.helpers import enum
from mypy.lightfield import io as lfio
from mypy.visualization.imshow import imoverlay, imshow
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor2D as st2d

import scipy.misc as misc
COLORSPACE = enum(RGB=0, LAB=1, LUV=2)
PREFILTER = enum(NO=0, IMGD=1, EPID=2, IMGD2=3, EPID2=4)

import mypy.pointclouds.depthToCloud as dtc

#path to data and result directory
path_horizontal = "/home/swanner/rexHome/Zeiss/Zeiss_MetalDummy1_20_03_2014/15x15_cross_190514/h/"
path_vertical = "/home/swanner/rexHome/Zeiss/Zeiss_MetalDummy1_20_03_2014/15x15_cross_190514/v/"
centerview_path = "/home/swanner/rexHome/Zeiss/Zeiss_MetalDummy1_20_03_2014/15x15_cross_190514/left.png"
result_path = "/home/swanner/rexHome/Zeiss/Zeiss_MetalDummy1_20_03_2014/15x15_cross_190514/results/"
result_label = "LaplaceImgPrefilter"

#list of global shifts in px
global_shifts = [8, 9]

# distance between two cameras in m
base_line = 0.001

# depth range possible
min_depth = 0.1
max_depth = 1.0

# focal length in pixel
focal_length = 5740.38

#lf is rgb
rgb = True

#color space RGB,LAB,LUV
color_space = COLORSPACE.RGB

#pefltering NO,IMGD,EPID,IMGD2,EPID2
prefilter=PREFILTER.IMGD2

#structure tensor scales
inner_scale = 0.6
outer_scale = 2.2




#============================================================================================================
#============================================================================================================
#============================================================================================================

if not result_path.endswith("/"):
    result_path += "/"
if not result_label.endswith("/"):
    result_label += "/"
if not os.path.isdir(result_path+result_label):
    os.makedirs(result_path+result_label)


compute_h = False
compute_v = False
lf_shape = None
if path_horizontal is not None:
    if not path_horizontal.endswith("/"):
        path_horizontal += "/"
    lf3dh = lfio.load_3d(path_horizontal, rgb=rgb)
    compute_h = True
    lf_shape = lf3dh.shape
if path_vertical is not None:
    if not path_vertical.endswith("/"):
        path_vertical += "/"
    compute_v = True
    lf3dv = lfio.load_3d(path_vertical, rgb=rgb)
    if lf_shape is None:
        lf_shape = lf3dv.shape

orientation = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
coherence = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
cam_labels = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.uint8)

for shift in global_shifts:

    if compute_h:
        lf3d = np.copy(lf3dh)
        lf3d = lfhelpers.refocus_3d(lf3d, shift, 'h')

        if color_space:
            lf3d = st2d.changeColorSpace(lf3d, color_space)

        if prefilter > 0:
            if prefilter == PREFILTER.IMGD:
                    lf3d = st2d.preImgDerivation(lf3d, scale=0.4, direction='h')
            if prefilter == PREFILTER.EPID:
                lf3d = st2d.preEpiDerivation(lf3d, scale=0.4, direction='h')
            if prefilter == PREFILTER.IMGD2:
                lf3d = st2d.preImgLaplace(lf3d, scale=0.4)
            if prefilter == PREFILTER.EPID2:
                lf3d = st2d.preEpiLaplace(lf3d, scale=0.4, direction='h')


        st3d = st2d.structureTensor2D(lf3d, inner_scale=inner_scale, outer_scale=outer_scale, direction='h')
        orientation_h, coherence_h = st2d.evaluateStructureTensor(st3d)
        orientation_h[:] += shift
        misc.imsave(result_path+result_label+"orientation_h_shift_{0}.png".format(shift), orientation_h[lf_shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"coherence_h_{0}.png".format(shift), coherence_h[lf_shape[0]/2, :, :])

    if compute_v:
        lf3d = np.copy(lf3dv)
        lf3d = lfhelpers.refocus_3d(lf3d, shift, 'v')

        if color_space:
            lf3d = st2d.changeColorSpace(lf3d, color_space)

        if prefilter > 0:
            if prefilter == PREFILTER.IMGD:
                lf3d = st2d.preImgDerivation(lf3d, scale=0.4, direction='v')
            if prefilter == PREFILTER.EPID:
                lf3d = st2d.preEpiDerivation(lf3d, scale=0.4, direction='v')
            if prefilter == PREFILTER.IMGD2:
                lf3d = st2d.preImgLaplace(lf3d, scale=0.4)
            if prefilter == PREFILTER.EPID2:
                lf3d = st2d.preEpiLaplace(lf3d, scale=0.4, direction='v')

        st3d = st2d.structureTensor2D(lf3d, inner_scale=inner_scale, outer_scale=outer_scale, direction='v')
        orientation_v, coherence_v = st2d.evaluateStructureTensor(st3d)
        orientation_v[:] += shift
        misc.imsave(result_path+result_label+"orientation_v_shift_{0}.png".format(shift), orientation_v[lf_shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"coherence_v_{0}.png".format(shift), coherence_v[lf_shape[0]/2, :, :])

    if compute_h and compute_v:
        orientation_tmp, coherence_tmp, cam_labels_tmp = st2d.mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
        orientation, coherence, cam_labels = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)
        misc.imsave(result_path+result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :])

    else:
       if compute_h:
           orientation, coherence, cam_labels = st2d.mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
       if compute_v:
           orientation, coherence, cam_labels = st2d.mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)


invalids = np.where(coherence < 0.01)
orientation[invalids] = 0
misc.imsave(result_path+result_label+"camLabels_final.png", cam_labels[lf_shape[0]/2, :, :])
misc.imsave(result_path+result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :])
misc.imsave(result_path+result_label+"coherence_final.png", coherence[lf_shape[0]/2, :, :])

depth = dtc.disparity_to_depth(orientation[lf_shape[0]/2, :, :], base_line, focal_length, min_depth, max_depth)
color = misc.imread(centerview_path)

tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
tmp[:, :, 0] = orientation[lf_shape[0]/2, :, :]
tmp[:, :, 1] = coherence[lf_shape[0]/2, :, :]
tmp[:, :, 2] = depth[:]
tmp[:, :, 3] = cam_labels[lf_shape[0]/2, :, :]
vim = vigra.RGBImage(tmp)
vim.writeImage(result_path+result_label+"final.exr")

dtc.save_pointcloud(result_path+result_label+"pointcloud.ply", depth_map=depth, color=color, focal_length=focal_length)
