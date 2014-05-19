import os
import vigra
import numpy as np

from mypy.lightfield import io as lfio
from mypy.visualization.imshow import imoverlay, imshow
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor2D as st2d

import scipy.misc as misc

path_horizontal = "/home/swanner/rexHome/Zeiss/Zeiss_MetalDummy1_20_03_2014/noise_15x15_cross_190514/h/"
path_vertical = "/home/swanner/rexHome/Zeiss/Zeiss_MetalDummy1_20_03_2014/noise_15x15_cross_190514/v/"
result_path = "/home/swanner/rexHome/Zeiss/Zeiss_MetalDummy1_20_03_2014/noise_15x15_cross_190514/results/"
result_label = "noPrefilter"
rgb = True

prefilter = False
inner_scale = 0.6
outer_scale = 2.0
global_shifts = [8, 9]


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
        #lf3d = st2d.preDerivation(lf3d)
        lf3d = lfhelpers.refocus_3d(lf3d, shift, 'h')
        if prefilter:
            lf3d = st2d.epiPreDerivation(lf3d, scale=0.1, direction='h')
        st3d = st2d.structureTensor2D(lf3d, inner_scale=inner_scale, outer_scale=outer_scale, direction='h')
        orientation_h, coherence_h = st2d.evaluateStructureTensor(st3d)
        orientation_h[:] += shift
        imshow(orientation_h[lf_shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"orientation_h_shift_{0}.png".format(shift), orientation_h[lf_shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"coherence_h_{0}.png".format(shift), coherence_h[lf_shape[0]/2, :, :])

    if compute_v:
        lf3d = np.copy(lf3dv)
        #lf3d = st2d.preDerivation(lf3d)
        lf3d = lfhelpers.refocus_3d(lf3d, shift, 'v')
        if prefilter:
            lf3d = st2d.epiPreDerivation(lf3d, scale=0.1, direction='v')
        st3d = st2d.structureTensor2D(lf3d, inner_scale=inner_scale, outer_scale=outer_scale, direction='v')
        orientation_v, coherence_v = st2d.evaluateStructureTensor(st3d)
        orientation_v[:] += shift
        imshow(orientation_h[lf_shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"orientation_v_shift_{0}.png".format(shift), orientation_v[lf_shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"coherence_v_{0}.png".format(shift), coherence_v[lf_shape[0]/2, :, :])

    if compute_h and compute_v:
        orientation_tmp, coherence_tmp, cam_labels_tmp = st2d.mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
        misc.imsave(result_path+result_label+"camLabels_shift_{0}.png".format(shift), cam_labels_tmp[lf3dh.shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"orientation_merged_shift_{0}.png".format(shift), orientation_tmp[lf_shape[0]/2, :, :])
        misc.imsave(result_path+result_label+"coherence_merged_{0}.png".format(shift), coherence_tmp[lf_shape[0]/2, :, :])

        orientation, coherence, cam_labels = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

    else:
       if compute_h:
           orientation, coherence, cam_labels = st2d.mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
       if compute_v:
           orientation, coherence, cam_labels = st2d.mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)


invalids = np.where(coherence < 0.01)
orientation[invalids] = 0
misc.imsave(result_path+result_label+"camLabels_final.png", cam_labels[lf_shape[0]/2, :, :])
tmp = vigra.colors.linearRangeMapping(orientation[lf_shape[0]/2, :, :], newRange=(0.0, 255.0))
misc.imsave(result_path+result_label+"orientation_final.png", tmp)
tmp = vigra.colors.linearRangeMapping(coherence[lf_shape[0]/2, :, :], newRange=(0.0, 255.0))
misc.imsave(result_path+result_label+"coherence_final.png", tmp)
