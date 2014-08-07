# -*- coding: utf-8 -*-

import vigra
import numpy as np
import scipy as sp

from mypy.tools.cg import transformations as transformations


########################################################################################################################
#                               C A M E R A   T R A I L   C O M P U T A T I O N
########################################################################################################################

def compute_linear_trail_from_translation(pos1, num_of_sampling_points, baseline, translation_vector):
    """
    Returns dictionary with camera index and spacial position of the
    camera based on the initial camera position, number of sampling points,
    baseline and a camera tranlation unit vector.
    Further the baseline, the translation direction and the trail length
    is returned
    """

    assert isinstance(pos1, np.ndarray)
    assert pos1.shape[0] == 3
    assert isinstance(num_of_sampling_points, int)
    assert isinstance(baseline, float)
    assert isinstance(translation_vector, np.ndarray)
    assert translation_vector.shape[0] == 3

    length = baseline * (num_of_sampling_points-1)

    trail = {}
    for n in range(num_of_sampling_points):
        trail[n] = pos1 + n * baseline * translation_vector

    return trail, baseline, translation_vector, length


def compute_linear_trail_from_positions(pos1, pos2, num_of_sampling_points):
    """
    Returns dictionary with camera index and spacial position of the
    camera based on two positions and the number of sampling points.
    Further the baseline, the translation direction and the trail length
    is returned
    """

    assert isinstance(pos1, np.ndarray)
    assert pos1.shape[0] == 3
    assert isinstance(pos2, np.ndarray)
    assert pos2.shape[0] == 3
    assert isinstance(num_of_sampling_points, int)

    trail_vec = pos2-pos1
    length = transformations.vector_length(trail_vec)
    baseline = length/(num_of_sampling_points-1)
    direction = transformations.vector_normed(trail_vec)

    trail = {}
    for n in range(num_of_sampling_points):
        trail[n] = pos1 + n * baseline * direction

    return trail, baseline, direction, length



#=======================================================================================================================
#                               T E S T   C A M E R A   T R A I L   C O M P U T A T I O N
def test_compute_linear_trail():
    trail_gt = {0: np.array([ 2.63167, -5.28107,  4.63259]), 1: np.array([2.64222135, -5.2698763 ,  4.63242361]), 230: np.array([ 5.05848, -2.70652,  4.59432])}
    baseline_gt = 0.015383674768
    translation_vector_gt = np.array([ 0.68587954  ,0.72763471, -0.0108161 ])
    length_gt = 3.53824519663

    #TODO: maximum depth of scene is maximum_scene_depth = 12 m

    initial_position = np.array([2.63167, -5.28107, 4.63259])
    final_position = np.array([5.05848, -2.70652, 4.59432])
    number_of_sampling_points = 231
    trail, baseline, translation_vector, length = compute_linear_trail_from_positions(initial_position, final_position, number_of_sampling_points)

    assert np.allclose(baseline, baseline_gt)
    assert np.allclose(length_gt, length)
    assert np.allclose(translation_vector, translation_vector_gt)
    for key in trail_gt.keys():
        assert np.allclose(trail[key], trail_gt[key])

    trail, baseline, translation_vector, length = compute_linear_trail_from_translation(initial_position, number_of_sampling_points, baseline, translation_vector)

    assert np.allclose(baseline, baseline_gt)
    assert np.allclose(length_gt, length)
    assert np.allclose(translation_vector, translation_vector_gt)
    for key in trail_gt.keys():
        assert np.allclose(trail[key], trail_gt[key])
#
#=======================================================================================================================







########################################################################################################################
#                           V I S I B L E   S C E N E F R A M E    C O M P U T A T I O N
########################################################################################################################

def compute_visible_sceneframe(camera_position, look_at_vector, focal_length_mm, sensor_width_mm, sensor_resolution_yx, maximum_depth):
    """
    """
    assert isinstance(camera_position, np.ndarray)
    assert camera_position.shape[0] == 3
    assert isinstance(look_at_vector, np.ndarray)
    assert look_at_vector.shape[0] == 3
    assert isinstance(focal_length_mm, float)
    assert isinstance(sensor_width_mm, float)
    assert isinstance(sensor_resolution_yx, type([]))
    assert isinstance(maximum_depth, float)

    sensorSize_y = sensor_width_mm * sensor_resolution_yx[0]/float(sensor_resolution_yx[1])
    fov_v = np.arctan2(sensor_width_mm, 2.0*focal_length_mm)
    fov_h = np.arctan2(sensorSize_y, 2.0*focal_length_mm)
    # vwsx = 2.0*parameter["camInitialPos"][2]*np.tan(parameter["fov"])+parameter["maxBaseline_mm"]/100.0
    # vwsy = 2.0*parameter["camInitialPos"][2]*np.tan(fov_h)

