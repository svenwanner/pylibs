# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.misc import imsave

from mypy.streaming.camera import Camera
from mypy.streaming.plyWriter import PlyWriter
from mypy.streaming.INIReader import Parameter


from mypy.streaming.globals import DEBUG





class DepthAccumulator(object):
    def __init__(self):
        self.cameras = None
        self.parameter = None
        self.depthProjector = None
        self.plyWriter = PlyWriter()
        self.disparity_counter = 0

    def reset(self):
        self.cameras = []
        self.parameter = None
        self.disparity_counter = 0

    def setCounter(self, index):
        self.disparity_counter = index

    def initWorldGrid(self):
        assert self.parameter is not None, "Missing parameter object!"

    def setParameter(self, parameter):
        if parameter is not None:
            assert isinstance(parameter, Parameter), "Need a instance of the Parameter class!"
        assert isinstance(self.cameras, type([])), "Camera object is not a empty list, reset depthProjector before setting new parameter!"
        assert len(self.cameras) == 0, "Camera object is not empty, reset depthProjector before setting new parameter!"
        self.parameter = parameter

        # let the depthProjector compute all camera objects for camera trail
        self.cameras = self.camerasFromPointAndDirection(self.parameter.initial_camera_pos_m,
                                                         self.parameter.number_of_sampling_points,
                                                         self.parameter.baseline_mm,
                                                         self.parameter.camera_translation_vector,
                                                         self.parameter.focal_length_mm,
                                                         self.parameter.sensor_width_mm,
                                                         self.parameter.resolution_yx,
                                                         self.parameter.euler_rotation_xyz)

        self.plyWriter.setReferenceCamera(self.cameras[len(self.cameras)/2])
        self.plyWriter.setFilename(os.path.dirname(self.parameter.result_folder[0:-1]) + "/pointcloud.ply")
        if DEBUG >= 2: print "save pointcloud to:", self.plyWriter.filename

    def addDisparity(self, disparity, reliability, color):
        depth = self.disparity2Depth(disparity, reliability)
        if DEBUG >= 2:
            imsave(self.parameter.result_folder+"depth_%4.4i.png" % self.disparity_counter, depth)
            imsave(self.parameter.result_folder+"coherence_%4.4i.png" % self.disparity_counter, reliability)
            imsave(self.parameter.result_folder+"color_%4.4i.png" % self.disparity_counter, color)
        self.save(depth, reliability, color)

    def disparity2Depth(self, disparity, reliability):
        depth = np.zeros_like(disparity)
        depth[:] = self.parameter.focal_length_px * self.parameter.baseline_mm/(disparity[:]+1e-28)
        depth /= 1000.0
        np.place(depth, depth > self.parameter.max_depth_m, 0.0)
        np.place(depth, depth < self.parameter.min_depth_m, 0.0)
        np.place(depth, reliability < 0.01, 0.0)
        return depth

    def camerasFromPointAndDirection(self, cam_pos1, num_of_sampling_points, baseline, translation_vector, focal_length_mm, sensor_width_mm, resolution, euler_rotation_xyz):
        """
        Computes camera objects for a linear trail automatically from a start camera position,
        the number of sampling points, the baseline  a translation vector and the euler rotation
        of the cameras.
        :param cam_pos1: <ndarray> camera start position
        :param num_of_sampling_points: <int> number of cameras
        :param baseline: <float> distance between 2 cameras
        :param translation_vector: <ndarray> camera translation vector
        :param focal_length_mm: <float>
        :param sensor_width_mm: <float>
        :param resolution:  <[]>
        :param euler_rotation_xyz: <ndarray> rotational euler angles of the cameras
        """
        assert isinstance(cam_pos1, np.ndarray)
        assert cam_pos1.shape[0] == 3
        assert isinstance(num_of_sampling_points, int)
        assert isinstance(focal_length_mm, float)
        assert isinstance(sensor_width_mm, float)
        assert isinstance(resolution, np.ndarray)
        assert resolution.shape[0] == 2
        assert isinstance(resolution[0], int)
        assert isinstance(euler_rotation_xyz, np.ndarray)
        assert euler_rotation_xyz.shape[0] == 3

        cameras = []
        trail, baseline, direction, length = compute_linear_trail_from_translation(cam_pos1, num_of_sampling_points, baseline, translation_vector)
        for n in range(len(trail.keys())):
            cameras.append(Camera(focal_length_mm, sensor_width_mm, resolution))
            cameras[n].setPosition(trail[n])
            cameras[n].setRotation(euler_rotation_xyz)
        return cameras

    def save(self, depth, reliability, color):
        self.plyWriter.setDepthmap(depth, self.cameras[self.disparity_counter], reliability, color)
        self.plyWriter.save()


def compute_linear_trail_from_translation(pos1, num_of_sampling_points, baseline, translation_vector):
    """
    Returns dictionary with camera index and spacial position of the
    camera based on the initial camera position, number of sampling points,
    baseline and a camera tranlation unit vector.
    Further the baseline, the translation direction and the trail length
    is returned

    :param pos1: ndarray initial camera coordinates x,y,z
    :param num_of_sampling_points: int number of camera positions
    :param baseline: float distance between two neighbored cameras
    :param translation_vector: ndarray direction of camera movement
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
        trail[n] = pos1 + n * baseline/1000.0 * translation_vector

    return trail, baseline, translation_vector, length
