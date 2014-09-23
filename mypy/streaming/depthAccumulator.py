# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.misc import imsave
from mypy.streaming.INIReader import Parameter
from mypy.streaming.depthProjector import DepthProjector

from mypy.streaming.globals import DEBUG


class BackProjector(object):

    def __init__(self):
        pass



    def addDepth(self):


class DepthAccumulator(object):
    def __init__(self):
        self.cameras = None
        self.parameter = None
        self.depthProjector = None
        self.disparity_counter = 0

    def reset(self):
        self.cameras = []
        self.parameter = None
        self.backProjector = BackProjector()
        self.disparity_counter = 0

    def setCounter(self, index):
        self.disparity_counter = index


    def initWorldGrid(self):
        assert self.parameter is not None, "Missing parameter object!"


    def setParameter(self, parameter):
        if parameter is not None:
            assert isinstance(parameter, Parameter), "Need a instance of the Parameter class!"
        assert isinstance(self.depthProjector, DepthProjector), "Wrong type depthProjector, reset depthProjector before setting new parameter!"
        assert isinstance(self.cameras, type([])), "Camera object is not a empty list, reset depthProjector before setting new parameter!"
        assert len(self.cameras) == 0, "Camera object is not empty, reset depthProjector before setting new parameter!"
        self.parameter = parameter

        # let the depthProjector compute all camera objects for camera trail
        self.cameras = self.depthProjector.camerasFromPointAndDirection(self.parameter.initial_camera_pos_m,
                                                         self.parameter.number_of_sampling_points,
                                                         self.parameter.baseline_mm,
                                                         self.parameter.camera_translation_vector,
                                                         self.parameter.focal_length_mm,
                                                         self.parameter.sensor_width_mm,
                                                         self.parameter.resolution_yx,
                                                         self.parameter.euler_rotation_xyz)

        pc_filename = os.path.dirname(self.parameter.result_folder[0:-1]) + "/pointcloud.ply"
        if DEBUG >= 2: print "save pointcloud to:", pc_filename
        self.depthProjector.cloud_filename = pc_filename

    def addDisparity(self, disparity, reliability, color):
        depth = self.disparity2Depth(disparity, reliability)
        if DEBUG >= 2:
            imsave(self.parameter.result_folder+"depth_%4.4i.png" % self.disparity_counter, depth)
            imsave(self.parameter.result_folder+"coherence_%4.4i.png" % self.disparity_counter, reliability)
            imsave(self.parameter.result_folder+"color_%4.4i.png" % self.disparity_counter, color)

        # self.depthProjector.cameras.append(self.cameras[self.disparity_counter])
        # self.depthProjector.addDepthMap(depth, reliability)
        # self.depthProjector.addColor(color)
        # self.depthProjector.reconstruct()


    def disparity2Depth(self, disparity, reliability):
        depth = np.zeros_like(disparity)
        depth[:] = self.parameter.focal_length_px * self.parameter.baseline_mm/(disparity[:]+1e-28)
        depth /= 1000.0
        np.place(depth, depth > self.parameter.max_depth_m, 0.0)
        np.place(depth, depth < self.parameter.min_depth_m, 0.0)
        np.place(depth, reliability < 0.01, 0.0)
        return depth


    # def save(self):
    #     self.depthProjector.reconstruct()
    #     pc_filename = os.path.dirname(self.parameter.result_folder[0:-1]) + "/pointcloud.ply"
    #     if DEBUG >= 2: print "save pointcloud to:", pc_filename
    #     self.depthProjector.save(pc_filename)


