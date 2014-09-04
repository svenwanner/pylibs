# -*- coding: utf-8 -*-

import os
import glob
import vigra
import numpy as np

from mypy.streaming.camera import Camera
from mypy.streaming.plyWriter import PlyWriter
from mypy.tools.cg import transformations as transformations

from mypy.streaming.globals import DEBUG

########################################################################################################################
#                              D E P T H   T O   W O R L D   P R O J E C T O R
########################################################################################################################

class DepthProjector(object):
    """
    This class reprojects multiple depth maps from corresponding
    camera instances. The class need a list of depth maps and a list
    of cameras. Calling the class or using reproject(...) computes a
    cloud which can be saved using save(...) .
    """

    def __init__(self):
        self.depth_maps = []
        self.colors = []
        self.cameras = []
        self.cloud = np.zeros((0, 4), dtype=np.float64)

    def __call__(self, min_reliability=0.0):
        """
        Calling the class calls reconstruct
        :param min_reliability: <float> coherence threshold, points below are ignored
        """
        assert isinstance(min_reliability, float)
        self.reconstruct(min_reliability)

    def loadFromFiles(self, path, ftype="exr"):
        """
        Loads a list of depth maps from file
        :param path: <str> path to depth map files
        :param ftype: <str> file type default: "exr"
        """
        assert isinstance(path, str)
        assert isinstance(ftype, str)
        if not path.endswith(os.path.sep):
            path += os.path.sep
        if not ftype.startswith("."):
            ftype = "."+ftype

        filenames = []
        for f in glob.glob(path+"*"+ftype):
            filenames.append(f)
        filenames.sort()

        for f in filenames:
            d = np.transpose(np.array(vigra.readImage(f))[:, :, 0]).astype(np.float64)
            depth = np.zeros((d.shape[0], d.shape[1], 2), dtype=np.float32)
            depth[:, :, 0] = d[:]
            depth[:, :, 1] = np.ones_like(d)
            self.depth_maps.append(depth)

    def addDepthMapFromFile(self, filename):
        """
        Add a depth map. Ensure that number of cameras is the same as number of depth maps.
        :param filename: <str>
        """
        assert isinstance(filename, str)
        assert os.path.isfile(filename), "file does not exist!"
        d = np.transpose(np.array(vigra.readImage(filename))[:, :, 0]).astype(np.float64)
        depth = np.zeros((d.shape[0], d.shape[1], 2), dtype=np.float32)
        depth[:, :, 0] = d[:]
        depth[:, :, 1] = np.ones_like(d)
        self.depth_maps.append(depth)

    def addDepthMap(self, depth_map):
        """
        Add a depth map from ndarray, if depth_map has 2 channels the second is interpreted
        as reliability. If not reliability is set to 1.
        :param depth_map: <ndarray>
        """
        assert isinstance(depth_map, np.ndarray)
        depth = np.zeros((depth_map.shape[0], depth_map.shape[1], 2), dtype=np.float32)
        if len(depth_map.shape) > 2:
            if depth_map.shape[2] == 2:
                depth = depth_map
            elif depth_map.shape[2] > 2:
                depth[:, :, 0] = depth_map[:, :, 0]
                depth[:, :, 1] = 1
        else:
            depth[:, :, 0] = depth_map[:, :]
            depth[:, :, 1] = 1
        self.depth_maps.append(depth)

    def addColor(self, img):
        """
        Add a color image. Ensure that number of color images is the same as number of cameras and number of depth maps.
        :param img: <ndarray>
        """
        assert isinstance(img, np.ndarray)
        self.colors.append(img)

    def addCamera(self, focal_length_mm, sensor_size_mm, resolution, position, rotation, name="Camera"):
        """
        Add a camera. Ensure that number of cameras is the same as number of depth maps.
        :param focal_length_mm: <float>
        :param sensor_size_mm: <float>
        :param resolution: <[]>
        :param position: <[]>
        :param rotation: <[]>
        :param name: <str> camera name, default is "Camera"
        """
        cam = Camera(focal_length_mm, sensor_size_mm, resolution)
        cam.setPosition(position)
        cam.setRotation(rotation)
        cam.name = name+"_%4.4i" % len(self.cameras)
        self.cameras.append(cam)

    def camerasFrom2Points(self, cam_pos1, cam_pos2, num_of_sampling_points, focal_length_mm, sensor_width_mm, resolution, euler_rotation_xyz):
        """
        Computes camera objects for a linear trail automatically from a start and a
        final camera position, the number of sampling points, and the euler rotation
        of the cameras.
        :param cam_pos1: <ndarray> camera start position
        :param cam_pos2: <ndarray> camera final position
        :param num_of_sampling_points: <int> number of cameras
        :param focal_length_mm: <float>
        :param sensor_width_mm: <float>
        :param resolution:  <[]>
        :param euler_rotation_xyz: <ndarray> rotational euler angles of the cameras
        """
        assert isinstance(cam_pos1, np.ndarray)
        assert cam_pos1.shape[0] == 3
        assert isinstance(cam_pos2, np.ndarray)
        assert cam_pos2.shape[0] == 3
        assert isinstance(num_of_sampling_points, int)
        assert isinstance(focal_length_mm, float)
        assert isinstance(sensor_width_mm, float)
        assert isinstance(resolution, type([]))
        assert isinstance(resolution[0], int)
        assert isinstance(euler_rotation_xyz, np.ndarray)
        assert euler_rotation_xyz.shape[0] == 3

        trail, baseline, direction, length = compute_linear_trail_from_positions(cam_pos1, cam_pos2, num_of_sampling_points)
        for n in range(len(trail.keys())):
            self.cameras.append(Camera(focal_length_mm, sensor_width_mm, resolution))
            self.cameras[n].setPosition(trail[n])
            self.cameras[n].setRotation(euler_rotation_xyz)

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

        trail, baseline, direction, length = compute_linear_trail_from_translation(cam_pos1, num_of_sampling_points, baseline, translation_vector)
        for n in range(len(trail.keys())):
            self.cameras.append(Camera(focal_length_mm, sensor_width_mm, resolution))
            self.cameras[n].setPosition(trail[n])
            self.cameras[n].setRotation(euler_rotation_xyz)

    def transform(self, point, cam_index):
        """
        Transforms a point using the camera world matrix
        :param point: <list> 4D point
        :param cam_index: <int> index of camera list
        :return: <list> transformed point
        """
        wm = np.mat(self.cameras[cam_index].world_matrix, dtype=np.float64)
        return wm * point

    def reconstruct(self, min_reliability=0.0):
        """
        Reconstructs a point cloud from depth maps and cameras. Number of both
        has to be same. Result is saved in self.cloud which can be saved using save()
        :param min_reliability: <float> coherence threshold, points below are ignored
        """
        assert isinstance(min_reliability, float)
        assert 0.0 <= min_reliability < 1.0, "invalid min reliability range!"
        assert len(self.depth_maps) == len(self.cameras), "number of depth maps and cameras unequal!"
        if len(self.colors) > 0:
            assert len(self.depth_maps) == len(self.colors), "number of depth maps and colors unequal!"

        #loop over depth maps
        for cam_index, depth_grid in enumerate(self.depth_maps):
            #get positions of valid depth values
            valid_coh = np.where(depth_grid[:, :, 1] > min_reliability)
            #m is the number of points already in the cloud
            m = self.cloud.shape[0]
            #resize the cloud array by the new number of points
            if len(self.colors) > 0:
                self.cloud.resize((self.cloud.shape[0]+valid_coh[0].shape[0], 7))
            else:
                self.cloud.resize((self.cloud.shape[0]+valid_coh[0].shape[0], 4))
            #loop over pixel domain
            for u in xrange(depth_grid.shape[0]):
                for v in xrange(depth_grid.shape[1]):
                    #check if coherence is valid
                    coh = depth_grid[u, v, 1]
                    if coh > min_reliability:
                        #write coherence into 4th dimension of the cloud
                        self.cloud[m, 3] = depth_grid[u, v, 1]
                        #get depth value
                        _z = -depth_grid[u, v, 0]
                        #ignore z=0 and z=inf
                        if not np.isinf(_z) and _z != 0.0:
                            #reproject to real coordinates
                            _y = (float(u) - depth_grid.shape[0]/2.0) * _z/self.cameras[cam_index].f_px
                            _x = -(float(v) - depth_grid.shape[1]/2.0) * _z/self.cameras[cam_index].f_px
                            point = np.mat([_x, _y, _z, 1], dtype=np.float64).T
                            #if world_matrix exist, transform point from camera to world cs
                            if self.cameras[cam_index].world_matrix is not None:
                                point = self.transform(point, cam_index)
                            self.cloud[m, 0] = point[0, 0]
                            self.cloud[m, 1] = point[1, 0]
                            self.cloud[m, 2] = point[2, 0]
                            #if color data available set rgb
                            if len(self.colors) > 0:
                                self.cloud[m, 4] = self.colors[0][u, v, 0]
                                self.cloud[m, 5] = self.colors[0][u, v, 1]
                                self.cloud[m, 6] = self.colors[0][u, v, 2]

                        m += 1

    def save(self, filename, cformat="EN"):
        """
        Saves the cloud to file. The format parameter controls the
        type of number saving. "EN": using . 0.01, "DE": using , 0,01
        :param filename: <str> filename to save cloud
        :param cformat: <str> number coding
        """
        writer = PlyWriter(filename, self.cloud, format=cformat)
        writer.setColor()
        writer.save()



#=======================================================================================================================
#                            T E S T   D E P T H   T O   W O R L D   P R O J E C T O R
def test_depthProjector():
    initial_position = np.array([2.63167, -5.28107, 4.63259])
    final_position = np.array([5.05848, -2.70652, 4.59432])
    number_of_sampling_points = 231
    baseline = 0.015383674768
    translation_vector = np.array([0.68587954, 0.72763471, -0.0108161])
    focal_length = 22.0
    sensor_width = 32.0
    resolution = [540, 960]
    euler_rotation_xyz = np.array([63.559/180.0*np.pi, 0.62/180.0*np.pi, 46.692/180.0*np.pi])
    positions_gt = [np.array([2.63167, -5.28107,  4.63259]),
                    np.array([2.64222135, -5.2698763, 4.63242361]),
                    np.array([5.05848, -2.70652, 4.59432])]
    world_matrices_gt = [
        np.array([(0.6858805418014526, -0.31737014651298523, 0.6548619270324707, 2.6316704750061035),
                    (0.7276337742805481, 0.31246858835220337, -0.6106656193733215, -5.281065464019775),
                    (-0.01081676036119461, 0.8953433036804199, 0.4452453553676605, 4.632586479187012),
                    (0.0, 0.0, 0.0, 1.0)]),
        np.array([(0.6858805418014526, -0.31737014651298523, 0.6548619270324707, 2.6422219276428223),
                    (0.7276337742805481, 0.31246858835220337, -0.6106656193733215, -5.269871711730957),
                    (-0.01081676036119461, 0.8953433036804199, 0.4452453553676605, 4.632420063018799),
                    (0.0, 0.0, 0.0, 1.0)]),
        np.array([(0.6858805418014526, -0.31737014651298523, 0.6548619270324707, 5.058481693267822),
                    (0.7276337742805481, 0.31246858835220337, -0.6106656193733215, -2.7065224647521973),
                    (-0.01081676036119461, 0.8953433036804199, 0.4452453553676605, 4.594315052032471),
                    (0.0, 0.0, 0.0, 1.0)])
    ]

    depthProjector = DepthProjector()
    depthProjector.camerasFrom2Points(initial_position, final_position, number_of_sampling_points, focal_length, sensor_width, resolution, euler_rotation_xyz)
    m = 0
    for n in [0, 1, 230]:
        assert np.allclose(depthProjector.cameras[n].position, positions_gt[m])
        assert np.allclose(depthProjector.cameras[n].world_matrix, world_matrices_gt[m], rtol=1e-04, atol=1e-05)
        m += 1

    depthProjector.camerasFromPointAndDirection(initial_position, number_of_sampling_points, baseline, translation_vector, focal_length, sensor_width, resolution, euler_rotation_xyz)
    m = 0
    for n in [0, 1, 230]:
        assert np.allclose(depthProjector.cameras[n].position, positions_gt[m])
        assert np.allclose(depthProjector.cameras[n].world_matrix, world_matrices_gt[m], rtol=1e-04, atol=1e-05)
        m += 1
#
#=======================================================================================================================





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
        trail[n] = pos1 + n * baseline * translation_vector

    return trail, baseline, translation_vector, length


def compute_linear_trail_from_positions(pos1, pos2, num_of_sampling_points):
    """
    Returns dictionary with camera index and spacial position of the
    camera based on two positions and the number of sampling points.
    Further the baseline, the translation direction and the trail length
    is returned

    :param pos1: ndarray initial camera coordinates x,y,z
    :param pos2: ndarray final camera coordinates x,y,z
    :param num_of_sampling_points: int number of camera positions
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
    trail_gt = {0: np.array([2.63167, -5.28107,  4.63259]), 1: np.array([2.64222135, -5.2698763, 4.63242361]), 230: np.array([5.05848, -2.70652, 4.59432])}
    baseline_gt = 0.015383674768
    translation_vector_gt = np.array([0.68587954 ,0.72763471, -0.0108161])
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
