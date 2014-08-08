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
#                                             C A M E R A  O B J E C T
########################################################################################################################

class Camera(object):
    """
    This class represents a camera by providing all important parameter.
    Accessible are:
    f_mm -> focal length in mm
    f_px -> focal length in px
    resolution -> sensor resolution
    position -> world position
    look_at -> camera look at vector
    euler_rotation_xyz -> euler rotation angles sorted x,y,z
    rotation_matrix -> camera rotation matrix
    world_matrix -> camera world matrix
    max_depth -> maximum distance the camera can see
    """

    def __init__(self, focal_length_mm, sensor_width_mm, resolution):

        assert isinstance(focal_length_mm, float)
        assert isinstance(sensor_width_mm, float)
        assert isinstance(resolution, type([]))
        assert len(resolution) == 2
        assert isinstance(resolution[0], int)

        self.f_mm = focal_length_mm
        self.f_px = focal_length_mm/sensor_width_mm * resolution[1]
        self.resolution = resolution
        self.sensor = [float(self.resolution[0])/float(self.resolution[1])*sensor_width_mm, sensor_width_mm]
        self.position = None
        self.look_at = None
        self.euler_rotation_xyz = None
        self.rotation_matrix = None
        self.world_matrix = None
        self.max_depth = None

    def setPosition(self, position):
        """
        set the camera position.

        :param position: ndarray camera coordinates x,y,z
        """
        self.position = position

    def setRotation(self, euler_rotation_xyz):
        """
        set the rotation angles. The function automatically computes
        the rotation and the world matrix and the look at vector as well.

        :param euler_rotation_xyz: ndarray of euler angles x,y,z
        """
        assert isinstance(euler_rotation_xyz, np.ndarray)
        assert euler_rotation_xyz.shape[0] == 3
        self.euler_rotation_xyz = euler_rotation_xyz
        ax = euler_rotation_xyz[0]
        ay = euler_rotation_xyz[1]
        az = euler_rotation_xyz[2]
        self.rotation_matrix = transformations.euler_matrix(ax, ay, az, axes='sxyz')
        self.world_matrix = transformations.compose_matrix(angles=euler_rotation_xyz, translate=self.position)
        self.compute_look_at()

    def compute_look_at(self):
        """
        computes the cameras look at vector
        """
        assert self.euler_rotation_xyz is not None
        assert self.rotation_matrix is not None
        self.look_at = np.mat([0.0, 0.0, -1.0, 0.0])
        self.look_at = self.look_at * np.linalg.inv(np.mat(self.rotation_matrix))
        self.look_at = np.array(self.look_at[0, 0:3])

    def changePosition(self, position, euler_rotation_xyz=None):
        """
        changes the cameras position and if set the rotation
        :param position: ndarray camera coordinates x,y,z
        :param euler_rotation_xyz: ndarray of euler angles x,y,z [default=None]
        """
        assert isinstance(position, np.ndarray)
        assert position.shape[0] == 3

        if euler_rotation_xyz is not None:
            self.setRotation(euler_rotation_xyz)

        self.position = position

#=======================================================================================================================
#
def test_Camera():
    focal_length = 22.0
    sensor_width = 32.0
    resolution = [540, 960]
    position = np.array([2.67388, -5.23629, 4.63192])
    euler_rotation_xyz = np.array([63.559/180.0*np.pi, 0.62/180.0*np.pi, 46.692/180.0*np.pi])

    look_at_gt = np.array([-0.65486208,  0.61066204, -0.44524995])
    rotation_matrix_gt = np.array([[0.6858798, -0.31737131, 0.65486208, 0.0],
                                   [0.72763439, 0.3124741, -0.61066204, 0.0],
                                   [-0.01082083, 0.89534093, 0.44524995, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])
    world_matrix_gt = np.array([[0.6858798, -0.31737131, 0.65486208, 2.67388],
                                [0.72763439, 0.3124741, -0.61066204, -5.23629],
                                [-0.01082083, 0.89534093, 0.44524995, 4.63192],
                                [0.0, 0.0, 0.0, 1.0]])


    cam = Camera(focal_length, sensor_width, resolution)
    cam.setPosition(position)
    cam.setRotation(euler_rotation_xyz)

    assert cam.f_mm == 22.0
    assert cam.f_px == 660.0
    assert cam.resolution == [540, 960]
    assert cam.sensor == [18.0, 32.0]
    assert np.allclose(cam.rotation_matrix, rotation_matrix_gt)
    assert np.allclose(cam.world_matrix, world_matrix_gt)
    assert np.allclose(cam.look_at, look_at_gt)
#
#=======================================================================================================================