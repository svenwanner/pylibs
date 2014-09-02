# -*- coding: utf-8 -*-

import numpy as np

from mypy.tools.cg import transformations as transformations




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
        assert isinstance(resolution, np.ndarray)
        assert resolution.shape[0] == 2
        assert isinstance(resolution[0], int)

        self.name = "Camera"
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

    def __str__(self):
        props = "\n========== " + self.name + " ==============\n"
        props += "focal_length: " + str(self.f_mm) + "mm "+ str(self.f_px) + "px\n"
        props += "resolution: " + str(self.resolution)+ "\n"
        props += "sensor: " + str(self.sensor) + "\n"
        props += "position: " + str(self.position) + "\n"
        props += "look_at: " + str(self.look_at) + "\n"
        props += "euler_rotation_xyz: " + str(self.euler_rotation_xyz) + "\n"
        props += "rotation_matrix: " + str(self.rotation_matrix) + "\n"
        props += "world_matrix: " + str(self.world_matrix) + "\n"
        props += "max_depth: " + str(self.max_depth) + "\n"
        props += "===========================================\n"
        return props

    def setPosition(self, position):
        """
        changes the cameras position and if set the rotation
        :param position: ndarray camera coordinates x,y,z
        :param euler_rotation_xyz: ndarray of euler angles x,y,z [default=None]
        """
        assert isinstance(position, np.ndarray)
        assert position.shape[0] == 3

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
    assert np.allclose(cam.euler_rotation_xyz, euler_rotation_xyz)
    assert np.allclose(cam.rotation_matrix, rotation_matrix_gt)
    assert np.allclose(cam.world_matrix, world_matrix_gt)
    assert np.allclose(cam.look_at, look_at_gt)
#
#=======================================================================================================================
