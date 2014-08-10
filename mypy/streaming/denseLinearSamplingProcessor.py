# -*- coding: utf-8 -*-

import os
import sys
import glob
import vigra
import numpy as np
import scipy as sp
import scipy.misc as misc

from mypy.tools.cg import transformations as transformations

########################################################################################################################
#                                           P L Y   W R I T E R
########################################################################################################################

class PlyWriter(object):
    """
    The PlyWriter saves a pointcloud to file in the .ply format
    the class needs a filename<str> and a cloud<ndarray[n,4]>
    cloud dimensions are 0:x, 1:y, 2:z, 3:coherence
    additionally dimensions 4,5,6 can be set, these are interpreted as rgb
    rgb values can also be passed as ndarray with the same shape[0] as cloud in the constructor
    if no color is set in one of the ways described, the coherence is used as color
    """
    def __init__(self, filename=None, cloud=None, color=None, format="EN"):
        self.filename = filename
        self.cloud = cloud
        self.format = format
        self.color = None
        self.num_of_vertices = 0

    def __call__(self):
        self.save()

    def save(self, append=False):
        if not self.filename.endswith(".ply"):
            self.filename += ".ply"

        if append:
            f = open(self.filename, 'r')
            lines = f.readlines()
            f.close()
            f = open(self.filename, 'w')
            self.write_header(f)
            for n, line in enumerate(lines):
                if n > 10:
                    f.write(line)
        else:
            f = open(self.filename, 'w')

        self.write(f)
        f.close()

    def setColor(self, color=None):
        # if parameter is not None and is ndarray set this as color
        if type(color) is np.ndarray:
            assert self.cloud.shape[0] == color.shape[0]
            self.color = color
        else:
            # if cloud has 7 dimensions interpret last 3 as rgb
            if self.cloud.shape[1] == 7:
                self.color = self.cloud[:, 4:]
            # else use coherence as color
            else:
                self.color = np.zeros((self.cloud.shape[0], 1))
                self.color[:, 0] = self.cloud[:, 3]

        # re-range data to [0,1]
        min_col = np.amin(self.color)
        self.color = self.color.astype(np.float32)
        if min_col < 0:
            self.color[:] -= min_col
        max_color = np.amax(self.color)
        if max_color < 1.0:
            pass
        elif max_color > 1.0 and (max_color <= 10 or max_color > 255):
            self.color[:] /= max_color
        elif max_color > 1.0 and max_color <= 255:
            self.color[:] /= 255.0
        else:
            assert False, "color range cannot be handled!"

    def write_header(self, f):
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % self.num_of_vertices)
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if self.color is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')

    def write_points(self):
        string = ""
        for n in range(self.cloud.shape[0]):
            if self.cloud[n][2] > 0:
                line = ""
                line += "{0} {1} {2}".format(self.cloud[n, 0], self.cloud[n, 1], self.cloud[n, 2])
                if self.color is not None:
                    if self.color.shape[1] == 3:
                        line += " {0} {1} {2}".format(int(np.round(self.color[n, 0]*255)),
                                                      int(np.round(self.color[n, 1]*255)),
                                                      int(np.round(self.color[n, 2]*255)))
                    else:
                        line += " {0} {1} {2}".format(int(np.round(self.color[n, 0]*255)),
                                                      int(np.round(self.color[n, 0]*255)),
                                                      int(np.round(self.color[n, 0]*255)))
                line += "\n"
                if self.format == "DE":
                    line = line.replace(".", ",")
                string += line
                self.num_of_vertices += 1
        return string

    def write(self, f):
        line = self.write_points()
        self.write_header(f)
        f.write(line)

########################################################################################################################
#                              D E P T H   T O   W O R L D   P R O J E C T O R
########################################################################################################################

class DepthProjector(object):

    def __init__(self):
        self.depth_maps = []
        self.cameras = []
        self.cloud = np.zeros((0, 4), dtype=np.float64)

    def __call__(self, min_reliability=0.0):
        assert isinstance(min_reliability, float)
        self.reconstruct(min_reliability)

    def loadFromFiles(self, path, ftype="exr"):
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
            depth = np.transpose(np.array(vigra.readImage(f))[:, :, 0]).astype(np.float64)
            self.depth_maps.append(depth)

    def camerasFrom2Points(self, cam_pos1, cam_pos2, num_of_sampling_points, focal_length_mm, sensor_width_mm, resolution, euler_rotation_xyz):
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
        assert isinstance(cam_pos1, np.ndarray)
        assert cam_pos1.shape[0] == 3
        assert isinstance(num_of_sampling_points, int)
        assert isinstance(focal_length_mm, float)
        assert isinstance(sensor_width_mm, float)
        assert isinstance(resolution, type([]))
        assert isinstance(resolution[0], int)
        assert isinstance(euler_rotation_xyz, np.ndarray)
        assert euler_rotation_xyz.shape[0] == 3

        trail, baseline, direction, length = compute_linear_trail_from_translation(cam_pos1, num_of_sampling_points, baseline, translation_vector)
        for n in range(len(trail.keys())):
            self.cameras.append(Camera(focal_length_mm, sensor_width_mm, resolution))
            self.cameras[n].setPosition(trail[n])
            self.cameras[n].setRotation(euler_rotation_xyz)

    def transform(self, point, cam_index):
        #wm = np.linalg.inv(np.mat(self.cameras[n].world_matrix))
        wm = np.mat(self.cameras[cam_index].world_matrix, dtype=np.float64)
        return wm * point

    def reconstruct(self, min_reliability=0.0):
        assert isinstance(min_reliability, float)
        assert 0.0 <= min_reliability < 1.0

        for cam_index, depth_grid in enumerate(self.depth_maps):
            valid_coh = np.where(depth_grid[:, :, 1] > min_reliability)
            m = self.cloud.shape[0]
            self.cloud.resize((self.cloud.shape[0]+valid_coh[0].shape[0], 4))
            for y in xrange(depth_grid.shape[0]):
                for x in xrange(depth_grid.shape[1]):
                    coh = depth_grid[y, x, 1]
                    if coh > min_reliability:
                        self.cloud[m, 3] = depth_grid[y, x, 1]
                        _z = -depth_grid[y, x, 0]
                        if not np.isinf(_z):
                            _y = (float(y) - depth_grid.shape[0]/2.0) * _z/self.cameras[cam_index].f_px
                            _x = (float(x) - depth_grid.shape[1]/2.0) * _z/self.cameras[cam_index].f_px
                            point = np.mat([_x, _y, _z, 1], dtype=np.float64).T
                            if self.cameras[cam_index].world_matrix is not None:
                                point = self.transform(point, cam_index)
                            self.cloud[m, 0] = point[0, 0]
                            self.cloud[m, 1] = point[1, 0]
                            self.cloud[m, 2] = point[2, 0]
                        m += 1

    def save(self, filename, cformat="EN"):
        writer = PlyWriter(filename, self.cloud, format=cformat)
        writer.setColor(color=np.random.randint(0, 255, self.cloud.shape[0]*3).reshape((self.cloud.shape[0], 3)))
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
    #depthProjector.loadFromFiles("/home/swanner/Desktop/denseSampledTestScene/rendered5/low_res/depth")
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


if __name__ == "__main__":


    depthProjector = DepthProjector()

    d = np.transpose(np.array(vigra.readImage("/home/swanner/Desktop/tmp/depth/0001.exr")[:, :, 0]))
    c = np.random.randint(0, 1000, d.shape[0]*d.shape[1]).reshape((d.shape[0], d.shape[1])).astype(np.float32)
    c /= 1000.0
    depth1 = np.zeros((d.shape[0], d.shape[1], 2))
    depth1[:, :, 0] = np.copy(d[:])
    depth1[:, :, 1] = np.copy(c[:])
    cam1 = Camera(35.0, 32.0, [540, 960])
    cam1.setPosition(np.array((4.9950, -4.1860, 3.9597)))
    cam1.setRotation(np.array((1.1500, 0.0000, 0.8197)))
    cam1.name = "Camera 1"
    print cam1
    depthProjector.depth_maps.append(depth1)
    depthProjector.cameras.append(cam1)

    d = np.transpose(np.array(vigra.readImage("/home/swanner/Desktop/tmp/depth/0002.exr")[:, :, 0]))
    c = np.random.randint(0, 1000, d.shape[0]*d.shape[1]).reshape((d.shape[0], d.shape[1])).astype(np.float32)
    c /= 1000.0
    depth2 = np.zeros((d.shape[0], d.shape[1], 2))
    depth2[:, :, 0] = np.copy(d[:])
    depth2[:, :, 1] = np.copy(c[:])
    cam2 = Camera(35.0, 32.0, [540, 960])
    cam2.setPosition(np.array((-3.2872, 4.5016, 5.4028)))
    cam2.setRotation(np.array((0.8993, 0.0000, 3.7588)))
    cam2.name = "Camera 2"
    print cam2
    depthProjector.depth_maps.append(depth2)
    depthProjector.cameras.append(cam2)

    d = np.transpose(np.array(vigra.readImage("/home/swanner/Desktop/tmp/depth/0003.exr")[:, :, 0]))
    c = np.random.randint(0, 1000, d.shape[0]*d.shape[1]).reshape((d.shape[0], d.shape[1])).astype(np.float32)
    c /= 1000.0
    depth3 = np.zeros((d.shape[0], d.shape[1], 2))
    depth3[:, :, 0] = d[:]
    depth3[:, :, 1] = c[:]
    cam3 = Camera(35.0, 32.0, [540, 960])
    cam3.setPosition(np.array((-3.3543, -5.7408, 4.9266)))
    cam3.setRotation(np.array((1.0444, 0.0000, 5.7277)))
    cam3.name = "Camera 3"
    print cam3
    depthProjector.depth_maps.append(depth3)
    depthProjector.cameras.append(cam3)

    # d = np.transpose(np.array(vigra.readImage("/home/swanner/Desktop/tmp/depth/0004.exr")[:, :, 0]))
    # c = np.random.randint(0, 1000, d.shape[0]*d.shape[1]).reshape((d.shape[0], d.shape[1])).astype(np.float32)
    # c /= 1000.0
    # depth4 = np.zeros((d.shape[0], d.shape[1], 2))
    # depth4[:, :, 0] = d[:]
    # depth4[:, :, 1] = c[:]
    # cam4 = Camera(35.0, 32.0, [540, 960])
    # cam4.setPosition(np.array((-5.5117, 5.0170, 3.9854)))
    # cam4.setRotation(np.array((1.1575, 0.0000, 3.9534)))
    # depthProjector.depth_maps.append(depth4)
    # depthProjector.cameras.append(cam4)

    depthProjector(0.5)
    depthProjector.save("/home/swanner/Desktop/tmp/cloud.ply")

    print depthProjector.cloud.shape



