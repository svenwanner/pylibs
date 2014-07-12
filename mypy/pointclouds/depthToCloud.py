import numpy as np
import pylab as plt

def save_pointcloud(filename, depth_map=None, disparity_map=None, color=None, intensity=None, confidence=None, focal_length=None, base_line=None, min_depth=0.01, max_depth=10.0):
    """
    saves a pointcloud from a passed depth or disparity map, one is obligatory.

    :param filename: str filename of the resulting pointcloud
    :param depth_map: ndarray depth image
    :param disparity_map: ndarray disparity image
    :param color: ndarray image to be mapped onto pointcloud
    :param intensity: ndarray intensity array of the pointcloud
    :param confidence: ndarray confidence array of the pointcloud
    :param focal_length: float camera focal_length in px
    :param base_line:float distance between two cameras
    :param min_depth: float lower depth clipping value
    :param max_depth: float upper depth clipping value
    """
    assert isinstance(filename, str)

    print "save point cloud..."
    if depth_map is not None:
        assert isinstance(depth_map, np.ndarray)
    if disparity_map is not None:
        assert isinstance(disparity_map, np.ndarray)
    if color is not None:
        assert isinstance(color, np.ndarray)
    if intensity is not None:
        assert isinstance(intensity, np.ndarray)
    if confidence is not None:
        assert isinstance(confidence, np.ndarray)
    if focal_length is not None:
        assert isinstance(focal_length, float)
    if base_line is not None:
        assert isinstance(base_line, float)

    if depth_map is None and disparity_map is None:
        assert False, "need either a depth_map or a disparity_map!"
    if disparity_map is not None and (focal_length is None or base_line is None):
        assert False, "disparity_maps need focal_length and base_line as additional input!"
    if depth_map is not None and focal_length is None:
        assert False, "depth_map need focal_length as additional input!"

    if depth_map is None:
        depth_map = disparity_to_depth(disparity_map, base_line, focal_length, min_depth, max_depth)

    np.place(depth_map, depth_map < min_depth, -1.0)
    np.place(depth_map, depth_map > max_depth, -1.0)
    cloud = cloud_from_depth(depth_map, focal_length)
    plyWriter = PlyWriter(filename, cloud, color, intensity, confidence)


def disparity_to_depth(disparity, base_line, focal_length, min_depth=0.1, max_depth=1):
    """
    computes depth from disparity map

    :param disparity: ndarray disparity image
    :param base_line: float distance between two cameras
    :param focal_length: float camera focal_length in px
    :param min_depth: float lower depth clipping value
    :param max_depth: float upper depth clipping value
    :return: ndarray depth image
    """

    print "convert disparity to depth..."
    depth = np.zeros_like(disparity)
    for y in range(disparity.shape[0]):
        for x in range(disparity.shape[1]):
            depth[y, x] = focal_length*base_line/(disparity[y, x]+1e-16)
            if depth[y, x] > max_depth or depth[y, x] < min_depth or np.isinf(depth[y, x]):
                depth[y, x] = -1.0
    return depth


def cloud_from_depth(depth_map, focal_length):
    """
    computes a point cloud from a depth image by reprojecting from
    image space to world space. The returned cloud represents 3D
    position cloud[x,y,z].

    :param depth_map: ndarray depth data
    :param focal_length: float focal length in px
    :rtype : ndarray cloud
    """
    print "make cloud from depth..."
    cloud = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
    cloud[:, :, 2] = depth_map[:]

    for y in xrange(cloud.shape[0]):
        for x in xrange(cloud.shape[1]):
            if cloud[y, x, 2] > 0.0:
                cloud[y, x, 0] = (x-depth_map.shape[1]/2.0)*depth_map[y, x]/focal_length
                cloud[y, x, 1] = (y-depth_map.shape[0]/2.0)*depth_map[y, x]/focal_length
            else:
                cloud[y, x, 2] = -1.0

    return cloud


class PlyWriter(object):
    def __init__(self, name=None, cloud=None, colors=None, intensity=None, confidence=None):
        self.name = name
        self.cloud = cloud
        self.colors = colors
        self.intensity = intensity
        self.confidence = confidence

        if name is not None and cloud is not None:
            self.save()

    def save(self):
        assert isinstance(self.cloud, np.ndarray)
        assert isinstance(self.name, str)

        if not self.name.endswith(".ply"):
            self.name += ".ply"

        points = []
        colors = None
        intensity = None
        confidence = None

        if self.colors is not None:
            colors = []
            assert isinstance(self.colors, np.ndarray)
            assert (self.colors.shape[0] == self.cloud.shape[0] and self.colors.shape[1] == self.cloud.shape[1]), "Shape mismatch between color and cloud!"

        if self.intensity is not None:
            intensity = []
            assert isinstance(self.intensity, np.ndarray)
            assert (self.intensity.shape[0] == self.cloud.shape[0] and self.intensity.shape[1] == self.cloud.shape[1]), "Shape mismatch between intensity and cloud!"

        if self.confidence is not None:
            confidence = []
            assert isinstance(self.confidence, np.ndarray)
            assert (self.confidence.shape[0] == self.cloud.shape[0] and self.confidence.shape[1] == self.cloud.shape[1]), "Shape mismatch between confidence and cloud!"


        for y in xrange(self.cloud.shape[0]):
            for x in xrange(self.cloud.shape[1]):

                if self.cloud[y, x, 2] > 0.0:
                    points.append([self.cloud[y, x, 0], self.cloud[y, x, 1], self.cloud[y, x, 2]])

                    if self.colors is not None:
                        colors.append([self.colors[y, x, 0], self.colors[y, x, 1], self.colors[y, x, 2]])
                    if self.intensity is not None:
                        intensity.append(self.intensity[y, x])
                    if self.confidence is not None:
                        confidence.append(self.confidence[y, x])

        f = open(self.name, 'w')

        self.write_header(f, points)
        self.write_points(f, points, colors, confidence, intensity)

        f.close()

    def write_header(self, f, points):
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % len(points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if self.colors is not None:
            self.add_color_header(f)
        if self.intensity is not None:
            self.add_intensity_header(f)
        if self.confidence is not None:
            self.add_confidence_header(f)
        f.write('end_header\n')

    def add_color_header(self, f):
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')

    def add_confidence_header(self, f):
        f.write('property float confidence\n')

    def add_intensity_header(self, f):
        f.write('property float intensity\n')

    def write_points(self, f, points, colors=None, confidence=None, intensity=None):
        for n, point in enumerate(points):
            f.write("{0} {1} {2}".format(point[0], point[1], point[2]))
            if colors is not None:
                f.write(" {0} {1} {2}".format(colors[n][0], colors[n][1], colors[n][2]))
            if intensity is not None:
                f.write(" {0}".format(intensity[n]))
            if confidence is not None:
                f.write(" {0}".format(confidence[n]))
            f.write('\n')



