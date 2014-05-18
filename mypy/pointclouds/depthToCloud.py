import numpy as np
import scipy.ndimage as nd
import vigra


def loadEXR(filename):
    return vigra.readImage(filename)[:, :, 0].transpose()


def cloud_from_depth(depth_map, focal_length):
    """
    computes a point cloud from a depth image by reprojecting from
    image space to world space. The returned cloud represents 3D
    position cloud[x,y,z].

    :param depth_map: ndarray depth data
    :param focal_length: float focal length in px
    :rtype : ndarray cloud
    """

    cloud = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
    cloud[:, :, 2] = depth_map[:]
    for y in xrange(depth_map.shape[0]):
        for x in xrange(depth_map.shape[1]):
            cloud[y, x, 0] = (x-depth_map.shape[1]/2.0)*depth_map[y, x]/focal_length
            cloud[y, x, 1] = (y-depth_map.shape[0]/2.0)*depth_map[y, x]/focal_length

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



