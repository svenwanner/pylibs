from mypy.pointclouds.PointCloud import Cloud
import numpy as np

class PlyWriter(object):
    def __init__(self, name=None, cloud=None):

        if name is not None:
            assert isinstance(name, str)
        if cloud is not None:
            assert isinstance(cloud, Cloud)

        self.name = name
        self.cloud = cloud

        self.save()

    def save(self):
        colors = None
        intensity = None
        confidence = None

        if self.cloud.has_color:
            assert isinstance(self.cloud.colors, np.ndarray)
            colors = True

        if self.intensity is not None:
            assert isinstance(self.intensity, np.ndarray)
            intensity = True

        if self.confidence is not None:
            assert isinstance(self.confidence, np.ndarray)
            confidence = True

        for p in xrange(self.cloud.points.width):
                #TODO here what should be done
                if self.cloud[y, x, 2] > 0:
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
