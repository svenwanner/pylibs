try:
    import pcl
except Exception as e:
    print "Warning: seems you haven't installed the python-pcl v1.0 - http://strawlab.github.io/python-pcl/"

import numpy as np



class PlyWriter(object):
    def __init__(self, name=None, cloud=None):

        if name is not None:
            assert isinstance(name, str)
        if cloud is not None:
            assert isinstance(cloud, Cloud)

        self.name = name
        self.cloud = cloud

        if self.name is not None and self.cloud is not None:
            self.save()


    def save(self):
        assert self.name is not None, "no filename set, cannot save pointcloud!"
        assert self.cloud is not None, "no cloud set, cannot save pointcloud!"

        f = open(self.name, 'w')
        self.write_header(f)
        self.write_points(f)
        f.close()


    def write_header(self, f):
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % self.cloud.points.width)
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if self.cloud.has_color:
            self.add_color_header(f)
        if self.cloud.has_intensities:
            self.add_intensity_header(f)
        if self.cloud.has_confidence:
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

    def write_points(self, f):
        for n in xrange(self.cloud.points.width):
            f.write("{0} {1} {2}".format(float(self.cloud.points[n][0]), float(self.cloud.points[n][1]), float(self.cloud.points[n][2])))
            if self.cloud.has_color:
                f.write(" {0} {1} {2}".format(int(self.cloud.colors[n][0]), int(self.cloud.colors[n][1]), int(self.cloud.colors[n][2])))
            if self.cloud.has_intensities:
                f.write(" {0}".format(float(self.cloud.intensity[n])))
            if self.cloud.has_confidence:
                f.write(" {0}".format(float(self.cloud.confidence[n])))
            f.write('\n')


class Cloud(object):

    def __init__(self):
        self.points = None
        self.has_points = False
        self.colors = None
        self.has_color = False
        self.intensities = None
        self.has_intensities = False
        self.confidence = None
        self.has_confidence = False
        self.vertices = 0


    def save(self, filename):
        assert isinstance(filename, str)

        if filename.endswith(".pcd"):
            self.points.to_file(filename)
        if filename.endswith(".ply"):
            writer = PlyWriter(filename, self)


    def transform(self, translation=np.array([0, 0, 0]), rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, atype="rad"):

        assert isinstance(translation, np.ndarray), "need translation as parameter!"
        assert isinstance(rotation_x, float), "need rotation_x as parameter!"
        assert isinstance(rotation_y, float), "need rotation_y as parameter!"
        assert isinstance(rotation_z, float), "need rotation_z as parameter!"
        assert self.has_points is True, "no points available, transform failed!"

        if atype != "rad":
            rotation_x = rotation_x/180.0*np.pi
            rotation_y = rotation_y/180.0*np.pi
            rotation_z = rotation_z/180.0*np.pi

        trans_mat = np.mat([[1, 0, 0, translation[0]], [0, 1, 0, translation[1]], [0 ,0, 1, translation[2]], [0, 0, 0, 1]], dtype=np.float32)
        rot_mat_x = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        rot_mat_y = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        rot_mat_z = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        rot_mat_x[1, 1] = np.cos(rotation_x)
        rot_mat_x[1, 2] = -np.sin(rotation_x)
        rot_mat_x[2, 1] = np.sin(rotation_x)
        rot_mat_x[2, 2] = np.cos(rotation_x)

        rot_mat_y[0, 0] = np.cos(rotation_y)
        rot_mat_y[0, 2] = np.sin(rotation_y)
        rot_mat_y[2, 0] = -np.sin(rotation_y)
        rot_mat_y[2, 2] = np.cos(rotation_y)

        rot_mat_z[0, 0] = np.cos(rotation_z)
        rot_mat_z[0, 1] = -np.sin(rotation_z)
        rot_mat_z[1, 0] = np.sin(rotation_z)
        rot_mat_z[1, 1] = np.cos(rotation_z)

        tmp_points = np.zeros((self.vertices, 3), dtype=np.float32)

        for n in xrange(self.points.width):
            p = np.mat([0, 0, 0, 1],dtype=np.float32)
            p[0, 0:3] = self.points[n][:]

            p = rot_mat_x * p.T
            p = rot_mat_y * p
            p = rot_mat_z * p
            p = trans_mat * p
            p = p.T

            tmp_points[n, 0] = p[0, 0]
            tmp_points[n, 1] = p[0, 1]
            tmp_points[n, 2] = p[0, 2]

        self.points.from_array(tmp_points)


    def from_file(self, filename):
        assert isinstance(filename, str)

        p = pcl.PointCloud()

        if filename.endswith(".pcd"):
            p.from_file(filename)
        elif filename.endswith(".ply"):
            f = open(filename, "r")
            found_x = 0; found_y = 0; found_z = 0
            found_red = 0; found_green = 0; found_blue = 0

            for line in f:
                if line.startswith("element vertex"):
                    self.vertices = int(line[15:-1])
                if line.startswith("property float x"):
                    found_x = 1
                if line.startswith("property float y"):
                    found_y = 1
                if line.startswith("property float z"):
                    found_z = 1
                if line.startswith("property uchar red"):
                    found_red = 1
                if line.startswith("property uchar green"):
                    found_green = 1
                if line.startswith("property uchar blue"):
                    found_blue = 1
                if found_x+found_y+found_z == 3:
                    self.has_points = True
                if found_red+found_green+found_blue == 3:
                    self.has_color = True
                if line.startswith("end_header"):
                    break

            point_index = 0
            tmp_points = None

            if self.has_points:
                tmp_points = np.zeros((self.vertices, 3), dtype=np.float32)
            if self.has_color:
                self.colors = np.zeros((self.vertices, 3), dtype=int)

            for line in f:
                numbers = []
                for i in line.split(" "):
                    numbers.append(float(i))

                if self.has_points:
                    for k in range(3):
                        tmp_points[point_index, k] = numbers[k]
                if self.has_color:
                    for k in range(3):
                        self.colors[point_index, k] = int(numbers[k+3])

                point_index += 1

            self.points = pcl.PointCloud()
            self.points.from_array(tmp_points)



