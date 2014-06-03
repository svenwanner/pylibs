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
        self.mat_world = None
        self.mat_world_is_inverted = False

    def save(self, filename):
        """
        save point cloud depending on filetype as .pcd or .ply
        :param filename:
        """
        assert isinstance(filename, str), "save point cloud from file failed, argument error!"
        if filename.endswith(".pcd"):
            self.points.to_file(filename)
        if filename.endswith(".ply"):
            writer = PlyWriter(filename, self)

    def set_world_matrix(self, matrix_world):
        """
        set a 4 by 4 camera world matrix as list, as ndarray or as np.matrix type
        :param matrix_world: 4 by 4 matrix
        """
        if isinstance(matrix_world, type([])) or isinstance(matrix_world, type(())):
            assert len(matrix_world) == 4, "set world matrix failed, unknown matrix_world format!"
            assert len(matrix_world[0]) == 4, "set world matrix failed, unknown matrix_world format!"
            self.mat_world = np.mat(matrix_world)
        if isinstance(matrix_world, np.ndarray):
            assert matrix_world.shape == (4, 4), "set world matrix failed, unknown matrix_world format!"
            self.mat_world = np.mat(matrix_world)
        if isinstance(matrix_world, np.matrix):
            assert matrix_world.shape == (4, 4), "set world matrix failed, unknown matrix_world format!"
            self.mat_world = matrix_world


    def compute_world_matrix(self, translation=np.array([0, 0, 0]), rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, atype="rad"):
        """
        set the camera world matrix via translation vector and rotations around xyz axis
        :param translation: ndarray translation vector
        :param rotation_x: float rotation around x axis
        :param rotation_y: float rotation around y axis
        :param rotation_z: float rotation around z axis
        :param atype: str angle type ["rad"] or "deg"
        """
        assert isinstance(translation, np.ndarray), "transform point cloud failed, translation argument error!"
        assert isinstance(rotation_x, float), "transform point cloud failed, rotation_x argument error!"
        assert isinstance(rotation_y, float), "transform point cloud failed, rotation_y argument error!"
        assert isinstance(rotation_z, float), "transform point cloud failed, rotation_z argument error!"

        self.mat_world = np.mat([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.float32)

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

        self.mat_world = trans_mat*rot_mat_z*rot_mat_y*rot_mat_x

    def invert_world_matrix(self):
        """
        inverts the world matrix
        """
        assert isinstance(self.mat_world, np.matrix), "invert world matrix failed, translation argument error!"
        self.mat_world = self.mat_world.I
        self.mat_world_is_inverted = not self.mat_world_is_inverted

    def transform(self, translation=np.array([0, 0, 0]), rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, atype="rad"):
        """
        transforms a point cloud depending on the world matrix set. If no camera world matrix was set before the passed
        translation vector and rotations are used to compute it.
        :param translation: ndarray translation vector
        :param rotation_x: float rotation around x axis
        :param rotation_y: float rotation around y axis
        :param rotation_z: float rotation around z axis
        :param atype: str angle type ["rad"] or "deg"
        """
        assert self.has_points is True, "transform point cloud failed, no points available!"

        if self.mat_world is None:
            self.compute_world_matrix(translation, rotation_x, rotation_y, rotation_z, atype)

        tmp_points = np.zeros((self.vertices, 3), dtype=np.float32)

        for n in xrange(self.points.width):
            p = np.mat([0, 0, 0, 1], dtype=np.float32)
            p[0, 0:3] = self.points[n][:]

            rot_mat = np.mat([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            p = rot_mat * p.T
            p = self.mat_world * p
            p = p.T

            tmp_points[n, 0] = p[0, 0]
            tmp_points[n, 1] = p[0, 1]
            tmp_points[n, 2] = p[0, 2]

        self.points.from_array(tmp_points)

    def set(self, points, colors=None, intensity=None, confidence=None):

        if isinstance(points, np.ndarray):
            assert len(points.shape) == 3
            assert points.shape[2] == 3
            self.points = pcl.PointCloud()

            self.vertices = points.shape[0]*points.shape[1]
            tmp_points = np.zeros((self.vertices, 3), dtype=np.float32)

            if colors is not None and isinstance(colors, np.ndarray):
                assert (colors.shape[0] == points.shape[0] and colors.shape[1] == points.shape[1]), "failed to set color, shape mismatch!"
                self.colors = np.zeros((self.vertices, 3), dtype=int)
                self.has_color = True

            if intensity is not None and isinstance(intensity, np.ndarray):
                assert (intensity.shape[0] == intensity.shape[0] and intensity.shape[1] == intensity.shape[1]), "failed to set confidence, shape mismatch!"
                self.intensities = np.zeros(self.vertices, dtype=float)
                self.has_intensities = True

            if confidence is not None and isinstance(confidence, np.ndarray):
                assert (confidence.shape[0] == confidence.shape[0] and confidence.shape[1] == confidence.shape[1]), "failed to set confidence, shape mismatch!"
                self.confidence = np.zeros(self.vertices, dtype=float)
                self.has_confidence = True

            n = 0
            for y in range(points.shape[0]):
                for x in range(points.shape[1]):
                    tmp_points[n, :] = points[y, x, :]
                    if self.has_color:
                        if len(colors.shape) == 3 and colors.shape[2] >= 3:
                            self.colors[n, :] = colors[y, x, 0:3]
                        elif len(colors.shape) == 3 and colors.shape[2] == 1:
                            self.colors[n, 0] = colors[y, x, 0]
                            self.colors[n, 1] = colors[y, x, 0]
                            self.colors[n, 2] = colors[y, x, 0]
                        elif len(colors.shape) == 2:
                            self.colors[n, 0] = colors[y, x]
                            self.colors[n, 1] = colors[y, x]
                            self.colors[n, 2] = colors[y, x]
                        else:
                            assert False, "failed to set color, unknown array type!"
                    if self.has_intensities:
                        self.intensities[n] = intensity[y, x]
                    if self.has_confidence:
                        self.confidence[n] = confidence[y, x]
                    n += 1
                    
            self.points.from_array(tmp_points)

    def from_file(self, filename):
        """
        read a pointcloud from file
        :param filename: str filename of the point cloud. Possible types .pcd, .ply
        """
        assert isinstance(filename, str), "read point cloud from file failed, argument error!"

        p = pcl.PointCloud()

        if filename.endswith(".pcd"):
            p.from_file(filename)
        elif filename.endswith(".ply"):
            f = open(filename, "r")
            found_x = 0; found_y = 0; found_z = 0
            found_red = 0; found_green = 0; found_blue = 0
            intensity_index = 0
            confidence_index = 0

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
                if line.startswith("property float intensity"):
                    self.has_intensities = True
                if line.startswith("property float confidence"):
                    self.has_confidence = True
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
            if self.has_confidence:
                self.confidence = np.zeros(self.vertices, dtype=np.float32)
                if self.has_intensities:
                    confidence_index = 5
                else:
                    confidence_index = 4
            if self.has_intensities:
                intensity_index = 4
                self.intensities = np.zeros(self.vertices, dtype=np.float32)

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
                if self.intensities:
                    self.intensities[point_index] = int(numbers[k+intensity_index])
                if self.has_confidence:
                    self.confidence[point_index] = int(numbers[k+confidence_index])

                point_index += 1

            self.points = pcl.PointCloud()
            self.points.from_array(tmp_points)



