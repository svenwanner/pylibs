try:
    import pcl
except Exception as e:
    print "Warning: seems you haven't installed the python-pcl v1.0 - http://strawlab.github.io/python-pcl/"

import numpy as np

class Cloud(object):

    def __init__(self):
        self.points = None
        self.has_points = False
        self.colors = None
        self.has_color = False
        self.vertices = 0


    def save(self, filename):
        assert isinstance(filename, str)

        if filename.endswith(".pcd"):
            self.points.to_file(filename)

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
                    if point_index < 20:
                        print "read ", float(i)

                if self.has_points:
                    for k in range(3):
                        tmp_points[point_index, k] = numbers[k]
                if self.has_color:
                    for k in range(3):
                        self.colors[point_index, k] = int(numbers[k+3])

                point_index += 1

            self.points = pcl.PointCloud()
            self.points.from_array(tmp_points)
