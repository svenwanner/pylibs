# -*- coding: utf-8 -*-

import numpy as np
from mypy.streaming.camera import Camera

########################################################################################################################
#                                           P L Y   W R I T E R
########################################################################################################################

class PlyWriter(object):

    def __init__(self, filename=None):
        if filename is not None:
            self.setFilename(filename)
        self.num_of_vertices = 0
        self.cloud = None
        self.color = None
        self.reliability = None
        self.format = "EN"
        self.header_exist = False
        self.save_round = 1
        self.save_reliability_as_color = False
    def setFilename(self, filename):
        assert isinstance(filename, str)
        if not filename.endswith(".ply"):
            filename += ".ply"
        self.filename = filename
        self.outfile = open(self.filename, "w")


    def saveReliabilityAsColor(self, status=True):
        assert isinstance(status, bool)
        self.save_reliability_as_color = status

    def setFormat2German(self):
        self.format = "DE"

    def setFormat2English(self):
        self.format = "EN"

    def setCloud(self, cloud):
        assert isinstance(cloud, np.ndarray)
        self.cloud = cloud

    def setReliability(self, reliability):
        assert isinstance(reliability, np.ndarray)
        self.reliability = reliability

    def setColor(self, color=None):
        assert isinstance(color, np.ndarray)
        self.color = color

    def getPointTransform(self, ref_cam, cam):
        assert isinstance(ref_cam, Camera)
        assert isinstance(cam, Camera)

    def transform(self, point, camera):
        wm = np.mat(camera.world_matrix, dtype=np.float64)
        return wm * point

    def setDepthmap(self, depth, camera, reliability=None, color=None):
        assert isinstance(depth, np.ndarray)
        assert isinstance(camera, Camera)

        valid = None
        self.cloud = None
        self.reliability = None
        self.cloud = None

        if reliability is not None:
            valid = np.where(reliability != 0)
            self.cloud = np.zeros((len(valid[0]), 3), dtype=np.float32)
            self.reliability = np.zeros((len(valid[0])), dtype=np.float32)
            if color is not None:
                self.color = np.zeros((len(valid[0]), 3), dtype=np.uint8)
        else:
            reliability = np.ones_like(depth)
            self.cloud = np.zeros((depth.shape[0]*depth.shape[1], 3), dtype=np.float32)
            self.reliability = np.zeros((len(valid[0])), dtype=np.float32)
            if color is not None:
                self.color = np.zeros((depth.shape[0]*depth.shape[1], 3), dtype=np.uint8)

        #transform = self.getPointTransform(self.ref_cam, camera)

        n = 0
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                if reliability[y, x] > 0:
                    _x = -float((float(x)-depth.shape[1]/2.0)*depth[y, x]/camera.f_px)
                    _y = float((float(y)-depth.shape[0]/2.0)*depth[y, x]/camera.f_px)
                    _z = -depth[y, x]
                    point = np.mat([_x, _y, _z, 1], dtype=np.float64).T
                    point = self.transform(point, camera)
                    self.cloud[n, 0] = point[0, 0]
                    self.cloud[n, 1] = point[1, 0]
                    self.cloud[n, 2] = point[2, 0]
                    if reliability is not None:
                        self.reliability[n] = reliability[y, x]
                    if color is not None:
                        self.color[n, 0] = color[y, x, 0]
                        self.color[n, 1] = color[y, x, 1]
                        self.color[n, 2] = color[y, x, 2]
                    n += 1


    def write_header(self, outfile):
        outfile.write('ply\n')
        outfile.write('format ascii 1.0\n')
        outfile.write('element vertex %d\n' % self.num_of_vertices)
        outfile.write('property float x\n')
        outfile.write('property float y\n')
        outfile.write('property float z\n')
        if self.color is not None or self.save_reliability_as_color:
            outfile.write('property uchar red\n')
            outfile.write('property uchar green\n')
            outfile.write('property uchar blue\n')
        if self.reliability is not None:
            outfile.write('property float reliability\n')
        outfile.write('end_header\n')

    def write_points(self):
        string = ""
        for n in range(self.cloud.shape[0]):
            reliability = 1.0
            if self.reliability is not None:
                reliability = self.reliability[n]
            if reliability != 0:
                line = ""
                line += "{0} {1} {2}".format(self.cloud[n, 0], self.cloud[n, 1], self.cloud[n, 2])
                if self.color is not None or self.save_reliability_as_color:
                    if self.save_reliability_as_color:
                        line += " {0} {1} {2}".format(int(np.round(reliability*255)),
                                                      int(np.round(reliability*255)),
                                                      int(np.round(reliability*255)))
                    elif self.color.shape[1] == 3:
                        line += " {0} {1} {2}".format(int(np.round(self.color[n, 0])),
                                                      int(np.round(self.color[n, 1])),
                                                      int(np.round(self.color[n, 2])))
                    elif self.color.shape[1] == 1:
                        line += " {0} {1} {2}".format(int(np.round(self.color[n, 0])),
                                                      int(np.round(self.color[n, 0])),
                                                      int(np.round(self.color[n, 0])))


                if self.reliability is not None:
                    line += " {0}".format(float(reliability))
                line += "\n"
                if self.format == "DE":
                    line = line.replace(".", ",")
                string += line
                self.num_of_vertices += 1
        return string

    def save(self):
        self.save_round += 1
        self.num_of_vertices = 0
        line = self.write_points()
        if self.header_exist:
            tmp_file = open(self.filename, 'r')
            tmp = tmp_file.readlines()
            tmp_file.close()

            self.outfile = open(self.filename, 'w')
            old_num_of_vertices = int(tmp[2].split(" ")[2])
            self.num_of_vertices += old_num_of_vertices
            self.write_header(self.outfile)
            while True:
                if tmp[0].startswith("end"):
                    tmp.pop(0)
                    break
                else:
                    tmp.pop(0)

            tmp_line = ""
            for l in tmp:
                tmp_line += l
            line = tmp_line + line
        else:
            self.write_header(self.outfile)

        self.outfile.write(line)
        self.header_exist = True
        self.outfile.close()







# class PlyWriter(PlyWriterBase):
#     """
#     The PlyWriter saves a pointcloud to file in the .ply format
#     the class needs a filename<str> and a cloud<ndarray[n,4]>
#     cloud dimensions are 0:x, 1:y, 2:z, 3:coherence
#     additionally dimensions 4,5,6 can be set, these are interpreted as rgb
#     rgb values can also be passed as ndarray with the same shape[0] as cloud in the constructor
#     if no color is set in one of the ways described, the coherence is used as color
#     """
#     def __init__(self, filename=None, cloud=None, color=None, format="EN"):
#         PlyWriterBase.__init__(self)
#
#         self.filename = filename
#         self.cloud = cloud
#         self.format = format
#         self.color = None
#         self.num_of_vertices = 0
#
#     def __call__(self):
#         self.save()
#
#     def save(self, append=False):
#         if not self.filename.endswith(".ply"):
#             self.filename += ".ply"
#
#         if append:
#             f = open(self.filename, 'r')
#             lines = f.readlines()
#             f.close()
#             f = open(self.filename, 'w')
#             self.write_header(f)
#             for n, line in enumerate(lines):
#                 if n > 10:
#                     f.write(line)
#         else:
#             f = open(self.filename, 'w')
#
#         self.write(f)
#         f.close()
#
#     def setColor(self, color=None):
#         # if parameter is not None and is ndarray set this as color
#         if type(color) is np.ndarray:
#             assert self.cloud.shape[0] == color.shape[0]
#             self.color = color
#         else:
#             # if cloud has 7 dimensions interpret last 3 as rgb
#             if self.cloud.shape[1] == 7:
#                 self.color = self.cloud[:, 4:]
#             # else use coherence as color
#             else:
#                 self.color = np.zeros((self.cloud.shape[0], 1))
#                 self.color[:, 0] = self.cloud[:, 3]
#
#         # re-range data to [0,1]
#         min_col = np.amin(self.color)
#         self.color = self.color.astype(np.float32)
#         if min_col < 0:
#             self.color[:] -= min_col
#         max_color = np.amax(self.color)
#         if max_color <= 1.0:
#             pass
#         elif max_color > 1.0 and (max_color <= 10 or max_color > 255):
#             self.color[:] /= max_color
#         elif max_color > 1.0 and max_color <= 255:
#             self.color[:] /= 255.0
#         else:
#             assert False, "color range cannot be handled!"
#
#
#

if __name__ == "__main__":
    class Parameter:
        def __init__(self):
            self.focal_length_px = 10.0

    depth = np.zeros((10, 10, 2), dtype=np.float32)
    depth2 = np.zeros((10, 10, 2), dtype=np.float32)
    cloud = np.zeros((100, 3), dtype=np.float32)
    reb = np.zeros((100), dtype=np.float32)
    color = np.zeros((100, 3), dtype=np.float32)
    color2 = np.zeros((100, 3), dtype=np.float32)
    color3 = np.zeros((100, 3), dtype=np.float32)
    parameter = Parameter()
    n = 0
    for y in range(10):
        for x in range(10):
            depth[y, x, 0] = 1.0
            depth[y, x, 1] = np.random.randint(50, 1000, 1)[0]/1000.0
            depth2[y, x, 0] = 0.2 + np.random.randint(0, 100, 1)[0]/1000.0
            depth2[y, x, 1] = np.random.randint(50, 1000, 1)[0]/1000.0
            cloud[n, 0] = float(x)/10.0-0.5
            cloud[n, 1] = float(y)/10.0-0.5
            cloud[n, 2] = float(x)/10.0-0.5
            reb[n] = np.random.randint(50, 1000, 1)[0]/1000.0
            color[n, 0] = 1.0-x/20.0
            color[n, 1] = np.random.randint(0, 200, 1)[0]/255.0
            color[n, 2] = np.random.randint(0, 200, 1)[0]/255.0
            color2[n, 1] = 1.0-x/20.0
            color2[n, 0] = np.random.randint(0, 200, 1)[0]/255.0
            color2[n, 2] = np.random.randint(0, 200, 1)[0]/255.0
            color3[n, 2] = 1.0-x/20.0
            color3[n, 0] = np.random.randint(0, 200, 1)[0]/255.0
            color3[n, 1] = np.random.randint(0, 200, 1)[0]/255.0
            n += 1

    filename = "/home/swanner/Desktop/testCloud.ply"


    writer = PlyWriter(filename)
    writer.saveReliabilityAsColor(True)
    writer.setCloud(cloud)
    writer.setColor(color)
    writer.setReliability(reb)
    writer.save()

    writer.setDepthmap(depth, parameter.focal_length_px)
    writer.setColor(color2)
    writer.save()

    writer.setDepthmap(depth2, parameter.focal_length_px)
    writer.setColor(color3)
    writer.save()
