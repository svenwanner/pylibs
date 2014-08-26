# -*- coding: utf-8 -*-

import numpy as np

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
        if max_color <= 1.0:
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
            if self.cloud[n][2] != 0:
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

