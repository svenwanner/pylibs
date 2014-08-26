# -*- coding: utf-8 -*-

import os
import vigra
from glob import glob
from scipy.misc import imread
import threading
import numpy as np
from mypy.streaming.depthProjector import DepthProjector


class Engine(object):

     def __init__(self):
         self.running = False

     def run(self):

        self.running = True
        while self.running:
            self.running = False


class FileReader(threading.Thread):

    def __init__(self, input_path, stack_size=11):
        assert isinstance(input_path, str)
        assert os.path.isdir(input_path), "input path does not exist!"
        if not input_path.endswith(os.sep):
            input_path += os.sep

        self.path = input_path
        self.filenames = []
        self.stack_size = stack_size
        self.valid_types = ("png", "PNG", "jpg", "JPG", "JPEG", "tif", "tiff", "TIFF", "bmp", "ppm", "exr")

        for f in glob(input_path+"*"):
            for ft in self.valid_types:
                if f.endswith(ft):
                    self.filenames.append(f)
        self.filenames.sort()
        assert len(self.filenames) > 0, "No image files found!"

        self.num_of_files = len(self.filenames)
        self.current_file = 0
        self.counter = 0

        tmp = self.loadImage(self.filenames[0])
        tmp = self.channelConverter(tmp)
        self.shape = (self.num_of_files, tmp[0], tmp[1])
        self.stack = np.zeros(self.shape, dtype=np.float32)


    def loadImage(self, fname):
        assert isinstance(fname, str)
        if fname.endswith("exr"):
            return np.transpose(np.array(vigra.readImage(fname))[:, :, 0]).astype(np.float32)
        else:
            return imread(fname)


    def channelConverter(self, img):
        if len(img.shape) == 2:
            return img
        out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        for c in range(img.shape[2]):
            out[:, :] += img[:, :, c]**(c+1)
        return out


    def bufferReady(self):
        if self.counter == self.stack_size-1:
            return True
        else: return False


    def getStack(self):
        if self.counter == self.stack_size-1:
            self.counter = 0
            return self.stack
        else: return None


    def run(self):
        while self.current_file < self.num_of_files:
            if self.counter == 0:
                self.stack = 0.0
            if self.counter < self.stack_size:
                tmp = self.loadImage(self.filenames[self.current_file])
                tmp = self.channelConverter(tmp)
                self.stack[self.current_file, :, :] = tmp[:]
                self.current_file += 1



if __name__ == "__main__":

    engine = Engine()
    engine.run()

    # #color = np.random.randint(0, 255, 540 * 960 * 3).reshape((540, 960, 3))
    # depthProjector = DepthProjector()
    #
    # depthProjector.addDepthMapFromFile("/home/swanner/Desktop/tmp/depth/0001.exr")
    # depthProjector.addCamera(35.0, 32.0, [540, 960], np.array((4.9950, -4.1860, 3.9597)), np.array((1.1500, 0.0000, 0.8197)))
    # #depthProjector.addColor(color)
    #
    # depthProjector.addDepthMapFromFile("/home/swanner/Desktop/tmp/depth/0002.exr")
    # depthProjector.addCamera(35.0, 32.0, [540, 960], np.array((-3.2872, 4.5016, 5.4028)), np.array((0.8993, 0.0000, 3.7588)))
    # #depthProjector.addColor(color)
    #
    # depthProjector.addDepthMapFromFile("/home/swanner/Desktop/tmp/depth/0003.exr")
    # depthProjector.addCamera(35.0, 32.0, [540, 960], np.array((-3.3543, -5.7408, 4.9266)), np.array((1.0444, 0.0000, 5.7277)))
    # #depthProjector.addColor(color)
    #
    # depthProjector(0.35)
    # depthProjector.save("/home/swanner/Desktop/tmp/cloud.ply")
    #
    # print "finished, created cloud with", depthProjector.cloud.shape[0], "points in total!"





