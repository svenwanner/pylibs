# -*- coding: utf-8 -*-

import os
import time
import vigra
import numpy as np
from glob import glob
from scipy.misc import imread, imsave
import pylab as plt
from mypy.streaming.depthProjector import DepthProjector



class Processor(object):

    def __init__(self):
        pass

class StructureTensorProcessor(Processor):
    def __init__(self):
        Processor.__init__(self)


class Engine():

    def __init__(self):
        self.running = False
        self.fileReader = None
        self.processor = None


    def setData(self, file_path, stack_size=11):
        self.fileReader = FileReader(file_path, stack_size)

    def setProcessor(self, processor):
        self.processor = processor

    def start(self):
        assert self.fileReader is not None, "No FileReader instance initialized!"
        assert self.processor is not None, "No Processor set!"
        
        self.running = True
        self.fileReader.start() #load first sub light field

        tmp = 0
        while self.running:
            if self.fileReader.bufferReady():
                print "counter bevore:", self.fileReader.counter
                print "current file bevore:", self.fileReader.current_file
                lf = self.fileReader.getStack()

                imsave("/home/swanner/Desktop/tmp_imgs/eng_%4.4i.png"%tmp, lf[0, :, :])
                tmp += 1
                print "counter:", self.fileReader.counter
                print "current file:", self.fileReader.current_file
                time.sleep(1)

                self.fileReader.start() #load next sub light field

            #if file reader finished break loop
            if self.fileReader.finished:
                self.running = False


class FileReader():

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
        self.ready = False
        self.finished = False

        tmp = self.loadImage(self.filenames[0])
        tmp = self.channelConverter(tmp)
        self.shape = (self.stack_size, tmp.shape[0], tmp.shape[1])
        self.stack = np.zeros(self.shape, dtype=np.float32)

    def loadImage(self, fname):
        print "loading image..."
        assert isinstance(fname, str)
        if fname.endswith("exr"):
            return np.transpose(np.array(vigra.readImage(fname))[:, :, 0]).astype(np.float32)
        else:
            return imread(fname)

    def channelConverter(self, img, ctype="exp_stretch"):
        print "converting image..."
        img = img.astype(np.float32)
        amax = np.amax(img)
        if amax > 1.0:
            img[:] /= 255.0
        if len(img.shape) == 2:
            return img
        out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        if ctype == "gray":
            out[:] = 0.3*img[:, :, 0]+0.59*img[:, :, 1]+0.11*img[:, :, 2]
        if ctype == "stretch":
            for c in range(img.shape[2]):
                out[:, :] += img[:, :, c]+c
        if ctype == "exp_stretch":
            for c in range(img.shape[2]):
                out[:, :] += (img[:, :, c])**(c+1)
        amax = np.amax(out)
        out[:] /= amax
        return out

    def bufferReady(self):
        return self.ready

    def getStack(self):
        if self.bufferReady():
            self.counter = 0
            self.ready = False
            return np.copy(self.stack)

    def start(self):
        while self.counter < self.stack_size-1 and not self.finished:
            self.ready = False
            if self.counter == 0:
                self.stack[:] = 0.0
            tmp = self.loadImage(self.filenames[self.current_file])
            tmp = self.channelConverter(tmp)
            self.stack[self.counter, :, :] = tmp[:]
            self.current_file += 1
            self.counter += 1
            if self.current_file == self.num_of_files-1:
                self.finished = True
                break
        self.ready = True



if __name__ == "__main__":

    data_path = "/home/swanner/Desktop/denseSampledTestScene/rendered_LR"
    processor = StructureTensorProcessor()

    engine = Engine()
    engine.setData(data_path)
    engine.setProcessor(processor)

    engine.start()

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





