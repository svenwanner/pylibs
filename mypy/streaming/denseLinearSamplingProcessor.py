# -*- coding: utf-8 -*-

import os
import time
import vigra
import numpy as np
from glob import glob
from multiprocessing import Pool
from scipy.misc import imread, imsave
import pylab as plt
from mypy.streaming.depthProjector import DepthProjector

from joblib import Parallel, delayed

########################################################################################################################
########################################################################################################################
######################  processing function needs to be defined globally to be accepted by the  ########################
######################  multiprocessing module. Create a new function with the same name and    ########################
######################  parameter signature and rename the old one to implement a new epi       ########################
######################  processing behaviour of the EpiProcessor class.                         ########################
########################################################################################################################
########################################################################################################################

def process(input):
    """
    this function is the main routine called on each epi through the
    Processor class. The input is a list containing the epi as first
    entry and the parameter dictionary as second. Ensure that the
    number of channels of your output array are correct.
    :param input: <[]> list [epi<ndarray>,parameter<{}>]
    :return: <ndarray> result
    """
    assert isinstance(input, type([]))
    out = np.zeros((input[0].shape[0], input[0].shape[1], input[1]["channels"]), dtype=np.float32)
    tensor = vigra.filters.structureTensor(input[0], input[1]["inner_scale"], input[1]["inner_scale"])

    ### compute coherence value ###
    up = np.sqrt((tensor[:, :, 2]-tensor[:, :, 0])**2 + 4*tensor[:, :, 1]**2)
    down = (tensor[:, :, 2]+tensor[:, :, 0] + 1e-25)
    coherence = up / down

    ### compute disparity value ###
    orientation = vigra.numpy.arctan2(2*tensor[:, :, 1], tensor[:, :, 2]-tensor[:, :, 0]) / 2.0
    orientation = vigra.numpy.tan(orientation[:])

    ### mark out of boundary orientation estimation ###
    invalid_ubounds = np.where(orientation > 1.1)
    invalid_lbounds = np.where(orientation < -1.1)
    if not input[1].has_key("min_coherence"):
        input[1]["min_coherence"] = 0.5
    invalid_coh = np.where(coherence < input[1]["min_coherence"])

    ### set coherence of invalid values to zero ###
    coherence[invalid_ubounds] = 0
    coherence[invalid_lbounds] = 0
    coherence[invalid_coh] = 0

    ### set orientation of invalid values to related maximum/minimum value
    orientation[invalid_ubounds] = -1.5
    orientation[invalid_lbounds] = -1.5
    orientation[invalid_coh] = -1.5

    out[:, :, 0] = orientation[:, :]
    out[:, :, 1] = coherence[:, :]
    return out

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

class EpiProcessor(object):
    """
    This class is used to be taking control over the epi iteration calling
    the process method defined globally above multithreaded for each epi.
    It is necessary to set the parameter object containing the number of
    channels the result array should have and all parameter used in the
    process method. The rest ist done by the Engine class.
    """
    def __init__(self):
        self.data = None
        self.result = None
        self.parameter = None
        self.result_channels = 1

    def setParameter(self, parameter):
        """
        sets the parameter object. A key channel is obligatory defining
        the number of output channels.
        :param parameter: <{}> dictionary keeping the parameters needed by process()
        """
        assert isinstance(parameter, type({}))
        self.parameter = parameter
        assert parameter.has_key("channels"), "Parameter object need a key called channel defining number of result channels!"
        self.result_channels = parameter["channels"]

    def setData(self, data):
        """
        sets the data array to be processed.
        :param data: <ndarray> input data
        """
        assert isinstance(data, np.ndarray)
        assert len(data.shape) == 3
        self.data = data

    def getResult(self):
        """
        returns the result array
        """
        return self.result

    def start(self):
        """
        this is the main routine of the class calling process
        on each epi in parallel.
        """
        assert self.data is not None, "Need data before process can be started!"

        self.result = np.zeros((self.data.shape[0], self.data.shape[1], self.data.shape[2], self.result_channels), dtype=np.float32)

        assert self.result is not None, "No result array is defined!"
        assert self.parameter is not None, "No parameter object is defined!"

        inputs = []
        for n in range(self.data.shape[1]):
            inputs.append([self.data[:, n, :], self.parameter])

        result = Parallel(n_jobs=4)(delayed(process)(inputs[i]) for i in range(len(inputs)))
        for m, res in enumerate(result):
            for c in range(self.result_channels):
                self.result[:, m, :, c] = res[:, :, c]






class Engine():
    """
    This class is the Engine taking care of the different pipeline
    processes like reading the image files, passing the loaded sub-
    lightfields to the processor and so on...
    """
    def __init__(self):
        self.running = False
        self.fileReader = None
        self.processor = None


    def setData(self, file_path, stack_size=11):
        """
        set the data path and the image volume stack size by
        instantiating a FileReader object. Calling this function
        is obligatory for running the engine.
        :param file_path: <str> path to directory containing image files
        :param stack_size: <int> number of images in a single stack
        """
        self.fileReader = FileReader(file_path, stack_size)

    def setProcessor(self, processor):
        """
        set a processor object handling the sub light fields.
        Calling this function is obligatory for running the engine.
        :param processor: <Processor> a image volume processor
        """
        self.processor = processor

    def start(self):
        """
        starts the engine which consecutively does file reading, processing
        data accumulating for all possible sub light fields.
        """
        assert self.fileReader is not None, "No FileReader instance initialized!"
        assert self.processor is not None, "No Processor set!"

        self.running = True
        self.fileReader.start()
        while self.running:
            if self.fileReader.bufferReady():
                print "processing stack..."
                self.processor.setData(self.fileReader.getStack())
                self.processor.start()
                self.fileReader.start()


                print "done!"

            #if file reader finished break loop
            if self.fileReader.finished:
                self.running = False


class FileReader():
    """
    This class reads all image filenames from a given directory,
    loads parts of it and returns the images as ndarray volumes.
    """
    def __init__(self, input_path, stack_size=11):
        """
        Constructor needs a filepath containing image files and
        defines the stack size.
        :param input_path: <str> path to directory containing image files
        :param stack_size: <int> number of images in a single stack
        """
        assert isinstance(input_path, str)
        assert os.path.isdir(input_path), "input path does not exist!"
        assert isinstance(stack_size, int)
        assert stack_size > 0, "stack size needs to be at least 1!"

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
        """
        loads a single image
        :param fname: <str> filename
        :return: <ndarray> image
        """
        assert isinstance(fname, str)
        if fname.endswith("exr"):
            return np.transpose(np.array(vigra.readImage(fname))[:, :, 0]).astype(np.float32)
        else:
            return imread(fname)

    def channelConverter(self, img, ctype="exp_stretch"):
        """
        converts an image to a grayscale image depending on
        the conversion type.
        :param img: <ndarray> image
        :param ctype: <str> conversion type ["gray","stretch","exp_stretch"]
        :return: <ndarray> single band image
        """
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
        """
        returns the buffer is filled flag state
        :return: <bool> flag if buffer is filled
        """
        return self.ready

    def getStack(self):
        """
        returns the filled buffer and resets all counters
        :return: <ndarray> image volume buffer
        """
        if self.bufferReady():
            self.counter = 0
            self.ready = False
            return np.copy(self.stack)

    def start(self):
        """
        runs the buffer filling process
        """
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
    processor = EpiProcessor()
    processor.setParameter({"channels": 2, "inner_scale": 0.6, "outer_scale": 1.3, "min_coherence": 0.95})

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





