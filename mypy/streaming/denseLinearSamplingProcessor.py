# -*- coding: utf-8 -*-

import sys, os
from glob import glob
from mypy.streaming.INIReader import Parameter
from mypy.streaming.fileReader import FileReader
from mypy.streaming.processors import EpiProcessor
from mypy.streaming.depthAccumulator import DepthAccumulator

from mypy.streaming.globals import DEBUG

#???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
#???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
#
#                                               D E S C R I P T I O N
#
#
#???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
#???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????


class Engine():
    """
    This class is the Engine taking care of the different pipeline
    processes like reading the image files, passing the loaded sub-
    lightfields to the processor and so on...
    """
    def __init__(self, project_path=None):
        assert isinstance(project_path, str)
        self.ini_files = None
        self.setProjectPath(project_path)
        self.running = False
        self.fileReader = None
        self.processor = None
        self.depthAccumulator = DepthAccumulator()


    def setProjectPath(self, project_path):
        if project_path is not None:
            assert isinstance(project_path, str)
            assert os.path.isdir(project_path)
            self.project_path = project_path
            if not self.project_path.endswith(os.sep):
                self.project_path += os.sep
            self.ini_files = []

            for p in glob(self.project_path + "*.ini"):
                self.ini_files.append(p)
            self.ini_files.sort()


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

        for ini in self.ini_files:
            # create a Parameter object
            self.parameter = Parameter(ini)
            if DEBUG > 0: print self.parameter

            #create a Processor object
            self.processor = EpiProcessor(self.parameter)

            # pass the parameter to the depthAccumulator and reset the image index counter
            self.depthAccumulator.reset()
            self.depthAccumulator.setParameter(self.parameter)

            # create a FileReader from the current ini file
            self.fileReader = FileReader(self.parameter.image_files_location, self.parameter.stack_size, self.parameter.swap_files_order)
            self.fileReader.read()
            self.running = True

            # count the subLFs
            subLF_counter = 0

            while self.running:
                if self.fileReader.bufferReady():
                    if DEBUG > 0: print "\nprocessing stack..."
                    # the fileReader reads the next subLF
                    stack, cv_color = self.fileReader.getStack()
                    # pass the data to the processor and start orientation estimation
                    self.processor.setData(stack)
                    self.processor.start()
                    # get the disparity or orientation map from the processor
                    orientation = self.processor.getResult()
                    # set the current index of the disparity map that is added to the depthProjector
                    self.depthAccumulator.setCounter(self.parameter.stack_size*subLF_counter+self.parameter.stack_size/2)
                    # pass the disparity map to the depthAccumulator
                    self.depthAccumulator.addDisparity(orientation[:, :, 0], orientation[:, :, 1], cv_color)
                    #read the next subLF
                    self.fileReader.read()
                    subLF_counter += 1
                    if DEBUG > 0: print "done!"

                #if file reader finished break loop
                if self.fileReader.finished:
                    self.running = False

            self.depthAccumulator.save()






if __name__ == "__main__":

    if len(sys.argv) == 2:
        project_path = sys.argv[1]
        assert os.path.exists(project_path), "Path does not exist!"
    else:
        print "Please pass a path to a project folder containing ini files!"
        sys.exit()

    engine = Engine(project_path)
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





