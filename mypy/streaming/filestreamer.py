import sys
import vigra
import numpy as np
from multiprocessing import Process,Queue, cpu_count

from mypy.tools.linearAlgebra import normalize_vec
from mypy.lightfield.helpers import getFilenameList
from mypy.lightfield.depth.structureTensor2D import evaluateStructureTensor
from mypy.lightfield.io import load_3d

from scipy.misc import imshow

cpus_available = cpu_count()


#########################################################################################################
#########################################################################################################
#########################################################################################################


class Processor(object):

    def __init__(self, parameter):
        assert isinstance(parameter, type({}))
        self.params = parameter
        self.lf = None
        self.world = None

    def __load__(self, filelist):
        assert isinstance(filelist, type([]))
        roi = None
        swor = False
        if self.params.has_key("roi"):
            roi = self.params["roi"]
        if self.params.has_key("switchFilesOrder"):
            swor = self.params["switchFilesOrder"]
        return load_3d(filelist, roi=roi, switchOrder=swor)

    def setData(self, subLF):
        if isinstance(subLF, type([])):
            print "set file list - ", subLF
            self.lf = self.__load__(subLF)
        elif isinstance(subLF, np.ndarray):
            print "set ndarray - ", subLF.shape
            self.lf = subLF
        else:
            print "Abort, unknown subLF type!"
            sys.exit()

        print "Processor has loaded data successfully!"

    def worldContainerShape(self):
        """
        Here a world container needs to be specified. The world container is the full sensor
        area over all cameras projected into the world space using the camera distance and the
        accuracy or discretization of the sensor area. The world array stores all computations
        from each camera position or sub light field and thus has as many result layer as iterations
        are necessary. At each world position for each iteration a vector is stored which length
        needs to be specified here but is at least one for the depth value. The general form of the
        world array is as follows:
        world.shape = (x,y,layer,values) values by default are assumed as 0:depth
        """
        pass

    def start(self):
        self.preprocess()
        self.process()
        self.postprocess()


    def preprocess(self):
        pass

    def process(self):
        pass

    def postprocess(self):
        pass




class StructureTensorClassic(Processor):

    def __init__(self, parameter):
        Processor.__init__(self, parameter)

    def orientation(self, epi, inner_scale, outer_scale):
        tensor_channels = np.zeros((self.lf.shape[3], epi.shape[0], epi.shape[1], 3), dtype=np.float32)
        for c in range(self.lf.shape[3]):
            tensor_channels[c, :, :, :] = vigra.filters.structureTensor(epi[:, :, c], inner_scale, outer_scale)

        orientation, coherence = evaluateStructureTensor(np.sum(tensor_channels,axis=0))


    def worldContainerShape(self):
        return ()

    def preprocess(self):
        print "preprocess data..."
        print "finished"

    def process(self):
        print "process data..."
        y = 0
        iscale = self.params["innerScale"]
        oscale = self.params["outerScale"]
        while True:
            jobs = []
            for i in range(cpus_available):
                if y >= self.lf.shape[1]: break
                epi = self.lf[:, y, :, :]
                y += 1

                p = Process(target=self.orientation, args=(epi, iscale, oscale))
                jobs.append(p)
                p.start()

            for j in jobs:
                j.join()



        print "finished"

    def postprocess(self):
        pass



#########################################################################################################
#########################################################################################################
#########################################################################################################


def computeMissingParameter(parameter):
    # if no frameShift set use sub volume site as shift
    if not parameter.has_key("frameShift"):
        parameter["frameShift"] = parameter["subImageVolumeSize"]
    # compute the field of view of the camera
    parameter["fov"] = np.arctan2(parameter["sensorSize_mm"], 2.0*parameter["focalLength_mm"])
    # compute focal length is pixel
    parameter["focalLength_px"] = float(parameter["focalLength_mm"])/float(parameter["sensorSize_mm"])*parameter["sensorSize_px"][1]
    ## compute the total number of frames
    #parameter["totalNumOfFrames"] = parameter["subImageVolumeSize"]+(parameter["numOfSubImageVolumes"]-1)*parameter["subImageVolumeSize"]
    # compute number of sub image volumes
    parameter["numOfSubImageVolumes"] = int(np.floor(parameter["totalNumOfFrames"]/float(parameter["frameShift"])))
    # compute the maximum traveling distance of the camera
    parameter["maxBaseline_mm"] = float(parameter["totalNumOfFrames"]-1)*parameter["baseline_mm"]
    # compute the final camera position and the center position of the camera track
    parameter["camInitialPos"] = np.array(parameter["camInitialPos"])
    if not parameter.has_key("frameShift"):
        parameter["frameShift"] = parameter["subImageVolumeSize"]
    if not parameter.has_key("camTransVector") or not parameter.has_key("camLookAtVector"):
        print "Warning, either camTransVector or camLookAtVector is missing, default values are used instead!"
        parameter["camTransVector"] = np.array([1.0, 0.0, 0.0])
        parameter["camLookAtVector"] = np.array([0.0, 0.0, -1.0])
    else:
        parameter["camTransVector"] = np.array(parameter["camTransVector"])
        parameter["camLookAtVector"] = np.array(parameter["camLookAtVector"])
        parameter["camTransVector"] = normalize_vec(parameter["camTransVector"])
        parameter["camLookAtVector"] = normalize_vec(parameter["camLookAtVector"])
    parameter["camFinalPos"] = parameter["camInitialPos"] + parameter["maxBaseline_mm"]*parameter["camTransVector"]/100.0
    # define camera z-coordinate as horopter distance
    parameter["horopter_m"] = parameter["camInitialPos"][2]
    # define horopter vector
    parameter["horopter_vec"] = parameter["camLookAtVector"]*float(parameter["horopter_m"])

    # compute vertical field of view
    sensorSize_y = float(parameter["sensorSize_mm"])*parameter["sensorSize_px"][0]/float(parameter["sensorSize_px"][1])
    fov_h = np.arctan2(sensorSize_y, 2.0*parameter["focalLength_mm"])
    # compute real visible scene width and height
    vwsx = 2.0*parameter["camInitialPos"][2]*np.tan(parameter["fov"])+parameter["maxBaseline_mm"]/100.0
    vwsy = 2.0*parameter["camInitialPos"][2]*np.tan(fov_h)
    parameter["visibleWorldArea_m"] = [vwsx, vwsy]

    if not parameter.has_key("worldAccuracy_m") or parameter["worldAccuracy_m"] <= 0.0:
        parameter["worldAccuracy_m"] = vwsy/parameter["sensorSize_px"][0]

    for key in parameter.keys():
        print key, ":", parameter[key]

    return parameter



class Engine(object):

    def __init__(self, parameter):
        # read available image filenames
        self.fnames = getFilenameList(parameter["filesPath"], parameter["switchFilesOrder"])
        # compute missing parameter
        parameter["totalNumOfFrames"] = len(self.fnames)
        self.params = computeMissingParameter(parameter)

        self.world = None

        # set processor instance
        if self.params["processor"] == "structureTensorClassic":
            self.processor = StructureTensorClassic(self.params)

        # check if number of frames to compute and images available is consistent
        if len(self.fnames) < self.params["totalNumOfFrames"]:
            print "number of image files found is less than computed number of frames check parameter settings",
            print " subImageVolumeSize, numOfSubImageVolumes, frameShift and the number of your image files!"
            sys.exit()

        self.world = np.zeros(self.processor.worldContainerShape(), dtype=np.float32)

    def computeListIndices(self, n):
        return n*self.params["frameShift"], n*self.params["frameShift"]+self.params["subImageVolumeSize"]

    def run(self):

        self.processor.world = self.world
        for n in range(self.params["numOfSubImageVolumes"]):

            sindex, findex = self.computeListIndices(n)
            fnames = self.fnames[sindex:findex]
            if len(fnames) < self.params["subImageVolumeSize"]:
                break

            self.processor.setData(fnames)
            self.processor.world = self.world
            self.processor.start()







#########################################################################################################
#########################################################################################################
#########################################################################################################

def main(parameter):

    engine = Engine(parameter)
    engine.run()


if __name__ == "__main__":

    parameter = {
        "filesPath": "/home/swanner/Desktop/denseSampledTestScene/rendered3/fullRes/",
        "switchFilesOrder": False,
        "resultsPath": "/home/swanner/Desktop/denseSampledTestScene/results3_FR/",
        "rgb": True,
        "processor": "structureTensorClassic",
        "innerScale": 0.6,
        "outerScale": 1.3,
        "sensorSize_mm": 32,
        "focalLength_mm": 16,
        "focuses": [2, 3],
        "baseline_mm": 0.8695652173913043,
        "sensorSize_px": [540, 960],
        "subImageVolumeSize": 11,
        "frameShift": 11,
        "camInitialPos": [-1.0, 0.0, 2.6],
        "camTransVector": [1.0, 0.0, 0.0],
        "camLookAtVector": [0.0, 0.0, -1.0],
        "roi": {"pos": [270-150, 480-150], "size": [300, 300]}
    }

    main(parameter)