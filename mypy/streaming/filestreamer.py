import sys
import vigra
import numpy as np
from scipy.ndimage import shift
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool

from mypy.tools.linearAlgebra import normalize_vec
from mypy.lightfield.helpers import getFilenameList
from mypy.lightfield.io import load_3d

import pylab as plt
from scipy.misc import imsave

cpus_available = cpu_count()



#########################################################################################################
###################            H E L P E R   F U N C T I O N S                   ########################
#########################################################################################################

def imshow(im, cmap="jet"):
    cmaps = {"jet":plt.cm.hot, "hot":plt.cm.hot, "gray":plt.cm.gray}
    if len(im.shape) == 2:
        plt.imshow(im, cmap=cmaps[cmap])
    else:
        plt.imshow(im[:, :, 0:3])
    plt.title("range:"+str(np.amin(im))+","+str(np.amax(im)))
    plt.show()



#########################################################################################################
##################    O R I E N T A T I O N C O M P U T A T I O N   C L A S S    ########################
#########################################################################################################



class Orientation(object):
    def __init__(self,  epi, inner_scale, outer_scale, focuses):
        self.epi = epi
        self.inner_scale = inner_scale
        self.outer_scale = outer_scale
        self.focuses = focuses

    def __call__(self):
        return self.compute()

    def compute(self):
        tmp_orientation = None
        tmp_coherence = None
        final_orientation = np.zeros((self.epi.shape[0], self.epi.shape[1]), dtype=np.float32)
        final_coherence = np.zeros((self.epi.shape[0], self.epi.shape[1]), dtype=np.float32)
        for focus in self.focuses:
            tensor_channels = np.zeros((self.lf.shape[3], self.epi.shape[0], self.epi.shape[1], 3), dtype=np.float32)
            for c in range(self.lf.shape[3]):
                repi = self.refocusEpi(self.epi, focus)
                tensor_channels[c, :, :, :] = vigra.filters.structureTensor(repi[:, :, c], self.inner_scale, self.outer_scale)
            tmp_orientation, tmp_coherence = self.evaluateStructureTensor(np.sum(tensor_channels, axis=0))
            final_orientation, final_coherence = self.mergeOrientations_wta(final_orientation, final_coherence, tmp_orientation+focus, tmp_coherence)
        return final_orientation, final_coherence



#########################################################################################################
###################            P R O C E S S O R   T E M P L A T E S             ########################
#########################################################################################################


class Processor(object):

    def __init__(self, parameter):
        assert isinstance(parameter, type({}))
        self.params = parameter
        self.lf = None
        self.world = None
        self.orientation_lf = None
        self.coherence_lf = None
        self.ID = 0

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

    def orientationToWorld(self):
        shape = self.orientation_lf.shape
        cv_orientation = self.orientation_lf[shape[0]/2, :, :]
        cv_coherence = self.coherence_lf[shape[0]/2, :, :]
        self.world = np.zeros((shape[1], shape[2], 4), dtype=np.float32)
        #invalids = np.where(cv_coherence == 0)

        print "reproject data..."
        for y in xrange(self.world.shape[0]):
            for x in xrange(self.world.shape[1]):
                depth = self.params["focalLength_px"]*self.params["baseline_mm"]/(cv_orientation[y, x]+1e-16)/100.0
                self.world[y, x, 0] = float((float(x)-cv_orientation.shape[1]/2.0)*depth/float(self.params["focalLength_px"]))
                self.world[y, x, 1] = float((float(y)-cv_orientation.shape[0]/2.0)*depth/float(self.params["focalLength_px"]))
                if  cv_coherence[y, x] == 0:
                    self.world[y, x, 2] = 0.0
                else:
                    self.world[y, x, 2] = depth
                self.world[y, x, 3] = cv_coherence[y, x]

    def refocusEpi(self, epi, focus):
        repi = np.zeros_like(epi)
        for c in range(epi.shape[2]):
            for y in range(epi.shape[0]):
                repi[y, :, c] = shift(epi[y, :, c], shift=(y - epi.shape[0]/2)*focus)
        return repi

    def start(self):
        self.preprocess()
        self.process()
        self.postprocess()
        self.orientationToWorld()


    def preprocess(self):
        pass

    def process(self):
        pass

    def postprocess(self):
        pass




class StructureTensorClassic(Processor):

    def __init__(self, parameter):
        Processor.__init__(self, parameter)

    def orientation(self, epi, inner_scale, outer_scale, focuses):
        tmp_orientation = None
        tmp_coherence = None
        final_orientation = np.zeros((epi.shape[0], epi.shape[1]), dtype=np.float32)
        final_coherence = np.zeros((epi.shape[0], epi.shape[1]), dtype=np.float32)
        for focus in focuses:
            tensor_channels = np.zeros((self.lf.shape[3], epi.shape[0], epi.shape[1], 3), dtype=np.float32)
            for c in range(self.lf.shape[3]):
                repi = self.refocusEpi(epi, focus)
                tensor_channels[c, :, :, :] = vigra.filters.structureTensor(repi[:, :, c], inner_scale, outer_scale)
            tmp_orientation, tmp_coherence = self.evaluateStructureTensor(np.sum(tensor_channels, axis=0))
            final_orientation, final_coherence = self.mergeOrientations_wta(final_orientation, final_coherence, tmp_orientation+focus, tmp_coherence)
        return final_orientation, final_coherence

    def mergeOrientations_wta(self, orientation1, coherence1, orientation2, coherence2):
        winner = np.where(coherence2 > coherence1)
        orientation1[winner] = orientation2[winner]
        coherence1[winner] = coherence2[winner]
        return orientation1, coherence1

    def evaluateStructureTensor(self, tensor):
        ### compute coherence value ###
        up = np.sqrt((tensor[ :, :, 2]-tensor[:, :, 0])**2 + 4*tensor[ :, :, 1]**2)
        down = (tensor[ :, :, 2]+tensor[:, :, 0] + 1e-25)
        coherence = up / down
        ### compute disparity value ###
        orientation = vigra.numpy.arctan2(2*tensor[:, :, 1], tensor[:, :, 2]-tensor[:, :, 0]) / 2.0
        orientation = vigra.numpy.tan(orientation[:])
        ### mark out of boundary orientation estimation ###
        invalid_ubounds = np.where(orientation > 1.1)
        invalid_lbounds = np.where(orientation < -1.1)
        ### set coherence of invalid values to zero ###
        coherence[invalid_ubounds] = 0
        coherence[invalid_lbounds] = 0
        ### set orientation of invalid values to related maximum/minimum value
        orientation[invalid_ubounds] = 1.1
        orientation[invalid_lbounds] = -1.1
        return orientation, coherence

    def preprocess(self):
        pass

    def process(self):
        print "process data..."
        iscale = self.params["innerScale"]
        oscale = self.params["outerScale"]
        focuses = self.params["focuses"]

        self.orientation_lf = np.zeros((self.lf.shape[0], self.lf.shape[1], self.lf.shape[2]), dtype=np.float32)
        self.coherence_lf = np.zeros((self.lf.shape[0], self.lf.shape[1], self.lf.shape[2]), dtype=np.float32)

        for y in xrange(self.lf.shape[1]):
            epi = self.lf[:, y, :, :].astype(np.float32)
            if self.params.has_key("prefilter"):
                if self.params["prefilter"] > 0:
                    for c in range(epi.shape[2]):
                        epi[:, :, c] = vigra.filters.gaussianGradient(epi[:, :, c], float(self.params["prefilter"]))[:, :, 0]
            orientation, coherence = self.orientation(epi, iscale, oscale, focuses)
            self.orientation_lf[:, y, :] = orientation[:]
            self.coherence_lf[:, y, :] = coherence[:]

        #imshow(self.orientation_lf[self.orientation_lf.shape[0]/2, :, :])
        imsave(self.params["resultsPath"]+"disp_%4.4i.png"%self.ID,  self.orientation_lf[self.orientation_lf.shape[0]/2, :, :])

    def postprocess(self):
        pass





#########################################################################################################
###################                P R O C E S S I N G    E N G I N E            ########################
#########################################################################################################

class Engine(object):

    def __init__(self, parameter):
        # read available image filenames
        self.fnames = getFilenameList(parameter["filesPath"], parameter["switchFilesOrder"])
        # compute missing parameter
        parameter["totalNumOfFrames"] = len(self.fnames)
        self.params = self.computeMissingParameter(parameter)
        self.global_processors = {}

        global cpus_available
        if self.params.has_key("numOfProcessors") and (0 < self.params["numOfProcessors"] < cpus_available):
            cpus_available = self.params["numOfProcessors"]

        # check if number of frames to compute and images available is consistent
        if len(self.fnames) < self.params["totalNumOfFrames"]:
            print "number of image files found is less than computed number of frames check parameter settings",
            print " subImageVolumeSize, numOfSubImageVolumes, frameShift and the number of your image files!"
            sys.exit()

        wx = int(np.ceil(parameter["visibleWorldArea_m"][0]/parameter["worldAccuracy_m"]))
        wy = int(np.ceil(parameter["visibleWorldArea_m"][1]/parameter["worldAccuracy_m"]))
        self.worldgrid_shape = (wy, wx, 4, self.params["numOfSubImageVolumes"])
        self.world_size = (parameter["visibleWorldArea_m"][0], parameter["visibleWorldArea_m"][1])
        print "created world grid of shape:", self.worldgrid_shape
        print "visible world size is:", self.world_size
        self.world = np.zeros(self.worldgrid_shape, np.float32)

    def computeMissingParameter(self, parameter):
        # if no frameShift set use sub volume site as shift
        if not parameter.has_key("frameShift"):
            parameter["frameShift"] = parameter["subImageVolumeSize"]
        # compute the field of view of the camera
        parameter["fov"] = np.arctan2(parameter["sensorSize_mm"], 2.0*parameter["focalLength_mm"])
        # compute focal length is pixel
        parameter["focalLength_px"] = float(parameter["focalLength_mm"])/float(parameter["sensorSize_mm"])*parameter["sensorSize_px"][1]
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
        parameter["camCenterPos"] = parameter["camInitialPos"] + parameter["maxBaseline_mm"]*parameter["camTransVector"]/200.0
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

    def computeListIndices(self, n):
        return n*self.params["frameShift"], n*self.params["frameShift"]+self.params["subImageVolumeSize"]

    def process2thread(self, params, index):
        # set processor instance
        if self.params["processor"] == "structureTensorClassic":
            self.global_processors[index] = StructureTensorClassic(self.params)
        self.global_processors[index].ID = index
        self.global_processors[index].setData(params)
        self.global_processors[index].start()

    def computeCurrentCamPosition(self, index):
        slfs = self.params["subImageVolumeSize"]
        b = self.params["baseline_mm"]
        ipos = self.params["camInitialPos"]
        tvec = self.params["camTransVector"]
        return ipos + (index * slfs + slfs/2 - 1) * b/100.0 * tvec

    def computeCamShiftVector(self, currentPos):
        return self.params["camCenterPos"]-currentPos

    def projectPointsToCenter(self, processor):
        camp_pos = self.computeCurrentCamPosition(processor.ID)
        cam_shift = self.computeCamShiftVector(camp_pos)
        print "camp_pos of index", processor.ID, ":", camp_pos
        print "cam shift:", cam_shift

        for y in range(processor.world.shape[0]):
            for x in range(processor.world.shape[1]):
                for i in range(3):
                    processor.world[y, x, i] += cam_shift[i]

        imsave(self.params["resultsPath"]+"depth_%4.4i.png" % processor.ID,  processor.world[:, :, 2])
        imsave(self.params["resultsPath"]+"coherence_%4.4i.png" % processor.ID,  processor.world[:, :, 3])

    def world2grid(self, x, y):
        if -self.world_size[1]/2.0 <= y <= self.world_size[1]/2.0:
            if -self.world_size[0]/2.0 <= x <= self.world_size[0]/2.0:
                n = int((self.world_size[1]/2.0-y)/self.world_size[1]*self.worldgrid_shape[0])
                m = int(self.worldgrid_shape[1]-int((self.world_size[0]/2.0-x)/self.world_size[0]*self.worldgrid_shape[1]))
                return n, m
            else:
                print "x coordinate out of range!"
                return None, None
        else:
            print "y coordinate out of range!"
            return None, None


    def grid2world(self, n, m):
        if 0 <= n <= self.N:
            if 0 <= m <= self.M:
                return [(float(self.worldgrid_shape[0])/2.0-float(n))/float(self.worldgrid_shape[0])*self.world_size[0],
                        (float(m)-float(self.worldgrid_shape[1])/2.0)/float(self.worldgrid_shape[1])*self.world_size[1]]
            else:
                return None
        else:
            return None

    def addPointsToWorld(self, points):
        print "type points:", type(points)
        for y in range(points.shape[0]):
            for x in range(points.shape[1]):
                #print "try to project point", y, x, "with values", points[y, x, 0], points[y, x, 1]
                n, m = self.world2grid(points[y, x, 0], points[y, x, 1])
                #print "projected to index:", n, m
                if n is not None and m is not None and 0 <= n < self.world.shape[0] and m >= 0 and m < self.world.shape[1]:
                    for i in range(4):
                        self.world[n, m, i] = points[y, x, i]

    def run(self):
        global global_index

        n = 0
        while True:

            if n >= self.params["numOfSubImageVolumes"]:
                break

            fname_list = {}
            print "cpus_available", cpus_available
            for i in range(cpus_available):
                sindex, findex = self.computeListIndices(n)
                fname_list[n] = self.fnames[sindex:findex]
                if len(fname_list[n]) < self.params["subImageVolumeSize"]:
                    if len(fname_list[n]) > 0:
                        fname_list[n] = None
                n += 1

            if len(fname_list.keys()) == 0:
                break

            pool = Pool(len(fname_list.keys()))

            for index in fname_list.keys():
                if fname_list[index] is not None and len(fname_list[index]) > 0:
                    pool.apply_async(self.process2thread, (fname_list[index], index))

            pool.close()
            pool.join()

            k=0
            for key in self.global_processors.keys():
                self.projectPointsToCenter(self.global_processors[key])
                self.addPointsToWorld(self.global_processors[key].world)
                imsave(self.params["resultsPath"]+"reprojected_%4.4i.png" % self.global_processors[key].ID, self.world[:, :, 2, k])
                k+=1

            self.global_processors.clear()





#########################################################################################################
###################                 M A I N   R O U T I N E S                    ########################
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
        "roi": {"pos": [270-150, 480-150], "size": [300, 300]},
        "prefilter": 0.4,
        "numOfProcessors:": 0
    }

    main(parameter)