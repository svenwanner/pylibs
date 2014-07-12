import sys, os
from glob import glob
import numpy as np
import vigra
from scipy.misc import imread, imsave, imshow
import mypy.visualization.imshow as ims

from mypy.lightfield.io import load_lf3d_fromFiles
from mypy.lightfield.helpers import changeColorspace
from mypy.pointclouds.depthToCloud import disparity_to_depth, cloud_from_depth, PlyWriter


class SubLFProcessor(object):
    def __init__(self, parameter):
        assert isinstance(parameter, type({}))

        self.fpath = parameter["filepath"]
        self.focus = parameter["focus"]
        self.numOfCams = parameter["subLF_size"]
        self.colorpace = parameter["colorspace"]
        self.results = None

    def showEpi(self, index):
        epiImg = np.zeros((self.shape[0]*self.shape[3], self.shape[1], self.shape[2]))
        for c in range(self.shape[3]):
            epiImg[c*self.shape[0]:(c+1)*self.shape[3]] = np.copy(self.lf[:, index, :, :])

    def load(self, camIndex):
        assert isinstance(camIndex, int)

        try:
            self.lf = load_lf3d_fromFiles(self.fpath, camIndex, self.numOfCams, self.focus, dtype=np.float32)
            self.shape = self.lf.shape
            self.results = np.zeros((self.shape[1], self.shape[2], 2), dtype=np.float32)
            self.cv = self.lf[self.shape[0]/2, :, :, 0:self.shape[3]]
            self.lf = changeColorspace(self.lf, self.colorpace)
            return True
        except Exception as e:
            print e
            return False

    def getResults(self):
        return self.results

    def compute(self):
        pass



class StructureTensorProcessor(SubLFProcessor):

    def __init__(self, parameter):
        SubLFProcessor.__init__(self, parameter)
        self.parameter = parameter

    def compute(self):
        print "\n<-- compute..."

        for y in range(self.shape[1]):

            ### compute tensor for each y coordinate and each color channel ###
            tensor = np.zeros((self.shape[3], self.shape[0], self.shape[2], 3), dtype=np.float32)
            for c in range(self.shape[3]):
                tensor[c, :, :, :] = vigra.filters.structureTensor(self.lf[:, y, :, c],
                                                                   self.parameter["inner_scale"],
                                                                   self.parameter["outer_scale"])

            ### mean over color channels ###
            tensor = np.mean(tensor, axis=0)

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
            invalid = np.where(coherence < self.parameter["min_coherence"])

            orientation += self.focus

            ### set coherence of invalid values to zero ###
            coherence[invalid_ubounds] = 0
            coherence[invalid_lbounds] = 0
            coherence[invalid] = 0

            ### set orientation of invalid values to related maximum/minimum value
            orientation[invalid_ubounds] = -1
            orientation[invalid_lbounds] = -1
            orientation[invalid] = -1

            self.results[y, :, 0] = orientation[self.shape[0]/2, :]
            self.results[y, :, 1] = coherence[self.shape[0]/2, :]

        print "done -->"


class DenseLightFieldProcessor(object):

    def __init__(self, parameter, processor):
        assert isinstance(parameter, type({}))
        assert isinstance(processor, SubLFProcessor)
        assert isinstance(parameter["total_frames"], int)
        assert isinstance(parameter["subLF_size"], int)

        self.parameter = parameter
        self.processor = processor

        self.iterations = self.parameter["total_frames"]/self.parameter["subLF_size"]


    def run(self):
        cam_index = 0
        for i in range(self.iterations):
            if self.processor.load(cam_index):
                processor.compute()
                results = processor.getResults()
                depth = disparity_to_depth(results[:, :, 0], parameter["baseline"], parameter["focal_length"], parameter["min_depth"], parameter["max_depth"])
                cloud = cloud_from_depth(depth, parameter["focal_length"])
                pcWriter = PlyWriter(name=self.parameter["resultpath"]+"_%4.4i" % i,
                                    cloud=cloud, colors=processor.cv, confidence=results[:, :, 1])

            cam_index += self.parameter["subLF_size"]



if __name__ == "__main__":

    parameter = {"filepath": "/home/swanner/Desktop/highSampledTestScene/rendered/imgs",
                 "resultpath": "/home/swanner/Desktop/highSampledTestScene/results/cloud",
                 "subLF_size": 7,
                 "total_frames": 21,
                 "focus": 2,
                 "colorspace": "hsv",
                 "inner_scale": 0.6,
                 "outer_scale": 1.3,
                 "min_coherence": 0.98,
                 "focal_length": 480,
                 "baseline": 0.01,
                 "min_depth": 1.7,
                 "max_depth": 2.7}

    processor = StructureTensorProcessor(parameter)

    main = DenseLightFieldProcessor(parameter, processor)
    main.run()



    #ims.imshow(lf[:, 250, :, 0:lf.shape[3]])