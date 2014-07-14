import sys, os
from glob import glob
import numpy as np
import vigra
from scipy.misc import imread, imsave, imshow
import mypy.visualization.imshow as ims

from mypy.lightfield.io import load_lf3d_fromFiles
from mypy.lightfield.helpers import changeColorspace
from mypy.pointclouds.depthToCloud import disparity_to_depth, cloud_from_depth, PlyWriter, transformCloud


class SubLFProcessor(object):
    def __init__(self, parameter):
        assert isinstance(parameter, type({}))

        self.fpath = parameter["filepath"]
        self.focus = parameter["focus"]
        self.numOfCams = parameter["num_of_cams"]
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
            self.cv = np.copy(self.lf[self.shape[0]/2, :, :, 0:self.shape[3]])
            if self.cv.dtype != np.uint8:
                self.cv *= 255.0
                self.cv = self.cv.astype(np.uint8)

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
        assert isinstance(parameter["num_of_cams"], int)
        parameter["sensor_size"] = float(parameter["sensor_size"])
        parameter["inner_scale"] = float(parameter["inner_scale"])
        parameter["outer_scale"] = float(parameter["outer_scale"])
        parameter["focal_length_mm"] = float(parameter["focal_length_mm"])
        parameter["baseline"] = float(parameter["baseline"])
        parameter["min_depth"] = float(parameter["min_depth"])
        parameter["max_depth"] = float(parameter["max_depth"])

        print "\n<-- set up DenseLightFieldProcessor..."

        self.parameter = parameter
        self.processor = processor

        # compute additional parameter necessary for computation
        self.parameter["focal_length_per_pixel"] = parameter["focal_length_mm"]/parameter["sensor_size"]
        self.parameter["focal_length_px"] = self.parameter["focal_length_per_pixel"]*parameter["resolution"][0]
        self.parameter["center_of_scene_m"] = [self.parameter["cam_initial_pos"][1]-self.parameter["cam_final_pos"][1],
                                               self.parameter["cam_initial_pos"][0]-self.parameter["cam_final_pos"][0]]

        # compute amount of iterations possible
        self.iterations = self.parameter["total_frames"]/self.parameter["num_of_cams"]
        print self.iterations, "iterations are necessary"

        #compute real world scene grid
        self.visibleSceneWidth, self.visibleSceneHeight = self.computeVisibleSceneArea()
        print "computed visible scene size h =", self.visibleSceneHeight, "m w =", self.visibleSceneWidth, "m"
        self.worldGrid = np.zeros((self.visibleSceneHeight/parameter["world_accuracy_m"], self.visibleSceneWidth/parameter["world_accuracy_m"], 2), dtype=np.float32)
        print "created world grid of size (h,w):", self.worldGrid.shape[0], self.worldGrid.shape[1]

        self.parameter["center_of_scene_px"] = [self.worldGrid.shape[0]/2, self.worldGrid.shape[1]/2]


        print "done -->"

    def computeVisibleSceneArea(self):

        # compute field of view angles
        alpha = np.arctan2(float(self.parameter["sensor_size"]), 2.0*self.parameter["focal_length_mm"])
        alpha2 = np.arctan2(self.parameter["sensor_size"]/(float(self.parameter["resolution"][0])/self.parameter["resolution"][1]), 2.0*parameter["focal_length_mm"])

        # compute distance between cameras
        d_tot = np.abs(float(self.parameter["cam_initial_pos"][0])-float(self.parameter["cam_final_pos"][0]))

        # compute real visible scene width and height
        vsw = 2*self.parameter["cam_initial_pos"][2]*np.tan(alpha)+d_tot
        vsh = 2*self.parameter["cam_initial_pos"][2]*np.tan(alpha2)
        return vsw, vsh


    def getWorldCoordFromIndex(self, h, w):
        pass


    def getIndexFromWorldCoord(self, x, y):
        pass

    def run(self):
        cam_index = 0
        append_points = False
        target_cam_shift = self.parameter["baseline"]*self.parameter["total_frames"]/2

        plyWriter = PlyWriter(self.parameter["resultpath"], format="EN")

        final_cloud = None
        final_result = None

        for i in range(self.iterations):
            print "\n<-- compute iteration step", i, "..."

            if i > 0:
                append_points = True

            if self.processor.load(cam_index):
                cloudshift = self.parameter["cam_initial_pos"][0] + i*self.parameter["baseline"]*self.parameter["num_of_cams"]+self.parameter["baseline"]*self.parameter["num_of_cams"]/2
                translate = [cloudshift, 0.0, self.parameter["cam_initial_pos"][2]]

                print "transform cloud..."
                print "translation:", translate, "..."
                print "rotate:", self.parameter["cam_rotation"], "..."

                processor.compute()
                results = processor.getResults()

                if i==0:
                    final_result = np.zeros_like(results)

                depth = disparity_to_depth(results[:, :, 0],
                                           self.parameter["baseline"],
                                           self.parameter["focal_length_px"],
                                           self.parameter["min_depth"],
                                           self.parameter["max_depth"])

                cloud = cloud_from_depth(depth, self.parameter["focal_length_px"])
                cloud = transformCloud(cloud,
                                       rotate_x=self.parameter["cam_rotation"][0],
                                       rotate_y=self.parameter["cam_rotation"][1],
                                       rotate_z=self.parameter["cam_rotation"][2],
                                       translate=translate)

                if i==0:
                    final_cloud = np.zeros_like(cloud)

                plyWriter.cloud = cloud
                plyWriter.colors = processor.cv
                plyWriter.confidence = results[:, :, 1]

                plyWriter.save(append=append_points)

                imsave(self.parameter["resultpath"]+"_depth_%4.4i.png" % i, processor.results[:, :, 0])
                imsave(self.parameter["resultpath"]+"_coherence_%4.4i.png" % i, processor.results[:, :, 1])

                print "done -->"
            else:
                print "\nFinished...!"
                sys.exit()

            cam_index += self.parameter["num_of_cams"]







if __name__ == "__main__":

    parameter = {"filepath": "/home/swanner/Desktop/denseSampledTestScene/rendered/imgs",
                 "resultpath": "/home/swanner/Desktop/denseSampledTestScene/results/cloud",
                 "num_of_cams": 9,
                 "total_frames": 200,
                 "focus": 2,
                 "cam_rotation": [180.0, 0.0, 0.0],
                 "cam_initial_pos": [-1.0, 0.0, 2.6],
                 "cam_final_pos": [1.0, 0.0, 2.6],
                 "world_accuracy_m": 0.01,
                 "resolution": [960, 540],
                 "sensor_size": 32,
                 "colorspace": "hsv",
                 "inner_scale": 0.6,
                 "outer_scale": 1.0,
                 "min_coherence": 0.98,
                 "focal_length_mm": 16,
                 "baseline": 0.01,
                 "min_depth": 1.4,
                 "max_depth": 2.8}

    processor = StructureTensorProcessor(parameter)

    main = DenseLightFieldProcessor(parameter, processor)
    main.run()



    #ims.imshow(lf[:, 250, :, 0:lf.shape[3]])