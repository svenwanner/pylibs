import sys, os
from glob import glob
import numpy as np
import vigra
from scipy.misc import imread, imsave, imshow
import mypy.visualization.imshow as ims

from mypy.lightfield.io import load_lf3d_fromFiles
from mypy.lightfield.helpers import changeColorspace, refocus_epi
from mypy.streaming.discreteWorld import discreteWorldSpace
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
            self.lf = load_lf3d_fromFiles(self.fpath, camIndex, self.numOfCams, dtype=np.float32)
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

        for y in xrange(self.shape[1]):

            epi = np.copy(self.lf[:, y, :, :])
            tmp_results = np.zeros((len(self.parameter["focus"]), epi.shape[1], 5), np.float32)

            for n, focus in enumerate(self.parameter["focus"]):

                ### compute tensor for each y coordinate and each color channel ###
                tensor = np.zeros((self.shape[3], self.shape[0], self.shape[2], 3), dtype=np.float32)

                repi = refocus_epi(epi, focus)
                for c in range(self.shape[3]):
                    tensor[c, :, :, :] = vigra.filters.structureTensor(repi[:, :, c],
                                                                       self.parameter["inner_scale"],
                                                                       self.parameter["outer_scale"])

                ### mean over color channels ###
                tensor = np.mean(tensor, axis=0)

                ### compute coherence value ###
                up = np.sqrt((tensor[self.shape[0]/2, :, 2]-tensor[self.shape[0]/2, :, 0])**2 + 4*tensor[self.shape[0]/2, :, 1]**2)
                down = (tensor[self.shape[0]/2, :, 2]+tensor[self.shape[0]/2, :, 0] + 1e-25)
                coherence = up / down

                ### compute disparity value ###
                orientation = vigra.numpy.arctan2(2*tensor[self.shape[0]/2, :, 1], tensor[self.shape[0]/2, :, 2]-tensor[self.shape[0]/2, :, 0]) / 2.0
                orientation = vigra.numpy.tan(orientation[:])

                ### mark out of boundary orientation estimation ###
                invalid_ubounds = np.where(orientation > 1.1)
                invalid_lbounds = np.where(orientation < -1.1)
                invalid = np.where(coherence < self.parameter["min_coherence"])

                orientation += focus

                ### set coherence of invalid values to zero ###
                coherence[invalid_ubounds] = 0
                coherence[invalid_lbounds] = 0
                coherence[invalid] = 0

                ### set orientation of invalid values to related maximum/minimum value
                orientation[invalid_ubounds] = -1
                orientation[invalid_lbounds] = -1
                orientation[invalid] = -1

                tmp_results[n, :, 0] = orientation[:]
                tmp_results[n, :, 1] = coherence[:]

            max_c = np.amax(tmp_results[:, :, 1], axis=0)
            arg_max_c = np.argmax(tmp_results[:, :, 1], axis=0)
            x = np.arange(0, tmp_results.shape[1])
            ind = [arg_max_c, x]
            tmp_ori = tmp_results[:, :, 0]
            best_orientations = tmp_ori[ind]

            self.results[y, :, 0] = best_orientations[:]
            self.results[y, :, 1] = max_c[:]
            #self.results[y, :, 0:self.shape[3]] = epi[self.shape[0]/2, :, 0:self.shape[3]]

        print "done -->"


class DenseLightFieldEngine(object):

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
        parameter["fov"] = np.arctan2(float(parameter["sensor_size"]), 2.0*parameter["focal_length_mm"])

        optimal_accuracy = parameter["cam_initial_pos"][2]*np.tan(2*parameter["fov"]/parameter["resolution"][1])
        if not parameter.has_key("world_accuracy_m") or parameter["world_accuracy_m"] <= 0.0:
            parameter["world_accuracy_m"] = optimal_accuracy
        print "optimal accuracy is:", optimal_accuracy, ", you set to:", parameter["world_accuracy_m"]

        print "\n<-- set up DenseLightFieldProcessor..."

        self.parameter = parameter
        self.processor = processor

        self.cam_max_baseline = (self.parameter["total_frames"]-1)*self.parameter["baseline"]
        if not self.parameter.has_key("cam_final_pos"):
            self.parameter["cam_final_pos"] = [self.parameter["cam_initial_pos"][0]+self.cam_max_baseline, 0.0, self.parameter["cam_initial_pos"][2]]

        # compute additional parameter necessary for computation
        self.parameter["focal_length_per_pixel"] = parameter["focal_length_mm"]/parameter["sensor_size"]
        self.parameter["focal_length_px"] = self.parameter["focal_length_per_pixel"]*parameter["resolution"][0]
        self.parameter["center_of_scene_m"] = [self.parameter["cam_final_pos"][0] - self.parameter["cam_initial_pos"][0],
                                               self.parameter["cam_final_pos"][1] - self.parameter["cam_initial_pos"][1]]

        # compute amount of iterations possible
        self.iterations = self.parameter["total_frames"]/self.parameter["num_of_cams"]
        print self.iterations, "iterations are necessary"

        #compute real world scene grid
        self.visibleSceneWidth, self.visibleSceneHeight = self.computeVisibleSceneArea()
        self.final_results = np.zeros((self.iterations, self.visibleSceneHeight, self.visibleSceneWidth, 2))
        # create disceteWorldSpace instance storing all reconstructed points and generating a point cloud when finished
        self.worldGrid = discreteWorldSpace([self.visibleSceneHeight, self.visibleSceneWidth], parameter["world_accuracy_m"], self.iterations)

        print "done -->"

    def computeVisibleSceneArea(self):

        # compute field of view angles
        fov_w = self.parameter["fov"]
        fov_h = np.arctan2(float(self.parameter["sensor_size"])/(float(self.parameter["resolution"][0])/float(self.parameter["resolution"][1])), 2.0*self.parameter["focal_length_mm"])
        print "computed fov_w:", self.parameter["fov"]/np.pi*180.0
        print "computed fov_h:", fov_h/np.pi*180.0

        # compute distance between cameras
        d_tot = self.cam_max_baseline

        # compute real visible scene width and height
        vsw = 2*self.parameter["cam_initial_pos"][2]*np.tan(fov_w)+d_tot
        vsh = 2*self.parameter["cam_initial_pos"][2]*np.tan(fov_h)

        print "visible scene size is: width =", vsw, "height =", vsh, "m"
        return vsw, vsh



    def run(self):
        cam_index = 0

        plyWriter = PlyWriter(self.parameter["resultpath"], format="EN")

        for i in range(self.iterations):
            print "\n<-- compute iteration step", i, "..."

            if self.processor.load(cam_index):
                current_cam_pos = self.parameter["cam_initial_pos"][0] + i*self.parameter["baseline"]*self.parameter["num_of_cams"]+self.parameter["baseline"]*self.parameter["num_of_cams"]/2
                cloud_destination = self.parameter["cam_initial_pos"][0]+(self.parameter["total_frames"]-1)*self.parameter["baseline"]/2.0
                cloudshift = cloud_destination-current_cam_pos
                translate = [-cloudshift, 0.0, self.parameter["cam_initial_pos"][2]]

                print "transform cloud..."
                print "translation:", translate, "..."
                print "rotate:", self.parameter["cam_rotation"], "..."

                processor.compute()
                results = processor.getResults()



                # compute depth from disparity
                depth = disparity_to_depth(results[:, :, 0],
                                           self.parameter["baseline"],
                                           self.parameter["focal_length_px"],
                                           self.parameter["min_depth"],
                                           self.parameter["max_depth"])



                #compute cloud and transform it to center camera
                cloud = cloud_from_depth(depth, self.parameter["focal_length_px"])
                cloud = transformCloud(cloud,
                                       rotate_x=self.parameter["cam_rotation"][0],
                                       rotate_y=self.parameter["cam_rotation"][1],
                                       rotate_z=self.parameter["cam_rotation"][2],
                                       translate=translate)

                # push all points from the cloud into the current iteration layer of the worldGrid instance
                ### TODO: check y,x dimension of cloud is equal to results?
                for n in range(cloud.shape[0]):
                    for m in range(cloud.shape[1]):
                        self.worldGrid.setWorldValue(cloud[n, m, 0], cloud[n, m, 1], i, np.array([cloud[n, m, 2], results[n, m, 1]]))

                # save all disparity steps as image
                imsave(self.parameter["resultpath"]+"_layer_%4.4i.png" % i, self.worldGrid.grid[:, :, i, 0])
                imsave(self.parameter["resultpath"]+"_depth_%4.4i.png" % i, processor.results[:, :, 0])
                imsave(self.parameter["resultpath"]+"_coherence_%4.4i.png" % i, processor.results[:, :, 1])

                print "done -->"
            else:
                print "\nFinished...!"
                sys.exit()

            cam_index += self.parameter["num_of_cams"]



        self.worldGrid.save(self.parameter["resultpath"]+"_worldGrid")

        plyWriter = PlyWriter(self.parameter["resultpath"]+"_final", format="EN")
        cloud = self.worldGrid.getResult()
        cloud = transformCloud(cloud,
                                       rotate_x=self.parameter["cam_rotation"][0],
                                       rotate_y=self.parameter["cam_rotation"][1],
                                       rotate_z=self.parameter["cam_rotation"][2],
                                       translate=[0, 0, 0])

        imsave(self.parameter["resultpath"]+"_finalDepth.png", cloud[:, :, 2])
        imsave(self.parameter["resultpath"]+"_finalCoherence.png", cloud[:, :, 3])
        plyWriter.cloud = cloud

        plyWriter.save()





if __name__ == "__main__":

    parameter = {"filepath": "/home/swanner/Desktop/denseSampledTestScene/rendered/fullRes",
                 "resultpath": "/home/swanner/Desktop/denseSampledTestScene/results_FR/cloud",
                 "num_of_cams": 11,
                 "total_frames": 200,
                 "focus": [2],
                 "cam_rotation": [180.0, 0.0, 0.0],
                 "cam_initial_pos": [-1.0, 0.0, 2.6],
                 "cam_final_pos": [1.0, 0.0, 2.6],
                 "world_accuracy_m": 0.0,
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


    parameter2 = {"filepath": "/home/swanner/Desktop/denseSampledTestScene/rendered2/fullRes",
             "resultpath": "/home/swanner/Desktop/denseSampledTestScene/results2_FR/cloud",
             "num_of_cams": 11,
             "total_frames": 231,
             "focus": [3, 4],
             "cam_rotation": [180.0, 0.0, 0.0],
             "cam_initial_pos": [0.0, 0.0, 2.0],
             #"cam_final_pos": [2.31, 0.0, 2.0],
             "world_accuracy_m": 0.0,
             "resolution": [960, 540],
             "sensor_size": 32,
             "colorspace": "rgb",
             "inner_scale": 0.6,
             "outer_scale": 1.1,
             "min_coherence": 0.95,
             "focal_length_mm": 16,
             "baseline": 0.01004347826086956522,
             "min_depth": 1.40,
             "max_depth": 2.1}


    processor = StructureTensorProcessor(parameter2)

    engine = DenseLightFieldEngine(parameter2, processor)
    engine.run()