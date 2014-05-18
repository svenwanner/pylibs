import vigra
import inspect, os
import numpy as np
import scipy.misc as misc

import mypy.pointclouds.depthToCloud as dtc


main_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
testfiles_path = main_path+"/../pointclouds/PointcloudTests/"





if __name__ == "__main__":
    depth1 = dtc.loadEXR(testfiles_path+"/pos1.exr")
    depth2 = dtc.loadEXR(testfiles_path+"/pos2.exr")
    depth3 = dtc.loadEXR(testfiles_path+"/pos3.exr")
    depth4 = dtc.loadEXR(testfiles_path+"/pos4.exr")

    cloud1 = dtc.cloud_from_depth(depth1, 1350.0)
    color1 = misc.imread(testfiles_path+"/pos1.png")
    cloud2 = dtc.cloud_from_depth(depth2, 1350.0)
    color2 = misc.imread(testfiles_path+"/pos2.png")
    cloud3 = dtc.cloud_from_depth(depth3, 1350.0)
    color3 = misc.imread(testfiles_path+"/pos3.png")
    cloud4 = dtc.cloud_from_depth(depth4, 1350.0)
    color4 = misc.imread(testfiles_path+"/pos4.png")

    confidence = np.random.randint(0, 1000, cloud1.shape[0]*cloud1.shape[1]).reshape((cloud1.shape[0],cloud1.shape[1])).astype(np.float32)
    confidence[:] /= 1000.0

    plyWriter = dtc.PlyWriter(testfiles_path+"/pos1.ply", cloud1, color1, confidence, confidence)
    plyWriter = dtc.PlyWriter(testfiles_path+"/pos2.ply", cloud2, color2, confidence)
    plyWriter = dtc.PlyWriter(testfiles_path+"/pos3.ply", cloud3, color3)
    plyWriter = dtc.PlyWriter(testfiles_path+"/pos4.ply", cloud4)
