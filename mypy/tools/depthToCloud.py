from __future__ import print_function
import sys
import numpy as np
from pylab import imread
from types import IntType, FloatType, StringType, ListType, TupleType
import vigra
import mypy.pointclouds.depthToCloud as dtc



if __name__ == "__main__":

    if len(sys.argv) >= 4 and len(sys.argv) <= 5:

        color_img = vigra.readImage(sys.argv[2])
        assert isinstance(color_img, np.ndarray)
        print("ColorImageSize: ")
        print(color_img.shape)

        if color_img.dtype  == np.uint16:
            color_img = color_img/256
        # img = np.zeros([color_img.shape[0],color_img.shape[1],color_img.shape[2]],dtype=np.uint8)

        img = color_img.astype(dtype=np.uint8)

        depth_img = vigra.readImage(sys.argv[1])
        depth_img = depth_img[:, :, 2]
        assert isinstance(depth_img, np.ndarray)
        print("DepthMapSize: ")
        print(depth_img.shape)

        assert depth_img.shape != color_img.shape

        print("make pointcloud...")
        if isinstance(color_img, np.ndarray):
            dtc.save_pointcloud(sys.argv[3], depth_map=depth_img, color=img, focal_length=float(sys.argv[4]))
        else:
            dtc.save_pointcloud(sys.argv[3], depth_map=depth_img, focal_length=float(sys.argv[4]))

        print("ok")

    else:
        print("\033[93m Convert depthmap to pointcloud\033[0m")
        print("arg1: depthimage file")
        print("arg2: colorimage file")
        print("arg3: point cloud save location and name including extension")
        print("arg4: focallength")


  
