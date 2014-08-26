# -*- coding: utf-8 -*-

import numpy as np
from mypy.streaming.depthProjector import DepthProjector





if __name__ == "__main__":

    #color = np.random.randint(0, 255, 540 * 960 * 3).reshape((540, 960, 3))
    depthProjector = DepthProjector()

    depthProjector.addDepthMapFromFile("/home/swanner/Desktop/tmp/depth/0001.exr")
    depthProjector.addCamera(35.0, 32.0, [540, 960], np.array((4.9950, -4.1860, 3.9597)), np.array((1.1500, 0.0000, 0.8197)))
    #depthProjector.addColor(color)

    depthProjector.addDepthMapFromFile("/home/swanner/Desktop/tmp/depth/0002.exr")
    depthProjector.addCamera(35.0, 32.0, [540, 960], np.array((-3.2872, 4.5016, 5.4028)), np.array((0.8993, 0.0000, 3.7588)))
    #depthProjector.addColor(color)

    depthProjector.addDepthMapFromFile("/home/swanner/Desktop/tmp/depth/0003.exr")
    depthProjector.addCamera(35.0, 32.0, [540, 960], np.array((-3.3543, -5.7408, 4.9266)), np.array((1.0444, 0.0000, 5.7277)))
    #depthProjector.addColor(color)

    depthProjector(0.35)
    depthProjector.save("/home/swanner/Desktop/tmp/cloud.ply")

    print "finished, created cloud with", depthProjector.cloud.shape[0], "points in total!"





