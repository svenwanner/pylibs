"""
1.) Copy this script anywhere and specifiy all paths to your data relative to this script.
2.) Define all standard config parameter
3.) Define your tasks by variing the paramter you want to change, don't forget to give every
   experiment a unique result_label and run the workflow.
4.) By calling multiple of these scripts in a sh script via python oneOfTheseScripts.py you can set up
   huge experiments.
"""

import logging
import inspect, os
from mypy.lightfield.helpers import checkDebug
import mypy.workflows.StructureTensor25D as st25d
import mypy.lightfield.depth.prefilter as prefilter

context = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

### Activate Debugging in user input 'debug' is set
checkDebug(sys.argv)


logging.info(context)

#============================================================================================================
#==================        Set Parameter and start disparity estimation framework        ====================
#============================================================================================================


#create a config object
config = st25d.Config()

#                                change only below here
#-----------------------------------------------------------------------------------



#================================
#=== 2.) set config parameter ===
#================================

# path to store the results [necessary]
config.result_path = context + "/results"

# name of the results folder [necessary]
config.result_label = "mytest"

# path to the horizontal images [default None]
config.path_horizontal = context + "/h"

# path to the vertical images [default None]
config.path_vertical = context + "/v"

# path to the center view image to get color for pointcloud [default None]
config.centerview_path = context + "/h/0007.tif"

#region of interest to process roi = {"pos":[y,x],"size":[sy,sx]
config.roi = None

config.structure_tensor_type = "classic"  # type of the structure tensor class to be used

############## not used in 2.5D
config.inner_scale = 0.6  # structure tensor inner scale [default 0.6]
config.outer_scale = 1.3  # structure tensor outer scale [default 1.3]


########  2.5D filter
config.volume_pre_smoothing_scale = 0.5 # Smooth each structure tensor compnent in the 3D volume
config.volume_post_smoothing_scale = 1.1 # Smooth each structure tensor compnent in the 3D volume


config.coherence_threshold = 0.1  # if coherence less than value the disparity is set to invalid
config.focal_length = 5740.38  # focal length in pixel [default Nikon D800 f=28mm]
config.global_shifts = [8, 9 ]  # list of horopter shifts in pixel [default [0]]
config.base_line = 0.001  # camera baseline [default 0.001]

config.color_space = prefilter.COLORSPACE.RGB  # colorscape to convert the images into possible RGB,LAB,LUV [default RGB]

######
config.prefilter_scale = 0.4  # scale of the prefilter [0.4] and for the derivative filter
config.prefilter = prefilter.PREFILTER.IMGD2  # type of the prefilter possible NO,IMGD, EPID, IMGD2, EPID2 [default IMGD2}

config.median = 5  # apply median filter on disparity map
config.selective_gaussian = 2.0  # apply a selective gaussian post filter
config.tv = {"alpha": 1.0, "steps": 1000}  # apply total variation to depth map [default {"alpha": 1.0, "steps": 1000}]

config.min_depth = 0.01  # minimum depth possible [default 0.01]
config.max_depth = 1.0  # maximum depth possible [default 1.0]

config.rgb = True  # forces grayscale if False [default True]

config.output_level = 3  # level of detail for file output possible 1,2,3 [default 2]


#=====================================================
#=== 3.) run workflow for multiple cofigurations  ====
#=====================================================

config.result_label = "2.5D_StructureTensor"
st25d.structureTensor25D(config)


#config.result_label = "gradientImgPrefilter"
#config.prefilter = st2d.PREFILTER.IMGD2
#st2d.structureTensor2D(config)
#config.saveLog()

#config.result_label = "gradientEpiPrefilter"
#config.prefilter = st2d.PREFILTER.EPID2
#config.saveLog()

#config.result_label = "laplaceImgPrefilter"
#config.prefilter = st2d.PREFILTER.IMGD2
#st2d.structureTensor2D(config)
#config.saveLog()

#config.result_label = "laplaceEpiPrefilter"
#config.prefilter = st2d.PREFILTER.EPID2
#config.saveLog()
