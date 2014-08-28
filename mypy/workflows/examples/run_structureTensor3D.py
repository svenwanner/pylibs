"""
1.) Copy this script anywhere and specifiy all paths to your data relative to this script.
2.) Define all standard config parameter
3.) Define your tasks by variing the paramter you want to change, don't forget to give every
   experiment a unique result_label and run the workflow.
4.) By calling multiple of these scripts in a sh script via python oneOfTheseScripts.py you can set up
   huge experiments.
"""

import logging
import inspect, os, sys
from mypy.lightfield.helpers import checkDebug
import mypy.workflows.StructureTensor3D as st3d
import mypy.lightfield.depth.prefilter as PREF

context = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

### Activate Debugging in user input 'debug' is set
checkDebug(sys.argv)


logging.info(context)

#============================================================================================================
#==================        Set Parameter and start disparity estimation framework        ====================
#============================================================================================================


#create a config object
config = st3d.Config()

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
config.centerview_path = context + "/h/0LFcam_h_006.png"

#region of interest to process roi = {"pos":[y,x],"size":[sy,sx]
config.roi = None

config.structure_tensor_type = "classic"  # type of the structure tensor class to be used

config.inner_scale = 0.3  # structure tensor inner scale [default 0.6]
config.outer_scale = 0.9  # structure tensor outer scale [default 1.3]

config.coherence_threshold = 0.1  # if coherence less than value the disparity is set to invalid
config.focal_length = 5740.38  # focal length in pixel [default Nikon D800 f=28mm]
config.global_shifts = [6,5,4,3]  # list of horopter shifts in pixel [default [0]]
config.base_line = 0.001  # camera baseline [default 0.001]

config.color_space = PREF.COLORSPACE.RGB  # colorscape to convert the images into possible RGB,LAB,LUV [default RGB]

######
config.prefilter_scale = 0.4  # scale of the prefilter [0.4] and for the derivative filter
config.prefilter = PREF.PREFILTER.NO  # type of the prefilter possible NO,IMGD, EPID, IMGD2, EPID2 [default IMGD2}

config.min_depth = 0.01  # minimum depth possible [default 0.01]
config.max_depth = 1.0  # maximum depth possible [default 1.0]

config.rgb = True  # forces grayscale if False [default True]

config.output_level = 3  # level of detail for file output possible 1,2,3 [default 2]


#=====================================================
#=== 3.) run workflow for multiple cofigurations  ====
#=====================================================

config.result_label = "3D_StructureTensor"
st3d.structureTensor3D(config)