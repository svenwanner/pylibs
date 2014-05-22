import mypy.workflows.StructureTensor2D as st2d


config = st2d.Config

#                                change only below here
#-----------------------------------------------------------------------------------

#============================
#=== set config parameter ===
#============================

# path to store the results [necessary]
config.result_path = "/home/swanner/Desktop/ZeissData/results"

# name of the results folder [necessary]
config.result_label = "results_nld_1_7"

# path to the horizontal images [default None]
config.path_horizontal = "/home/swanner/Desktop/ZeissData/h"

# path to the vertical images [default None]
config.path_vertical = "/home/swanner/Desktop/ZeissData/v"

# path to the center view image to get color for pointcloud [default None]
config.centerview_path = "/home/swanner/Desktop/ZeissData/results/"+config.result_label+"/coherence_final.png"
config.centerview_path = "/home/swanner/Desktop/ZeissData/cv.png"

#region of interest to process roi = {"pos":[y,x],"size":[sy,sx]
config.roi = None
#config.roi = {"pos": [110, 90], "size": [300, 434]}

config.inner_scale = 0.6                    # structure tensor inner scale [default 0.6]
config.outer_scale = 1.8                    # structure tensor outer scale [default 1.3]
config.double_tensor = 0.0                  # if > 0.0 a second structure tensor with the outerscale specified is applied
config.coherence_threshold = 0.45           # if coherence less than value the disparity is set to invalid
config.focal_length = 5740.38               # focal length in pixel [default Nikon D800 f=28mm]
config.global_shifts = [8, 9]               # list of horopter shifts in pixel [default [0]]
config.base_line = 0.001                    # camera baseline [default 0.001]

config.color_space = st2d.COLORSPACE.RGB    # colorscape to convert the images into possible RGB,LAB,LUV [default RGB]
config.prefilter_scale = 0.4                # scale of the prefilter [0.4]
config.prefilter = st2d.PREFILTER.IMGD2     # type of the prefilter possible NO,IMGD, EPID, IMGD2, EPID2 [default IMGD2}
config.median = 3                           # apply median filter on disparity map
config.nonlinear_diffusion = None              # apply nonlinear diffusion [0] edge threshold, [1] scale

config.min_depth = 0.01                     # minimum depth possible [default 0.01]
config.max_depth = 1.0                      # maximum depth possible [default 1.0]
config.rgb = True                           # forces grayscale if False [default True]

config.output_level = 2                     # level of detail for file output possible 1,2,3 [default 2]

#-----------------------------------------------------------------------------------
#                                change only above here

#============================
#===     run workflow     ===
#============================
st2d.structureTensor2D(config)

