import mypy.workflows.StructureTensor2D as st2d


config = st2d.Config()

#                                change only below here
#-----------------------------------------------------------------------------------

#============================
#=== set config parameter ===
#============================

# path to store the results [necessary]
config.result_path = "/path/to/your/results"

# name of the results folder [necessary]
config.result_label = "myResults"

# path to the horizontal images [default None]
config.path_horizontal = "/path/to/your/horizontal/images"

# path to the vertical images [default None]
config.path_vertical = "/path/to/your/vertical/images"

# path to the center view image to get color for pointcloud [default None]
config.centerview_path = "/path/to/your/image/that/should/be/mapped/onto/pointcloud/cv_img.png"

#region of interest to process roi = {"pos":[y,x],"size":[sy,sx]} [Default None]
config.roi = {"pos": [10, 10], "size": [100, 100]}

config.structure_tensor_type = "hour-glass" # type of the structure tensor class to be used ["classic" , "hour-glass", "Scharr" , "Experimental"]

config.inner_scale = 0.6                    # structure tensor inner scale [default 0.6]
config.outer_scale = 1.3                    # structure tensor outer scale [default 1.3]
#Optional:
config.hourglass_scale = 0.6                # if hour glass filter is activated

config.coherence_threshold = 0.45           # if coherence less than value the disparity is set to invalid
config.focal_length = 5740.38               # focal length in pixel [default Nikon D800 f=28mm]

config.base_line = 0.001                    # camera baseline [default 0.001]

config.color_space = st2d.COLORSPACE.RGB    # colorscape to convert the images into possible RGB,LAB,LUV [default RGB]
config.prefilter_scale = 0.4                # scale of the prefilter [0.4]
config.prefilter = st2d.PREFILTER.IMGD2     # type of the prefilter possible NO,IMGD, EPID, IMGD2, EPID2, SCHARR [default IMGD2}


config.interpolation = st2d.INTERPOLATE.SPLINE  # {NONE,SPLINE} while NONE is pixelwise shift
config.global_shifts = [8, 9]               # list of horopter shifts in pixel [default [0]]

config.median = 3                           # apply median filter on disparity map
config.nonlinear_diffusion = None           # apply nonlinear diffusion [0] edge threshold, [1] scale
config.selective_gaussian = 2.0             # apply a selective gaussian post filter
config.tv = {"alpha": 1.0, "steps": 1000}   # apply total variation to depth map [default {"alpha": 1.0, "steps": 1000}]

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
config.saveLog()
