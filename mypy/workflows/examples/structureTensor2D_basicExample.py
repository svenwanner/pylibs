import mypy.workflows.StructureTensor2D as st2d

#create a config object
config = st2d.Config

#                                Change below here
#-----------------------------------------------------------------------------------



#============================
#=== set config parameter ===
#============================

# path to store the results [necessary]
config.result_path = "/home/to/my/results"
# name of the results folder [necessary]
config.result_label = "myResultPathName"
# path to the horizontal images [default None]
config.path_horizontal = "/home/to/my/horizontal/data"
# path to the vertical images [default None]
config.path_vertical = "/home/to/my/vertical/data"
# path to the center view image to get color for pointcloud [default None]
config.centerview_path = "/home/to/my/horizontal/centerviewImage.png"
config.inner_scale = 0.6                    # structure tensor inner scale [default 0.6]
config.outer_scale = 1.3                    # structure tensor outer scale [default 1.3]
config.focal_length = 5740.38               # focal length in pixel [default Nikon D800 f=28mm]
config.global_shifts = [2,3,4,5,6,7,8]      # list of horopter shifts in pixel [default [0]]
config.base_line = 0.001                    # camera baseline [default 0.001]
config.color_space = st2d.COLORSPACE.RGB    # colorscape to convert the images into possible RGB,LAB,LUV [default RGB]
config.prefilter_scale = 0.4                # scale of the prefilter [0.4]
config.prefilter = st2d.PREFILTER.IMGD2     # type of the prefilter possible NO,IMGD, EPID, IMGD2, EPID2 [default IMGD2}
config.min_depth = 0.01                     # minimum depth possible [default 0.01]
config.max_depth = 1.0                      # maximum depth possible [default 1.0]
config.rgb = True                           # forces grayscale if False [default True]
config.output_level = 2                     # level of detail for file output possible 1,2,3 [default 2]


#============================
#===     run workflow     ===
#============================
st2d.structureTensor2D(config)
