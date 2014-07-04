import os
import vigra
import numpy as np
import pylab as plt
from mypy.lightfield.depth import structureTensor2D as st2d

import mypy.pointclouds.depthToCloud as dtc
from mypy.lightfield.depth.prefilter import COLORSPACE
from mypy.lightfield.depth.prefilter import PREFILTER

from mypy.lightfield import io as lfio
from mypy.lightfield.depth.structureTensor_ComputationClass import Compute


#============================================================================================================
#============================================================================================================
#============================================================================================================

class Config:
    def __init__(self):

        self.result_path = None                 # path to store the results
        self.result_label = None                # name of the results folder

        self.path_horizontal = None             # path to the horizontal images [optional]
        self.path_vertical = None               # path to the vertical images [optional]

        self.roi = None                         # region of interest

        self.centerview_path = None             # path to the center view image to get color for pointcloud [optional]

        self.structure_tensor_type = "classic"  # type of the structure tensor class to be used
        self.inner_scale = 0.6                  # structure tensor inner scale
        self.outer_scale = 0.9                  # structure tensor outer scale
        self.hourglass_scale = 0
        self.coherence_threshold = 0.7          # if coherence less than value the disparity is set to invalid
        self.focal_length = 5740.38             # focal length in pixel [default Nikon D800 f=28mm]
        self.global_shifts = [0]                # list of horopter shifts in pixel
        self.base_line = 0.001                  # camera baseline

        self.color_space = COLORSPACE.RGB       # colorscape to convert the images into [RGB,LAB,LUV]
        self.prefilter_scale = 0.4              # scale of the prefilter
        self.prefilter = PREFILTER.IMGD2        # type of the prefilter [NO,IMGD, EPID, IMGD2, EPID2]

        self.median = 5                         # apply median filter on disparity map
        self.nonlinear_diffusion = [0.5, 5]     # apply nonlinear diffusion [0] edge threshold, [1] scale
        self.selective_gaussian = 2.0           # apply a selective gaussian post filter
        self.tv = {"alpha": 1.0, "steps": 1000} # apply total variation to depth map

        self.min_depth = 0.01                   # minimum depth possible
        self.max_depth = 10.0                    # maximum depth possible

        self.rgb = True                         # forces grayscale if False

        self.output_level = 2                   # level of detail for file output possible 1,2,3

    def saveLog(self, filename=None):
        if filename is not None:
            f = open(filename, "w")
        else:
            f = open(self.result_path+self.result_label+"/log.txt", "w")
        f.write("roi : "); f.write(str(self.roi)+"\n")
        f.write("inner_scale : "); f.write(str(self.inner_scale)+"\n")
        f.write("outer_scale : "); f.write(str(self.outer_scale)+"\n")
        if self.structure_tensor_type == "hour-glass":
            f.write("hourglass_scale : "); f.write(str(self.hourglass_scale)+"\n")
        f.write("coherence_threshold : "); f.write(str(self.coherence_threshold)+"\n")
        f.write("focal_length : "); f.write(str(self.focal_length)+"\n")
        f.write("global_shifts : "); f.write(str(self.global_shifts)+"\n")
        f.write("base_line : "); f.write(str(self.base_line)+"\n")
        f.write("color_space : "); f.write(str(self.color_space)+"\n")
        f.write("prefilter_scale : "); f.write(str(self.prefilter_scale)+"\n")
        f.write("prefilter : "); f.write(str(self.prefilter)+"\n")
        f.write("median : "); f.write(str(self.median)+"\n")
        f.write("nonlinear_diffusion : "); f.write(str(self.nonlinear_diffusion)+"\n")
        f.write("selective_gaussian : "); f.write(str(self.selective_gaussian)+"\n")
        f.write("total variation : "); f.write(str(self.tv)+"\n")
        f.write("min_depth : "); f.write(str(self.min_depth)+"\n")
        f.write("max_depth : "); f.write(str(self.max_depth)+"\n")
        f.close()


#============================================================================================================
#============================================================================================================
#============================================================================================================



def structureTensor2D(config):

########################################################################################################################
##################################   Check the correctness of the parent path
########################################################################################################################

    if not config.result_path.endswith("/"):
        config.result_path += "/"
    if not config.result_label.endswith("/"):
        config.result_label += "/"
    if not os.path.isdir(config.result_path+config.result_label):
        os.makedirs(config.result_path+config.result_label)
    if config.output_level >3:
        print('config result path: ' + str(config.result_label))

########################################################################################################################
##################################   Initialize light field descriptors
########################################################################################################################

    compute_h = False
    compute_v = False
    lf_shape = None
    lf3dh = None
    lf3dv = None

########################################################################################################################
##################################   Load Image Data
########################################################################################################################

    print("load data...")
    try:
        if not config.path_horizontal.endswith("/"):
            config.path_horizontal += "/"
        lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
        compute_h = True
        lf_shape = lf3dh.shape
        if config.output_level >3:
            print('Image shape of horizontal images: ' + str(lf3dh.shape))

    except:
        pass

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        compute_v = True
        if lf_shape is None:
            lf_shape = lf3dv.shape
        if config.output_level >3:
            print('Image shape of vertical images: ' + str(lf3dv.shape))

    except:
        pass


    print "ok"

########################################################################################################################
##################################  Initialize memory for disparity and coherence values
########################################################################################################################

    coherence = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    orientation = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)


########################################################################################################################
##################################  Thread split computation for horizontal and vertical light field
########################################################################################################################

    for shift in config.global_shifts:

        threads = []

    ### generate one thread for the horizontal computation ###
        if compute_h:
            thread = Compute(lf3dh, shift, config, direction='h')
            threads += [thread]
            thread.start()

    #    if compute_v:
    #        thread = Compute(lf3dv, shift, config, direction='v')
    #        threads += [thread]
    #        thread.start()

    ### Initialize Pointer for Solution Array ###

        orientation_h = None
        coherence_h = None
        orientation_v = None
        coherence_v = None

    ### Join threads and get solution of it ###
        for x in threads:
            x.join()
            if x.direction == 'h':
                orientation_h, coherence_h = x.get_results()
            if x.direction == 'v':
                orientation_v, coherence_v = x.get_results()

        if compute_h and compute_v:

            print("merge vertical/horizontal ...")

            orientation_tmp, coherence_tmp = st2d.mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
            orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :])
                plt.imsave(config.result_path+config.result_label+"coherence_merged_shift_{0}.png".format(shift), coherence[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            print "ok"

        else:

            print("merge shifts")

            if compute_h:
                orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)

            if compute_v:
                orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)

            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
                plt.imsave(config.result_path+config.result_label+"coherence_merged_shift_{0}.png".format(shift), coherence[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            print "ok"

    mask = coherence[lf_shape[0]/2, :, :]

    if config.output_level >= 2:
        plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence_final.png", mask, cmap=plt.cm.jet)

    depth = dtc.disparity_to_depth(orientation[lf_shape[0]/2, :, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)

    #if isinstance(config.nonlinear_diffusion, type([])):
    #    print "apply nonlinear diffusion",
    #    vigra.filters.nonlinearDiffusion(depth, config.nonlinear_diffusion[0], config.nonlinear_diffusion[1])
    #    print "ok"
    #if isinstance(config.selective_gaussian, float) and config.selective_gaussian > 0:
    #    print "apply masked gauss...",
    #    gauss = vigra.filters.Kernel2D()
    #    vigra.filters.Kernel2D.initGaussian(gauss, config.selective_gaussian)
    #    gauss.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_CLIP)
    #    depth = vigra.filters.normalizedConvolveImage(depth, mask, gauss)
    #    print "ok"
    #if isinstance(config.median, int) and config.median > 0:
    #    print "apply median filter ...",
    #    depth = median_filter(depth, config.median)
    #    print "ok"
    #if isinstance(config.tv, type({})):
    #    print "apply total variation...",
    #    assert depth.shape == mask.shape
    #    drange = config.max_depth-config.min_depth
    #    depth = vigra.filters.totalVariationFilter(depth.astype(np.float64), mask.astype(np.float64), 0.01*drange*config.tv["alpha"], config.tv["steps"], 0)
    #    print "ok"

    invalids = np.where(mask == 0)
    depth[invalids] = 0

    if config.output_level >= 1:
        plt.imsave(config.result_path+config.result_label+"depth_final.png", depth, cmap=plt.cm.jet)

    #if config.output_level >= 1:
    #    if isinstance(config.centerview_path, str):
    #        color = misc.imread(config.centerview_path)
    #        if isinstance(config.roi, type({})):
    #            sposx = config.roi["pos"][0]
    #            eposx = config.roi["pos"][0] + config.roi["size"][0]
    #            sposy = config.roi["pos"][1]
    #            eposy = config.roi["pos"][1] + config.roi["size"][1]
    #            color = color[sposx:eposx, sposy:eposy, 0:3]

        # tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
        # tmp[:, :, 0] = orientation[lf_shape[0]/2, :, :]
        # tmp[:, :, 1] = coherence[lf_shape[0]/2, :, :]
        # tmp[:, :, 2] = depth[:]
        # vim = vigra.RGBImage(tmp)
        # vim.writeImage(config.result_path+config.result_label+"final.exr")
        # myshow.finalsViewer(config.result_path+config.result_label+"final.exr", save_at=config.result_path+config.result_label)

    #    print "make pointcloud...",
    #    if isinstance(color, np.ndarray):
    #        dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, color=color, confidence=coherence[lf_shape[0]/2, :, :], focal_length=config.focal_length)
    #    else:
    #        dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, confidence=coherence[lf_shape[0]/2, :, :], focal_length=config.focal_length)

    #    print "ok"
