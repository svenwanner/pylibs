import os
import vigra
import numpy as np
import pylab as plt
import scipy.misc as misc
import threading
from scipy.ndimage.filters import median_filter


from mypy.lightfield import io as lfio
from mypy.lightfield.helpers import enum
import mypy.pointclouds.depthToCloud as dtc
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor2D as st2d
from mypy.visualization.imshow import epishow


COLORSPACE = enum(RGB=0, LAB=1, LUV=2)
PREFILTER = enum(NO=0, IMGD=1, EPID=2, IMGD2=3, EPID2=4)



#============================================================================================================
#============================================================================================================
#============================================================================================================

class Config:
    def __init__(self):

        self.result_path = None             # path to store the results
        self.result_label = None            # name of the results folder

        self.path_horizontal = None         # path to the horizontal images [optional]
        self.path_vertical = None           # path to the vertical images [optional]

        self.roi = None                     # region of interest

        self.centerview_path = None         # path to the center view image to get color for pointcloud [optional]

        self.inner_scale = 0.6              # structure tensor inner scale
        self.outer_scale = 0.9              # structure tensor outer scale
        self.double_tensor = 2.0            # if > 0.0 a second structure tensor with the outerscale specified is applied
        self.coherence_threshold = 0.7      # if coherence less than value the disparity is set to invalid
        self.focal_length = 5740.38         # focal length in pixel [default Nikon D800 f=28mm]
        self.global_shifts = [0]            # list of horopter shifts in pixel
        self.base_line = 0.001              # camera baseline

        self.color_space = COLORSPACE.RGB   # colorscape to convert the images into [RGB,LAB,LUV]
        self.prefilter_scale = 0.4          # scale of the prefilter
        self.prefilter = PREFILTER.IMGD2    # type of the prefilter [NO,IMGD, EPID, IMGD2, EPID2]

        self.median = 5                     # apply median filter on disparity map
        self.nonlinear_diffusion = [0.5, 5] # apply nonlinear diffusion [0] edge threshold, [1] scale
        self.selective_gaussian = 2.0       # apply a selective gaussian post filter

        self.min_depth = 0.01               # minimum depth possible
        self.max_depth = 1.0                # maximum depth possible

        self.rgb = True                     # forces grayscale if False

        self.output_level = 2               # level of detail for file output possible 1,2,3

    def saveLog(self, filename=None):
        if filename is not None:
            f = open(filename, "w")
        else:
            f = open(self.result_path+self.result_label+"/log.txt", "w")
        f.write("roi : ")
        f.write(str(self.roi)+"\n")
        f.write("inner_scale : ")
        f.write(str(self.inner_scale)+"\n")
        f.write("outer_scale : ")
        f.write(str(self.outer_scale)+"\n")
        f.write("double_tensor : ")
        f.write(str(self.double_tensor)+"\n")
        f.write("coherence_threshold : ")
        f.write(str(self.coherence_threshold)+"\n")
        f.write("focal_length : ")
        f.write(str(self.focal_length)+"\n")
        f.write("global_shifts : ")
        f.write(str(self.global_shifts)+"\n")
        f.write("base_line : ")
        f.write(str(self.base_line)+"\n")
        f.write("color_space : ")
        f.write(str(self.color_space)+"\n")
        f.write("prefilter_scale : ")
        f.write(str(self.prefilter_scale)+"\n")
        f.write("prefilter : ")
        f.write(str(self.prefilter)+"\n")
        f.write("median : ")
        f.write(str(self.median)+"\n")
        f.write("nonlinear_diffusion : ")
        f.write(str(self.nonlinear_diffusion)+"\n")
        f.write("selective_gaussian : ")
        f.write(str(self.selective_gaussian)+"\n")
        f.write("min_depth : ")
        f.write(str(self.min_depth)+"\n")
        f.write("max_depth : ")
        f.write(str(self.max_depth)+"\n")
        f.close()


#============================================================================================================
#============================================================================================================
#============================================================================================================


class Compute(threading.Thread):
    lock = threading.Lock()

    def __init__(self, lf3d, shift, config, direction):
        threading.Thread.__init__(self)
        self.lf3d = lf3d
        self.shift = shift
        self.config = config
        self.direction = direction
        self.orientation = None
        self.coherence = None

    def run(self):
        if self.direction == 'h':
            self.orientation, self.coherence = compute_horizontal(self.lf3d, self.shift, self.config)
        if self.direction == 'v':
            self.orientation, self.coherence = compute_vertical(self.lf3d, self.shift, self.config)

    def get_results(self):
        return self.orientation, self.coherence



def compute_horizontal(lf3dh, shift, config):
    print "compute horizontal shift {0}".format(shift), "...",
    lf3d = np.copy(lf3dh)
    lf3d = lfhelpers.refocus_3d(lf3d, shift, 'h')

    if config.color_space:
        lf3d = st2d.changeColorSpace(lf3d, config.color_space)

    if config.prefilter > 0:
        if config.prefilter == PREFILTER.IMGD:
            lf3d = st2d.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='h')
        if config.prefilter == PREFILTER.EPID:
            lf3d = st2d.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='h')
        if config.prefilter == PREFILTER.IMGD2:
            lf3d = st2d.preImgLaplace(lf3d, scale=config.prefilter_scale)
        if config.prefilter == PREFILTER.EPID2:
            lf3d = st2d.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='h')


    st3d = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.outer_scale, direction='h')
    if config.double_tensor > 0.0:
        tmp = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.double_tensor, direction='h')
        st3d[:] += tmp[:]
        st3d /= 2.0

    orientation_h, coherence_h = st2d.evaluateStructureTensor(st3d)
    orientation_h[:] += shift

    if config.coherence_threshold > 0.0:
        invalids = np.where(coherence_h < config.coherence_threshold)
        coherence_h[invalids] = 0.0

    if config.output_level == 3:
        misc.imsave(config.result_path+config.result_label+"orientation_h_shift_{0}.png".format(shift), orientation_h[lf_shape[0]/2, :, :])
    if config.output_level == 3:
        misc.imsave(config.result_path+config.result_label+"coherence_h_{0}.png".format(shift), coherence_h[lf_shape[0]/2, :, :])
    print "ok"

    return orientation_h, coherence_h



def compute_vertical(lf3dv, shift, config):
    print "compute vertical shift {0}".format(shift), "...",
    lf3d = np.copy(lf3dv)
    lf3d = lfhelpers.refocus_3d(lf3d, shift, 'v')

    if config.color_space:
        lf3d = st2d.changeColorSpace(lf3d, config.color_space)

    if config.prefilter > 0:
        if config.prefilter == PREFILTER.IMGD:
            lf3d = st2d.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='v')
        if config.prefilter == PREFILTER.EPID:
            lf3d = st2d.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='v')
        if config.prefilter == PREFILTER.IMGD2:
            lf3d = st2d.preImgLaplace(lf3d, scale=config.prefilter_scale)
        if config.prefilter == PREFILTER.EPID2:
            lf3d = st2d.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='v')

    st3d = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.outer_scale, direction='v')
    if config.double_tensor > 0.0:
        tmp = st2d.structureTensor2D(lf3d, inner_scale=config.inner_scale, outer_scale=config.double_tensor, direction='v')
        st3d[:] += tmp[:]
        st3d /= 2.0

    orientation_v, coherence_v = st2d.evaluateStructureTensor(st3d)
    orientation_v[:] += shift

    if config.coherence_threshold > 0.0:
        invalids = np.where(coherence_v < config.coherence_threshold)
        coherence_v[invalids] = 0.0

    if config.output_level == 3:
        misc.imsave(config.result_path+config.result_label+"orientation_v_shift_{0}.png".format(shift), orientation_v[lf_shape[0]/2, :, :])
    if config.output_level == 3:
        misc.imsave(config.result_path+config.result_label+"coherence_v_{0}.png".format(shift), coherence_v[lf_shape[0]/2, :, :])
    print "ok"

    return orientation_v, coherence_v



#============================================================================================================
#============================================================================================================
#============================================================================================================


def structureTensor2D(config):
    if not config.result_path.endswith("/"):
        config.result_path += "/"
    if not config.result_label.endswith("/"):
        config.result_label += "/"
    if not os.path.isdir(config.result_path+config.result_label):
        os.makedirs(config.result_path+config.result_label)

    compute_h = False
    compute_v = False
    lf_shape = None
    lf3dh = None
    lf3dv = None

    print "load data...",
    try:
        if not config.path_horizontal.endswith("/"):
            config.path_horizontal += "/"
        lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
        compute_h = True
        lf_shape = lf3dh.shape
    except:
        pass

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        compute_v = True
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        if lf_shape is None:
            lf_shape = lf3dv.shape
    except:
        pass

    orientation = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    coherence = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    print "ok"

    for shift in config.global_shifts:

        threads = []

        if compute_h:
            thread = Compute(lf3dh, shift, config, direction='h')
            threads += [thread]
            thread.start()

        if compute_v:
            thread = Compute(lf3dv, shift, config, direction='v')
            threads += [thread]
            thread.start()

        orientation_h = None
        coherence_h = None
        orientation_v = None
        coherence_v = None

        for x in threads:
            x.join()
            if x.direction == 'h':
                orientation_h, coherence_h = x.get_results()
            if x.direction == 'v':
                orientation_v, coherence_v = x.get_results()




        if compute_h and compute_v:
            print "merge vertical/horizontal ...",
            orientation_tmp, coherence_tmp = st2d.mergeOrientations_wta(orientation_h, coherence_h, orientation_v, coherence_v)
            orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_tmp, coherence_tmp)

            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            print "ok"

        else:
            print "merge shifts"
            if compute_h:
                orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_h, coherence_h)
            if compute_v:
                orientation, coherence = st2d.mergeOrientations_wta(orientation, coherence, orientation_v, coherence_v)
            if config.output_level >= 2:
                plt.imsave(config.result_path+config.result_label+"orientation_merged_shift_{0}.png".format(shift), orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
                plt.imsave(config.result_path+config.result_label+"coherence_merged_shift_{0}.png".format(shift), coherence[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
            print "ok"

    invalids = np.where(coherence < config.coherence_threshold)
    orientation[invalids] = 0
    coherence[invalids] = 0

    mask = coherence[lf_shape[0]/2, :, :]

    if config.output_level >= 2:
        plt.imsave(config.result_path+config.result_label+"orientation_final.png", orientation[lf_shape[0]/2, :, :], cmap=plt.cm.jet)
        plt.imsave(config.result_path+config.result_label+"coherence_final.png", mask, cmap=plt.cm.jet)

    depth = dtc.disparity_to_depth(orientation[lf_shape[0]/2, :, :], config.base_line, config.focal_length, config.min_depth, config.max_depth)

    if isinstance(config.nonlinear_diffusion, type([])):
        print "apply nonlinear diffusion", config.nonlinear_diffusion[0], ",", config.nonlinear_diffusion[1],
        vigra.filters.nonlinearDiffusion(depth, config.nonlinear_diffusion[0], config.nonlinear_diffusion[1])
        print "ok"
    if config.selective_gaussian > 0:
        print "apply masked gauss...",
        gauss = vigra.filters.Kernel2D()
        vigra.filters.Kernel2D.initGaussian(gauss, config.selective_gaussian)
        gauss.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_CLIP)
        depth = vigra.filters.normalizedConvolveImage(depth, mask, gauss)
        print "ok"
    if config.median > 0:
        print "apply median filter ...",
        depth = median_filter(depth, config.median)
        print "ok"

    invalids = np.where(mask == 0)
    depth[invalids] = 0

    if config.output_level >= 1:
        plt.imsave(config.result_path+config.result_label+"depth_final.png", depth, cmap=plt.cm.jet)

    if config.output_level >= 1:
        if isinstance(config.centerview_path, str):
            color = misc.imread(config.centerview_path)
            if isinstance(config.roi, type({})):
                sposx = config.roi["pos"][0]
                eposx = config.roi["pos"][0] + config.roi["size"][0]
                sposy = config.roi["pos"][1]
                eposy = config.roi["pos"][1] + config.roi["size"][1]
                color = color[sposx:eposx, sposy:eposy, 0:3]

        tmp = np.zeros((lf_shape[1], lf_shape[2], 4), dtype=np.float32)
        tmp[:, :, 0] = orientation[lf_shape[0]/2, :, :]
        tmp[:, :, 1] = coherence[lf_shape[0]/2, :, :]
        tmp[:, :, 2] = depth[:]
        vim = vigra.RGBImage(tmp)
        vim.writeImage(config.result_path+config.result_label+"final.exr")

        print "make pointcloud...",
        if isinstance(color, np.ndarray):
            dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, color=color, focal_length=config.focal_length)
        else:
            dtc.save_pointcloud(config.result_path+config.result_label+"pointcloud.ply", depth_map=depth, focal_length=config.focal_length)

        print "ok"
