import threading
import numpy as np
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor2D as st2d
from mypy.lightfield.depth.prefilter import PREFILTER
import mypy.lightfield.depth.prefilter as prefilter
import scipy.misc as misc
import pylab as plt



########################################################################################################################
##################################  Computation Class for the horizontal and vertical direction
########################################################################################################################


class Compute(threading.Thread):
    lock = threading.Lock()

    def __init__(self, lf3d, shift, config, direction):
        """
        :param lf3d:
        :param shift:
        :param config:
        :param direction:
        """
        threading.Thread.__init__(self)
        self.lf3d = lf3d
        self.shift = shift
        self.config = config
        self.direction = direction
        self.orientation = None
        self.coherence = None

    def run(self):
        print('Compute: ' + str(self.direction))
        if self.direction == 'h':
            self.orientation, self.coherence = compute_horizontal(self.lf3d, self.shift, self.config)
        if self.direction == 'v':
           self.orientation, self.coherence = compute_vertical(self.lf3d, self.shift, self.config)

    def get_results(self):
        return self.orientation, self.coherence





########################################################################################################################
##################################  Compute the horizontal light field structure tensor
########################################################################################################################

def compute_horizontal(lf3dh, shift, config):

    print("compute horizontal shift {0}".format(shift))

    ### initialize Pointers to result arrays ###
    orientation_h = None
    coherence_h = None

    ### shift images to horoptor ###
    lf3d = lfhelpers.refocus_3d(lf3dh, shift, 'h')


    # ### Output to evaluate shift (only for pixel shift)
    # if config.output_level > 3:
    #
    #     print("Control if shift is made correctly, comparison between shifted image and original image")
    #     #print(lf3d.shape[0]/2+1)
    #     for t in range(lf3d.shape[0]/2+1):
    #         print(t)
    #         pos = t
    #         tmp_A = lf3dh[lf3d.shape[0]/2-pos, :, shift*pos:lf3d.shape[2], :]
    #         plt.imsave(config.result_path+config.result_label+"tmpA{0}.png".format(str(lf3d.shape[0]/2-t).zfill(2)), tmp_A)
    #         #print(tmp_A.shape)
    #         tmp_B = lf3d[lf3d.shape[0]/2-pos, :, 0:lf3d.shape[2]-shift*pos, :]
    #         plt.imsave(config.result_path+config.result_label+"tmpB{0}.png".format(str(lf3d.shape[0]/2-t).zfill(2)), tmp_B)
    #         #print(tmp_B.shape)
    #         tmp = np.subtract(tmp_A, tmp_B)
    #         plt.imsave(config.result_path+config.result_label+"Subtract.png", tmp)
    #         S = np.sum(tmp)
    #         print(S)
    #
    #     for t in range(lf3d.shape[0]/2+1):
    #         print(-t)
    #         pos = -t
    #         tmp_A = lf3dh[lf3d.shape[0]/2-pos, :, 0:lf3d.shape[2]+shift*pos, :]
    #         plt.imsave(config.result_path+config.result_label+"tmpA{0}.png".format(str(t+lf3d.shape[0]/2).zfill(2)), tmp_A)
    #         #print(tmp_A.shape)
    #         tmp_B = lf3d[lf3d.shape[0]/2-pos, :, -shift*pos:lf3d.shape[2], :]
    #         plt.imsave(config.result_path+config.result_label+"tmpB{0}.png".format(str(t+lf3d.shape[0]/2).zfill(2)), tmp_B)
    #         #print(tmp_B.shape)
    #         tmp = np.subtract(tmp_A, tmp_B)
    #         plt.imsave(config.result_path+config.result_label+"Subtract.png", tmp)
    #         S = np.sum(tmp)
    #         print(S)


    ### change colorspace if necessary ###
    if config.color_space:
        lf3d = st2d.changeColorSpace(lf3d, config.color_space)

    ### Prefilter of imput images (optional) ###
    if config.prefilter == PREFILTER.IMGD:
        lf3d = prefilter.preImgDerivation(lf3d, scale=config.prefilter_scale, direction='h')
    if config.prefilter == PREFILTER.EPID:
        lf3d = prefilter.preEpiDerivation(lf3d, scale=config.prefilter_scale, direction='h')
    if config.prefilter == PREFILTER.IMGD2:
        lf3d = prefilter.preImgLaplace(lf3d, scale=config.prefilter_scale)
    if config.prefilter == PREFILTER.EPID2:
        lf3d = prefilter.preEpiLaplace(lf3d, scale=config.prefilter_scale, direction='h')
    if config.prefilter == PREFILTER.SCHARR:
        lf3d = prefilter.preImgScharr(lf3d, config, direction='h')

    ### compute structure tensor ###
    structureTensor = None
    if config.structure_tensor_type == "classic":
        structureTensor = st2d.StructureTensorClassic()
    if config.structure_tensor_type == "hour-glass":
        structureTensor = st2d.StructureTensorHourGlass()
    if config.structure_tensor_type == "scharr":
        structureTensor = st2d.StructureTensorScharr()

    params = {"direction": 'h', "inner_scale": config.inner_scale, "outer_scale": config.outer_scale, "hour-glass": config.hourglass_scale}
    structureTensor.compute(lf3d, params)
    st3d = structureTensor.get_result()

    ### compute disparity map and coherence map with structure tensor components ###
    orientation_h, coherence_h = st2d.evaluateStructureTensor(st3d)
    orientation_h[:] += shift

    if config.coherence_threshold > 0.0:
        print('Apply Coherence Threshold')
        invalids = np.where(coherence_h < config.coherence_threshold)
        coherence_h[invalids] = 0.0

    if config.output_level > 3:
        misc.imsave(config.result_path+config.result_label+"orientation_h_shift_{0}.png".format(shift), orientation_h[orientation_h.shape[0]/2, :, :])
    if config.output_level > 3:
        misc.imsave(config.result_path+config.result_label+"coherence_h_{0}.png".format(shift), coherence_h[coherence_h.shape[0]/2, :, :])

    return orientation_h, coherence_h



def compute_vertical(lf3dv, shift, config):
    print "compute vertical shift {0}".format(shift), "...",
    lf3d = lfhelpers.refocus_3d(lf3dv, shift, 'v')

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

    structureTensor = None
    if config.structure_tensor_type == "classic":
        structureTensor = st2d.StructureTensorClassic()
    if config.structure_tensor_type == "hour-glass":
        structureTensor = st2d.StructureTensorHourGlass()
    if config.structure_tensor_type == "scharr":
        structureTensor = st2d.StructureTensorScharr()

    params = {"direction": 'v', "inner_scale": config.inner_scale, "outer_scale": config.outer_scale, "hour-glass": config.hourglass_scale}
    structureTensor.compute(lf3d, params)
    st3d = structureTensor.get_result()

    orientation_v, coherence_v = st2d.evaluateStructureTensor(st3d)
    orientation_v[:] += shift

    if config.coherence_threshold > 0.0:
        invalids = np.where(coherence_v < config.coherence_threshold)
        coherence_v[invalids] = 0.0

    if config.output_level > 3:
        misc.imsave(config.result_path+config.result_label+"orientation_v_shift_{0}.png".format(shift), orientation_v[orientation_v[0]/2, :, :])
    if config.output_level > 3:
        misc.imsave(config.result_path+config.result_label+"coherence_v_{0}.png".format(shift), coherence_v[coherence_v[0]/2, :, :])


    return orientation_v, coherence_v



#============================================================================================================
#============================================================================================================
#============================================================================================================
