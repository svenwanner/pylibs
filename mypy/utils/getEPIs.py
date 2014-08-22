import os
import pylab as plt

from mypy.lightfield import io as lfio
from mypy.lightfield import helpers as lfhelpers
import numpy as np
import Image
import PIL

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
        self.height = 0
        self.centerview_path = None             # path to the center view image to get color for pointcloud [optional]
        self.global_shifts = 5.0               # list of horopter shifts in pixel

        self.rgb = True                         # forces grayscale if False

        self.output_level = 2                   # level of detail for file output possible 1,2,3

#============================================================================================================
#============================================================================================================
#============================================================================================================


def getEPI(config):

    compute_h = False
    compute_v = False


    if not config.result_path.endswith("/"):
        config.result_path += "/"
    if not config.result_label.endswith("/"):
        config.result_label += "/"
    if not os.path.isdir(config.result_path+config.result_label):
        os.makedirs(config.result_path+config.result_label)

    print(config.path_horizontal)

    print "load data...",
    try:
        if not config.path_horizontal.endswith("/"):
            config.path_horizontal += "/"
        lf3dh = lfio.load_3d(config.path_horizontal, rgb=config.rgb, roi=config.roi)
        compute_h = True
        print(lf3dh.shape)
    except:
        pass

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        compute_v = True
        print(lf3dv.shape)
    except:
        pass

    if compute_h == True:

        lf3dh = lfhelpers.refocus_3d(lf3dh, float(config.global_shifts[0]), 'h')

        if config.height == 0:
            config.height = lf3dh.shape[0]

        for i in range(lf3dh.shape[1]):
            img = lf3dh[:,i,config.global_shifts[0]*lf3dh.shape[0]/2:lf3dh.shape[2]-config.global_shifts[0]*lf3dh.shape[0]/2,:]
            img[:] = img[:]*255
            img = img.astype(np.uint8)
            shape = img.shape[1]
            img = PIL.Image.fromarray(img)
            img = img.resize((shape,config.height), Image.BICUBIC)
            img = np.array(img)
            plt.imsave(config.result_path+config.result_label+"EPI_horizontal_at_{0}.tif".format(i), img)

        # for i in range(lf3dh.shape[1]):
        #     img = lf3dh[:,config.global_shifts[0]*lf3dh.shape[0]/2:lf3dh.shape[1]-config.global_shifts[0]*lf3dh.shape[0]/2,i,:]
        #     img[:] = img[:]*255
        #     img = img.astype(np.uint8)
        #     shape = img.shape[1]
        #     img = PIL.Image.fromarray(img)
        #     img = img.resize((shape,50), Image.BICUBIC)
        #     img = np.array(img)
        #     plt.imsave(config.result_path+config.result_label+"Fake_EPI_horizontal_at_{0}.tif".format(i), img, cmap=plt.cm.jet)

    if compute_v == True:

        lf3dv = lfhelpers.refocus_3d(lf3dv, float(config.global_shifts[0]), 'v')

        if config.height == 0:
            config.height = lf3dv.shape[0]

        for i in range(lf3dh.shape[1]):
            img = lf3dv[:,config.global_shifts[0]*lf3dv.shape[0]/2:lf3dv.shape[1]-config.global_shifts[0]*lf3dv.shape[0]/2,i,:]
            img[:] = img[:]*255
            img = img.astype(np.uint8)
            shape = img.shape[1]
            img = PIL.Image.fromarray(img)
            img = img.resize((shape,50), Image.BICUBIC)
            img = np.array(img)
            plt.imsave(config.result_path+config.result_label+"EPI_vertical_at_{0}.tif".format(i), img, cmap=plt.cm.jet)

        # for i in range(lf3dh.shape[2]):
        #     img = lf3dv[:,i,config.global_shifts[0]*lf3dh.shape[0]/2:lf3dh.shape[2]-config.global_shifts[0]*lf3dh.shape[0]/2,:]
        #     img[:] = img[:]*255
        #     img = img.astype(np.uint8)
        #     shape = img.shape[1]
        #     img = PIL.Image.fromarray(img)
        #     img = img.resize((shape,50), Image.BICUBIC)
        #     img = np.array(img)
        #     plt.imsave(config.result_path+config.result_label+"Fake_EPI_vertical_at_{0}.tif".format(i), img, cmap=plt.cm.jet)



    # for i in range(lf3dh.shape[0]):
    #     img = lf3dh[i,config.global_shifts[0]*lf3dv.shape[0]/2:lf3dh.shape[1]-config.global_shifts[0]*lf3dv.shape[0]/2,config.global_shifts[0]*4:lf3dh.shape[2]-config.global_shifts[0]*4,:]
    #     img[:] = img[:]*255
    #     img = img.astype(np.uint8)
    #     plt.imsave(config.result_path+config.result_label+"Img_horizontal_cropped_{0}.tif".format(i), img, cmap=plt.cm.jet)
    #     img = lf3dv[i,config.global_shifts[0]*lf3dv.shape[0]/2:lf3dh.shape[1]-config.global_shifts[0]*lf3dv.shape[0]/2,config.global_shifts[0]*4:lf3dh.shape[2]-config.global_shifts[0]*4,:]
    #     img[:] = img[:]*255
    #     img = img.astype(np.uint8)
    #     plt.imsave(config.result_path+config.result_label+"Img_vertical_cropped_{0}.tif".format(i), img, cmap=plt.cm.jet)
