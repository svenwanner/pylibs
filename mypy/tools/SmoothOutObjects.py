__author__ = 'mdiebold'

import sys, os
import inspect
import numpy as np
from glob import glob
from scipy.misc import imsave, imread
from mypy.visualization.imshow import imshow
from mypy.lightfield import io as lfio
import pylab as plt

context = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

inpath = context+"/imgs/"
outpath = context+"/merged/"
thresh = 0.5
roi = None
roi = {"pos":[1500, 2000],"size":[2000, 4000]}
output_level = 2

def mergeAndSave(inpath, thresh, outpath):

    # Load all images into list
    print "try to load horizontal data ...\n",
    try:
        if not inpath.endswith("/"):
            inpath += "/"
        tmp = lfio.load_3d(inpath, rgb=True, roi=roi)
        print "ok"
    except:
        print "could not load all data"

    print('Image shape of horizontal images: ' + str(tmp.shape))

### Add artificial poission distributed noise
    for i in range(tmp.shape[0]):
        imagea = tmp[i, :, :, :]
        noisy = imagea + 2 * imagea.std() * np.random.random(imagea.shape)
        noisy = np.clip(noisy, 0, 1)
        tmp[i, :, :, :] = noisy
        if output_level == 2:
            plt.imsave(outpath+"noisy input_%4.4i.png" %i, noisy)

    median_img = np.median(tmp, axis=0)

    if output_level == 2:
        imsave(outpath+"median_image.png", median_img)

    mask = np.ones((tmp.shape[0],tmp.shape[1],tmp.shape[2],tmp.shape[3]),dtype=np.float32)

    for i in range(tmp.shape[0]):
        tmp_img = tmp[i, :, :, :]
        mask_img = mask[i, :, :, :]
        mask_up = np.where(tmp_img[:] > median_img[:]*(1.0+thresh))
        mask_img[mask_up] = 0.0
        mask_down = np.where(tmp_img[:] < median_img[:]*(1.0-thresh))
        mask_img[mask_down] = 0.0
        mask[i, :, :, :] = mask_img[:]

        if output_level == 2:
            plt.imsave(outpath+"mask_%4.4i.png" %i, mask[i, :, :, :])

        average_image = np.average(tmp[:], axis=0, weights=mask[:])

    plt.imsave(outpath+"averaged.png", average_image)


if __name__ == "__main__":

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    mergeAndSave(inpath, thresh, outpath)
