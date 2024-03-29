import numpy as np
import pylab as plt
from glob import glob
import scipy.misc as misc
import vigra

from mypy.image.io import loadEXR
from mypy.visualization.imshow import imshow

from mypy.lightfield.helpers import refocus_3d
from mypy.lightfield.helpers import getFilenames, loadSequence

def finalResultViewer(final_exr, save_to=None):

    try:
        final_res = loadEXR(final_exr)
        for i in range(3):
            imshow(final_res[:, :, i])

        if isinstance(save_to, str):
            for i in range(3):
                misc.imsave(save_to, final_res[:, :, i])
    except:
        print "Error reading final.exr!"


def load_lf3d_fromFiles(fpath, index=0, amount=-1, dtype=np.float32, ftype="png", switchOrder=False):
    """
    Load a filename list from path, start index and amount of filenames to load
    as well as the filetype can be specified. By default a list of all filenames
    of file type png are read. If focus is set the returned 3d light field is refocused.

    :rtype : object
    :param fpath: str location of image files
    :param index: int list index of filenames
    :param amount: int number of filenames to load
    :param focus: int or float refocus step in pixel
    :param dtype: numpy.dtype
    :param ftype: str "png","tif","ppm","bmp","jpg"
    :return: ndarray of range [numOfImgs, sy, sx, channels]
    """
    fnames = getFilenames(fpath, index, amount, ftype, switchOrder)
    return loadSequence(fnames, dtype)

def load_4d(path, horizontalCameras, verticalCameras, config, rgb=True, roi=None, switchOrder=False):
    """
    load a 3d light field from filesequence. The images need to be in .png, .tif or .jpg

    :rtype : light field 3D volume
    :param path: string path to load filesequence from
    :param rgb: bool to define number of channels in light field returned
    :param roi: dict to define a region of interest {"size":[h,w],"pos":[y,x]}
    :return lf: lf numpy array of structure [num_of_cams,height,width,channels]
    """
    assert isinstance(path, str)
    assert isinstance(rgb, bool)
    if roi is not None:
        assert isinstance(roi, dict)

    fnames = []
    for f in glob(path + "*.png"):
        fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.jpg"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.JPG"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.tif"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.TIF"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.exr"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.ppm"):
            fnames.append(f)

    fnames.sort()

    for i in fnames:
        print(i)

    if switchOrder:
        fnames.reverse()

    im = vigra.readImage(fnames[0], order='C')

    if len(im.shape) == 2:
        rgb = False

    lf = np.zeros((verticalCameras, horizontalCameras, im.shape[0], im.shape[1], 3), dtype=np.float32)

    for n in range(0, len(fnames)):
        im = vigra.readImage(fnames[n], order='C')
        a = (horizontalCameras-1) - n % horizontalCameras
        b = int(n / horizontalCameras)
        lf[b,a, :, :, :] = im[:,:,0:3]

    amax = np.amax(lf)
    if amax >= 1:
        lf[:] /= 255

    # for v in range(lf.shape[0]):
    #     for h in range(lf.shape[1]):
    #         plt.imsave(config.result_path+config.result_label+"image_{0}.png".format(h+9*v), lf[v,h, :, :, :])


    return lf


def load_3d(path, rgb=True, roi=None, switchOrder=False):
    """
    load a 3d light field from filesequence. The images need to be in .png, .tif or .jpg

    :rtype : light field 3D volume
    :param path: string path to load filesequence from
    :param rgb: bool to define number of channels in light field returned
    :param roi: dict to define a region of interest {"size":[h,w],"pos":[y,x]}
    :return lf: lf numpy array of structure [num_of_cams,height,width,channels]
    """
    assert isinstance(path, str)
    assert isinstance(rgb, bool)
    if roi is not None:
        assert isinstance(roi, dict)

    fnames = []
    for f in glob(path + "*.png"):
        fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.jpg"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.JPG"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.tif"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.TIF"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.exr"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(path + "*.ppm"):
            fnames.append(f)

    fnames.sort()

    for i in fnames:
        print(i)

    if switchOrder:
        fnames.reverse()

    sposx = 0
    eposx = 0
    sposy = 0
    eposy = 0

    im = vigra.readImage(fnames[0],order='C')
    # im = misc.imread(fnames[0])
    if len(im.shape) == 2:
        rgb = False

    if roi is not None:
        sposx = roi["pos"][0]
        eposx = roi["pos"][0] + roi["size"][0]
        sposy = roi["pos"][1]
        eposy = roi["pos"][1] + roi["size"][1]
        if rgb:
            lf = np.zeros((len(fnames), roi["size"][0], roi["size"][1], 3), dtype=np.float32)
        else:
            lf = np.zeros((len(fnames), roi["size"][0], roi["size"][1], 1), dtype=np.float32)

    else:
        if rgb:
            lf = np.zeros((len(fnames), im.shape[0], im.shape[1], 3), dtype=np.float32)
        else:
            lf = np.zeros((len(fnames), im.shape[0], im.shape[1], 1), dtype=np.float32)

    if roi is None:
        if rgb:
            if len(im.shape) == 3:
                lf[0, :, :, :] = im[:, :, 0:3]
            else:
                for c in range(3):
                    lf[0, :, :, c] = im[:]
        else:
            if len(im.shape) == 3:
                lf[0, :, :, 0] = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :, 2]
            else:
                lf[0, :, :, 0] == im[:]
    else:
        if rgb:
            if len(im.shape) == 3:
                lf[0, :, :, 0:3] = im[sposx:eposx, sposy:eposy, 0:3]
            else:
                for c in range(3):
                    lf[0, :, :, c] = im[sposx:eposx, sposy:eposy]
        else:
            if len(im.shape) == 3:
                lf[0, :, :, 0] = 0.3 * im[sposx:eposx, sposy:eposy, 0] + 0.59 * im[sposx:eposx, sposy:eposy, 1] + 0.11 * im[sposx:eposx, sposy:eposy, 2]
            else:
                lf[0, :, :, 0] = im[sposx:eposx, sposy:eposy]

    for n in range(1, len(fnames)):
        # im = misc.imread(fnames[n])
        im = vigra.readImage(fnames[n], order='C')
        if rgb:
            if roi is None:
                if len(im.shape) == 3:
                    lf[n, :, :, :] = im[:, :, 0:3]
                else:
                    for c in range(3):
                        lf[n, :, :, c] = im[:]
            else:
                if len(im.shape) == 3:
                    lf[n, :, :, :] = im[sposx:eposx, sposy:eposy, 0:3]
                else:
                    for c in range(3):
                        lf[n, :, :, c] = im[sposx:eposx, sposy:eposy]
        else:
            if roi is None:
                if len(im.shape) == 3:
                    lf[n, :, :, 0] = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :, 2]
                else:
                    lf[n, :, :, 0] = im[:]
            else:
                if len(im.shape) == 3:
                    lf[n, :, :, 0] = 0.3 * im[sposx:eposx, sposy:eposy, 0] + 0.59 * im[sposx:eposx, sposy:eposy, 1] + 0.11 * im[sposx:eposx, sposy:eposy, 2]
                else:
                    lf[n, :, :, 0] = im[sposx:eposx, sposy:eposy]

    amax = np.amax(lf)
    if amax >= 1:
        lf[:] /= 255

    return lf



def readEpi_fromSequence(fpath, position=0, direction='h'):
    """
    load a 3d light field from filesequence. The images need to be in .png, .tif or .jpg

    :rtype : light field 3D volume
    :param path: string path to load filesequence from
    :param rgb: bool to define number of channels in light field returned
    :param roi: dict to define a region of interest {"size":[h,w],"pos":[y,x]}
    :return lf: lf numpy array of structure [num_of_cams,height,width,channels]
    """
    assert isinstance(fpath, str)

    fnames = []
    for f in glob(fpath + "*.png"):
        fnames.append(f)
    if len(fnames) == 0:
        for f in glob(fpath + "*.jpg"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(fpath + "*.bmp"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(fpath + "*.ppm"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(fpath + "*.tif"):
            fnames.append(f)
    if len(fnames) == 0:
        for f in glob(fpath + "*.bmp"):
            fnames.append(f)
    fnames.sort()

    im = misc.imread(fnames[0])
    channels = 1
    if len(im.shape) == 3:
        channels = 3

    if direction == 'h':
        epi = np.zeros((len(fnames), im.shape[1], channels))
    if direction == 'v':
        epi = np.zeros((len(fnames), im.shape[0], channels))

    for n,f in enumerate(fnames):
        im = misc.imread(fnames[n])
        if direction == 'h':
            if len(im.shape) == 3:
                epi[n, :, 0:3] = im[position, :, 0:3]
            else:
                epi[n, :, 0] = im[position, :]
        if direction == 'v':
           if len(im.shape) == 3:
               epi[n, :, 0:3] = im[ :, position, 0:3]
           else:
               epi[n, :, 0] = im[:, position]

    return epi[:, :, 0:channels]
