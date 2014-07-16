import numpy as np
import pylab as plt
from glob import glob
import scipy.misc as misc

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


def load_lf3d_fromFiles(fpath, index=0, amount=-1, dtype=np.float32, ftype="png"):
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
    fnames = getFilenames(fpath, index, amount, ftype)
    return loadSequence(fnames, dtype)




def load_3d(path, rgb=False, roi=None):
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
        for f in glob(path + "*.tif"):
            fnames.append(f)
    fnames.sort()


    sposx = 0
    eposx = 0
    sposy = 0
    eposy = 0

    im = misc.imread(fnames[0])
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
            lf[0, :, :, :] = im[:, :, 0:3]
        else:
            lf[0, :, :, 0] = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :, 2]
    else:
        if rgb:
            lf[0, :, :, 0:3] = im[sposx:eposx, sposy:eposy, 0:3]
        else:
            lf[0, :, :, 0] = im[sposx:eposx, sposy:eposy]

    for n in range(1, len(fnames)):
        im = misc.imread(fnames[n])
        if rgb:
            if roi is None:
                lf[n, :, :, :] = im[:, :, 0:3]
            else:
                lf[n, :, :, :] = im[sposx:eposx, sposy:eposy, 0:3]
        else:
            if roi is None:
                lf[n, :, :, 0] = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :, 2]
            else:
                lf[n, :, :, 0] = im[sposx:eposx, sposy:eposy]

    amax = np.amax(lf)
    if amax >= 1:
        lf[:] /= 255

    return lf