import numpy as np
import pylab as plt
from glob import glob
import scipy.misc as misc



def epiViewer(lf3d, position, direction, cmap="gray"):

    """
    simple epi viewer

    :param lf3d: ndarray 3d light field of shape [cams,y,x,channels=[1]/[3]]
    :param position:  int fixed spatial position to extract the epi
    :param direction: int direction or type of 3d lightfield 'h' or 'v'
    :param cmap: string defining the colormap [default "gray"] "gray","hot","jet"
    """
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(position, int)
    assert isinstance(direction, type(''))
    assert isinstance(cmap, str)

    if direction == 'h':
        epi = lf3d[:, position, :, 0:lf3d.shape[3]]
    if direction == 'v':
        epi = lf3d[:, :, position, 0:lf3d.shape[3]]

    if lf3d.shape[3] == 3:
        plt.show(epi)
    elif lf3d.shape[3] == 1:
        if cmap == "gray":
            plt.show(epi, cmap=plt.cm.gray)
        elif cmap == "hot":
            plt.show(epi, cmap=plt.cm.hot)
        else:
            plt.show(epi, cmap=plt.cm.jet)
    else:
        assert False, "unsupported lf shape!"

    plt.show()


def load_3d(path, rgb=False, roi=None):
    """
    load a 3d light field from filesequence. The images need to be in .png, .tif or .jpg

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