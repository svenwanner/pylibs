import vigra
import sys, os
import logging
import numpy as np
from glob import glob
from scipy.ndimage import shift
from scipy.misc import imread

import skimage.color as color

def changeColorspace(lf3d, cspace="luv"):
    assert lf3d.shape[3] == 3, "only 3 channel light fields can be changed in colorspace"

    if lf3d.dtype == np.uint8:
        lf3d = lf3d.astype(np.float32)
    if np.amax(lf3d) > 1.0:
        lf3d[:] /= 255.0

    if cspace == "hsv":
        for i in range(lf3d.shape[0]):
            lf3d[i, :, :, :] = color.rgb2hsv(lf3d[i, :, :, :])
    elif cspace == "luv":
        for i in range(lf3d.shape[0]):
            #lf3d[i, :, :, :] = color.rgb2luv(lf3d[i, :, :, :])
            lf3d[i, :, :, :] = vigra.colors.transform_RGB2Luv(lf3d[i, :, :, :])
    elif cspace == "lab":
        for i in range(lf3d.shape[0]):
            #lf3d[i, :, :, :] = color.rgb2lab(lf3d[i, :, :, :])
            lf3d[i, :, :, :] = vigra.colors.transform_RGB2Lab(lf3d[i, :, :, :])
    return lf3d


def getFilenames(fpath, index=0, amount=-1, ftype="png"):
    """
    Load a filename list from path, start index and amount of filenames to load
    as well as the filetype can be specified. By default a list of all filenames
    of file type png are read.

    :rtype : object
    :param fpath: str location of image files
    :param index: int list index of filenames
    :param amount: int number of filenames to load
    :param ftype: str "png","tif","ppm","bmp","jpg"
    :return: list of filenames
    """
    assert isinstance(fpath, str)
    assert isinstance(index, int)
    assert isinstance(amount, int)

    print "\n<-- getFilenames..."

    if not fpath.endswith(os.path.sep):
        fpath += os.path.sep
    if not ftype.startswith("."):
        ftype = "."+ftype
    fnames = []
    for f in glob(fpath+"*"+ftype):
        fnames.append(f)
    fnames.sort()

    last_index = index + amount
    if amount == -1 or index + amount >= len(fnames):
        last_index = len(fnames)

    print "load filenames from path:", fpath, "..."
    print "will keep images from index:", index, "to", index+amount-1, "..."

    fnames = fnames[index:last_index]

    print "return list of", len(fnames), "images", "..."
    print "done -->"
    return fnames


def loadSequence(fnames, dtype=np.float32):
    """
    Load a filename list of image names and returns the image data as numpy array.
    If dtype is np.float32 the range is normed to [0,1], gray value images are stored
    in a 3d array, but from type [sy,sx,1] instead of [sy,sx,3] in case of rgb data.

    :param fnames: list of image file names
    :param dtype: numpy.dtype
    :return: ndarray of range [numOfImgs, sy, sx, channels]
    """
    assert isinstance(fnames, type([]))
    assert len(fnames) > 0, "empty filename list cannot be loaded!"
    assert isinstance(fnames[0], str)
    assert isinstance(dtype, type(np.float32))

    print "\n<-- loadSequence...",

    im = imread(fnames[0])
    channels = 1
    if len(im.shape) == 3:
        channels = 3
        if im.shape[2] == 4:
            im = im[:, :, 0:3]

    sequence = np.zeros((len(fnames), im.shape[0], im.shape[1], channels), dtype=dtype)
    sequence[0, :, :, 0:channels] = im[:]

    for i in range(1, len(fnames)):
        im = imread(fnames[i])
        if len(im.shape) == 3:
            if im.shape[2] == 4:
                im = im[:, :, 0:3]
        sequence[i, :, :, 0:channels] = im[:]

    if dtype == np.float32:
        if np.amax(sequence > 1.0):
            sequence[:] /= 255.0

    if len(fnames) == 0:
        print "failed!"
        sys.exit()
    else:
        print "ok"
        print "load sequene of shape:", sequence.shape, "..."
        print "range:", np.amin(sequence), np.amax(sequence), "..."
        print "done -->"
    return sequence






# ============================================================================================================
#======================              Activate Deugging modus with input argument          ====================
#=============================================================================================================
def checkDebug(argv):
    if len(argv) > 1:
        if argv[1] == 'debug':
            logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    #logging.debug('Example')
    #logging.info('Example')

    return 0




def enum(**enums):
    return type('Enum', (), enums)


# ============================================================================================================
#======================              Shift Light fied to horoptor depth                   ====================
#=============================================================================================================

def refocus_3d(lf, focus, lf_type='h'):
    """
    refocus a 3D light field by an integer pixel shift
    
    :param lf: numpy array of structure [num_of_cams,height,width,channels]
    :param focus: integer pixel value to refocus
    :param lf_type: char 'h' or 'v' to decide between horizontal and vertical light field
    :return lf: numpy array of structure [num_of_cams,height,width,channels]
    """
    assert isinstance(lf, np.ndarray)
    assert isinstance(focus, int) or isinstance(focus, float)
    assert isinstance(lf_type, type(''))

    tmp = np.copy(lf)
    if lf_type == 'h':
        for h in range(lf.shape[0]):
            for c in range(lf.shape[3]):
                if isinstance(focus, float):
                    tmp[h, :, :, c] = shift(lf[h, :, :, c], shift=[0, (h - lf.shape[0] / 2) * focus] )
                elif isinstance(focus, int):
                    tmp[h, :, :, c] = np.roll(lf[h, :, :, c], shift=(h - lf.shape[0] / 2) * focus, axis=1)
    elif lf_type == 'v':
        for v in range(lf.shape[0]):
            for c in range(lf.shape[3]):
                if isinstance(focus, float):
                    tmp[v, :, :, c] = shift(lf[v, :, :, c], shift=[(v - lf.shape[0] / 2) * focus, 0])
                elif isinstance(focus, int):
                    tmp[v, :, :, c] = np.roll(lf[v, :, :, c], shift=(v - lf.shape[0] / 2) * focus, axis=0)
    else:
        print "refocus undefined"

    return tmp



def refocus_epi(epi, focus):
    """
    refocus a 3D light field by an integer pixel shift

    :param epi: numpy array of structure [num_of_cams,width,channels]
    :param focus: integer pixel value to refocus
    :return repi: numpy array of structure [num_of_cams,width,channels]
    """
    assert isinstance(epi, np.ndarray)
    assert isinstance(focus, int) or isinstance(focus, float)

    tmp = np.copy(epi)
    for h in range(tmp.shape[0]):
        for c in range(tmp.shape[2]):
            if isinstance(focus, float):
                tmp[h, :, c] = shift(epi[h, :, c], shift=[0, (h - epi.shape[0] / 2) * focus])
            elif isinstance(focus, int):
                tmp[h, :, c] = np.roll(epi[h, :, c], shift=(h - epi.shape[0] / 2) * focus)

    return tmp