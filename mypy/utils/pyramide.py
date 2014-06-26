import os


import matplotlib.pyplot as plt

import types
import numpy as np
import pylab as plt
import OpenEXR, Imath
from glob import glob
from scipy.ndimage import convolve
import matplotlib.cm as cm

import vigra
from mypy.lightfield import io as lfio



HSCHARR_WEIGHTS = np.array([[ 3,  10,  3],
                            [ 0,   0,  0],
                            [-3, -10, -3]]) / 32.0
VSCHARR_WEIGHTS = HSCHARR_WEIGHTS.T

def hscharr(image, mask=None):

    result = convolve(image, HSCHARR_WEIGHTS)
    return result

def vscharr(image, mask=None):

    result = convolve(image, VSCHARR_WEIGHTS)
    return result


def pyramide_downscaling(config):
    print("Pyramide")

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
        lf_shape = lf3dh.shape
        print("horizontal data loaded successfully")
        compute_h = True
    except:
        print("no horizontal data available")

    try:
        if not config.path_vertical.endswith("/"):
            config.path_vertical += "/"
        lf3dv = lfio.load_3d(config.path_vertical, rgb=config.rgb, roi=config.roi)
        if lf_shape is None:
            lf_shape = lf3dv.shape
        print("vertical data loaded successfully")
        compute_v = True
    except:
        print("no vertical data available")

    orientation = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
    coherence = np.zeros((lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)

    print "ok"

    img = lf3dh[0,:,:,1]

    # print("Shape of one image: ")
    # print(img.shape)
    # print(img.dtype)
    #
    # # img[:] = img[:] * 255
    #
    # # img = img.astype(dtype=np.uint8)
    # img = img[0:img.shape[0],0:img.shape[1]]
    # img = vigra.Image(img)
    #
    #
    # print("shape if input image")
    # print(img.shape)
    #
    # pyr = vigra.sampling.ImagePyramid(img,0,0,2)
    # pyr.reduce(0,1)
    # print("shape of reduced image")
    # print(pyr[0].shape)
    # print(pyr[1].shape)
    #
    # pyr2 = vigra.sampling.ImagePyramid(pyr[1],1,0,1)
    # pyr2.expand(1,0)
    # print("shape of reduced image")
    # print(pyr2[0].shape)
    # print(pyr2[1].shape)
    #

    # grad = np.ndarray([img.shape[0],img.shape[1],2],dtype=np.float32)
    grad = vigra.filters.gaussianGradient(img,0.6)

    # tmp_a = vscharr(img)  ## in direction of first dimension
    # tmp_b = hscharr(img)  ## in direction of second dimension
    #
    # grad[:,:,1] = tmp_a[:,:]
    # grad[:,:,0] = tmp_b[:,:]


    print(grad.shape)
    tensor = vigra.filters.vectorToTensor(grad)
    print(tensor.shape)
    tensor_glass = vigra.filters.hourGlassFilter2D(tensor, sigma = 1.3, rho = 0.2)

    tensor_gauss = tensor
    tensor_gauss[:, :, 0] = vigra.filters.gaussianSmoothing(tensor[:,:,0],sigma = 1.3)
    tensor_gauss[:, :, 1] = vigra.filters.gaussianSmoothing(tensor[:,:,1],sigma = 1.3)
    tensor_gauss[:, :, 2] = vigra.filters.gaussianSmoothing(tensor[:,:,2],sigma = 1.3)

    orientation1 = 1/2.0*vigra.numpy.arctan2(2*tensor_gauss[:, :, 1], tensor_gauss[:, :, 2]-tensor_gauss[:, :, 0])
    orientation2 = 1/2.0*vigra.numpy.arctan2(2*tensor_glass[:, :, 1], tensor_glass[:, :, 2]-tensor_glass[:, :, 0])

    out =[]

    out.append(tensor_gauss[:, :, 0])
    out.append(tensor_gauss[:, :, 1])
    out.append(tensor_gauss[:, :, 2])

    show(out)

    out =[]

    out.append(tensor_glass[:, :, 0])
    out.append(tensor_glass[:, :, 1])
    out.append(tensor_glass[:, :, 2])

    show(out)

    out =[]

    out.append(orientation1[:, :])
    out.append(orientation2[:, :])

    show(out)

    plt.show()

    #


    #
    # img1 = pyr[1]
    # img2 = pyr2[1]
    #
    #
    # diff = img1[:] - img2[:]





    return


def filterStringList(inputlist,filters=[],filter_sign=1):
  """
  @author Sven Wanner
  @brief filters a list of strings for keywords, filter_sign can be used
  to decide wheter the filter words are hold or deleted
  @param inputlist [list[str]] string list to filter
  @param filters [list[str]] filter words DEFAULT:[]
  @param filter_sign [int] 1/-1 positiv/negativ filtering DEFAULT:1
  @return list[str]
  """
  assert type(inputlist) is type([]), "Input Error"
  assert type(filters) is type([]), "Input Error"
  assert filter_sign == 1 or filter_sign == -1, "Input Error"
  tmp = inputlist
  for f in filters:
    if filter_sign < 0:
      tmp = filter(lambda x: x.find(f) == -1, tmp)
    else:
      tmp = filter(lambda x: x.find(f) != -1, tmp)
  return tmp

def loadOpenEXR(filename):
  """
  @author Sven Wanner
  @brief loads OpenEXR files from filename.
  @param filename [str] filename of image files
  return returns ndarray(y,x)
  """
  assert type(filename) is type(""), "Input Error"
  pt = Imath.PixelType(Imath.PixelType.FLOAT)
  golden = OpenEXR.InputFile(filename)
  dw = golden.header()['dataWindow']
  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
  cstr = golden.channel('R', pt)
  c = np.fromstring(cstr, dtype = np.float32)
  c.shape = (size[1], size[0])
  return c

def readFileSequence(files_loc,str_filter=[],filter_sign=1):
  """
  @author Sven Wanner
  @brief reads filenames in directory and filters by keyword strings,
  filter_sign can be used to decide wheter the filter words are hold or deleted
  @param files_loc [str] path of directory to read files
  @param str_filter [list[str]] filter words DEFAULT:[]
  @param filter_sign [int] 1/-1 positiv/negativ filtering DEFAULT:1
  @return list[str]
  """
  assert type(files_loc) is type(""), "Input Error"
  assert type(str_filter) is type([]), "Input Error"
  assert filter_sign == 1 or filter_sign == -1, "Input Error"
  candidates = glob(files_loc+"/*")
  if len(candidates) == 0: return []
  candidates.sort()
  candidates = filterStringList(candidates,str_filter,filter_sign)
  return candidates


def show(imgs,labels=[""],interpolation="nearest",cmap="gray",show=True,transpose=False):
    """
    2D image viewer: arg1 image<ndarray> or list of them, arg2 labels<String> or list of them, arg3 interpolation (default:'nearest')
    """
    if transpose:
      for i in range(len(imgs)):
        imgs[i] = np.transpose(imgs[i])


    if cmap=="gray":
        cmap = cm.gray
    elif cmap=="jet":
        cmap=cm.jet
    elif cmap=="hot":
        cmap=cm.hot
    elif cmap=="autumn":
        cmap=cm.autumn

    fig = plt.figure()

    if str(type(imgs))=="<type 'numpy.ndarray'>":
        label = ""
        if type(labels)==types.ListType: label = labels[0]
        elif type(labels)==types.StringType: label = labels

        ax = fig.add_subplot(111)
        ax.set_title(label)
        ax.set_xlabel("("+str(np.amin(imgs))+","+str(np.amax(imgs))+") mean="+str(np.mean(imgs)) )
        ax.imshow(imgs,interpolation=interpolation,cmap=cmap)

    elif type(imgs)==types.ListType and len(imgs) < 5:
        if len(labels)<len(imgs):
            diff=len(imgs)-len(labels)
            for i in range(diff):
                labels.append("")

        n=111
        if len(imgs) == 2:
            n+=10
        if len(imgs) > 2:
            n+=110

        for i in range(len(imgs)):
            ax = fig.add_subplot(n)
            ax.set_title(labels[i])
            ax.set_xlabel("("+str(np.amin(imgs[i]))+","+str(np.amax(imgs[i]))+") mean="+str(np.mean(imgs[i])) )
            ax.imshow(imgs[i],interpolation=interpolation,cmap=cmap)
            n+=1
    else: print "Wrong parameter type, need ndarray or list of them"

    # if show: plt.show()
