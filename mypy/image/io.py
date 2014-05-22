import vigra
import numpy as np
import scipy.misc as misc

def loadEXR(filename):
    return vigra.readImage(filename)[:, :, 0].transpose()

# def load(filename, roi=None, bw=True, normed=True):
#     img = misc.imread(filename)
#     dtype = np.uint8
#     if normed:
#         dtype = np.float32
#         img = img.astype(np.float32)
#         img = img[:]/255.0
#
#     if len(img.shape) == 2:
#         img = img[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1]]
#         if bw:
#             return img
#         else:
#             tmp = np.zeros((img.shape[0],img.shape[1],3))
#     else:
#         pass
#
#     if roi is not None:
#
#         if len(img.shape) > 2:
#             img = img[roi[0][0]:roi[1][0],roi[0][1]:roi[1][1],:]
#         elif len(img.shape) == 3:
#             if roi is None:
#                 img = img
#
#     else:
#         if len(img.shape) == 2:
#             if roi is None:
#                 pass
#         elif len(img.shape) == 3:
#             if roi is None:
#                 pass