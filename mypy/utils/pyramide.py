import os
import numpy as np

import vigra
from mypy.lightfield import io as lfio

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

    img = lf3dh[0,:,:,:]

    print("Shape of one image: ")
    print(img.shape)
    print(img.dtype)

    # img[:] = img[:] * 255

    # img = img.astype(dtype=np.uint8)

    img = vigra.Image(img)


    print("shape if input image")
    print(img.shape)

    pyr = vigra.sampling.ImagePyramid(img,0,0,2)
    pyr.reduce(0,1)
    print("shape of reduced image")
    print(pyr[0].shape)
    print(pyr[1].shape)


    img = img[0:img.shape[0]-1,0:img.shape[1]]
    pyr2 = vigra.sampling.ImagePyramid(img,0,0,2)
    pyr2.reduce(0,1)
    print("shape of reduced image")
    print(pyr2[0].shape)
    print(pyr2[1].shape)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(pyr[1])
    fig, ax = plt.subplots()
    ax.imshow(pyr2[1])

    img1 = pyr[1]
    img2 = pyr2[1]


    diff = img1[:] - img2[:]


    fig, ax = plt.subplots()
    ax.imshow(diff)
    plt.show()



    return