import vigra
import numpy as np

import pylab as plt


def imshow(im, cmap="gray"):
    if cmap == "gray": 
        cmap = plt.cm.gray
    elif cmap == "jet": 
        cmap = plt.cm.jet
    elif cmap == "hot": 
        cmap = plt.cm.hot
    else: 
        print "cmap not available use gray as default"
        cmap = plt.cm.gray
        
    amax = np.amax(im)
    amin = np.amin(im)

    plt.imshow(im, cmap=cmap)
    plt.title("range: ("+str(amin)+","+str(amax)+")")
    plt.show()


def imoverlay(im, overlay, alpha=0.5, show=True):
    assert isinstance(im, np.ndarray)
    assert isinstance(overlay, np.ndarray)
    assert isinstance(alpha, float)
    assert isinstance(show, bool)

    assert 0.0 < alpha < 1.0
    assert im.shape[0] == overlay.shape[0]
    assert im.shape[1] == overlay.shape[1]
    assert len(im.shape) == 2 or len(im.shape) == 3
    assert len(overlay.shape) == 2 or len(overlay.shape) == 3

    img = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    if len(im.shape) == 2:
        tmp = vigra.colors.linearRangeMapping(im, newRange=(0, 255)).astype(np.uint8)
        for i in range(3):
            img[:, :, i] = tmp[:]
    else:
        for i in range(3):
            if im.shape[2] == 3:
                tmp = vigra.colors.linearRangeMapping(im[:, :, i].astype(np.float32), newRange=(0, 255)).astype(
                    np.uint8)
            else:
                tmp = vigra.colors.linearRangeMapping(im[:, :, 0], newRange=(0, 255)).astype(np.uint8)
            img[:, :, i] = tmp[:]

    olay = np.zeros((overlay.shape[0], overlay.shape[1], 3), dtype=np.uint8)
    if len(overlay.shape) == 2:
        tmp = vigra.colors.linearRangeMapping(overlay.astype(np.float32), newRange=(0, 255)).astype(np.uint8)
        olay[:, :, 0] = tmp[:]
    else:
        if overlay.shape[2] == 3:
            for i in range(3):
                tmp = vigra.colors.linearRangeMapping(overlay[:, :, i].astype(np.float32), newRange=(0, 255)).astype(
                    np.uint8)
                olay[:, :, i] = tmp[:]
        else:
            tmp = vigra.colors.linearRangeMapping(overlay[:, :, 0], newRange=(0, 255)).astype(np.uint8)
            olay[:, :, 0] = tmp[:]

    img_fac = 1.0 - alpha
    img = img[:].astype(np.float32) * img_fac + olay[:].astype(np.float32) * alpha
    out = np.zeros_like(img).astype(np.uint8)

    for i in range(3):
        out[:, :, i] = vigra.colors.linearRangeMapping(img[:, :, i], newRange=(0, 255)).astype(np.uint8)

    plt.imshow(out)
    if show:
        plt.show()
