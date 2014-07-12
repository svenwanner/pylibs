import vigra
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from mypy.lightfield import helpers as lfhelpers



def finalsViewer(filename, ranges=None, save_at=None, fontsize=8):
    # try:
    t = vigra.readImage(filename)
    disp = t[:, :, 0]
    coh = t[:, :, 1]
    depth = t[:, :, 2]

    if ranges is not None:
        assert isinstance(ranges, type({})), "ranges needs to be None or a dictionary"

        if "disp" in ranges and isinstance(ranges["disp"], type([])):
            np.place(disp, disp < ranges["disp"][0], ranges["disp"][0])
            np.place(disp, disp > ranges["disp"][1], ranges["disp"][1])
        if "coh" in ranges and isinstance(ranges["coh"], type([])):
            np.place(coh, coh < ranges["coh"][0], ranges["coh"][0])
            np.place(coh, coh > ranges["coh"][1], ranges["coh"][1])
        if "depth" in ranges and isinstance(ranges["depth"], type([])):
            np.place(depth, depth < ranges["depth"][0], ranges["depth"][0])
            np.place(depth, depth > ranges["depth"][1], ranges["depth"][1])

    plt.figure()

    sb = plt.subplot(2, 2, 1)
    plt.imshow(disp, interpolation='nearest', cmap=cm.jet)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('disparity')

    amax = np.amax(disp)
    amin = np.amin(disp)
    num_range = np.linspace(amin, amax, num=20)

    cb = plt.colorbar(ticks=num_range)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    plt.subplot(2, 2, 2)
    plt.imshow(coh, interpolation='nearest', cmap=cm.hot)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('coherence')

    amax = np.amax(coh)
    amin = np.amin(coh)
    num_range = np.linspace(amin, amax, num=20)

    cb = plt.colorbar(ticks=num_range)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(10)

    plt.subplot(2, 2, 3)
    plt.imshow(depth, interpolation='nearest', cmap=cm.jet)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('depth')

    amax = np.amax(depth)
    amin = np.amin(depth)
    num_range = np.linspace(amin, amax, num=20)

    cb = plt.colorbar(ticks=num_range)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    if save_at is None:
        plt.show()
    elif isinstance(save_at, str):
        if not save_at.endswith("/"):
            save_at += "/"
        save_at += "final.svg"
        plt.savefig(save_at)



def epishow(lf3d, position, direction, shift=0, cmap="gray"):
    """
    simple epi viewer

    :param lf3d: ndarray 3d light field of shape [cams,y,x,channels=[1]/[3]]
    :param position:  int fixed spatial position to extract the epi
    :param direction: int direction or type of 3d lightfield 'h' or 'v'
    :param shift: int refocus shift in pixel
    :param cmap: string defining the colormap [default "gray"] "gray","hot","jet"
    """
    assert isinstance(lf3d, np.ndarray)
    assert isinstance(position, int)
    assert isinstance(direction, type(''))
    assert isinstance(shift, int)
    assert isinstance(cmap, str)

    if direction == 'h':
        lf = lfhelpers.refocus_3d(lf3d, shift, lf_type='h')
        epi = lf[:, position, :, 0:lf3d.shape[3]]
    if direction == 'v':
        lf = lfhelpers.refocus_3d(lf3d, shift, lf_type='v')
        epi = lf[:, :, position, 0:lf3d.shape[3]]

    if lf3d.shape[3] == 3:
        plt.imshow(epi)
    elif lf3d.shape[3] == 1:
        if cmap == "gray":
            plt.imshow(epi, cmap=plt.cm.gray)
        elif cmap == "hot":
            plt.imshow(epi, cmap=plt.cm.hot)
        else:
            plt.imshow(epi, cmap=plt.cm.jet)
    else:
        assert False, "unsupported lf shape!"

    plt.show()


def imshow(im, cmap="gray", show=True):
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

    plt.imshow(im, cmap=cmap, interpolation='nearest')
    plt.title("range: ("+str(amin)+","+str(amax)+")")
    if show:
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
