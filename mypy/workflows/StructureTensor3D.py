from mypy.lightfield import io as lfio
from mypy.visualization.imshow import imoverlay, imshow
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor3D as st3d


def compute(path, results, name, roi, inner_scale=0.6, outer_scale=1.0, focus=0):
    """
    computes disparity using the 3D structure tensor for the
    center view of a 3D light field loaded from a file sequence

    :param path: string path to load filesequence from
    :param results: string directory to store the results
    :param name: string name used for result storing
    :param roi: dict to define a region of interest {"size":[h,w],"pos":[y,x]}
    :param inner_scale: float inner scale of the structure tensor
    :param outer_scale: float outer scale of the structure tensor
    :param focus: integer pixel value to refocus
    """

    assert isinstance(path, str)
    assert isinstance(results, str)
    assert isinstance(name, str)
    assert isinstance(roi, dict)
    assert isinstance(inner_scale, float)
    assert isinstance(outer_scale, float)
    assert isinstance(focus, int)

    lf = lfio.load_3d(path, roi=roi)
    lf = lfhelpers.refocus_3d(lf, focus)

    evals, evecs = st3d.structure_tensor3d(lf[:, :, :, 0], inner_scale, outer_scale)
    disparity, coherence = st3d.structure_tensor3d_conditioner(evals, evecs)
    imshow(disparity, cmap="jet")
    imshow(coherence, cmap="hot")
    #imoverlay(lf[4, :, :, :], disparity, alpha=0.3)


compute("/home/swanner/Desktop/TestLF/render/9x1/imgs_bw/",
        "/home/swanner/Desktop/TestLF/3DTensor/results",
        "pearplanepyr",
        {"size": [128, 128], "pos": [280, 440]},
        0.6, 1.3, 4)