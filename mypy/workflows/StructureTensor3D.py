from mypy.lightfield import io as lfio
from mypy.visualization.imshow import imshow
from mypy.lightfield import helpers as lfhelpers
from mypy.lightfield.depth import structureTensor3D as st3d
        

def compute( path = "/home/swanner/Desktop/TestLF/render/9x1/imgs_bw/",
            results = "/home/swanner/Desktop/TestLF/3DTensor/results",
            name = "pearplanepyr",
            roi = {"size":[220,320],"pos":[250,320]},
            inner_scale = 0.6,
            outer_scale = 1.3,
            focus = 5 ):
    
    lf = lfio.load3D(path,roi=roi)
    lf = lfhelpers.refocus3D(lf,focus)
    
    evals,evecs = st3d.structureTensor3D(lf[:,:,:,0],inner_scale,outer_scale)
    disparity = st3d.structureTensor3DConditioner(evals,evecs)
    imshow(disparity,cmap="jet")    