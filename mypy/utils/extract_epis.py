import glob
import inspect, os
import numpy as np
import scipy.misc as misc

context = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

epi_gap = 10
refocus = 7

for d in ['h','v']:
    fnames = []
    for f in glob.glob(context+"/"+d+"/*.tif"):
        fnames.append(f)
    fnames.sort()
    
    im = misc.imread(fnames[0])
    lf = np.zeros((len(fnames),im.shape[0],im.shape[1],3),dtype=np.uint8)    
    
    for n,f in enumerate(fnames):
        im = misc.imread(f)
        lf[n,:,:,:] = im[:,:,0:3]

    if refocus is not None:
        for n in range(lf.shape[0]):
            for c in range(3):
                if d == 'h':
                    tmp = lf[n,:,:,c]
                    lf[n,:,:,c] = np.roll(tmp,refocus*(n-lf.shape[0]/2),1)
                else:
                    tmp = lf[n,:,:,c]
                    lf[n,:,:,c] = np.roll(tmp,refocus*(n-lf.shape[0]/2),0)
        
    if d == 'h':
        for y in range(lf.shape[1]):
            if y%epi_gap == 0:
                misc.imsave(context+"/epis_h/%4.4i.png"%y,lf[:,y,:,:])
    else:
        for x in range(lf.shape[2]):
            if x%epi_gap == 0:
                misc.imsave(context+"/epis_v/%4.4i.png"%x,lf[:,:,x,:])
        
