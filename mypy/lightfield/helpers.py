import sys
import numpy as np


def refocus3D(lf,focus,lftype='h'):
    tmp=np.copy(lf)
    if lftype=='h':
        for h in range(lf.shape[0]):
            for c in range(lf.shape[3]):
                lf[h,:,:,c] = np.roll(tmp[h,:,:,c],shift=(h-lf.shape[0]/2)*focus,axis=1)
    elif lftype=='v':
        for v in range(lf.shape[0]):
            for c in range(lf.shape[3]):
                lf[v,:,:,c] = np.roll(tmp[v,:,:,c],shift=(v-lf.shape[0]/2)*focus,axis=0)
    else:
        print "refocus undefined"
        sys.exit()
    return lf