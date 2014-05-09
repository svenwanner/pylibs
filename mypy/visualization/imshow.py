# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt

def imshow(im,cmap="gray"):
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
    
    plt.imshow(im,cmap=cmap)
    plt.title("range: ("+str(amin)+","+str(amax)+")")
    plt.show()