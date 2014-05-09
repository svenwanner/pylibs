# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
import scipy.misc as misc


def load3D(path,rgb=False,roi=None):
    fnames = []
    for f in glob(path+"*.png"):
        fnames.append(f)
    fnames.sort()
    
    sposx = 0;eposx = 0;sposy = 0;eposy = 0
    
    im = misc.imread(fnames[0])
    if len(im.shape)==2:
        rgb = False
    lf = None    
    
    if roi is not None:
        sposx = roi["pos"][0]
        eposx = roi["pos"][0]+roi["size"][0]
        sposy = roi["pos"][1]
        eposy = roi["pos"][1]+roi["size"][1]
        if rgb:
            lf = np.zeros((len(fnames),roi["size"][0],roi["size"][1],3),dtype=np.float32)
        else:
            lf = np.zeros((len(fnames),roi["size"][0],roi["size"][1],1),dtype=np.float32)
            
    else: 
        if rgb:
            lf = np.zeros((len(fnames),im.shape[0],im.shape[1],3),dtype=np.float32)
        else:
            lf = np.zeros((len(fnames),im.shape[0],im.shape[1],1),dtype=np.float32)
  
    if roi is None: 
        if rgb:            
            lf[0,:,:,:] = im[:,:,0:3]
        else:
            lf[0,:,:,0] = 0.3*im[:,:,0]+0.59*im[:,:,1]+0.11*im[:,:,2]
    else: 
        if rgb:  
            lf[0,:,:,0:3] = im[sposx:eposx,sposy:eposy,0:3]
        else:
            lf[0,:,:,0] = im[sposx:eposx,sposy:eposy]
            
    for n in range(1,len(fnames)):
        im = misc.imread(fnames[n])
        if rgb:
            if roi is None: 
                lf[0,:,:,:] = im[:,:,0:3]
            else: 
                lf[0,:,:,:] = im[sposx:eposx,sposy:eposy,0:3]
        else: 
            if roi is None: 
                lf[n,:,:,0] = im[:,:]
            else: 
                lf[n,:,:,0] = im[sposx:eposx,sposy:eposy]
                
    amax = np.amax(lf)
    if amax >= 1:
        lf[:]/=255
        
    return lf