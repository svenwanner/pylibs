# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
from glob import glob 

def getFilenamesFromDirectory(directory,file_pattern):
    """
    arg1 directory: path to files \n
    arg2 file_pattern: file type allowed \n
    returns a list of filenames from a given 
    directory and the file type specified
    """
    
    if not file_pattern.startswith("."):
        file_pattern = "."+file_pattern
    filenames = []
    for fname in glob(directory+"*"+file_pattern):
        filenames.append(fname)
        filenames.sort()
    return filenames
    
    
def readImagesFromDirectory(directory,file_pattern,rgb=True):
    """
    arg1 directory: path to files \n
    arg2 file_pattern: file type allowed \n
    arg3 rgb[True]: specifies the number of channels in the output images \n \n
    returns a list of images and the filenames read from a given 
    directory.
    """
    
    images = []
    filenames = getFilenamesFromDirectory(directory,file_pattern)
    for fname in filenames:
        img = plt.imread(fname)
    
        if len(img.shape)>2:
            if not rgb:
                img = 0.3*img[:,:,0]+0.59*img[:,:,1]+0.11*img[:,:,2]
            else:
                if img.shape[2] == 4:
                    img = img[:,:,0:3]
        else:
            tmp = np.zeros((img.shape[0],img.shape[1],3),dtype=float)
            for i in range(0,3):
                tmp[:,:,i] = img[:,:]
                
        images.append(img)
    return images,filenames