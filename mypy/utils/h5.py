# -*- coding: utf-8 -*-

import numpy as np 
import h5py as h5


def saveHdf5(filename=None, datasets=[], attributes={}):
   
  if type(datasets) is type([]) and len(datasets)>0 and type(datasets[0]) is type({}): 
    if filename is not None:
      try:
        print "saving hdf5 at:",filename
        out_file = h5.File(filename, 'w')
      except:
        print "saving failed, no file created please check filename"
        return False
      
      
      for n,ds in enumerate(datasets):
        try:
          dtype = ds['dtype']
        except:
          dtype = np.float32
          print "warning no dtype specified, use float32"
        try:
          name = ds['name']
        except:
          name = str("%3.3i"%n)
          print "warning no name specified, saved as",name
        try:
          data = ds['data']
        except:
          print "saving failed, no data given for dataset",name
          return False
          
        out_file.create_dataset(name, data=data, dtype=dtype, compression='gzip')
        
      for key in attributes.keys():
        out_file.attrs[key] = attributes[key]
    
      out_file.close()
    else:
      print "saving failed, no filename is given"
      return False
  else:
    print "saving hdf5 failed, datasets is of unidentified type"
    return False