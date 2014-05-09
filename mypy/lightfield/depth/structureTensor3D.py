


def structureTensor3D(lf,inner_scale,outer_scale):
    
    st = vigra.filters.structureTensor(lf,0.6,1.3)
    evals = np.zeros((lf.shape[1],lf.shape[2],3),dtype=np.float32)
    evecs = np.zeros((lf.shape[1],lf.shape[2],3,3),dtype=np.float32)
    
    for y in xrange(lf.shape[1]):
        for x in xrange(lf.shape[2]):
            mat = np.zeros((3,3),dtype=np.float64)
            mat[0,0] = st[lf.shape[0]/2,y,x,0]
            mat[0,1] = st[lf.shape[0]/2,y,x,1]
            mat[0,2] = st[lf.shape[0]/2,y,x,2]
            mat[1,0] = st[lf.shape[0]/2,y,x,1]
            mat[1,1] = st[lf.shape[0]/2,y,x,3]
            mat[1,2] = st[lf.shape[0]/2,y,x,4]
            mat[2,0] = st[lf.shape[0]/2,y,x,3]
            mat[2,1] = st[lf.shape[0]/2,y,x,4]
            mat[2,2] = st[lf.shape[0]/2,y,x,5]
            
            evals[y,x,:],evecs[y,x,:,:] = LA.eigh(mat)

    return evals,evecs
    
    
def structureTensor3DConditioner(evals,evecs):
    disparity = np.zeros((evals.shape[0],evals.shape[1]))
    for y in xrange(evals.shape[0]):
        for x in xrange(evals.shape[1]):
            axis=np.array([evals[y,x,0],evals[y,x,1],evals[y,x,2]])
            stype,order = shapeEstimator(axis)
            
            #Todo handle eigenvectors depending stype and order
            if(stype==0):
                disparity[y,x] = 0
            elif(stype==1):
                disparity[y,x] = 1
            elif(stype==2):
                disparity[y,x] = 2 
            elif(stype==3):
                disparity[y,x] = 3
                
    return disparity
    
    
def shapeEstimator(axis,dev=0.2):
    order = axis.argsort()
    axis[order[0]]=axis[order[0]]/axis[order[2]]
    axis[order[1]]=axis[order[1]]/axis[order[2]]
    axis[order[2]]=1
    
    high = axis[order[2]]
    mid = axis[order[1]]
    low = axis[order[0]]
    
    if high-mid<dev and high-low<dev:
        return 0,order
    elif (high-mid<dev and high-low>=dev) or (high-mid>=dev and high-low<dev):
        return 1,order
    elif (high-mid>=dev and high-low>=dev):
        return 2,order
    else:
        return 3,order