import numpy as np
from scipy.stats.mstats import mquantiles

try:
    import h5py as h5
except:
    print "missing h5py package"


class discreteWorldSpace(object):

    def __init__(self, world_size, precision, result_layer=1):
        assert isinstance(world_size, type([]))
        assert len(world_size) == 2
        assert isinstance(precision, float)
        assert isinstance(result_layer, int)

        self.world_size = world_size
        self.precision = precision
        self.N = int(world_size[0]/precision)
        self.M = int(world_size[1]/precision)
        self.result_layer = result_layer
        self.shape = (self.N+1, self.M+1, result_layer, 5)

        self.grid = np.zeros(self.shape, dtype=np.float32)
        print "created world grid of size (w,h):", self.shape[1], self.shape[0]


    def setWorldValue(self, x, y, layer, value, color=None):
        assert isinstance(x, np.float32)
        assert isinstance(y, np.float32)
        assert isinstance(layer, int)
        assert 0 <= layer < self.result_layer
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 1
        assert value.shape[0] == 2

        index = self.world2grid(x, y)
        if index is not None:
            self.grid[index[0], index[1], layer, 0:2] = value[:]
            if color is not None:
                self.grid[index[0], index[1], layer, 2:self.grid.shape[3]] = color[:]
        else:
            pass
            #print "Warning, projected position (",y,",",x,") out of grid!"


    def getResult(self, rtype="median"):
        if rtype == "mean":
            return self.getResultMean()
        if rtype == "median":
            return self.getResultMedian()
        if rtype == "median_3x3":
            return self.getResultsRegionMedian()


    def getResultMean(self):
        cloud = np.zeros((self.N+1, self.M+1, 4))
        color = np.zeros((self.N+1, self.M+1, 3))

        for n in range(self.N+1):
            for m in range(self.M+1):
                confidences = self.grid[n, m, :, 1]
                depths = self.grid[n, m, :, 0]
                csum = np.sum(confidences)
                if csum > 0:
                    confidences /= csum
                    depth = np.sum(confidences*depths)
                    wpos = self.grid2world(n, m)
                    cloud[n, m, 0] = wpos[1]
                    cloud[n, m, 1] = wpos[0]
                    cloud[n, m, 2] = depth
                    cloud[n, m, 3] = np.mean(confidences)

                    for c in range(3):
                        color[n, m, c] = self.grid[n, m, c+2]
                else:
                    cloud[n, m, 3] = 0
                    cloud[n, m, 2] = -1

        return cloud, color, None


    def getResultMedian(self):
        cloud = np.zeros((self.N+1, self.M+1, 4))
        color = np.zeros((self.N+1, self.M+1, 3))
        doubleDepthProp  = np.zeros((self.N+1, self.M+1))

        for n in range(self.N+1):
            for m in range(self.M+1):
                confidences = self.grid[n, m, :, 1]
                depths = self.grid[n, m, :, 0]
                csum = np.sum(confidences)
                if csum > 0:
                    wpos = self.grid2world(n, m)
                    cloud[n, m, 0] = wpos[1]
                    cloud[n, m, 1] = wpos[0]
                    dq = mquantiles(depths)
                    cloud[n, m, 2] = dq[1]
                    cloud[n, m, 3] = mquantiles(confidences)[1]
                    doubleDepthProp[n, m] = dq[2]-dq[0]

                    for c in range(3):
                        color[n, m, c] = np.median(self.grid[n, m, :, c+2])
                else:
                    cloud[n, m, 3] = 0
                    cloud[n, m, 2] = -1

        return cloud, color, doubleDepthProp


    def getResultsRegionMedian(self):
        cloud = np.zeros((self.N+1, self.M+1, 4))
        color = np.zeros((self.N+1, self.M+1, 3))
        doubleDepthProp  = np.zeros((self.N+1, self.M+1))

        for n in range(self.N+1):
            for m in range(self.M+1):
                confidences = None
                depths = None
                if n>0 and n<self.N and m>0 and m<self.M:
                    confidences = self.grid[n-1:n+2, m-1:m+2, :, 1].flatten()
                    depths = self.grid[n-1:n+2, m-1:m+2, :, 0].flatten()
                else:
                    confidences = self.grid[n, m, :, 1].flatten()
                    depths = self.grid[n, m, :, 0].flatten()
                csum = np.sum(confidences)
                if csum > 0:
                    wpos = self.grid2world(n, m)
                    cloud[n, m, 0] = wpos[1]
                    cloud[n, m, 1] = wpos[0]
                    dq = mquantiles(depths)
                    cloud[n, m, 2] = dq[1]
                    cloud[n, m, 3] = mquantiles(confidences)[1]
                    doubleDepthProp[n, m] = dq[2]-dq[0]

                    for c in range(3):
                        color[n, m, c] = np.median(self.grid[n, m, :, c+2])
                else:
                    cloud[n, m, 3] = 0
                    cloud[n, m, 2] = -1

        doubleDepthProp[:] /= np.amax(doubleDepthProp)
        return cloud, color, doubleDepthProp


    def world2grid(self, x, y):
        if -self.world_size[0]/2.0 <= y <= self.world_size[0]/2.0:
            if -self.world_size[1]/2.0 <= x <= self.world_size[0]/2.0:
                return [int((self.world_size[0]/2.0-y)/self.world_size[0]*self.N),
                        int(self.M-int((self.world_size[1]/2.0-x)/self.world_size[1]*self.M))]
            else:
                return None
        else:
            return None


    def grid2world(self, n, m):
        if 0 <= n <= self.N:
            if 0 <= m <= self.M:
                return [(float(self.N)/2.0-float(n))/float(self.N)*self.world_size[0],
                        (float(m)-float(self.M)/2.0)/float(self.M)*self.world_size[1]]
            else:
                return None
        else:
            return None


    def save(self, filename):
        assert isinstance(filename, str)

        if not filename.endswith(".h5"):
            filename += ".h5"

        f = h5.File(filename, "w")
        dset = f.create_dataset("grid", data=np.copy(self.grid), dtype=np.float32)
        f.close()
