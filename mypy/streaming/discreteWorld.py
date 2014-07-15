import numpy as np

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
        self.shape = (self.N+1, self.M+1, result_layer, 2)


        self.grid = np.zeros(self.shape, dtype=np.float32)

    def setWorldValue(self, y, x, layer, value):
        assert isinstance(y, float)
        assert isinstance(x, float)
        assert isinstance(layer, int)
        assert 0 <= layer < self.result_layer
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 1
        assert value.shape[0] == 2

        index = self.world2grid(y, x)
        if index is not None:
            self.grid[index[0], index[1], layer] = value[:]
        else:
            pass
            #print "Warning, projected position (",y,",",x,") out of grid!"


    def getResult(self):
        cloud = np.zeros((self.N+1, self.M+1, 4))
        #confidence = np.zeros((self.N+1, self.M+1))

        for n in range(self.N+1):
            for m in range(self.M+1):
                confidences = self.grid[n, m, :, 1]
                depths = self.grid[n, m, :, 0]
                csum = np.sum(confidences)
                if csum > 0:
                    #confidence[n, m] = np.mean(confidences)
                    confidences /= csum
                    depth = np.sum(confidences*depths)
                    wpos = self.grid2world(n, m)
                    cloud[n, m, 0] = wpos[0]
                    cloud[n, m, 1] = wpos[1]
                    cloud[n, m, 2] = depth
                    cloud[n, m, 3] = np.mean(confidences)
                else:
                    cloud[n, m, 3] = 0
                    cloud[n, m, 2] = -1

        return cloud

    def world2grid(self, y, x):
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
                #print "grid2world project m :", m, ", n :", n, " to x :", int((m-self.M/2)/self.M*self.world_size[1]), " to y :", int((self.N/2-n)/self.N*self.world_size[0])
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
        dset = f.create_dataset("cloud")
        dset = np.copy(self.grid)
        f.close()
