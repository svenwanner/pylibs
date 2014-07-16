import vigra
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab as plt

def compute_MSE(depth, gt, upperBorderError, lowerBorderError):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """

    mse = (depth[:] - gt[:])*(depth[:] - gt[:])
    mask = np.where(gt > 0)
    mse = np.mean(mse[mask])
    return mse


def compute_MAE(depth, gt, upperBorderError, lowerBorderError):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """

    mae = abs(depth[:] - gt[:])
    mask = np.where(gt > 0)
    mae = np.mean(mae[mask])
    return mae

def compute_MRE(depth, gt, upperBorderError, lowerBorderError):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """
    mask = np.where(gt > 0)
    mre = (depth[mask] - gt[mask])*(depth[mask] - gt[mask])
    mre = np.mean(mre)
    mre_pc = mre *100

    return mre, mre_pc


if __name__ == "__main__":

    if 6 <= len(sys.argv) <= 7:

        import numpy as np
        import matplotlib.pyplot as plt
        import pylab

        # Come up with x and y
        x = np.arange(0, 5, 0.1)
        y = np.sin(x)

        # Just print x and y for fun
        print x
        print y

        # plot the x and y and you are supposed to see a sine curve
        plt.plot(x, y)

        # without the line below, the figure won't show
        pylab.show()


        depth_img = vigra.readImage(sys.argv[1])
        assert isinstance(depth_img, np.ndarray)

        print("DepthImageSize: ")
        print(depth_img.shape)

        depth = depth_img[:, :, 2]

        imgplot = plt.imshow(depth)
        plt.show()

        Z=np.array(((1,2,3,4,5),(4,5,6,7,8),(7,8,9,10,11)))
        im = plt.imshow(Z, cmap='hot')
        plt.colorbar(im, orientation='horizontal')
        plt.show()


        #GT_img = vigra.readImage(sys.argv[2])
        #assert isinstance(GT_img, np.ndarray)

        #print("GTImageSize: ")
        #print(GT_img.shape)

        #upperBorderDepth = sys.argv[3]
        #print('Upper Border:' + str(upperBorderDepth))
        #lowerBorderDepth = sys.argv[4]
        #print('Lower Border:' + str(lowerBorderDepth))

        #upperBorderError = sys.argv[3]
        #print('Upper Border:' + str(upperBorderError))
        #lowerBorderError = sys.argv[4]
        #print('Lower Border:' + str(lowerBorderError))

        #mae = compute_MAE(depth_img,GT_img,upperBorderError,lowerBorderError)

        #mse = compute_MSE(depth_img,GT_img,upperBorderError,lowerBorderError)

        #[mre, mre_pc] = compute_MRE(depth_img,GT_img,upperBorderError,lowerBorderError)

        #print('MAE [m]: '+ str(mae))
        #print('MSE [m]: '+ str(mse))
        #print('MRE [ ]: '+ str(mre))
        #print('MRE [%]: '+ str(mre_pc))

    else:
        print("\033[93m Computes the Mean Absolute Error, Mean Square Error and the Mean Relative Error of a given Depth map with the GT image \033[0m")
        print("arg1: depthimage file")
        print("arg2: GT file")
        print("arg3: Upper Border Depth")
        print("arg4: Lower Border Depth")
        print("arg5: Upper Border GT")
        print("arg6: Lower Border GT")











