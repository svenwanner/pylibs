import vigra
import sys
import numpy as np


def compute_MSE(depth, gt):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """

    mse = (depth[:] - gt[:])*(depth[:] - gt[:]) / gt[:]
    mse_pc = mse*100

    return mse, mse_pc



if __name__ == "__main__":

    if 2 <= len(sys.argv) <= 3:

        depth_img = vigra.readImage(sys.argv[1])
        assert isinstance(depth_img, np.ndarray)

        print("ColorImageSize: ")
        print(depth_img.shape)

        GT_img = vigra.readImage(sys.argv[2])
        assert isinstance(GT_img, np.ndarray)

        GT_img = depth_img[:, :, 1]
        print("DepthMapSize: ")
        print(depth_img.shape)

        [mse, mse_pc] = compute_MSE(depth_img,GT_img)

    else:
        print("\033[93m Computes the Mean Square Error of a given Depth map with the GT image \033[0m")
        print("arg1: depthimage file")
        print("arg2: GT file")













