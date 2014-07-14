import vigra
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compute_MSE(depth, gt, upperBorderError, lowerBorderError):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """

    mse = (depth[:] - gt[:])*(depth[:] - gt[:])
    mask = np.where(gt > 0)
    mse_no = np.mean(mse[mask])

    return mse, mse_no


def compute_MAE(depth, gt):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """

    mae = abs(depth[:] - gt[:])
    mask = np.where(gt > 0)
    mae_no = np.mean(mae[mask])

    return mae, mae_no

def compute_MRE(depth, gt, upperBorderError, lowerBorderError):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """
    mre = (depth[:] - gt[:])*(depth[:] - gt[:]) / gt[:]
    mask = np.where(gt > 0)
    mre_no = np.mean(mre[mask])
    mre_pc = mre_no *100

    return mre, mre_no, mre_pc


def plot(figure_no, data, min, max, title, bartitle, save=False):

    array = np.copy(data)
    print("Original Range: " + str(array.min()) + " - " + str(array.max()))

    np.place(array, array > max, max)
    np.place(array, array < min, min)

    array[0, 0] = float(min)
    array[array.shape[0]-1, array.shape[1]-1] = float(max)

    array = array.transpose()

    plt.figure(figure_no,figsize=(9.5, 7.5),dpi=80, facecolor='w',)

    v = np.linspace(-.1, 2.0, 15, endpoint=True)
    plt.title(title,fontsize = 22)
    plt.xlabel("x [px]",fontsize = 20)
    plt.ylabel("y [px]",fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    a = plt.imshow(array)

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    ticks_range = 10

    step = (max - min)/ticks_range
    d = ["" for x in range(ticks_range+1)]
    d[0] = str(min)

    tmp = min
    for i in range(ticks_range):
        tmp = round(tmp + step,2)
        d[i+1] = str(int(tmp*100)/100.0)

    if (max < np.max(data) ):
         d[10] = "> " + str(max)
    if (np.min(data) < min ):
         d[0] = "< " + str(max)

    b = plt.colorbar(cax=cax)
    b.ax.set_yticklabels(d)
    b.set_label(bartitle, fontsize=20)
    b.ax.tick_params(labelsize=15)

    if save == True:
        plt.savefig(title + '.png', bbox_inches='tight')


if __name__ == "__main__":

    if 6 <= len(sys.argv) <= 7:

        #### Load depth estimation and ground truth data ###

        depth_img = vigra.readImage(sys.argv[1])
        GT = vigra.readImage(sys.argv[2])

        ### Exr files are 4dimensional extract layer with depth information ###

        depth = np.copy(depth_img[:, :, 2])
        GT = np.copy(GT[:, :, 2])

        ### check if size of both arrays is the same, if not reduce the ground truth ###

        if (GT.shape != depth_img.shape):
            diff0 = abs(GT.shape[0]-depth_img.shape[0])/2
            diff1 = abs(GT.shape[1]-depth_img.shape[1])/2
            GT_img = GT[diff0:GT.shape[0]-diff0, diff1:GT.shape[1]-diff1]
        else:
            GT_img = GT

        ### compute the MAE, MSE, MRE ###


        mae, mae_no = compute_MAE(depth, GT_img)
        mse, mse_no = compute_MSE(depth, GT_img, sys.argv[5], sys.argv[6])
        [mre, mre_no, mre_pc] = compute_MRE(depth, GT_img, sys.argv[5], sys.argv[6])

        print('\033[91mMAE [m]: '+ str(mae_no))
        print('MSE [m]: '+ str(mse_no))
        print('MRE [ ]: '+ str(mre_no))
        print('MRE [%]: '+ str(mre_pc))
        print("\033[92m")

        #### Display MAE,MSE MRE ###
        save = True

        plot(3, mae,float(sys.argv[5]),float(sys.argv[6]), "Mean Absolute Error"+ str(mae_no), "MAE [px]",save)
        plot(4, mse,float(sys.argv[5]),float(sys.argv[6]), "Mean Squared Error"+ str(mse_no), "MSE [px]",save)
        plot(5, mre,float(sys.argv[5]),float(sys.argv[6]), "Mean Relative Error"+ str(mre_pc), "MRE [px]",save)

        print("\033[0m")
        plot(1, depth, float(sys.argv[3]),float(sys.argv[4]), "Depth Estimation", "Depth [m]",save)
        plot(2, GT_img, float(sys.argv[3]),float(sys.argv[4]), "Ground Truth", "Depth [m]",save)


        plt.show()


    else:
        print("\033[93m Computes the Mean Absolute Error, Mean Square Error and the Mean Relative Error of a given Depth map with the GT image \033[0m")
        print("arg1: depthimage file")
        print("arg2: GT file (optional if in different file safed)")
        print("arg3: Upper Border Depth")
        print("arg4: Lower Border Depth")
        print("arg5: Upper Border Comparison")
        print("arg6: Lower Border Comparison")











