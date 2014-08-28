import vigra
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compute_MSE(depth, gt):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """

    mse = (depth[:] - gt[:])*(depth[:] - gt[:])
    mask = np.where(gt > 0)
    np.place(mse, gt == 0 , 0)
    # Comment out for digital data having negative disparities , otherwise not.
    # mask = np.where(depth > 0)
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
    np.place(mae, gt == 0 , 0)
    # Comment out for digital data having negative disparities , otherwise not.
    # mask = np.where(depth > 0)
    mae_no = np.mean(mae[mask])

    return mae, mae_no

def compute_MRE(depth, gt):
    """

    :param depth: Computed depth map
    :param gt: Ground Truth data
    :return mse: mean squared error
    :return mse_pc: mean relative error in percent
    """
    mre = (depth[:] - gt[:])*(depth[:] - gt[:]) / abs(gt[:])
    mask = np.where(gt > 0)
    np.place(mre, gt == 0 , 0)
    # Comment out for digital data having negative disparities , otherwise not.
    # mask = np.where(depth > 0)
    mre_no = np.mean(mre[mask])
    mre_pc = mre_no *100

    return mre, mre_no, mre_pc


def plot(figure_no, data, min, max, title, bartitle, save=False):

    array = np.copy(data)
    print( title + " Range: " + str(array.min()) + " - " + str(array.max()))

    np.place(array, array > max, max)
    np.place(array, array < min, min)

    array[0, 0] = float(min)
    array[array.shape[0]-1, array.shape[1]-1] = float(max)

    array = array.transpose()

    plt.figure(figure_no,figsize=(9.5, 7.5),dpi=80, facecolor='w',)

    #v = np.linspace(-.1, 2.0, 15, endpoint=True)
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
         d[0] = "< " + str(min)

    b = plt.colorbar(cax=cax)
    b.ax.set_yticklabels(d)
    b.set_label(bartitle, fontsize=20)
    b.ax.tick_params(labelsize=15)

    if save == True:
        plt.savefig(title + '.png', bbox_inches='tight')



def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":

    if 9 <= len(sys.argv) <= 9:

        #### Load depth estimation and ground truth data ###

        depth_img = vigra.readImage(sys.argv[1])
        print(depth_img.shape)
        name = sys.argv[2]
        if name.endswith('exr'):
            data = vigra.readImage(sys.argv[2])
            print(data.shape)
            GT = np.copy(data[:, :, 0])
            minDepth = float(sys.argv[3])
            maxDepth = float(sys.argv[4])
            minError = float(sys.argv[5])
            maxError = float(sys.argv[6])
            save = str2bool(sys.argv[7])
            plotFIG = save = str2bool(sys.argv[8])
        elif name.endswith('tif'):
            data = vigra.readImage(sys.argv[2])
            print(data.shape)
            GT = np.copy(data[:, :, 0])
            minDepth = float(sys.argv[3])
            maxDepth = float(sys.argv[4])
            minError = float(sys.argv[5])
            maxError = float(sys.argv[6])
            save = str2bool(sys.argv[7])
            plotFIG = str2bool(sys.argv[8])
        else:
            pass

        ### Exr files are 4dimensional extract layer with depth information and cut border with bad estimation values ###

        disparity = np.copy(depth_img[:, :, 0].T)
        # disparity = disparity[15:disparity.shape[0]-15, 15:disparity.shape[1]-15]

        ### check if size of both arrays is the same, if not reduce the ground truth ###

        if (GT.shape != disparity.shape):
            print('enter')
            diff0 = abs(GT.shape[0]-disparity.shape[0])/2
            diff1 = abs(GT.shape[1]-disparity.shape[1])/2
            GT_img = GT[diff0:GT.shape[0]-diff0, diff1:GT.shape[1]-diff1]
        else:
            GT_img = GT

        ### compute the MAE, MSE, MRE ###

        mae, mae_no = compute_MAE(disparity, GT_img)
        mse, mse_no = compute_MSE(disparity, GT_img)
        [mre, mre_no, mre_pc] = compute_MRE(disparity, GT_img)

        print('\033[91mMAE [m]: '+ str(mae_no))
        print('MSE [m]: '+ str(mse_no))
        print('MRE [ ]: '+ str(mre_no))
        print('MRE [%]: '+ str(mre_pc))
        print("\033[92m")

        #### Display MAE,MSE MRE ###

        plot(3, mae, minError, maxError, "Mean_Absolute_Error:_"+ "{0:.5f}".format(mae_no), "MAE [px]",save)
        plot(4, mse, minError, maxError, "Mean_Squared_Error:_"+ "{0:.5f}".format(mse_no), "MSE [px]",save)
        plot(5, mre, minError, maxError, "Mean_Relative_Error:_"+ "{0:.2f}".format(mre_pc) , "MRE [%]",save)

        print("\033[0m")
        plot(1, disparity, minDepth, maxDepth, "Disparity_Estimation", "Disparity [px]", save)
        plot(2, GT_img, minDepth, maxDepth, "Ground_Truth", "Disparity [px]", save)

        if (plotFIG == 'True'):
            plt.show()


    else:
        print("\033[93m Computes the Mean Absolute Error, Mean Square Error and the Mean Relative Error of a given Disparity map with the GT disparity image \033[0m")
        print("arg1: disparity image file")
        print("arg2: GT disparity file")
        print("arg3: min disparity range in image")
        print("arg4: max disparity range in image")
        print("arg5: min Border Comparison")
        print("arg6: max Border Comparison")
        print("arg7: save result (True/False)")
        print("arg8: display result (True/False)")
        print("eg. python ErrorEstimation_Disparity.py 0 5 0 1 True True")











