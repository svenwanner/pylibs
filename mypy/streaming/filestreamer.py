import numpy as np
from multiprocessing import Process,Queue, cpu_count

from mypy.tools.linearAlgebra import normalize_vec

cpus_available = cpu_count()


def computeMissingParameter(parameter):
    # compute the field of view of the camera
    parameter["fov"] = np.arctan2(parameter["sensorSize_mm"], 2.0*parameter["focalLength_mm"])
    # compute focal length is pixel
    parameter["focalLength_px"] = float(parameter["focalLength_mm"])/float(parameter["sensorSize_mm"])*parameter["sensorSize_px"][1]
    # compute the total number of frames
    parameter["totalNumOfFrames"] = parameter["subImageVolumeSize"]+(parameter["numOfSubImageVolumes"]-1)*parameter["frameShift"]
    # compute the maximum traveling distance of the camera
    parameter["maxBaseline_mm"] = float(parameter["totalNumOfFrames"]-1)*parameter["baseline_mm"]
    # compute the final camera position and the center position of the camera track
    parameter["camInitialPos"] = np.array(parameter["camInitialPos"])
    if not parameter.has_key("camTransVector") or not parameter.has_key("camLookAtVector"):
        print "Warning, either camTransVector or camLookAtVector is missing, default values are used instead!"
        parameter["camTransVector"] = np.array([1.0, 0.0, 0.0])
        parameter["camLookAtVector"] = np.array([0.0, 0.0, -1.0])
    else:
        parameter["camTransVector"] = np.array(parameter["camTransVector"])
        parameter["camLookAtVector"] = np.array(parameter["camLookAtVector"])
        parameter["camTransVector"] = normalize_vec(parameter["camTransVector"])
        parameter["camLookAtVector"] = normalize_vec(parameter["camLookAtVector"])
    parameter["camFinalPos"] = parameter["camInitialPos"] + parameter["maxBaseline_mm"]*parameter["camTransVector"]/100.0
    # define camera z-coordinate as horopter distance
    parameter["horopter_m"] = parameter["camInitialPos"][2]
    # define horopter vector
    parameter["horopter_vec"] = parameter["camLookAtVector"]*float(parameter["horopter_m"])

    # compute vertical field of view
    sensorSize_y = float(parameter["sensorSize_mm"])*parameter["sensorSize_px"][0]/float(parameter["sensorSize_px"][1])
    fov_h = np.arctan2(sensorSize_y, 2.0*parameter["focalLength_mm"])
    # compute real visible scene width and height
    vwsx = 2.0*parameter["camInitialPos"][2]*np.tan(parameter["fov"])+parameter["maxBaseline_mm"]/100.0
    vwsy = 2.0*parameter["camInitialPos"][2]*np.tan(fov_h)
    parameter["visibleWorldArea"] = [vwsx, vwsy]

    for key in parameter.keys():
        print key, ":", parameter[key]

    return parameter




def main(parameter):

    is_processing = True

    #compute missing parameter and final storage container
    parameter = computeMissingParameter(parameter)
    #final_storage = np.zeros(, dtype=np.float32)

    #set up filestreaming object

    #set up sub light field processor



    while is_processing:
        # if filestreaming object is ready pass data to sub light field processor
        is_processing = False



if __name__ == "__main__":

    parameter = {
        "sensorSize_mm": 32,
        "focalLength_mm": 35,
        "baseline_mm": 1.020408163265306,
        "sensorSize_px": [540, 960],
        "subImageVolumeSize": 9,
        "frameShift": 9,
        "numOfSubImageVolumes": 11,
        "camInitialPos": [0.0, 0.0, 2.0],
        "camTransVector": [1.0, 0.0, 0.0],
        "camLookAtVector": [0.0, 0.0, -1.0]
    }
    main(parameter)