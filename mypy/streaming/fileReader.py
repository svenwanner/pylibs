import os
import vigra
import numpy as np
from glob import glob
from scipy.misc import imread


class FileReader():
    """
    This class reads all image filenames from a given directory,
    loads parts of it and returns the images as ndarray volumes.
    """
    def __init__(self, input_path, stack_size=11, swap_files_order=False):
        """
        Constructor needs a filepath containing image files and
        defines the stack size.
        :param input_path: <str> path to directory containing image files
        :param stack_size: <int> number of images in a single stack
        """
        assert isinstance(input_path, str)
        assert os.path.isdir(input_path), "input path does not exist!"
        assert isinstance(stack_size, int)
        assert stack_size > 0, "stack size needs to be at least 1!"

        if not input_path.endswith(os.sep):
            input_path += os.sep

        self.path = input_path
        self.filenames = []
        self.stack_size = stack_size
        self.swap_files_order = swap_files_order
        self.valid_types = ("png", "PNG", "jpg", "JPG", "JPEG", "tif", "tiff", "TIFF", "bmp", "ppm", "exr")

        for f in glob(input_path+"*"):
            for ft in self.valid_types:
                if f.endswith(ft):
                    self.filenames.append(f)
        self.filenames.sort()
        assert len(self.filenames) > 0, "No image files found!"

        self.num_of_files = len(self.filenames)
        self.current_file = 0
        self.counter = 0
        self.ready = False
        self.finished = False

        tmp = self.loadImage(self.filenames[0])
        tmp = self.channelConverter(tmp)
        self.shape = (self.stack_size, tmp.shape[0], tmp.shape[1])
        self.stack = np.zeros(self.shape, dtype=np.float32)
        self.cv_color = np.zeros((tmp.shape[0], tmp.shape[1], 3), dtype=np.uint8)

    def loadImage(self, fname):
        """
        loads a single image
        :param fname: <str> filename
        :return: <ndarray> image
        """
        assert isinstance(fname, str)
        if fname.endswith("exr"):
            return np.transpose(np.array(vigra.readImage(fname))[:, :, 0]).astype(np.float32)
        else:
            return imread(fname)

    def channelConverter(self, img, ctype="exp_stretch"):
        """
        converts an image to a grayscale image depending on
        the conversion type.
        :param img: <ndarray> image
        :param ctype: <str> conversion type ["gray","stretch","exp_stretch"]
        :return: <ndarray> single band image
        """
        img = img.astype(np.float32)
        amax = np.amax(img)
        if amax > 1.0:
            img[:] /= 255.0
        if len(img.shape) == 2:
            return img
        out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        if ctype == "gray":
            out[:] = 0.3*img[:, :, 0]+0.59*img[:, :, 1]+0.11*img[:, :, 2]
        if ctype == "stretch":
            for c in range(img.shape[2]):
                out[:, :] += img[:, :, c]+c
        if ctype == "exp_stretch":
            for c in range(img.shape[2]):
                out[:, :] += (img[:, :, c])**(c+1)
        amax = np.amax(out)
        out[:] /= amax
        return out

    def bufferReady(self):
        """
        returns the buffer is filled flag state
        :return: <bool> flag if buffer is filled
        """
        return self.ready

    def getStack(self):
        """
        returns the filled buffer and resets all counters
        :return: <ndarray> image volume buffer
        """
        if self.bufferReady():
            self.counter = 0
            self.ready = False
            return np.copy(self.stack), self.cv_color

    def read(self):
        """
        runs the buffer filling process
        """
        filesToRead = []
        stack_index = 0

        #if swap order is True count image index from the back
        if self.swap_files_order:
            stack_index = self.stack_size-1

        while self.counter < self.stack_size and not self.finished:
            self.ready = False
            if self.current_file == self.num_of_files:
                self.finished = True
                break

            # store filename for debug output only
            filesToRead.append(os.path.basename(self.filenames[self.current_file]))
            if self.counter == 0:
                self.stack[:] = 0.0
            # load image file
            tmp = self.loadImage(self.filenames[self.current_file])
            # save center view color
            if self.counter == self.stack_size/2:
                if len(tmp.shape) == 3:
                    self.cv_color[:, :, 0:3] = tmp[:, :, 0:3]
                elif len(tmp.shape) == 2:
                    for c in range(3):
                        self.cv_color[:, :, c] = tmp[:, :]
                else:
                    assert False, "Unknown image format!"
            # convert image to grayscale
            tmp = self.channelConverter(tmp)
            # put image into satck
            self.stack[stack_index, :, :] = tmp[:]

            # update counters
            self.current_file += 1
            self.counter += 1
            if self.swap_files_order:
                stack_index -= 1
            else:
                stack_index += 1

        if len(filesToRead) != 0:
            print "read files from", filesToRead[0], "to", filesToRead[-1]
        self.ready = True
