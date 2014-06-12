import random
import time
import logging

import os, sys
from glob import glob
import scipy.misc as misc

from threading import Thread


class Buffer(Thread):

    def __init__(self):
        print "initialize Buffer..."
        Thread.__init__(self)

        self.buffer_size = 17
        self.is_running = True
        self.finished = False
        self.data_available = False
        self.initialized = False

        self.data = None


    def enable(self):
        print "enable buffer..."
        self.is_running = True
        self.initialized = True

    def disable(self):
        print "disable buffer..."
        self.is_running = False

    def is_initialized(self):
        return self.initialized

    def end(self):
        return self.finished

    def execute(self):
        pass

    def run(self):

        while self.is_running:

            self.execute()

            if self.finished:
                self.is_running = False

    def get_data(self):
        if self.data_available:
            print "buffer returns data..."
            return self.data
        else:
            return None

    def clear(self):
        self.data_available = False
        self.data = []






class ImageBuffer(Buffer):

    def __init__(self, filepath):
        Buffer.__init__(self)
        print "initialize ImageBuffer..."

        self.buffer_size = 17
        self.filepath = filepath
        self.filenames = []
        self.data = []

        self.read_filenames(filepath)

    def read_filenames(self, filepath):
        assert isinstance(filepath, str)
        self.filenames = []
        if not filepath.endswith(os.sep):
            filepath += os.sep

        for f in glob(filepath+"*"):
            self.filenames.append(f)
        self.filenames.sort()
        print self.filenames

    def execute(self):
        self.data_available = False
        if len(self.filenames) == 0:
            self.finished = True
            self.is_running = False
            self.clear()

        if self.data is not None:
            if len(self.data) <= self.buffer_size:
                print "load image..."
                self.data.append(misc.imread(self.filenames.pop(0)))
            else:
                self.data_available = True
