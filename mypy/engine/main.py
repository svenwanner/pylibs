import sys
import time
import random
import logging
from threading import Thread

from Buffer.buffer import ImageBuffer


class MainThread(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.buffer = None

    def setBuffer(self, buffer):
        print "set a buffer..."
        self.buffer = buffer

    def run(self):
        ####################################
        ###  write your main code here   ###
        ####################################

        if not self.buffer.is_initialized():
            print "enable buffer and start buffer thread..."
            self.buffer.enable()
            self.buffer.start()



        while not self.buffer.end():
            data = self.buffer.get_data()
            if data is not None:
                self.buffer.clear()
                print "data sample", data[0].shape
                print "data available, do something..."


        ####################################
        ####################################


if __name__ == "__main__":

    main = MainThread()
    bufferObj = ImageBuffer("/home/swanner/Desktop/Linear4C/simple/cam1")
    main.setBuffer(bufferObj)
    main.start()


