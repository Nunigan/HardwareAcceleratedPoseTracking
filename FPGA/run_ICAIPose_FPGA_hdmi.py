#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:35:29 2021

@author: nunigan
"""

from ctypes import *
from typing import List
import cv2
import vart
import xir
import threading
import sys
import time
import numpy as np
from imutils.video import FPS

from ICAI_Pose_inference import ICAIPose
from overwrite_queue import OverwriteQueue



def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


class App:

    def __init__(self, video_capture_str, width, height, thread_count, ICAIPose_xmodel):

        self.width = width
        self.height = height

        global bQuit
        bQuit = False  # quit flag for threads

        # Initialize ICAI pose networks
        ICAIPose_graph = xir.Graph.deserialize(ICAIPose_xmodel)
        ICAIPose_subgraphs = get_child_subgraph_dpu(ICAIPose_graph)
        assert len(ICAIPose_subgraphs) == 1 # only one DPU kernel
        all_dpu_runners = []
        for i in range(int(thread_count)):
            all_dpu_runners.append(vart.Runner.create_runner(ICAIPose_subgraphs[0], "run"))


        # Init synchronous queues for inter-thread communication
        # Use queues that do not accumulate memory if input rate is higher than output rate
        queueIn = OverwriteQueue()
        queueOut = OverwriteQueue()

        # Create threads
        threadAll = []

        tc = threading.Thread(target=self.taskCapture, args=(video_capture_str, queueIn))
        threadAll.append(tc)

        for i in range(thread_count):
            tw = threading.Thread(target=self.taskWorker, args=(i, all_dpu_runners[i], queueIn, queueOut))
            threadAll.append(tw)

        td = threading.Thread(target=self.taskDisplay, args=(queueOut,))
        threadAll.append(td)

        # Launch all threads
        for x in threadAll:
            x.start()

        try:
            # Wait for all threads to stop
            for x in threadAll:
                x.join()
        except KeyboardInterrupt:
            bQuit = True
            # Wait for all threads to stop
            for x in threadAll:
                x.join()


    def taskCapture(self, inputId, queueIn):
        """ Reads frames from a camera input """

        global bQuit

        # Start the FPS counter
        fpsIn = FPS().start()

        # Initialize the camera input
        print("[INFO] taskCapture : starting camera input ...")
        cam = cv2.VideoCapture(inputId)

        if not (cam.isOpened()):
            print("[ERROR] taskCapture : Failed to open camera ", inputId )
            exit()

        while not bQuit:
            # Capture image from camera
            ret, frame = cam.read()

            fpsIn.update()
            # frame = cv2.resize(frame, (256,256))

            # Push captured image to input queue
            queueIn.put(frame)

        # Stop the timer and display FPS information
        fpsIn.stop()
        print("[INFO] taskCapture : elapsed time: {:.2f}".format(fpsIn.elapsed()))
        print("[INFO] taskCapture : elapsed FPS: {:.2f}".format(fpsIn.fps()))

    def taskWorker(self, worker, dpu, queueIn, queueOut):
        """ Processes a frame with ICAIPose """

        global bQuit

        print("[INFO] taskWorker[",worker,"] : starting thread ...")

        dpu_ICAIPose = ICAIPose(dpu, self.width, self.height)
        dpu_ICAIPose.start()

        while not bQuit:

            # Pop captured image from input queue
            frame = queueIn.get()

            # Process frame
            res = dpu_ICAIPose.process(frame)

            # Push processed image to output queue
            queueOut.put(res)

        # Stop the face detector (since there are multiple threads,
        # every thread must get an item to be able to quit)
        queueIn.put(frame)

    def taskDisplay(self, queueOut):

        global bQuit

        print("[INFO] taskDisplay : starting thread ...")

        # Start the FPS counter
        fpsOut = FPS().start()

        # out = cv2.VideoWriter('appsrc ! videoconvert ! video/x-raw, format=NV12! kmssink driver-name=xlnx plane-id=39 sync=false fullscreen-overlay=true',0, 30.0, (self.width, self.height))
        
        while not bQuit:
            # Pop processed image from output queue

            frame = queueOut.get()

            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            # Write frame to the sink
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # out.write(frame)
            # Update the FPS counter
            fpsOut.update()

        # Set Quit flag to inform all other running threads
        bQuit = True

        # Stop the timer and display FPS information
        fpsOut.stop()
        print("[INFO] taskDisplay : elapsed time: {:.2f}".format(fpsOut.elapsed()))
        print("[INFO] taskDisplay : elapsed FPS: {:.2f}".format(fpsOut.fps()))


def main():

    import os

    path = os.getcwd()

    cam_width = 1280
    cam_height = 720
    video_capture_str = "v4l2src device=/dev/video0 ! videoscale ! videoconvert ! video/x-raw, width={}, height={}, framerate=30/1 ! videoconvert ! appsink".format(cam_width, cam_height)
    thread_count = 3
    ICAIPose_xmodel = '/home/root/ICAIPose/ICAIPose320x240.xmodel'
    # ICAIPose_xmodel = '/home/root/ICAIPose/ICAIPose256x192.xmodel'

    app = App(
        video_capture_str,
        cam_width,
        cam_height,
        thread_count,
        ICAIPose_xmodel,
    )

if __name__ == "__main__":
    main()