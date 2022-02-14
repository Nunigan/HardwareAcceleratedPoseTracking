#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys
import argparse
import queue

from pose_parameter import PoseParameter
from imutils.video import FPS

from ICAI_Pose_inference import ICAIPose

global bQuit

def taskCapture(inputId,queueIn):

    global bQuit

    #print("[INFO] taskCapture : starting thread ...")

    # Start the FPS counter
    fpsIn = FPS().start()

    # Initialize the camera input
    print("[INFO] taskCapture : starting camera input ...")
    cam = cv2.VideoCapture(inputId, cv2.CAP_GSTREAMER)
    if not (cam.isOpened()):
        print("[ERROR] taskCapture : Failed to open camera ", inputId )
        exit()

    while not bQuit:
        # Capture image from camera
        ret,frame = cam.read()
        R, G, B = cv2.split(frame)
        frame = cv2.merge([R, G, B])        # Update the FPS counter
        fpsIn.update()

        # Push captured image to input queue
        queueIn.put(frame)

    # Stop the timer and display FPS information
    fpsIn.stop()
    print("[INFO] taskCapture : elapsed time: {:.2f}".format(fpsIn.elapsed()))
    print("[INFO] taskCapture : elapsed FPS: {:.2f}".format(fpsIn.fps()))

    #print("[INFO] taskCapture : exiting thread ...")


def taskWorker(worker,dpu,queueIn,queueOut):
    global bQuit

    print("[INFO] taskWorker[",worker,"] : starting thread ...")

    dpu_ICAIPose = ICAIPose(dpu)
    dpu_ICAIPose.start()

    
    while not bQuit:

        # Pop captured image from input queue
        frame = queueIn.get()

        res = dpu_ICAIPose.process(frame)

        # Push processed image to output queue
        queueOut.put(res)

    queueIn.put(frame)

    print("[INFO] taskWorker[",worker,"] : exiting thread ...")
    
    
def taskDisplay(queueOut):

    global bQuit

    print("[INFO] taskDisplay : starting thread ...")

    # Start the FPS counter
    fpsOut = FPS().start()

    while not bQuit:
        # Pop processed image from output queue
        frame = queueOut.get()

        # Display the processed image
        cv2.imshow("ICAIPose", frame)

        # Update the FPS counter
        fpsOut.update()

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    bQuit = True

    # Stop the timer and display FPS information
    fpsOut.stop()
    print("[INFO] taskDisplay : elapsed time: {:.2f}".format(fpsOut.elapsed()))
    print("[INFO] taskDisplay : elapsed FPS: {:.2f}".format(fpsOut.fps()))

    # Cleanup
    cv2.destroyAllWindows()

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
    
def main(argv):

    global bQuit
    bQuit = False
    inputId = 'mediasrcbin media-device=/dev/media0 v4l2src0::io-mode=dmabuf v4l2src0::stride-align=256  ! video/x-raw, width=1920, height=1080, format=NV12, framerate=30/1 ! videoconvert! appsink'
    threads = int(argv[1])
    ICAIPose_xmodel = argv[2]
    ICAIPose_graph = xir.Graph.deserialize(ICAIPose_xmodel)
    ICAIPose_subgraphs = get_child_subgraph_dpu(ICAIPose_graph)
    assert len(ICAIPose_subgraphs) == 1 # only one DPU kernel
    all_dpu_runners = [];
    for i in range(int(threads)):
        all_dpu_runners.append(vart.Runner.create_runner(ICAIPose_subgraphs[0], "run"));
    # Init synchronous queues for inter-thread communication
    queueIn = queue.Queue()
    queueOut = queue.Queue()

    # Launch threads
    threadAll = []

    tc = threading.Thread(target=taskCapture, args=(inputId,queueIn))
    threadAll.append(tc)

    for i in range(threads):
        tw = threading.Thread(target=taskWorker, args=(i,all_dpu_runners[i],queueIn,queueOut))
        threadAll.append(tw)
    td = threading.Thread(target=taskDisplay, args=(queueOut,))
    threadAll.append(td)
    for x in threadAll:
        x.start()

    # Wait for all threads to stop
    for x in threadAll:
        x.join()

    # Cleanup VART API
    del all_dpu_runners


if __name__ == "__main__":
    main(sys.argv)
