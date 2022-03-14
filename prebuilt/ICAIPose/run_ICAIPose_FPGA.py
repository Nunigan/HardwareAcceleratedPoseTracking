#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:35:29 2021

@author: nunigan
"""

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
import time
import gc


from pose_parameter import PoseParameter
from imutils.video import FPS

from ICAI_Pose_inference import ICAIPose

global bQuit

def taskCapture(inputId,queueIn):

    global bQuit



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
        frame = cv2.merge([R, G, B])      
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

    # Stop the face detector

    queueIn.put(frame)


    
    
def taskDisplay(queueOut):

    global bQuit
    start = time.time()

    print("[INFO] taskDisplay : starting thread ...")

    # Start the FPS counter
    fpsOut = FPS().start()

    # out = cv2.VideoWriter('appsrc ! videoconvert ! video/x-raw, format=NV12! kmssink driver-name=xlnx plane-id=39 sync=false fullscreen-overlay=true',0, 30.0, (width, height))
    while not bQuit:
        # Pop processed image from output queue
        frame = queueOut.get()
	
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        frame = cv2.copyMakeBorder(frame, height//2-512, height//2-512, width//2-512, width//2-512, cv2.BORDER_CONSTANT, value=[0,0,0])

        cv2.imshow('ICAIPose',frame)
        key = cv2.waitKey(1) & 0xFF

        # Update the FPS counter
        fpsOut.update()


        # if time.time() -start > 300:
        #     break
        
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
    global width
    global height
    global bQuit
    bQuit = False
    inputId = 'mediasrcbin media-device=/dev/media0 v4l2src0::io-mode=dmabuf v4l2src0::stride-align=256  ! video/x-raw, width=256, height=256, format=NV12, framerate=30/1 ! videoconvert! appsink'
    threads = int(argv[1])
    ICAIPose_xmodel = argv[2]
    width = int(argv[3])
    height = int(argv[4])
    ICAIPose_graph = xir.Graph.deserialize(ICAIPose_xmodel)
    ICAIPose_subgraphs = get_child_subgraph_dpu(ICAIPose_graph)
    assert len(ICAIPose_subgraphs) == 1 # only one DPU kernel
    all_dpu_runners = [];
    for i in range(int(threads)):
        all_dpu_runners.append(vart.Runner.create_runner(ICAIPose_subgraphs[0], "run"));
    # Init synchronous queues for inter-thread communication
    queueIn = queue.LifoQueue()
    queueOut = queue.LifoQueue()

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
    all_dpu_runners = None
    gc.collect()
    del threadAll
    threadAll = None
    gc.collect()

    
    
if __name__ == "__main__":
    while True:
        main(sys.argv)

