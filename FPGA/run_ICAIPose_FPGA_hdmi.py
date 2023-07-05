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

def taskAll(inputId, dpu):

    global bQuit
    dpu_ICAIPose = ICAIPose(dpu)
    dpu_ICAIPose.start()
    cam = cv2.VideoCapture(inputId, cv2.CAP_GSTREAMER)
    fpsIn = FPS().start()

    # while True:
    # t = time.time()
    ret, frame = cam.read()
    # print('cam', time.time()-t)
    # t = time.time()
    res = dpu_ICAIPose.process(frame)
    # print('dpu', time.time()-t)
    # t = time.time()
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    # Write frame to the sink
    fpsIn.update()
    cv2.imshow("window", res)
    # print('display', time.time()-t)
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # out.write(frame)
    # Update the FPS counter
    fpsIn.stop()
    print("[INFO] taskCapture : elapsed time: {:.2f}".format(fpsIn.elapsed()))
    print("[INFO] taskCapture : elapsed FPS: {:.2f}".format(fpsIn.fps()))

def taskCapture(inputId, queueIn):
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

    # img = cv2.imread('test_img.jpg')
    # img = cv2.resize(img, (256,256))

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

def taskWorker(worker, dpu, queueIn, queueOut):
    """ Processes a frame with ICAIPose """

    global bQuit

    print("[INFO] taskWorker[",worker,"] : starting thread ...")

    dpu_ICAIPose = ICAIPose(dpu)
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

class App:

    def __init__(self, video_capture_str, width, height, thread_count, ICAIPose_xmodel, width_scale=1):

        self.width = width
        self.height = height
        self.width_scale = width_scale

        global bQuit
        bQuit = False  # quit flag for threads

        # Initialize ICAI pose networks
        ICAIPose_graph = xir.Graph.deserialize(ICAIPose_xmodel)
        ICAIPose_subgraphs = get_child_subgraph_dpu(ICAIPose_graph)
        assert len(ICAIPose_subgraphs) == 1 # only one DPU kernel
        all_dpu_runners = []
        for i in range(int(thread_count)):
            all_dpu_runners.append(vart.Runner.create_runner(ICAIPose_subgraphs[0], "run"))

        # taskAll(0, all_dpu_runners[0])

        # Init synchronous queues for inter-thread communication
        # Use queues that do not accumulate memory if input rate is higher than output rate
        queueIn = OverwriteQueue()
        queueOut = OverwriteQueue()

        # Create threads
        threadAll = []

        tc = threading.Thread(target=taskCapture, args=(video_capture_str, queueIn))
        threadAll.append(tc)

        for i in range(thread_count):
            tw = threading.Thread(target=taskWorker, args=(i, all_dpu_runners[i], queueIn, queueOut))
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


    def taskDisplay(self, queueOut):

        global bQuit

        print("[INFO] taskDisplay : starting thread ...")

        # Start the FPS counter
        fpsOut = FPS().start()

        # out = cv2.VideoWriter('appsrc ! videoconvert ! video/x-raw, format=NV12! kmssink driver-name=xlnx plane-id=39 sync=false fullscreen-overlay=true',0, 30.0, (self.width, self.height))
        
        while not bQuit:
            # Pop processed image from output queue

            frame = queueOut.get()

            # Crop image to match the screen size aspect ratio
            # h, w = frame.shape[:2]
            # new_h = int(w/self.width*self.height)
            # y_offset = (h - new_h) // 2
            # frame = frame[y_offset:y_offset+new_h]

            # # Mirror frame since the viewer might find a mirror view more intuitive
            frame = cv2.flip(frame, 1)

            # Rescale frame to the screen size
            frame = cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )

            # assert frame.shape[:2] == (self.height, self.width)
            # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
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


def main(argv):
    video_capture_str = "v4l2src device=/dev/video0 ! image/jpeg,framerate=10/1,width=960, height=540,type=video ! jpegdec ! videoconvert ! video/x-raw ! appsink"
    # video_capture_str = 'vid.mp4'
    thread_count = int(argv[1])
    ICAIPose_xmodel = argv[2]
    width = int(argv[3])
    height = int(argv[4])

    app = App(
        video_capture_str,
        width,
        height,
        thread_count,
        ICAIPose_xmodel,
        width_scale=1.4, #correct video aspect ratio
    )

if __name__ == "__main__":
    main(sys.argv)