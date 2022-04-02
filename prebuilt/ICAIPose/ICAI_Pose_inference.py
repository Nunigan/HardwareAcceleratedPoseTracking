#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:40:18 2021

@author: nunigan
"""

from ctypes import *
import cv2
import os
import threading
import time
import sys
import numpy as np
from numpy import float32
import math

import vart
from pose_parameter import PoseParameter


class ICAIPose():
	  
  def __init__(self, dpu):

    self.dpu = dpu

    self.inputTensors = []
    self.outputTensors = []
    self.inputShape = []
    self.outputShape = []
 

  def start(self):

    dpu = self.dpu

    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    self.inputTensors = inputTensors
    self.outputTensors = outputTensors
    self.inputShape = input_ndim
    self.output0Shape = output_ndim



  def process(self,img):
    #print("[INFO] facedetect process")

    dpu = self.dpu
    #print("[INFO] facedetect runner=",dpu)

    input_ndim = self.inputShape
    output_ndim = self.output0Shape
    # inputShape = (3, 256, 256, 3)
    # output0Shape = (3, 256, 256, 16)

    im_org = img
    """ Image pre-processing """
    _B_MEAN = 103.939
    _G_MEAN = 116.779
    _R_MEAN = 123.6
    means = [_B_MEAN, _G_MEAN, _R_MEAN]
    scales = [1,1,1]
    R, G, B = cv2.split(img)
    B = (B - means[0]) * scales[0] * 0.5
    G = (G - means[1]) * scales[1] * 0.5
    R = (R - means[2]) * scales[2] * 0.5
    img = cv2.merge([B, G, R])
    img = img.astype(np.int8)
    """ Prepare input/output buffers """
    #print("[INFO] process - prep input buffer ")
    inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
    inputImage = inputData[0]
    inputImage[0,...] = img

    #print("[INFO] process - prep output buffer ")
    outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]


    """ Execute model on DPU """
    #print("[INFO] process - execute ")

    job_id = dpu.execute_async(inputData, outputData)

    dpu.wait(job_id)

    """ Retrieve output results """    
    #print("[INFO] process - get outputs ")
    OutputData0 = outputData
    np.save("test_out.npy", OutputData0)

    # Vars
    conf_thresh = 10
    drawObj = PoseDraw()

    conf_map = np.moveaxis(OutputData0[-1][0,:,:,:-1], [0,1,2], [1,2,0])
    kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
    kp_confidence = np.array([np.max(cm) for cm in conf_map])
    kp_location[kp_confidence < conf_thresh] = 0
    
    img_draw = drawObj.drawPose(im_org, kp_location[None])

    
    return img_draw

  def stop(self):
    #"""Destroy Runner"""
    del self.dpu
	
    self.dpu = []
    self.inputTensors = []
    self.outputTensors = []
    self.inputShape = []
    self.outputShape = []



class PoseDraw():
    def __init__(self):
        poseParamObj    = PoseParameter()
        self.limbs      = poseParamObj.getLimbs()
        self.limb_color = poseParamObj.getLimbColor()
        self.part_color = poseParamObj.getPartColor()

    def drawPose(self, inputImg, data):
        data_int = np.int32(data[...,::-1])

        # Draw limbs
        for data_pers in data_int:
            for i, pair in enumerate(self.limbs):
                if np.all(data_pers[pair]):
                    inputImg = cv2.line(inputImg, tuple(data_pers[pair[0]]),
                                        tuple(data_pers[pair[1]]),
                                        self.limb_color[i], 3)

        # Draw body parts
        non_zero_data = data_int[np.all(data_int, axis=-1)]
        idx = np.tile(np.arange(data_int.shape[1])[np.newaxis], (data_int.shape[0],1))[np.all(data_int, axis=-1)]
        for i, data_kp in zip(idx, non_zero_data):
            inputImg = cv2.circle(inputImg, tuple(data_kp), 5, self.part_color[i],
                                thickness=-1)

        return inputImg