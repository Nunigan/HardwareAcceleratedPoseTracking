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
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.draw import disk, circle_perimeter
import matplotlib.pyplot as plt
import vart
from pose_parameter import PoseParameter


class ICAIPose():

    def __init__(self, dpu, width, height):

        self.dpu = dpu

        self.inputTensors = []
        self.outputTensors = []
        self.inputShape = []
        self.outputShape = []
        self.drawObj = PoseDraw()
        self.color_part = self.drawObj.part_color
        self.color_limb = self.drawObj.limb_color
        self.limbs = self.drawObj.limbs
        self.width = width # cam
        self.height = height # cam 
        self.crop_left = (self.width-self.height)//2
        self.crop_right = self.width - (self.width-self.height)//2

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

        self.input_ndim = self.inputShape
        self.randint = np.random.randint(0,255,size=(self.input_ndim[-3],self.input_ndim[-2]), dtype=np.uint16)

    def pre_process(self, img):

        img = cv2.resize(img, (self.input_ndim[-2],self.input_ndim[-3]))
        """ Image pre-processing """

        offset = np.array([103.939, 116.779, 123.68], dtype='float32')[None,None,None,:]
        img = (img[...,::-1]-offset)*0.5
        img = img.astype(np.int8)

        return img

    def process(self,img):
        #print("[INFO] facedetect process")

        dpu = self.dpu
        #print("[INFO] facedetect runner=",dpu)

        input_ndim = self.inputShape
        output_ndim = self.output0Shape


        im_org = img
        img = self.pre_process(img)
        """ Prepare input/output buffers """
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        inputImage = inputData[0]

        inputImage[0,...] = img

        #print("[INFO] process - prep output buffer ")
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]


        """ Execute model on DPU """
        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)

        # img_draw = self.post_conf(outputData).astype(np.float32)
        img_draw = self.post_new(outputData, im_org)

        return img_draw

    def stop(self):
        #"""Destroy Runner"""
        del self.dpu
      
        self.dpu = []
        self.inputTensors = []
        self.outputTensors = []
        self.inputShape = []
        self.outputShape = []


    def post_new(self, conf, img):
        org_size = np.shape(img)
        conf_map = np.moveaxis(conf[-1][0,:,:,:-1], [0,1,2], [1,2,0])
        conf_thresh = 40

        keypoints = []
        for cm in conf_map:

            cm = np.abs(cm*255).astype(np.uint16)+self.randint
            dil = cv2.dilate(cm, np.ones((10,10),dtype=np.uint8), 1)
            cor = np.nonzero(np.logical_and(cm == dil, cm>conf_thresh*255))
            kp = np.swapaxes(np.array(cor), 0,1).astype(np.int32)[...,::-1]
            if len(kp) != 0:
                kp[:,1] = kp[:,1] * self.height/self.input_ndim[-3]
                kp[:,0] = kp[:,0] * self.width/self.input_ndim[-2]
            keypoints.append(kp)

        n_pers = []
        sorted = []
        for kps in keypoints:
            sorted.append(kps[np.argsort(kps,axis=0)[:,0]])
            n_pers.append(len(kps))
        n_pers = np.max(n_pers)
        
        cnt = 0
        for kps in sorted:
            for kp in kps:
                img = cv2.circle(img, tuple(kp), 3, self.color_part[cnt], thickness=-1)
            cnt += 1

        for k, (a,b) in enumerate(self.limbs):
            if len(sorted[a]) == len(sorted[b]):
                for i in range(len(sorted[a])):
                    img = cv2.line(img, sorted[a][i], sorted[b][i], self.color_limb[k], 2)
            else:
                dist_out = []
                pts = []
                for kp in sorted[a]:
                    dist = []
                    for j in range(len(sorted[b])):
                        dist.append(np.linalg.norm(kp-sorted[b][j]))
                        pts.append([kp, sorted[b][j]])
                    dist_out.append(dist)
                pts = np.array(pts)
                pts = pts[np.argsort(np.array(dist_out).flatten())[:min(len(sorted[a]), len(sorted[b]))]]
                for p1, p2 in pts:
                    img = cv2.line(img, p1, p2, self.color_limb[k], 2)

        img = cv2.flip(img, 1)

        # img = np.repeat(img, 3, axis = 0)
        # img = np.repeat(img, 3, axis = 1)

        # img = cv2.resize(
        #     img,
        #     (720, 400),
        #     interpolation=cv2.INTER_NEAREST,
        # )

        return img

    def post_conf(self, conf):
        conf_map = np.moveaxis(conf[-1][0,:,:,:-1], [0,1,2], [1,2,0])
        conf_map = np.sum(conf_map, axis=0)
        conf_map = (conf_map-np.min(conf_map))/(np.max(conf_map)-np.min(conf_map))
        return conf_map


class PoseDraw():
    def __init__(self):
        poseParamObj    = PoseParameter()
        self.limbs      = poseParamObj.getLimbs()
        self.limb_color = poseParamObj.getLimbColor()
        self.part_color = poseParamObj.getPartColor()

