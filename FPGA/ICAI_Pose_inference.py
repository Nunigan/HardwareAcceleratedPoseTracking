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

import vart
from pose_parameter import PoseParameter


class ICAIPose():

    def __init__(self, dpu):

        self.dpu = dpu

        self.inputTensors = []
        self.outputTensors = []
        self.inputShape = []
        self.outputShape = []
        self.drawObj = PoseDraw()
        self.color_part = self.drawObj.part_color
        self.color_limb = self.drawObj.limb_color
        self.limbs = self.drawObj.limbs


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

    def pre_process(self, img):

        img = img[:,210:750]

        img = cv2.resize(img, (256,256))

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
        return img

    def process(self,img):
        #print("[INFO] facedetect process")
        # t = time.time()

        im_org = img
        img = self.pre_process(img)
        dpu = self.dpu
        #print("[INFO] facedetect runner=",dpu)

        input_ndim = self.inputShape
        output_ndim = self.output0Shape
        # inputShape = (3, 256, 256, 3)
        # output0Shape = (3, 256, 256, 16)


        """ Prepare input/output buffers """
        #print("[INFO] process - prep input buffer ")
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        inputImage = inputData[0]
        inputImage[0,...] = img

        #print("[INFO] process - prep output buffer ")
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]


        """ Execute model on DPU """
        #print("[INFO] process - execute ")
        # print('pre',time.time()-t)
        # t = time.time()

        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)
        # print('dpu',time.time()-t)

        """ Retrieve output results """    
        #print("[INFO] process - get outputs ")
        # OutputData0 = outputData
        # np.save("test_out.npy", OutputData0)
        # t = time.time()
        # img_draw = self.post_old(outputData, im_org)

        # t = time.time()
        # conf = np.copy(outputData)
        # img_draw = self.post_conf(outputData).astype(np.float32)
        # img_conf = cv2.cvtColor(img_conf, cv2.COLOR_GRAY2RGB)
        img_draw = self.post_new(outputData, im_org)
        # img_draw = cv2.resize(img_draw, (256, 256))
        # img_conf = cv2.resize(img_conf, (256, 256))
        # img_draw = np.hstack((img_draw/255, img_conf))
        # h_img = cv2.hconcat([img_draw, img_conf])
        # Vars
        
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
        # np.save('conf.npy',conf[-1])
        # np.save('img.npy',img)
        org_size = np.shape(img)
        conf_map = np.moveaxis(conf[-1][0,:,:,:-1], [0,1,2], [1,2,0])
        keypoints = []
        conf_thresh = 15


        # for cm in conf_map:
        #     cm = np.where(cm>0.2, 1, 0)
        #     # cm = cm.astype(np.bool_)
        #     coordinates = peak_local_max(cm, min_distance=40, exclude_border=0, num_peaks=3)
        #     coordinates[:,0] = coordinates[:,0] * 1080/256
        #     coordinates[:,1] = coordinates[:,1] * 1920/256
        #     keypoints.append(coordinates.astype(np.int32)[...,::-1])

        for cm in conf_map:
            kp = []
            # cm[cm<20] = 0
            while True:
                max = np.unravel_index(np.argmax(cm), cm.shape)
                if cm[max] < conf_thresh:
                    break
                rr, cc = disk(max, 25, shape=(256,256))
                cm[rr, cc] = 0
                max = list(max)
                max[0] = max[0] * 540/256
                max[1] = max[1] * 540/256+210

                kp.append(max)
            keypoints.append((np.array(kp)).astype(np.int32)[...,::-1])

        cnt = 0
        for kps in keypoints:
            for kp in kps:
                img = cv2.circle(img, tuple(kp), 3, self.color_part[cnt], thickness=-1)
            cnt += 1
      

        for k, (a,b) in enumerate(self.limbs):
            pts = []
            flag = False
            dist_out = []
            for kp in keypoints[a]:
                if len(keypoints[a]) > len(keypoints[b]):
                    dist = []
                    flag = True
                    for j in range(len(keypoints[b])):
                        dist.append(np.linalg.norm(kp-keypoints[b][j]))
                        pts.append([kp, keypoints[b][j]])
                    dist_out.append(dist)
                else:
                    dist = []
                    for j in range(len(keypoints[b])):
                        dist.append(np.linalg.norm(kp-keypoints[b][j]))
                    pts.append([kp, keypoints[b][np.argmin(dist)]])
            if flag:
                pts = np.array(pts)
                pts = pts[np.argsort(np.array(dist_out).flatten())[:min(len(keypoints[a]), len(keypoints[b]))]]
                flag = False
            for p1, p2 in pts:
                if np.linalg.norm(p1-p2) < org_size[1]/3:
                    img = cv2.line(img, p1, p2, self.color_limb[k], 2)

        img = cv2.line(img, (210,0), (210,540), (255,255,255), 1)
        img = cv2.line(img, (750,0), (750,540), (255,255,255), 1)


        return img

    def post_old(self, conf, img):
        conf_thresh = 10
        org_size = np.shape(img)

        conf_map = np.moveaxis(conf[-1][0,:,:,:-1], [0,1,2], [1,2,0])
        kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
        kp_confidence = np.array([np.max(cm) for cm in conf_map])
        kp_location[kp_confidence < conf_thresh] = 0

        kp_location[:,0] = kp_location[:,0] * org_size[0]/256
        kp_location[:,1] = kp_location[:,1] * org_size[1]/256

        img_draw = self.drawObj.drawPose(img, kp_location[None])
        return img_draw
  

    def post_conf(self, conf):
        conf_map = np.moveaxis(conf[-1][0,:,:,:-1], [0,1,2], [1,2,0])
        # conf_map = np.where(conf_map<10,0,1)

        conf_map = np.sum(conf_map, axis=0)
        conf_map = (conf_map-np.min(conf_map))/(np.max(conf_map)-np.min(conf_map))
        return conf_map


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


def profile():
    from skimage.io import imread

    pose = ICAIPose('dpu_stub')

    conf = np.load('data/conf.npy')
    imgs = np.load('data/img.npy')
    conf = [conf]
    pose.post_new(conf,imgs)

if __name__ == "__main__":
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    profile()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()   
    # stats.dump_stats('data/cProfileExport')