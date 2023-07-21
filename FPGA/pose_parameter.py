# @Author: Simon Walser
# @Date:   2021-04-13 17:35:26
# @Last Modified by:   Simon Walser
# @Last Modified time: 2021-11-22 09:11:16


################################################################################
#
# Class definitions
#
################################################################################

import numpy as np

class PoseParameter():
    def __init__(self):
        self.__body_parts  =  { 0:   "Nose",
                                1:   "Neck",
                                2:   "RShoulder",
                                3:   "RElbow",
                                4:    "RWrist",
                                5:   "LShoulder",
                                6:   "LElbow",
                                7:   "LWrist",
                                8:   "MidHip",
                                9:   "RHip",
                                10:  "RKnee",
                                11:  "LAnkle",
                                12:  "LHip",
                                13:  "LKnee",
                                14:  "LAnkle"}
        self.__limbs   = np.array([[0,1],
                                   [1,2],
                                   [2,3],
                                   [3,4],
                                   [1,5],
                                   [5,6],
                                   [6,7],
                                   [1,8],
                                   [8,9],
                                   [9,10],
                                   [10,11],
                                   [8,12],
                                   [12,13],
                                   [13,14]], dtype='uint16')
        self.__color_limb =          [(153,0,51),  # Neck
                                      (153,51,0),  # RShoulder
                                      (153,102,0), # RUpperArm
                                      (153,153,0), # RForearm
                                      (102,153,0), # LShoulder
                                      (51,153,0),  # LUpperArm
                                      (0,153,0),   # LForearm
                                      (153,0,0),   # Back
                                      (0,153,51),  # RHip
                                      (0,153,102), # RTight
                                      (0,153,153), # RLowerLeg
                                      (0,102,153), # LHip
                                      (0,51,153),  # LTight
                                      (0,0,153)]   # LLowerLeg
        self.__color_part =          [(153,0,51),  # Nose
                                      (153,0,0),   # Neck
                                      (153,51,0),  # RShoulder
                                      (153,102,0), # RElbow
                                      (153,153,0), # RWrist
                                      (102,153,0), # LShoulder
                                      (51,153,0),  # LElbow
                                      (0,153,0),   # LWrist
                                      (153,0,0),   # MidHip
                                      (0,153,51),  # RHip
                                      (0,153,102), # RKnee
                                      (0,153,153), # RAnkle
                                      (0,102,153), # LHip
                                      (0,51,153),  # LKnee
                                      (0,0,153)]   # LAnkle

    def getBodyParts(self, poseType='BODY_25'):
        return self.__body_parts

    def getLimbs(self, poseType='BODY_25'):
        return self.__limbs

    def getLimbColor(self):
        return self.__color_limb

    def getPartColor(self):
        return self.__color_part
