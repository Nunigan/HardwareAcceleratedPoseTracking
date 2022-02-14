#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:44:56 2020

@author: nunigan
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import cv2

# image_list = []
# for filename in glob.glob('../images/*.jpg'): 
#     im=Image.open(filename)
#     # print(np.shape(im))
#     im = cv2.resize(np.array(im), (256,256))
#     im = im/np.max(im)*255
#     image_list.append(im)  
    
    
ims = np.load('../validation_data/imgs.npy')[:1000]   

#image = Image.open('out.jpg')

def calib_input(iter):
    print(iter)
    for i in range(100):
        im = np.array(ims[iter*100+i])
        im = np.reshape(im, (1,256,256,3))
    
    
    return {"input_1": im}

if __name__ == "__main__":
    calib_input(1)

