# -*- coding: utf-8 -*-
# @Author: Simon Walser
# @Date:   2021-11-22 08:26:53
# @Last Modified by:   Simon Walser
# @Last Modified time: 2021-11-22 15:05:08

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
import cv2
from skimage.transform import resize
from skimage.io import imread
import glob
from PIL import Image

from pose_parameter import PoseParameter

################################################################################
#
# Class / Function definitions
#
################################################################################

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

def preprocess_img(img, target_size):
    diff = np.max(img.shape[:-1]) - np.min(img.shape[:-1])
    border_1 = (diff + 1) // 2
    border_2 = diff // 2

    # Crop
    if np.argmax(img.shape[:-1]) == 0:
        img = img[border_1:img.shape[0]-border_2]
    elif np.argmax(img.shape[:-1]) == 1:
        img = img[:,border_1:img.shape[1]-border_2]
    else:
        raise Exception('img has invalid shape!')

    # Scaling
    img = resize(img, (target_size,target_size), anti_aliasing=True)

    # Correct dtype and shape
    if np.max(img) <= 1.0:
        img = np.float32(img * 255.0)[np.newaxis]
    else:
        img = np.float32(img)[np.newaxis]

    return img


def run_icaipose():

    img_size = 256
    conf_thresh = 0.05
    drawObj = PoseDraw()

    # Load model
    model = tf.keras.models.load_model('trained_model_ICAIPose/models/ICAIPose_best.h5')
    tf.keras.utils.plot_model(model, show_shapes=True)
    # Load validation image
    img = imread('out.jpg')
    img_pp = preprocess_img(img, img_size)

    # Run ICAIPose
    offset = np.array([103.939, 116.779, 123.68], dtype='float32')[None,None,None,:]
    # out = model(img_pp[...,::-1]-offset)

    out = np.load('out_FPGA.npy')


    conf_map = np.moveaxis(out[-1][0,:,:,:-1], [0,1,2], [1,2,0])
    kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
    kp_confidence = np.array([np.max(cm) for cm in conf_map])
    kp_location[kp_confidence < conf_thresh] = 0

    # Draw result
    img_draw = drawObj.drawPose(img_pp[0], kp_location[None])
    print(np.shape(img_pp[0]))
    print( kp_location[None])
    _, axis = plt.subplots(1,1)
    axis.imshow(img_draw/255)
    axis.axis('off')
    plt.show()
    model.save_weights('weights.h5')
    return img_draw


def run_Testbench():
    img_size = 256
    conf_thresh = 0.05
    drawObj = PoseDraw()

    # Load model
    model = tf.keras.models.load_model('trained_model_ICAIPose/models/ICAIPose_best.h5')

    offset = np.array([103.939, 116.779, 123.68], dtype='float32')[None,None,None,:]
    
    
    image_list = []
    image_list_pp = []
    for filename in glob.glob('images/singel/*.jpg'): 
        im=Image.open(filename)
        im = preprocess_img(np.array(im)/np.max(im)*255, img_size)
        image_list.append(np.array(im).reshape(256,256,3))
        im = im[...,::-1]-offset
        image_list_pp.append(im)

    res = []
    for i in range(12):
        out = model(image_list_pp[i])
    
        conf_map = np.moveaxis(out[-1][0,:,:,:-1], [0,1,2], [1,2,0])
        kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
        kp_confidence = np.array([np.max(cm) for cm in conf_map])
        kp_location[kp_confidence < conf_thresh] = 0
    
        # Draw result
        img_draw = drawObj.drawPose(image_list[i], kp_location[None])
        print(np.shape(image_list[i]))
        print( kp_location[None])
        _, axis = plt.subplots(1,1)
        axis.imshow(img_draw/255)
        axis.axis('off')
        plt.show()
        res.append(kp_location)

    return res

def compare_Testbenches():
    heat_FPGA = np.load('heatmaps_testbench.npy')
    kp_float = run_Testbench()
    
    kp_FPGA = []
    for i in range(12):
        conf_map = np.moveaxis(heat_FPGA[i][-1][0,:,:,:-1], [0,1,2], [1,2,0])
        kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
        kp_FPGA.append(kp_location)
    
    return kp_float, kp_FPGA
    
################################################################################
#
# Main functions
#
################################################################################


if __name__ == "__main__":
    # res = run_icaipose()
    # res = run_Testbench()
    kp_float, kp_FPGA = compare_Testbenches()