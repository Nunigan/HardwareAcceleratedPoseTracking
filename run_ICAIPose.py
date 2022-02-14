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
from matplotlib.widgets import Button
from pose_parameter import PoseParameter
import sys
from scipy import signal
import cmapy
import gc
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
        self.parts = poseParamObj.getBodyParts()

    def drawPose(self, inputImg, data):
        data_int = np.int32(data[...,::-1])

        # Draw limbs
        for data_pers in data_int:
            for i, pair in enumerate(self.limbs):
                if np.all(data_pers[pair]):
                    inputImg = cv2.line(inputImg, tuple(data_pers[pair[0]]),
                                        tuple(data_pers[pair[1]]),
                                        self.limb_color[i], 30)

        # Draw body parts
        non_zero_data = data_int[np.all(data_int, axis=-1)]
        idx = np.tile(np.arange(data_int.shape[1])[np.newaxis], (data_int.shape[0],1))[np.all(data_int, axis=-1)]
        for i, data_kp in zip(idx, non_zero_data):
            inputImg = cv2.circle(inputImg, tuple(data_kp),45, self.part_color[i],
                                thickness=-1)

        return inputImg


    def drawPose_multi(self, inputImg, data_in, upscale, n):        
    
        colors = [(235,5,142 ),(255,255 ,0),(55,255,153),(254,212,231)]
        device = ['FPGA', 'Truth', 'Leaky', 'Prelu']
        
        inputImg = cv2.resize(inputImg, (256*upscale, 256*upscale))
        for j in range(4):
            data = data_in[j]
            data_int = upscale*np.int32(data[...,::-1])

            # Draw limbs
            for data_pers in data_int:
                for i, pair in enumerate(self.limbs):
                    if np.all(data_pers[pair]):
                        inputImg = cv2.line(inputImg, tuple(data_pers[pair[0]]),
                                            tuple(data_pers[pair[1]]),
                                            colors[j], 4)
    
            # Draw body parts
            non_zero_data = data_int[np.all(data_int, axis=-1)]
            idx = np.tile(np.arange(data_int.shape[1])[np.newaxis], (data_int.shape[0],1))[np.all(data_int, axis=-1)]
            for i, data_kp in zip(idx, non_zero_data):
                inputImg = cv2.circle(inputImg, tuple(data_kp), 6, colors[j],
                                    thickness=-1)
                
                
            inputImg = cv2.circle(inputImg, (upscale*10, upscale*(10+j*10)), 10, colors[j],thickness=-1)
            inputImg = cv2.putText(inputImg, device[j] , (upscale*10+15, upscale*(12+j*10)), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[j], 4, cv2.LINE_AA)
        # inputImg = cv2.putText(inputImg, 'image ' +str(n) , (upscale*10+150, upscale*(12)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4, cv2.LINE_AA)

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
    conf_thresh = 0.00
    drawObj = PoseDraw()
    
    parts = drawObj.parts
    # Load model
    model = tf.keras.models.load_model('trained_model/ICAIPose/models/ICAIPose_best.h5')
    tf.keras.utils.plot_model(model, show_shapes=True)
    # Load validation image
    # img = imread('out.jpg')
    img = imread('test_img.jpg')

    img_pp = preprocess_img(img, img_size)

    # Run ICAIPose
    offset = np.array([103.939, 116.779, 123.68], dtype='float32')[None,None,None,:]
    out = model(img_pp[...,::-1]-offset)



    conf_map = np.moveaxis(out[-1][0,:,:,:-1], [0,1,2], [1,2,0])
    kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
    kp_confidence = np.array([np.max(cm) for cm in conf_map])
    kp_location[kp_confidence < conf_thresh] = 0

    # Draw result
    img_draw = drawObj.drawPose(img, 331/32*kp_location[None])
    _, axis = plt.subplots(1,1)
    axis.imshow(img_draw/255)
    axis.axis("off")

    plt.rcParams['font.family'] = 'STIXGeneral'
    fig, axis = plt.subplots(3,5)
    # axis.imshow(img_draw/255)
    for i in range(3):
        for j in range(5):
            axis[i,j].imshow(conf_map[j+5*i])
            axis[i,j].axis('off')
            axis[i,j].set_title(parts[j+5*i], fontsize=25)
    plt.figure()
    fig.tight_layout()
    plt.imshow(np.sum(conf_map, axis=0))
    plt.axis("off")
    plt.imsave('out_conf.png', np.sum(conf_map, axis=0))
    plt.imsave('out_ICAI.png', img_draw/255)

    # axis.axis('off')
    plt.tight_layout()
    plt.show()
    # model.save_weights('weights.h5')
    return img_draw


def run_Testbench():
    img_size = 256
    conf_thresh = 0.05
    drawObj = PoseDraw()


    offset = np.array([103.939, 116.779, 123.68], dtype='float32')[None,None,None,:]

    
    imgs = np.load('validation_data/imgs.npy')
    imgs_pp = (imgs[...,::-1]-offset)

    # Load model
    model = tf.keras.models.load_model('trained_model/ICAIPose/models/ICAIPose_best.h5')
    # model = tf.keras.models.load_model('/media/nunigan/SSD2/MSE/PA/PA_HardwareAcceleratedPoseTracking/code/ICAIPose_3/trained_model/ICAIPose/models/Prelu/ICAIPose_PReLU_each.h5')

    confs_inference = []
    kps = []
    for i in range(len(imgs)):
        out = model(imgs_pp[i].reshape(1,256,256,3))
        
        conf_map = np.moveaxis(out[-1][0,:,:,:-1], [0,1,2], [1,2,0])
        kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
        kp_confidence = np.array([np.max(cm) for cm in conf_map])
        kp_location[kp_confidence < conf_thresh] = 0
    
        # Draw result
        # img_draw = drawObj.drawPose(imgs[i], kp_location[None])
        # _, axis = plt.subplots(1,1)
        # axis.imshow(img_draw/255)
        # axis.axis('off')
        # plt.show()
        confs_inference.append(conf_map)
        kps.append(kp_location)
    return kps

def compare_Testbenches():
    drawObj = PoseDraw()
    heat_FPGA = np.load('heatmaps_testbench.npy')
    kp_float = run_Testbench()
    kp_FPGA = []
    for i in range(12):
        conf_map = np.moveaxis(heat_FPGA[i][-1][0,:,:,:-1], [0,1,2], [1,2,0])
        kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf_map])
        kp_FPGA.append(kp_location)
    
    for i in range(12):
        img = drawObj.drawPose(np.zeros((256,256,3)),kp_float[i][None]) + drawObj.drawPose(np.zeros((256,256,3)),kp_FPGA[i][None])
        plt.figure()
        plt.imshow(img)
        
    
    mse = np.mean(((np.array(kp_float)[:,:,0] - np.array(kp_FPGA)[:,:,0])**2 + (np.array(kp_float)[:,:,1] - np.array(kp_FPGA)[:,:,1])**2 )**0.5)
    mse_full = (((np.array(kp_float)[:,:,0] - np.array(kp_FPGA)[:,:,0])**2 + (np.array(kp_float)[:,:,1] - np.array(kp_FPGA)[:,:,1])**2 )**0.5)
    print(mse_full)
    return mse_full

def computre_MSE():

    confs_y = np.load('validation_data/confs_norm.npy')
    confs_x = np.load('validation_data/confs_FPGA_norm.npy')
    
    mse_FPGA = ((confs_y-confs_x)**2).mean()
    
    confs_x = None
    del confs_x
    gc.collect()

    confs_x = np.load('validation_data/confs_prelu_norm.npy')
    mse_perlu = ((confs_y-confs_x)**2).mean()

    confs_x = None
    del confs_x
    gc.collect()

    confs_x = np.load('validation_data/confs_Leaky_norm.npy')
    mse_leaky = ((confs_y-confs_x)**2).mean()
    
    confs_x = None
    del confs_x
    gc.collect()

    return mse_FPGA, mse_perlu, mse_leaky
def computre_kp_from_confs():

    
    confs_y = np.load('validation_data/confs_FPGA.npy')
    # confs_y = confs_y[:,:,:,:-1]
    # confs = (confs - confs.mean())/np.std(confs)

    kp =[]
    for i in range(len(confs_y)):
        # conf_map = np.moveaxis([i][-1][0,:,:,:-1], [0,1,2], [1,2,0])
        print(np.shape(confs_y[i]))
        conf = np.moveaxis(confs_y[i], 2,0)
        # conf = confs_y[i]
        print(np.shape(conf))

        kp_location = np.array([np.unravel_index(np.argmax(cm), cm.shape) for cm in conf])
        kp.append(kp_location)
    

    
    # imgs = np.load('validation_data/imgs.npy')
    # pafs = np.load('validation_data/pafs.npy')
    # imgs_pp = (imgs[...,::-1]-offset)
    
    
    return kp, confs_y

def MSE_kp():
    kp_FPGA = np.load('validation_data/kp_FPGA.npy')
    kp_truth = np.load('validation_data/kp_truth.npy')
    kp_leaky = np.load('validation_data/kp_Leaky.npy')
    kp_perlu = np.load('validation_data/kp_prelu.npy')
    
    
    print(np.shape(kp_FPGA))
    mse_FPGA = np.mean(((np.array(kp_truth)[:,:,0]  - np.array(kp_FPGA)[:,:,0])**2  + (np.array(kp_truth)[:,:,1] - np.array(kp_FPGA)[:,:,1])**2 )**0.5)
    mse_leaky = np.mean(((np.array(kp_truth)[:,:,0] - np.array(kp_leaky)[:,:,0])**2 + (np.array(kp_truth)[:,:,1] - np.array(kp_leaky)[:,:,1])**2 )**0.5)
    mse_prelu = np.mean(((np.array(kp_truth)[:,:,0] - np.array(kp_perlu)[:,:,0])**2 + (np.array(kp_truth)[:,:,1] - np.array(kp_perlu)[:,:,1])**2 )**0.5)

    mse_FPGA_full = ((np.array(kp_truth)[:,:,0]  - np.array(kp_FPGA)[:,:,0])**2  + (np.array(kp_truth)[:,:,1] - np.array(kp_FPGA)[:,:,1])**2 )**0.5
    mse_leaky_full = ((np.array(kp_truth)[:,:,0] - np.array(kp_leaky)[:,:,0])**2 + (np.array(kp_truth)[:,:,1] - np.array(kp_leaky)[:,:,1])**2 )**0.5
    mse_prelu_full = ((np.array(kp_truth)[:,:,0] - np.array(kp_perlu)[:,:,0])**2 + (np.array(kp_truth)[:,:,1] - np.array(kp_perlu)[:,:,1])**2 )**0.5    

    return mse_FPGA, mse_leaky, mse_prelu,mse_FPGA_full, mse_leaky_full, mse_prelu_full

def plots():
    drawObj = PoseDraw()
    
    imgs = np.load('validation_data/imgs.npy')
    kp_FPGA = np.load('validation_data/kp_FPGA.npy')
    kp_truth = np.load('validation_data/kp_truth.npy')
    kp_leaky = np.load('validation_data/kp_Leaky.npy')
    kp_perlu = np.load('validation_data/kp_prelu.npy')
    
    offset = np.array([103.939, 116.779, 123.68], dtype='float32')[None,None,:]

    # img_draw = drawObj.drawPose(imgs[n], kp_FPGA[n][None])
    # img_draw = drawObj.drawPose_multi(np.ones((1024,1024,3)),[kp_FPGA[n][None],kp_truth[n][None],kp_leaky[n][None],kp_perlu[n][None]])
    
    ims = []
    for i in range(2142):
        im = drawObj.drawPose_multi(imgs[i],[kp_FPGA[i][None],kp_truth[i][None],kp_leaky[i][None],kp_perlu[i][None]], 6, i).astype(np.uint8())
        im = np.flip(im, axis=-1) 
        ims.append(im)
    
    no_of_images = 500
    image_id = 0     # initially we are starting from 0th image (1st image)

    while image_id < no_of_images:  # iterating until we get at the end of the images list
        backward, forward = False, False   # they acts like a flag to store whether we are moving to next or previous image

        height = 1024   # height of our resized image      

        
        '''
        To add blur effect
        '''
        key, pause_key = None, None  # to store the key entered by user while slideshow

        if pause_key == ord('a') and image_id != 0: # If 'a' pressed (Previous Image)
            image_id -= 1    # decrementing image_id to get previous image id
            continue
        elif pause_key == ord('d'):  # If 'd' pressed (Next Image)
            image_id += 1   # incrementing image_id to get next image id
            continue

        cv2.imshow('Slideshow', ims[image_id])  # displaying clear image

        key = cv2.waitKey(10000000)   # taking key from user with 1000 ms delay
        
        if key == ord('q'):  # If 'q' pressed (User wants to quit when slideshow is displaying clear image)
            sys.exit(0)
        elif key == ord('s'):  # If 's' pressed (User wants to pause when slideshow is displaying clear image)
            pause_key = cv2.waitKey()
            if pause_key == ord('q'):   # If 'q' pressed (User wants to quit when slideshow is paused)
                sys.exit(0)
            elif key == ord('a') or key == ord('d'): # if user wants to go to next or previous image
                    break
        if key == ord('a') and image_id != 0: # If 'a' pressed (Previous Image)
            image_id -= 1    # decrementing image_id to get previous image id
            continue

        elif key == ord('d'):  # If 'd' pressed (Next Image)
            image_id += 1   # incrementing image_id to get next image id
            continue
            
        image_id += 1 # If no keys are pressed, then image_id incremented for next image
    cv2.destroyAllWindows() # when work id done, closing windows


def plot_conf():
    
    # ims_1 = np.load('validation_data/imgs.npy')
    # ims = np.copy(ims_1[:500]).astype(np.uint8)
    # ims_1 = None
    # del ims_1
    # gc.collect()

    ims_FPGA_1 = np.load('validation_data/confs_FPGA_new_norm.npy')
    ims_FPGA = np.copy(ims_FPGA_1[:500])
    ims_FPGA_1 = None
    del ims_FPGA_1
    gc.collect()
    ims_truth_1 = np.load('validation_data/confs_norm.npy')
    ims_truth = np.copy(ims_truth_1[:500])
    ims_truth_1 = None
    del ims_truth_1
    gc.collect()
    ims_Leaky_1 = np.load('validation_data/confs_Leaky_norm.npy')
    ims_Leaky = np.copy(ims_Leaky_1[:500])
    ims_Leaky_1 = None
    del ims_Leaky_1
    gc.collect()
    ims_prelu_1 = np.load('validation_data/confs_prelu_norm.npy')
    ims_prelu = np.copy(ims_prelu_1[:500])
    ims_prelu_1 = None
    del ims_prelu_1
    gc.collect()
    ims_FPGA = np.sum(ims_FPGA, axis=3)
    ims_truth= np.sum(ims_truth, axis=3)
    ims_Leaky = np.sum(ims_Leaky, axis=3)
    ims_prelu = np.sum(ims_prelu, axis=3)

    
    
    
    no_of_images = 500
    image_id = 0     # initially we are starting from 0th image (1st image)

    while image_id < no_of_images:  # iterating until we get at the end of the images list
        backward, forward = False, False   # they acts like a flag to store whether we are moving to next or previous image

        height = 1024   # height of our resized image      
        
        '''
        To add blur effect
        '''
        key, pause_key = None, None  # to store the key entered by user while slideshow

        if pause_key == ord('a') and image_id != 0: # If 'a' pressed (Previous Image)
            image_id -= 1    # decrementing image_id to get previous image id
            continue
        elif pause_key == ord('d'):  # If 'd' pressed (Next Image)
            image_id += 1   # incrementing image_id to get next image id
            continue
        
        
        fpga = ims_FPGA[image_id]
        fpga[fpga<0] = 0
        fpga = fpga/np.max(fpga)*255
        
        truth = ims_truth[image_id]
        truth[truth<0] = 0
        truth = truth/np.max(truth)*255
        
        leaky = ims_Leaky[image_id]
        leaky[leaky<0] = 0
        leaky = leaky/np.max(leaky)*255
        
        prelu = ims_prelu[image_id]
        prelu[prelu<0] = 0
        prelu = prelu/np.max(prelu)*255
        
        diff_fpga = (fpga-truth)**2
        diff_fpga = diff_fpga/np.max(diff_fpga)*255
        diff_leaky = (leaky-truth)**2
        diff_leaky = diff_leaky/np.max(diff_leaky)*255
        diff_prelu = (prelu-truth)**2
        diff_prelu = diff_prelu/np.max(diff_prelu)*255
        
        combined_fpga = np.c_[fpga,truth,diff_fpga]
        combined_leaky = np.c_[leaky,truth,diff_leaky]
        combined_prelu = np.c_[prelu,truth,diff_prelu]
        
        combined = np.r_[combined_fpga, combined_leaky, combined_prelu]

        combined = cv2.resize(combined, (2666,2000)).astype(np.uint8)
        
        
        
        
        window_name='Slideshow'
        img_colorized = cv2.applyColorMap(combined, cmapy.cmap('viridis'))
        
        img_colorized = cv2.putText(img_colorized, 'image ' +str(image_id), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        
        # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, img_colorized)  # displaying clear image

        key = cv2.waitKey(1000000)   # taking key from user with 1000 ms delay
        
        if key == ord('q'):  # If 'q' pressed (User wants to quit when slideshow is displaying clear image)
            sys.exit(0)
        elif key == ord('s'):  # If 's' pressed (User wants to pause when slideshow is displaying clear image)
            pause_key = cv2.waitKey()
            if pause_key == ord('q'):   # If 'q' pressed (User wants to quit when slideshow is paused)
                sys.exit(0)
            elif key == ord('a') or key == ord('d'): # if user wants to go to next or previous image
                    break
        if key == ord('a') and image_id != 0: # If 'a' pressed (Previous Image)
            image_id -= 1    # decrementing image_id to get previous image id
            continue
        elif key == ord('d'):  # If 'd' pressed (Next Image)
            image_id += 1   # incrementing image_id to get next image id
            continue
            
        image_id += 1 # If no keys are pressed, then image_id incremented for next image
    cv2.destroyAllWindows() # when work id done, closing windows
    
def norm_confs():
    ims = np.load('validation_data/confs_FPGA_new.npy')
    # ims = np.moveaxis(ims, 1,3)

    for i in range(2142):
        for j in range(15):
            ims[i,:,:,j] = (ims[i,:,:,j]-ims[i,:,:,j].mean())/(ims[i,:,:,j].std()+0.000000000001)
    np.save('validation_data/confs_FPGA_new_norm.npy', ims)     

def plot_images():
    imgs = np.load('validation_data/imgs.npy')
    confs = np.load('validation_data/confs.npy')
    confs = np.sum(confs, axis=3)
    imgs = imgs.astype(np.uint8) 

    
    fig, ax = plt.subplots(2,4)
    for j in range(0,2):
        ax[j,0].imshow(imgs[2*j])
        ax[j,1].imshow(confs[2*j])
        ax[j,0].axis('off')
        ax[j,1].axis('off')
        ax[j,2].imshow(imgs[2*j+1])
        ax[j,3].imshow(confs[2*j+1])
        ax[j,2].axis('off')
        ax[j,3].axis('off')

    plt.tight_layout()
    
    
def plot_ski():
    
    # ims_1 = np.load('validation_data/imgs.npy')
    # ims = np.copy(ims_1[:500]).astype(np.uint8)
    # ims_1 = None
    # del ims_1
    # gc.collect()

    ims_FPGA_1 = np.load('validation_data/confs_FPGA_new_norm.npy')
    ims_FPGA = np.copy(ims_FPGA_1[:500])
    ims_FPGA_1 = None
    del ims_FPGA_1
    gc.collect()
    ims_truth_1 = np.load('validation_data/confs_norm.npy')
    ims_truth = np.copy(ims_truth_1[:500])
    ims_truth_1 = None
    del ims_truth_1
    gc.collect()
    ims_Leaky_1 = np.load('validation_data/confs_Leaky_norm.npy')
    ims_Leaky = np.copy(ims_Leaky_1[:500])
    ims_Leaky_1 = None
    del ims_Leaky_1
    gc.collect()
    ims_prelu_1 = np.load('validation_data/confs_prelu_norm.npy')
    ims_prelu = np.copy(ims_prelu_1[:500])
    ims_prelu_1 = None
    del ims_prelu_1
    gc.collect()
    ims_FPGA = np.sum(ims_FPGA, axis=3)
    ims_truth= np.sum(ims_truth, axis=3)
    ims_Leaky = np.sum(ims_Leaky, axis=3)
    ims_prelu = np.sum(ims_prelu, axis=3)
    
    
         
    fpga = ims_FPGA[104]
    fpga[fpga<0] = 0
    fpga = fpga/np.max(fpga)*255
    
    truth = ims_truth[104]
    truth[truth<0] = 0
    truth = truth/np.max(truth)*255
    
    leaky = ims_Leaky[104]
    leaky[leaky<0] = 0
    leaky = leaky/np.max(leaky)*255
    
    prelu = ims_prelu[104]
    prelu[prelu<0] = 0
    prelu = prelu/np.max(prelu)*255
    
    diff_fpga = (fpga-truth)**2
    diff_fpga = diff_fpga/np.max(diff_fpga)*255
    diff_leaky = (leaky-truth)**2
    diff_leaky = diff_leaky/np.max(diff_leaky)*255
    diff_prelu = (prelu-truth)**2
    diff_prelu = diff_prelu/np.max(diff_prelu)*255
    
    combined_fpga = np.c_[fpga,truth,diff_fpga]
    combined_leaky = np.c_[leaky,truth,diff_leaky]
    combined_prelu = np.c_[prelu,truth,diff_prelu]
    
    combined = np.r_[combined_fpga, combined_leaky, combined_prelu]

    combined = cv2.resize(combined, (2666,2000)).astype(np.uint8)
    
    fig, ax = plt.subplots(1,3, figsize=(13,6))
    plt.rcParams['font.family'] = 'STIXGeneral'

    ax[0].imshow(fpga)
    ax[0].axis('off')
    ax[0].set_title('Conf Map Leaky ReLU', fontsize=19)
    
    ax[1].imshow(truth)
    ax[1].axis('off')
    ax[1].set_title('Conf Map Ground Truth', fontsize=19)

    ax[2].imshow(diff_fpga)
    ax[2].axis('off')
    ax[2].set_title('MSE', fontsize=19)
    plt.tight_layout()

################################################################################
#
# Main functions
#
################################################################################


if __name__ == "__main__":
    res = run_icaipose()
    # mse_FPGA, mse_perlu, mse_leaky = computre_MSE()
    # confs_inference = run_Testbench()
    # mse_full = compare_Testbenches()
    # kp, conf = computre_kp_from_confs()
    # f,l,p, f_f, l_f, p_f = MSE_kp()
    # plots() 
    # plot_conf()
    # norm_confs()
    # plot_images()
    # plot_ski()