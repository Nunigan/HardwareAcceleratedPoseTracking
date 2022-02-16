# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

import os
import os.path as osp
import sys
import numpy as np
import math
import json
import scipy
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='anno image path')
parser.add_argument('--caffe', help='load a model for training or evaluation')
parser.add_argument('--cpu', action='store_true', help='CPU ONLY')
parser.add_argument('--weights', help='weights path')
parser.add_argument('--model', help='model path')
parser.add_argument('--output', help='anno file path')
parser.add_argument('--name', help='output feature map name, default ConvNd_56', default='ConvNd_56')
parser.add_argument('--input', help='input name in the first layer', default='data')
args = parser.parse_args()

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
# Add caffe to PYTHONPATH
caffe_path = osp.join(args.caffe, 'python')
add_path(caffe_path)
import caffe

if args.cpu:
    caffe.set_mode_cpu()


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 0, 0], [0, 0, 0], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def color_normalize(x, mean, std):
    if x.shape[0] == 1:
        x = x.repeat(3, axis=0)
    x = np.float32(x)
    if x.max() > 1:
        x = x / 255.0
    for i in range(3):
        x[i,:] = x[i,:] - mean[i]
    return x

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]
    
    pad = 4 * [None]
    pad[0] = 0 if (h % stride == 0) else int((stride - (h % stride)) / 2)
    pad[1] = 0 if (w % stride == 0) else int((stride - (w % stride)) / 2)
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) - pad[0]
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) - pad[1]
    
    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis = 0)
    pad_left = np.tile(img_padded[:,0:1,:] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis = 1)
    pad_down = np.tile(img_padded[-2:-1,:,:] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis = 0)
    pad_right = np.tile(img_padded[:,-2:-1,:] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis = 1)
    
    return img_padded, pad

def resize(img, size):
    w,h = img.shape[0], img.shape[1]
    l = w if w > h else h
    resize_ratio = size / float(l)
    new_w = int(w * resize_ratio)
    new_h = int(h * resize_ratio)
    return cv2.resize(img, (new_h, new_w)), resize_ratio


if __name__ == '__main__':
    npoints = 16
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    output_name = args.name 
    img = load_image(args.data)
    img, resize_ratio = resize(img, 240)
    img, pad = padRightDownCorner(img, 256, 0)
    src_img = cv2.imread(args.data)
    #src_img = img.copy()
    img = color_normalize(img, [0.4404, 0.4440, 0.4327],[1,1,1])
    img = img.transpose([2,0,1])
    img_shape = img.shape
    net.blobs[args.input].reshape(1,img_shape[0],img_shape[1],img_shape[2])
    net.blobs[args.input].data[...] = img.reshape([1, img_shape[0], img_shape[1], img_shape[2]])
    output = net.forward()
    maps = output[args.name]
    pred_coordinate = np.zeros((npoints, 2), dtype=np.float)
    for j in range(npoints):
        a = maps[0,j,:,:]
        m = np.argmax(a)
        r, c = divmod(m, a.shape[1])
        val = np.max(a)
        if val <= 0:
            r = 0
            c = 0
        
        pred_coordinate[j, 0] = c 
        pred_coordinate[j, 1] = r
    print(pred_coordinate)
    for i in range(npoints):
        cv2.circle(src_img, (int((pred_coordinate[i,0]*4 - pad[1])/resize_ratio), int((pred_coordinate[i,1]*4-pad[0])/resize_ratio)), 4, colors[i], thickness=-1)
        #cv2.circle(src_img, (int(pred_coordinate[i,0]*4), int(pred_coordinate[i,1]*4)), 4, colors[i], thickness=-1)

    cv2.imwrite(args.output,src_img)
    print('success!')