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
parser.add_argument('--anno', help='anno file path')
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

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img)
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    if img.max() > 1:
        img /= 255.
    return img


def color_normalize(x, mean, std):
    if x.shape[0] == 1:
        x = x.repeat(3, axis=0)
    for i in range(3):
        x[i,:] = x[i,:] - mean[i]
    return x

def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def crop(img, center, scale, res, rot=0):

    img = img.transpose([1,2,0])
    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return np.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else np.zeros(res[0], res[1])
        else:
            img = cv2.resize(img, (new_wd, new_ht), interpolation=cv2.INTER_LINEAR)
            center = center * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, (res[1], res[0]), interpolation=cv2.INTER_LINEAR)
    new_img = new_img.transpose([2,0,1])
    new_img = np.float32(new_img)
    if new_img.max() > 1:
        new_img /= 255.
    return new_img
    
def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1

class Mpii():
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path'] # root image folders
        self.jsonfile   = kwargs['anno_path']
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.label_type = kwargs['label_type']

        # create train/val split
        with open(self.jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train_list, self.valid_list = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid_list.append(idx)
            else:
                self.train_list.append(idx)
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        return np.array([0.4404, 0.4440, 0.4327]), np.array([0.2458, 0.2410, 0.2468])

    def __getitem__(self, index):
        if self.is_train:
            a = self.anno[self.train_list[index]]
        else:
            a = self.anno[self.valid_list[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = np.array(a['joint_self'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = np.array(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.shape[0]
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            pass

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)
        
        # Generate ground truth
        tpts = pts.copy()
        target = np.zeros([nparts, self.out_res, self.out_res])
        target_weight = tpts[:, 2].copy().reshape([nparts, 1])

        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r)
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index' : index, 'center' : c, 'scale' : s,
        'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)

if __name__ == '__main__':
    npoints = 16
    print('-----begin evaluation-----')
    mpii = Mpii(is_train=False, image_path=args.data, anno_path=args.anno, inp_res=256, out_res=64, sigma=1,label_type='Gaussian')
    number=0
    precision = np.zeros(npoints+1)
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    output_name = args.name 
    acc =0
    for i in range(len(mpii.valid_list)):
        img,label,label_txt=mpii.__getitem__(i)
        pts = np.array(label_txt['tpts'])
        
        weights = pts[:,2]
        net.blobs['data'].data[...] = img.reshape((1, 3, 256, 256))
        output=net.forward()
        maps = output[output_name]
        pred_coordinate = np.zeros((npoints, 2), dtype=np.float)
        label_coordinate = np.zeros((npoints, 2), dtype=np.float)
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
            
        for j in range(npoints):
            a = label[j,:,:]
            val = np.max(a)
            m = np.argmax(a)
                
            r, c = divmod(m, a.shape[1])
            if val <= 0:
                r = -1
                c = -1
            label_coordinate[j, 0] = c
            label_coordinate[j, 1] = r 
        px = label_coordinate[:,0]
        py = label_coordinate[:,1]
        
        threshold = 6.4
        temp_precision = np.zeros(npoints+1)
        number += 1
        for j in range(len(px)):
            if px[j] == -1 or py[j] == -1:
                temp_precision[j] = 0
                continue
            
            temp_precision[j] = math.sqrt((px[j]-pred_coordinate[j, 0]) ** 2 + (py[j]-pred_coordinate[j, 1]) ** 2) < 0.5 * threshold
            
            temp_precision[-1] += 1
        precision += temp_precision
        temp_acc = np.sum(temp_precision[:-1]) / temp_precision[-1]
        print("Hit: {}, PCK@0.5: {}".format(temp_precision,temp_acc))

    print('-----last results-----')
    print('Hits: {}'.format(precision))
    acc = np.sum(precision[:-1]) / precision[-1]
    print('PCK@0.5: {}'.format(acc) )