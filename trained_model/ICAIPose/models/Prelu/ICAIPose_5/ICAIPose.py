# @Author: Simon Walser
# @Date:   2021-09-07 15:12:42
# @Last Modified by:   Simon Walser
# @Last Modified time: 2021-12-22 09:29:43

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model

################################################################################
#
# Class / Function Definitions
#
################################################################################



class InputBlock:
    def __init__(self):
        self.bias = tf.constant([[[[103.939, 116.779, 123.68]]]], dtype=tf.float32)
        # self.weight_path = '/workspace/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.weight_path = 'imagenet'
        self.__name__ = 'backbone_'

    def __call__(self, inputs):

        # x = Lambda(lambda inp: tf.stack([inp[...,2],inp[...,1],inp[...,0]], axis=-1))(inputs)
        # x = Lambda(lambda inp: tf.math.subtract(inp, self.bias))(x)
        x = tf.reverse(inputs, axis=[-1])
        x = tf.math.subtract(x, self.bias)
        backbone = VGG19(include_top=False, weights=self.weight_path, input_tensor=x)

        # Remove last ReLU activation function
        backbone.get_layer('block3_conv4').activation = tf.keras.activations.linear

        output = backbone.get_layer('block3_conv4').output

        return output



class DenseBlock:
    def __init__(self, channel_number, name):
        self.channel_number = channel_number
        self.name = name

    def __call__(self, inputs):

        x_1 = Conv2D(self.channel_number, 3, padding='same', name=self.name+'conv0')(inputs)
        x_1 = PReLU(name=self.name+'prelu0', shared_axes=[1, 2])(x_1)

        x_2 = Conv2D(self.channel_number, 3, padding='same', name=self.name+'conv1')(x_1)
        x_2 = PReLU(name=self.name+'prelu1', shared_axes=[1, 2])(x_2)

        x_3 = Conv2D(self.channel_number, 3, padding='same', name=self.name+'conv2')(x_2)
        x_3 = PReLU(name=self.name+'prelu2', shared_axes=[1, 2])(x_3)

        out = Concatenate(name=self.name+'concat0')([x_1,x_2,x_3])

        return out


class DetectionBlock:
    def __init__(self, channel_number, name):
        self.channel_number = channel_number
        self.name = name

    def __call__(self, inputs):

        x = DenseBlock(self.channel_number[0], name=self.name+'den0_')(inputs)
        x = DenseBlock(self.channel_number[0], name=self.name+'den1_')(x)
        x = DenseBlock(self.channel_number[0], name=self.name+'den2_')(x)
        x = DenseBlock(self.channel_number[0], name=self.name+'den3_')(x)
        x = DenseBlock(self.channel_number[0], name=self.name+'den4_')(x)

        x   = Conv2D(self.channel_number[1], 1, padding='same', name=self.name+'convexp')(x)
        x   = PReLU(name=self.name+'preluexp', shared_axes=[1, 2])(x)
        out = Conv2D(self.channel_number[2], 1, padding='same', name=self.name+'convproj')(x)

        return out


class OpenPose:
    def __init__(self, paf_channels, conf_channels, width_1, width_2, width_3, width_4, name=''):
        self.paf_channels = paf_channels
        self.conf_channels = conf_channels
        self.width_1 = width_1
        self.width_2 = width_2
        self.width_3 = width_3
        self.width_4 = width_4

        self.name = name

    def __call__(self, inputs):
        x = InputBlock()(inputs)
        x = PReLU(name=self.name+'prelubb', shared_axes=[1, 2])(x)

        x = Conv2D(256, 3, padding='same', name=self.name+'convstem0')(x)
        x = PReLU(name=self.name+'prelustem0', shared_axes=[1, 2])(x)
        x = Conv2D(128, 3, padding='same', name=self.name+'convstem1')(x)
        in_1 = PReLU(name=self.name+'prelustem1', shared_axes=[1, 2])(x)

        paf_1 = DetectionBlock(self.width_1+[self.paf_channels], name=self.name+'det0_')(in_1)

        in_2  = Concatenate(name=self.name+'conc1')([in_1,paf_1])
        paf_2 = DetectionBlock(self.width_2+[self.paf_channels], name=self.name+'det1_')(in_2)

        in_3  = Concatenate(name=self.name+'conc2')([in_1,paf_2])
        conf_1 = DetectionBlock(self.width_3+[self.conf_channels], name=self.name+'det2_')(in_3)

        in_4  = Concatenate(name=self.name+'conc3')([in_1,conf_1,paf_2])
        conf_2 = DetectionBlock(self.width_4+[self.conf_channels], name=self.name+'det3_')(in_4)

        # Upsampling
        up_1_1 = UpSampling2D((2,2), interpolation='bilinear', name=self.name+'upsampl0')(conf_2)
        up_1_2 = DepthwiseConv2D(4, 1, 'same', name=self.name+'upsamplconv0')(up_1_1)

        up_2_1 = UpSampling2D((2,2), interpolation='bilinear', name=self.name+'upsampl1')(up_1_2)
        conf_up = DepthwiseConv2D(4, 1, 'same', name=self.name+'upsamplconv1')(up_2_1)


        return paf_1, paf_2, conf_1, conf_2, conf_up


def get_model(config):
    img_res = config.pop('img_res')

    input = tf.keras.Input(shape=(img_res,img_res,3))
    out = OpenPose(**config)(input)
    model = Model(inputs=input, outputs=out)

    return model


################################################################################
#
# Main Functions
#
################################################################################

if __name__ == "__main__":
    from skimage.io import imread

    test_img = imread('../val_img.jpg')

    config = {}
    config['img_res'] = 256
    config['width_1'] = [96, 256] #[96, 256]
    config['width_2'] = [96, 256] #[128,512]
    config['width_3'] = [96, 256] #[96, 256]
    config['width_4'] = [96, 256] #[128,512]
    config['paf_channels']  = 28
    config['conf_channels'] = 15

    # Define model
    model = get_model(config)

    out = model(tf.random.uniform([16,256,256,3], maxval=255))
    tf.keras.utils.plot_model(model, show_shapes=True)
    model.summary()
