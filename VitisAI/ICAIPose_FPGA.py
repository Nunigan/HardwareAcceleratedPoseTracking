# @Author: Simon Walser
# @Date:   2021-09-07 15:12:42
# @Last Modified by:   Michael Schmid
# @Last Modified time: 2022-02-16

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model


# from pose_parameter import PoseParameter

################################################################################
#
# Class / Function Definitions
#
################################################################################



class InputBlock:
    def __init__(self):
        self.bias = tf.constant([[[[103.939, 116.779, 123.68]]]], dtype=tf.float32)
        self.__name__ = ''
    def __call__(self, inputs):


   
        backbone = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)

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
        x_1 = LeakyReLU(alpha=0.1,name=self.name+'leakyrelu0')(x_1)

        x_2 = Conv2D(self.channel_number, 3, padding='same', name=self.name+'conv1')(x_1)
        x_2 = LeakyReLU(alpha=0.1,name=self.name+'leakyrelu1')(x_2)

        x_3 = Conv2D(self.channel_number, 3, padding='same', name=self.name+'conv2')(x_2)
        x_3 = LeakyReLU(alpha=0.1,name=self.name+'leakyrelu2')(x_3)

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
        x   = LeakyReLU(alpha=0.1,name=self.name+'leakyreluexp')(x)
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
        x = LeakyReLU(alpha=0.1,name=self.name+'prelubb')(x)

        x = Conv2D(256, 3, padding='same', name=self.name+'convstem0')(x)
        x = LeakyReLU(alpha=0.1,name=self.name+'leakyrelustem0')(x)
        x = Conv2D(128, 3, padding='same', name=self.name+'convstem1')(x)
        in_1 = LeakyReLU(alpha=0.1,name=self.name+'leakyrelustem1')(x)

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

    config = {}
    config['img_res'] = 256
    config['width_1'] = [96, 256] #[96, 256]
    config['width_2'] = [96, 256] #[128,512]
    config['width_3'] = [96, 256] #[96, 256]
    config['width_4'] = [96, 256] #[128,512]
    config['paf_channels']  = 28
    config['conf_channels'] = 16


    model = get_model(config)
    model.summary()

    model.load_weights('../weights/weights_256.h5')
    model.save('saved_model.h5')
    

