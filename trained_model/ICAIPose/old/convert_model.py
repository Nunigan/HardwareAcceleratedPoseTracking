# -*- coding: utf-8 -*-
# @Author: Simon Walser
# @Date:   2021-11-22 08:07:23
# @Last Modified by:   Simon Walser
# @Last Modified time: 2021-11-22 15:04:24

import numpy as np
import os
import toml
import tensorflow as tf
from importlib.machinery import SourceFileLoader

from tensorflow.keras import Model

################################################################################
#
# Class / Function definitions
#
################################################################################

def weights2h5(src_path, dst_path):

    # Prepare config
    config = toml.load(os.path.join(src_path, 'config.toml'))
    img_res = config['training']['img_res']

    # Load module
    model_file = os.path.join(src_path, 'ICAIPose.py')
    module = SourceFileLoader('ICAIPose', model_file).load_module()

    # Create model
    input = tf.keras.Input(shape=(None,None,3))
    out = module.OpenPose(**config['network'])(input)
    model = Model(inputs=input, outputs=out)

    # Load weights
    model(tf.random.uniform(shape=(1,img_res,img_res,3), minval=0, maxval=255))
    model.load_weights(os.path.join(src_path, 'ICAIPose_best.h5'), by_name=True)

    # Save model
    model.save(os.path.join(dst_path, 'ICAIPose_best.h5'))


################################################################################
#
# Main functions
#
################################################################################


def main():
    src_path = '/home/simon/work/work_OST/PhysioAI/trained_models/poseestimationtrain/256/weights/ICAIPose_4/'
    dst_path = '/home/simon/work/work_OST/AIonFPGA/ICAIPose/models/'

    model = weights2h5(src_path, dst_path)


if __name__ == "__main__":
    main()
