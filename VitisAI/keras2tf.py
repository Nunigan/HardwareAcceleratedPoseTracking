#!/usr/bin/env python3leaky_re_lu_1/LeakyRelu
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:11:07 2020

@author: nunigan
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.keras.backend.set_learning_phase(0)


loaded_model = tf.keras.models.load_model('saved_model.h5')


print ('Keras model information:')
print (' Input names :',loaded_model.inputs)
print (' Output names:',loaded_model.outputs)
print('-------------------------------------')

tfckpt = 'train/tfchkpt.ckpt'


tf_session = tf.keras.backend.get_session()


# write out tensorflow checkpoint & meta graph
saver = tf.compat.v1.train.Saver()
save_path = saver.save(tf_session,tfckpt)
print (' Checkpoint created :',tfckpt)
