
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from skimage.io import imread
import numpy as np

quantized_model = tf.keras.models.load_model('quantized.h5') 

quantized_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics= tf.keras.metrics.SparseTopKCategoricalAccuracy())

eval_dataset = [imread('/workspace/data/im1.jpg'),imread('/workspace/data/im2.jpg'),imread('/workspace/data/im3.jpg')]

img = np.reshape(eval_dataset[0],(1,960,540,3))

offset = np.array([103.939, 116.779, 123.68], dtype='float32')[None,None,None,:]

# Run ICAIPose
img = (img[...,::-1]-offset)*0.5
img = img.astype(np.int8)

out = quantized_model(img)

conf_map = np.moveaxis(out[-1][0,:,:,:-1], [0,1,2], [1,2,0])

print(conf_map)