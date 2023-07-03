import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_inspect
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import numpy as np

model = tf.keras.models.load_model('../trained_model/ICAIPose/models/ICAIPose_best.h5')

train_images = np.load('../validation_data/imgs.npy')[:1000]

# inspector = vitis_inspect.VitisInspector(target=None)
# inspector.inspect_model(model,
#                         input_shape=[256,256,3],
#                         plot=True, 
#                         plot_file="model.svg", 
#                         dump_results=True, 
#                         dump_results_file="inspect_results.txt", 
#                         verbose=0)


quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(
    calib_dataset=train_images[0:10],
    include_cle=True,
    cle_steps=10,
    include_fast_ft=True)

quantized_model.save('quantized.h5')