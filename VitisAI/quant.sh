vai_q_tensorflow quantize \
        --input_frozen_graph ./train/frozen_graph.pb \
	--input_fn           image_input_fn.calib_input \
	--output_dir         ./quantize \
        --input_nodes        input_1\
	--output_nodes       upsamplconv1/BiasAdd \
	--input_shapes       1,256,256,3  \
	--calib_iter         9 \
        --gpu                0 \



