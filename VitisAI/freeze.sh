freeze_graph \
    --input_meta_graph  ./train/tfchkpt.ckpt.meta \
    --input_checkpoint  ./train/tfchkpt.ckpt \
    --output_graph      ./train/frozen_graph.pb \
    --output_node_names upsamplconv1/BiasAdd \
    --input_binary      true

#freeze_graph \
#    --input_meta_graph  ./train/keras_tf2_tf1/test.meta \
#    --input_checkpoint  ./train/keras_tf2_tf1/test \
#    --output_graph      ./train/keras_tf2_tf1/frozen_graph.pb \
#    --output_node_names leaky_re_lu_1/LeakyRelu \
#    --input_binary      true
