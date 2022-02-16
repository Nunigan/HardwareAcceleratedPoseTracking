vai_c_tensorflow \
    --frozen_pb  ./quantize/quantize_eval_model.pb \
    --arch       arch_kv.json \
    --output_dir ./compile/ \
    --net_name   own_network_256_KV260 \
    --option    "{'dump': 'all'}"
