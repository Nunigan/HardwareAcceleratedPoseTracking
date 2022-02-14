#!/bin/sh
python3 ../ICAIPose_FPGA.py
python3 keras2tf.py
source freeze.sh
source quant.sh
source comp.sh
#xdputil xmodel compile/own_network.xmodel -s test.svg
