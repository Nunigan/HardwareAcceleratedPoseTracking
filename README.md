# HardwareAcceleratedPoseTracking

- Project repository for the [hackster.io](https://www.hackster.io/) [Adaptive Computing Challenge 2021](https://www.hackster.io/contests/xilinxadaptivecomputing2021). 
with Xilinx.

## Project: [Hardware Accelerated Pose Tracking](https://www.hackster.io/michi_michi/hardware-accelerated-pose-tracking-d5ebb9)

- Prebuilt
  - All the files needed to run ICAIPose and the hourglass example on the KV260
  - prebuilt.tar.xz the same folder compressed for an easy download directly on the KV260
- Vitis AI
  - All data to compile the neural network with Vitis AI
  - Run the start docker.sh file to get the correct Vitis AI version
  - With the script "run_all.sh" all the scripts rquired to compile the network are executed
- Hourglass 
  - The hourglass network from Vitis AI Model Zoo
- Weights 
  - The pretrained weights for the network ICAIPose
- run_ICAIPose.py
  - Python script with various utility functions 
        
