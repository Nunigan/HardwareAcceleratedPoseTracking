# HardwareAcceleratedPoseTracking

- Project repository for the [hackster.io](https://www.hackster.io/) [Adaptive Computing Challenge 2021](https://www.hackster.io/contests/xilinxadaptivecomputing2021). 
with Xilinx.

## Project: [Hardware Accelerated Pose Tracking](https://www.hackster.io/michi_michi/hardware-accelerated-pose-tracking-d5ebb9)

- Vitis AI folder
 - All data to compile the neural network with Vitis AI
 - run the start docker.sh file to get the correct vitisAI version.
 - with the script "run_all.sh" all the scripts to compile the network are executed.
- hourglass folder
  - The hourglass network from VitisAI Model Zoo
- weights folder
  - The pretrained weights for the network ICAIPose
- prebuilt folder
  - all the files needed to run ICAIPose and the hourglass example on the KV260
  - prebuilt.tar.xz the same folder compressed for an easy download to the KV260
- run_ICAIPose.py
 - Python script with various utility functions 
        