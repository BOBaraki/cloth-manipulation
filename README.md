# Mujoco Cloth Manipulation Simulator
A mujoco simulator using bi-manual and unimanual pinch grapsing kinowa arms for cloth manipulation

```
conda create -n cloth-manipulation python=3.6 pip
conda activate cloth-manipulation
conda install -c conda-forge patchelf
git clone --recurse-submodules https://github.com/BOBaraki/cloth-manipulation
pip install -e mujoco-py
pip install -e sim_utils
pip install -e gym
pip install -r requirements.txt
```

In the initial folder run after changing the following saving paths:
```
python run.py RandomizedGen3LiftTwoHands-v0 --mode=demo --behavior lifting --max_steps 150 --data_path my_dataset
```
Keep in mind that you might need to run in the terminal first:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
```
This is for nvidia 384. You should probably find your own drivers version with **nvidia-smi** if you don't know it.
