# Mujoco Cloth Manipulation Simulator
A mujoco simulator using bi-manual and unimanual pinch grapsing kinowa arms for cloth manipulation

## To checkout the repo with gym:
First you need to install mujoco, mujoco-py and gym. Make sure you install mujoco-py from their [Github repo](https://github.com/openai/mujoco-py).
Also make a minimal gym installation:
```
pip install gym
```

Then clone this repo:
```
git clone https://github.com/BOBaraki/cloth-manipulation
```

You need to make sure that first you change some hardcoded paths to fit yours.
The files where you need to change the paths are in the files:
**cloth-manipulation/gym/gym/envs/robotics/assets/gen3/shared.xml**
**cloth-manipulation/gym/gym/envs/__init__.py**
**cloth-manipulation/datagen_domain_randomization.py**
**cloth-manipulation/gym/gym/envs/robotics/randomized_gen3_env.py**


Then in the initial folder run after changing the following saving paths:
```
python datagen_domain_randomization.py RandomizedGen3LiftTwoHands-v0 --mode=demo --behavior=lifting --max_steps=150
```
Keep in mind that you might need to run in the terminal first:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
```
This is for nvidia 384. You should probably find your own drivers version with **nvidia-smi** if you don't know it.


In case you get a segmentation error it's a mujoco-py bug you need to do the following:
First uninstall the current mujoco-py(not mujoco) and the install it by doing the following:

```
1. python setup.py build
2. python setup.py install
```

To see if it works, run again: 

```
python datagen_domain_randomization.py RandomizedGen3LiftTwoHands-v0 --mode=demo --behavior=lifting --max_steps=150
```
## Examples:
```
    # Run the data generation command 
    python datagen_domain_randomization.py RandomizedGen3LiftTwoHands-v0 --mode=demo --behavior=lifting --max_steps=150
    # Beucase mujoco captures the frames at each episode before any action is made clean the data after you generate them:
    python file_deletion.py 
```
