import os
from gym import utils
from gym.envs.robotics import gen3_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('gen3', 'sideways_fold.xml')


class Gen3SidewaysFoldEnv(gen3_env.Gen3Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        behaviors = ['diagonally', 'sideways']
        initial_qpos = {
            'robot1:Actuator1': 0.0,
            'robot1:Actuator2': 0.0,
            'robot1:Actuator3': 0.0,
            'robot1:Actuator4': 0.0,
            'robot1:Actuator5': 0.0,
            'robot1:Actuator6': 0.0,
            'robot1:Actuator7': 0.0,
        }
        gen3_env.Gen3Env.__init__(
            self, MODEL_XML_PATH, has_object=False, has_cloth=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.01, distance_threshold=0.05, cloth_length=11, behavior=behaviors[1],
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
