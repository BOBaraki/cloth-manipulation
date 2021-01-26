import numpy as np

from gym.envs.robotics import rotations, robot_env, utils
import math

DEBUG = False

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class DummyEnv(robot_env.RobotEnv):
    """Superclass for all Kinova Gen3 environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, has_cloth, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Kinova Gen3 environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            has_cloth ('True' or 'False'): whether or not the object has a cloth/textile
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.has_cloth = has_cloth

        super(DummyEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _step_callback(self):
        self.sim.forward()
        return None

    def _set_action(self, action):
        return None

    def _get_obs(self):
        """ returns the observations dict """
        # returns dummy numbers
        return {
            'observation': self.np_random.uniform(-0.15, 0.15, size=3),
            'achieved_goal': self.np_random.uniform(-0.15, 0.15, size=3),
            'desired_goal': self.np_random.uniform(-0.15, 0.15, size=3)
        }
        

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('B0_0')
        #body_id = self.sim.model.body_name2id('robot1:robotiq_85_base_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.

        self.sim.forward()
        return True

    def _sample_goal(self):
        return self.np_random.uniform(-0.15, 0.15, size=3)

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        
        self.sim.forward()


    def render(self, mode='human', width=500, height=500):
        return super(DummyEnv, self).render(mode, width, height)
