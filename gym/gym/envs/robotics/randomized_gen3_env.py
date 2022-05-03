import numpy as np
import os
import copy
import csv

from gym.envs.robotics import robot_env, utils
import math
from random import randint

import xml.etree.ElementTree as et

import mujoco_py
from mujoco_py.modder import TextureModder, MaterialModder, LightModder, CameraModder
from mujoco_utils.mujoco_py import get_camera_transform_matrix
from mujoco_utils.views import get_angles_hemisphere
import cv2
from PIL import Image

DEBUG = False
closed_pos = [1.12810781, -0.59798289, -0.53003607]
closed_angle = 0.45


def debug(msg, data):
    if DEBUG:
        print(msg, data)


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class RandomizedGen3Env(robot_env.RobotEnv):
    """Superclass for all Kinova Gen3 environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        has_object,
        has_cloth,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        cloth_length,
        behavior,
        initial_qpos,
        reward_type,
        **kwargs,
    ):
        """Initializes a new Kinova Gen3 environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above
                the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            has_cloth ('True' or 'False'): whether or not the object has a cloth/textile
        """
        self.data_path = kwargs["data_path"]
        self.last_saved_step = 0
        self.n_views = kwargs["n_views"]
        self.angles = None

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
        self.config_file = kwargs.get("config")
        self.color_from_id = {"0": "red", "1": "green", "2": "yellow", "3": "blue"}

        self.mode = "rgb_array"
        self.visual_randomize = False
        self.visual_data_recording = True
        self.track = False

        self.num_vertices = 4
        self.cloth_length = cloth_length
        self.randomize_cloth = 0.1
        self.behavior = behavior
        self.explicit_policy = True
        self.physical_params = [
            4.00000000e-03,
            8.00000000e-03,
            1.40000000e00,
            3.00000000e-03,
            1.00000000e-03,
            2.00000000e-03,
            1.00000000e-03,
            1.00000000e-02,
            3.00000000e-02,
            1.10000000e01,
        ]
        if self.behavior == "lifting-middle":
            self.vertex = np.random.randint(4, cloth_length - 3)

        # The number of n_actions need to change to 8 for a second agent
        super(RandomizedGen3Env, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=8,
            initial_qpos=initial_qpos,
            mode=self.mode,
            visual_randomize=self.visual_randomize,
            visual_data_recording=self.visual_data_recording,
        )

        # randomization
        self.config_file = kwargs.get("config")
        self.xml_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets", "gen3"
        )
        self.reference_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "assets",
            "gen3",
            kwargs.get("xml_name"),
        )
        self.reference_xml = et.parse(self.reference_path)
        self.dimensions = []
        self.dimension_map = []
        self.suffixes = []
        self._locate_randomize_parameters()
        self.initial_qpos = initial_qpos
        self.n_substeps = n_substeps
        if self.data_path is not None:
            self._index = len(
                [
                    name
                    for name in os.listdir(self.data_path)
                    if os.path.isfile(os.path.join(self.data_path, name))
                ]
            )

    # Randomization methods
    # ----------------------------
    def _locate_randomize_parameters(self):
        self.root = self.reference_xml.getroot()
        # locate all the randomization parameters you wish to randomize
        # self.object_joints = self.root.findall(".//body[@name='object0']/joint")
        # self.cloth = self.root.findall(".//body[@name='CB0_10']/composite")
        cloth = self.root.findall(".//body[@name='CB0_10']")[
            0
        ]  # list of all the composite objects
        self.composites = cloth.findall("./composite")
        # Initialize values for all physical parameters
        # for dim in self.unwrapped.dimensions:
        #     self.physical_params.append(dim.current_value)
        # print("physical_params", self.physical_params)

    def _randomize_size(self):
        size = self.dimensions[0].current_value
        print("Size to change to", size)

        for composite in self.composites:
            geom = composite.findall("./geom")[0]
            geom.set("size", "{:3f}".format(size))

    def _randomize_mass(self):
        mass = self.dimensions[1].current_value
        # print("Mass to change to", mass)

        for composite in self.composites:
            geom = composite.findall("./geom")[0]
            geom.set("mass", "{:3f}".format(mass))

    def _randomize_friction(self):
        friction = self.dimensions[2].current_value
        # print("Friction to change to",friction)

        for composite in self.composites:
            geom = composite.findall("./geom")[0]
            # print("friction actual", geom.get('friction'))
            geom.set("friction", "{:3f}{:f}{:f}".format(friction, friction, friction))

    def _randomize_joint_damping(self):
        joint_damping = self.dimensions[3].current_value
        # print("joint_damping to change to", joint_damping)

        for composite in self.composites:
            joint = composite.findall("./joint")[0]
            joint.set("damping", "{:3f}".format(joint_damping))

    def _randomize_joint_stiffness(self):
        joint_stiffness = self.dimensions[4].current_value
        # print("joint_stiffness to change to", joint_stiffness)

        for composite in self.composites:
            joint = composite.findall("./joint")[0]
            joint.set("stiffness", "{:3f}".format(joint_stiffness))

    def _randomize_tendon_damping(self):
        tendon_damping = self.dimensions[5].current_value
        # print("tendon_damping to change to", tendon_damping)

        for composite in self.composites:
            tendon = composite.findall("./tendon")[0]
            tendon.set("damping", "{:3f}".format(tendon_damping))

    def _randomize_tendon_stiffness(self):
        tendon_stiffness = self.dimensions[6].current_value
        # print("tendon_stiffness to change to", tendon_stiffness)

        for composite in self.composites:
            tendon = composite.findall("./tendon")[0]
            tendon.set("stiffness", "{:3f}".format(tendon_stiffness))

    def _randomize_flatinertia(self):
        flatinertia = self.dimensions[7].current_value
        # print("flatinertia to change to", flatinertia)

        for composite in self.composites:
            composite.set("flatinertia", "{:3f}".format(flatinertia))

    def _randomize_spacing(self):
        if self.behavior == "onehand-lifting":
            # spacing = np.clip(self.dimensions[8].current_value, 0.025, 0.03)
            spacing = np.random.uniform(0.022, 0.023)
        else:
            spacing = self.dimensions[8].current_value

        # print("spacing to change to", spacing)

        for composite in self.composites:
            composite.set("spacing", "{:3f}".format(spacing))

    # def _randomize_cloth_count(self):
    #     mass = self.dimensions[1].current_value
    #     print("Mass to change to", mass)

    #     for composite in self.composites:
    #         geom = composite.findall("./geom")[0]
    #         geom.set('mass', '{:3f}'.format(mass))

    def _create_xml(self):
        # self._randomize_size()
        self._randomize_mass()
        self._randomize_spacing()
        self._randomize_flatinertia()
        self._randomize_joint_damping()
        self._randomize_joint_stiffness()
        self._randomize_tendon_damping()
        self._randomize_tendon_stiffness()
        self._randomize_friction()
        return et.tostring(self.root, encoding="unicode", method="xml")

    def update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)
        # update values for all physical parameters
        self.physical_params = []
        for dim in self.unwrapped.dimensions:
            self.physical_params.append(dim.current_value)

    def _re_init(self, xml):
        # TODO: Now, likely needs rank
        randomized_path = os.path.join(self.xml_dir, "randomizedgen3.xml")

        with open(randomized_path, "wb") as fp:
            fp.write(xml.encode())
            fp.flush()

        try:
            self.model = mujoco_py.load_model_from_path(randomized_path)
        except Exception:
            print("Unable to load the xml file")

        self.sim = mujoco_py.MjSim(self.model, nsubsteps=self.n_substeps)

        self.modder = TextureModder(self.sim)
        self.mat_modder = MaterialModder(self.sim)
        self.light_modder = LightModder(self.sim)
        self.camera_modder = CameraModder(self.sim)

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        if self.viewer:
            self.viewer.update_sim(self.sim)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        # Probably needs to change due to the second agent but for the time we can
        # discard it completely until we use reinforcement learning
        if (
            self.behavior == "sideways"
            or self.behavior == "lifting"
            or self.behavior == "onehand-lifting"
            or self.behavior == "onehand"
        ):
            num_objects = 2
            if len(achieved_goal.shape) == 1:
                blocks_in_position = 0
                for x in range(num_objects):
                    if (
                        goal_distance(
                            achieved_goal[x * 3 : x * 3 + 3], goal[x * 3 : x * 3 + 3]
                        )
                        < self.distance_threshold
                    ):
                        # print(something)
                        print(blocks_in_position)
                        blocks_in_position += 1
                # reward = -1*self.num_objects + blocks_in_position
                reward = -(np.array(blocks_in_position != num_objects)).astype(
                    np.float32
                )  # non positive rewards
                # raward = 1
                return reward
            else:
                # reward = -np.ones(achieved_goal.shape[0])*self.num_objects
                reward = -np.ones(
                    achieved_goal.shape[0]
                )  # uncomment for totally sparse reward
                for x in range(achieved_goal.shape[0]):
                    blocks_in_position = 0
                    for i in range(num_objects):
                        if (
                            goal_distance(
                                achieved_goal[x][i * 3 : i * 3 + 3],
                                goal[x][i * 3 : i * 3 + 3],
                            )
                            < self.distance_threshold
                        ):
                            blocks_in_position += 1
                    # reward[x] = reward[x] + blocks_in_position
                    reward[x] = -(np.array(blocks_in_position != num_objects)).astype(
                        np.float32
                    )
                # print(reward)
                return reward
        elif self.behavior == "diagonally":
            d = goal_distance(achieved_goal, goal)
            debug("\tdistance to goal: ", d)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
                # return -(np.array(d > self.distance_threshold)).astype(np.float32)
            else:
                return -d

    # Gripper helper
    # ----------------------------
    def _gripper_sync(self):
        # move the left_spring_joint joint[14] and right_spring_joint(joint[10]) in the right angle
        self.sim.data.qpos[10] = self._gripper_consistent(self.sim.data.qpos[7:10])
        self.sim.data.qpos[14] = self._gripper_consistent(self.sim.data.qpos[11:14])
        self.sim.data.qpos[25] = self._gripper_consistent(self.sim.data.qpos[22:25])
        self.sim.data.qpos[29] = self._gripper_consistent(self.sim.data.qpos[26:29])

    def _gripper_consistent(self, angle):
        x = (
            -0.006496
            + 0.0315 * math.sin(angle[0])
            + 0.04787744772 * math.cos(angle[0] + angle[1] - 0.1256503306)
            - 0.02114828598 * math.sin(angle[0] + angle[1] + angle[2] - 0.1184899592)
        )
        y = (
            -0.0186011
            - 0.0315 * math.cos(angle[0])
            + 0.04787744772 * math.sin(angle[0] + angle[1] - 0.1256503306)
            + 0.02114828598 * math.cos(angle[0] + angle[1] + angle[2] - 0.1184899592)
        )
        return math.atan2(y, x) + 0.6789024115

    # RobotEnv methods
    # ----------------------------

    # gripper control
    def _step_callback(self):
        if self.block_gripper:
            for j in range(3):
                if (
                    self.behavior == "onehand"
                    or self.behavior == "onehand-lifting"
                    or self.behavior == "diagonally"
                ):
                    self.sim.data.qpos[22 + j] = closed_pos[j]
                    self.sim.data.qpos[26 + j] = closed_pos[j]
                else:
                    self.sim.data.qpos[7 + j] = closed_pos[j]
                    self.sim.data.qpos[11 + j] = closed_pos[j]
                    self.sim.data.qpos[22 + j] = closed_pos[j]
                    self.sim.data.qpos[26 + j] = closed_pos[j]
            # self.sim.data.set_joint_qpos('robot1:right_knuckle_joint', closed_angle)
            # self.sim.data.set_joint_qpos('robot1:left_knuckle_joint', closed_angle)
            self._gripper_sync()
            self.sim.forward()
        else:
            # sync the spring link
            self._gripper_sync()
            self.sim.forward()

    def find_closest_indice(self, gripper_position):
        cloth_points_all = np.array(
            [
                np.array(
                    [
                        "CB0_0",
                        "CB1_0",
                        "CB2_0",
                        "CB3_0",
                        "CB4_0",
                        "CB5_0",
                        "CB6_0",
                        "CB7_0",
                        "CB8_0",
                        "CB9_0",
                        "CB10_0",
                        "CB11_0",
                        "CB12_0",
                        "CB13_0",
                        "CB14_0",
                    ]
                ),
                np.array(
                    [
                        "CB0_1",
                        "CB1_1",
                        "CB2_1",
                        "CB3_1",
                        "CB4_1",
                        "CB5_1",
                        "CB6_1",
                        "CB7_1",
                        "CB8_1",
                        "CB9_1",
                        "CB10_1",
                        "CB11_1",
                        "CB12_1",
                        "CB13_1",
                        "CB14_1",
                    ]
                ),
                np.array(
                    [
                        "CB0_2",
                        "CB1_2",
                        "CB2_2",
                        "CB3_2",
                        "CB4_2",
                        "CB5_2",
                        "CB6_2",
                        "CB7_2",
                        "CB8_2",
                        "CB9_2",
                        "CB10_2",
                        "CB11_2",
                        "CB12_2",
                        "CB13_2",
                        "CB14_2",
                    ]
                ),
                np.array(
                    [
                        "CB0_3",
                        "CB1_3",
                        "CB2_3",
                        "CB3_3",
                        "CB4_3",
                        "CB5_3",
                        "CB6_3",
                        "CB7_3",
                        "CB8_3",
                        "CB9_3",
                        "CB10_3",
                        "CB11_3",
                        "CB12_3",
                        "CB13_3",
                        "CB14_3",
                    ]
                ),
                np.array(
                    [
                        "CB0_4",
                        "CB1_4",
                        "CB2_4",
                        "CB3_4",
                        "CB4_4",
                        "CB5_4",
                        "CB6_4",
                        "CB7_4",
                        "CB8_4",
                        "CB9_4",
                        "CB10_4",
                        "CB11_4",
                        "CB12_4",
                        "CB13_4",
                        "CB14_4",
                    ]
                ),
                np.array(
                    [
                        "CB0_5",
                        "CB1_5",
                        "CB2_5",
                        "CB3_5",
                        "CB4_5",
                        "CB5_5",
                        "CB6_5",
                        "CB7_5",
                        "CB8_5",
                        "CB9_5",
                        "CB10_5",
                        "CB11_5",
                        "CB12_5",
                        "CB13_5",
                        "CB14_5",
                    ]
                ),
                np.array(
                    [
                        "CB0_6",
                        "CB1_6",
                        "CB2_6",
                        "CB3_6",
                        "CB4_6",
                        "CB5_6",
                        "CB6_6",
                        "CB7_6",
                        "CB8_6",
                        "CB9_6",
                        "CB10_6",
                        "CB11_6",
                        "CB12_6",
                        "CB13_6",
                        "CB14_6",
                    ]
                ),
                np.array(
                    [
                        "CB0_7",
                        "CB1_7",
                        "CB2_7",
                        "CB3_7",
                        "CB4_7",
                        "CB5_7",
                        "CB6_7",
                        "CB7_7",
                        "CB8_7",
                        "CB9_7",
                        "CB10_7",
                        "CB11_7",
                        "CB12_7",
                        "CB13_7",
                        "CB14_7",
                    ]
                ),
                np.array(
                    [
                        "CB0_8",
                        "CB1_8",
                        "CB2_8",
                        "CB3_8",
                        "CB4_8",
                        "CB5_8",
                        "CB6_8",
                        "CB7_8",
                        "CB8_8",
                        "CB9_8",
                        "CB10_8",
                        "CB11_8",
                        "CB12_8",
                        "CB13_8",
                        "CB14_8",
                    ]
                ),
                np.array(
                    [
                        "CB0_9",
                        "CB1_9",
                        "CB2_9",
                        "CB3_9",
                        "CB4_9",
                        "CB5_9",
                        "CB6_9",
                        "CB7_9",
                        "CB8_9",
                        "CB9_9",
                        "CB10_9",
                        "CB11_9",
                        "CB12_9",
                        "CB13_9",
                        "CB14_9",
                    ]
                ),
                np.array(
                    [
                        "CB0_10",
                        "CB1_10",
                        "CB2_10",
                        "CB3_10",
                        "CB4_10",
                        "CB5_10",
                        "CB6_10",
                        "CB7_10",
                        "CB8_10",
                        "CB9_10",
                        "CB10_10",
                        "CB11_10",
                        "CB12_10",
                        "CB13_10",
                        "CB14_10",
                    ]
                ),
                np.array(
                    [
                        "CB0_11",
                        "CB1_11",
                        "CB2_11",
                        "CB3_11",
                        "CB4_11",
                        "CB5_11",
                        "CB6_11",
                        "CB7_11",
                        "CB8_11",
                        "CB9_11",
                        "CB10_11",
                        "CB11_11",
                        "CB12_11",
                        "CB13_11",
                        "CB14_11",
                    ]
                ),
                np.array(
                    [
                        "CB0_12",
                        "CB1_12",
                        "CB2_12",
                        "CB3_12",
                        "CB4_12",
                        "CB5_12",
                        "CB6_12",
                        "CB7_12",
                        "CB8_12",
                        "CB9_12",
                        "CB10_12",
                        "CB11_12",
                        "CB12_12",
                        "CB13_12",
                        "CB14_12",
                    ]
                ),
                np.array(
                    [
                        "CB0_13",
                        "CB1_13",
                        "CB2_13",
                        "CB3_13",
                        "CB4_13",
                        "CB5_13",
                        "CB6_13",
                        "CB7_13",
                        "CB8_13",
                        "CB9_13",
                        "CB10_13",
                        "CB11_13",
                        "CB12_13",
                        "CB13_13",
                        "CB14_13",
                    ]
                ),
                np.array(
                    [
                        "CB0_14",
                        "CB1_14",
                        "CB2_14",
                        "CB3_14",
                        "CB4_14",
                        "CB5_14",
                        "CB6_14",
                        "CB7_14",
                        "CB8_14",
                        "CB9_14",
                        "CB10_14",
                        "CB11_14",
                        "CB12_14",
                        "CB13_14",
                        "CB14_14",
                    ]
                ),
            ]
        )

        cloth_points_pos = []
        # slice the cloth points according to the number of cloth length
        cloth_points = cloth_points_all[: self.cloth_length, : self.cloth_length].copy()
        cloth_points = cloth_points.flatten()

        for point in cloth_points:
            cloth_points_pos.append(self.sim.data.get_body_xpos(point))
        clothMesh = np.asarray(cloth_points_pos)
        deltas = clothMesh - gripper_position
        dist_2 = np.einsum("ij,ij->i", deltas, deltas)
        closest = np.argmin(dist_2)

        return cloth_points[closest], dist_2[closest]

    def distance_from_indice(self, gripper_position, indice):
        point = self.sim.data.get_body_xpos(indice)
        distance = np.linalg.norm(point - gripper_position)

        return distance

    def find_point_coordinates(self):
        cloth_points_all = np.array(
            [
                np.array(
                    [
                        "CB0_0",
                        "CB1_0",
                        "CB2_0",
                        "CB3_0",
                        "CB4_0",
                        "CB5_0",
                        "CB6_0",
                        "CB7_0",
                        "CB8_0",
                        "CB9_0",
                        "CB10_0",
                        "CB11_0",
                        "CB12_0",
                        "CB13_0",
                        "CB14_0",
                    ]
                ),
                np.array(
                    [
                        "CB0_1",
                        "CB1_1",
                        "CB2_1",
                        "CB3_1",
                        "CB4_1",
                        "CB5_1",
                        "CB6_1",
                        "CB7_1",
                        "CB8_1",
                        "CB9_1",
                        "CB10_1",
                        "CB11_1",
                        "CB12_1",
                        "CB13_1",
                        "CB14_1",
                    ]
                ),
                np.array(
                    [
                        "CB0_2",
                        "CB1_2",
                        "CB2_2",
                        "CB3_2",
                        "CB4_2",
                        "CB5_2",
                        "CB6_2",
                        "CB7_2",
                        "CB8_2",
                        "CB9_2",
                        "CB10_2",
                        "CB11_2",
                        "CB12_2",
                        "CB13_2",
                        "CB14_2",
                    ]
                ),
                np.array(
                    [
                        "CB0_3",
                        "CB1_3",
                        "CB2_3",
                        "CB3_3",
                        "CB4_3",
                        "CB5_3",
                        "CB6_3",
                        "CB7_3",
                        "CB8_3",
                        "CB9_3",
                        "CB10_3",
                        "CB11_3",
                        "CB12_3",
                        "CB13_3",
                        "CB14_3",
                    ]
                ),
                np.array(
                    [
                        "CB0_4",
                        "CB1_4",
                        "CB2_4",
                        "CB3_4",
                        "CB4_4",
                        "CB5_4",
                        "CB6_4",
                        "CB7_4",
                        "CB8_4",
                        "CB9_4",
                        "CB10_4",
                        "CB11_4",
                        "CB12_4",
                        "CB13_4",
                        "CB14_4",
                    ]
                ),
                np.array(
                    [
                        "CB0_5",
                        "CB1_5",
                        "CB2_5",
                        "CB3_5",
                        "CB4_5",
                        "CB5_5",
                        "CB6_5",
                        "CB7_5",
                        "CB8_5",
                        "CB9_5",
                        "CB10_5",
                        "CB11_5",
                        "CB12_5",
                        "CB13_5",
                        "CB14_5",
                    ]
                ),
                np.array(
                    [
                        "CB0_6",
                        "CB1_6",
                        "CB2_6",
                        "CB3_6",
                        "CB4_6",
                        "CB5_6",
                        "CB6_6",
                        "CB7_6",
                        "CB8_6",
                        "CB9_6",
                        "CB10_6",
                        "CB11_6",
                        "CB12_6",
                        "CB13_6",
                        "CB14_6",
                    ]
                ),
                np.array(
                    [
                        "CB0_7",
                        "CB1_7",
                        "CB2_7",
                        "CB3_7",
                        "CB4_7",
                        "CB5_7",
                        "CB6_7",
                        "CB7_7",
                        "CB8_7",
                        "CB9_7",
                        "CB10_7",
                        "CB11_7",
                        "CB12_7",
                        "CB13_7",
                        "CB14_7",
                    ]
                ),
                np.array(
                    [
                        "CB0_8",
                        "CB1_8",
                        "CB2_8",
                        "CB3_8",
                        "CB4_8",
                        "CB5_8",
                        "CB6_8",
                        "CB7_8",
                        "CB8_8",
                        "CB9_8",
                        "CB10_8",
                        "CB11_8",
                        "CB12_8",
                        "CB13_8",
                        "CB14_8",
                    ]
                ),
                np.array(
                    [
                        "CB0_9",
                        "CB1_9",
                        "CB2_9",
                        "CB3_9",
                        "CB4_9",
                        "CB5_9",
                        "CB6_9",
                        "CB7_9",
                        "CB8_9",
                        "CB9_9",
                        "CB10_9",
                        "CB11_9",
                        "CB12_9",
                        "CB13_9",
                        "CB14_9",
                    ]
                ),
                np.array(
                    [
                        "CB0_10",
                        "CB1_10",
                        "CB2_10",
                        "CB3_10",
                        "CB4_10",
                        "CB5_10",
                        "CB6_10",
                        "CB7_10",
                        "CB8_10",
                        "CB9_10",
                        "CB10_10",
                        "CB11_10",
                        "CB12_10",
                        "CB13_10",
                        "CB14_10",
                    ]
                ),
                np.array(
                    [
                        "CB0_11",
                        "CB1_11",
                        "CB2_11",
                        "CB3_11",
                        "CB4_11",
                        "CB5_11",
                        "CB6_11",
                        "CB7_11",
                        "CB8_11",
                        "CB9_11",
                        "CB10_11",
                        "CB11_11",
                        "CB12_11",
                        "CB13_11",
                        "CB14_11",
                    ]
                ),
                np.array(
                    [
                        "CB0_12",
                        "CB1_12",
                        "CB2_12",
                        "CB3_12",
                        "CB4_12",
                        "CB5_12",
                        "CB6_12",
                        "CB7_12",
                        "CB8_12",
                        "CB9_12",
                        "CB10_12",
                        "CB11_12",
                        "CB12_12",
                        "CB13_12",
                        "CB14_12",
                    ]
                ),
                np.array(
                    [
                        "CB0_13",
                        "CB1_13",
                        "CB2_13",
                        "CB3_13",
                        "CB4_13",
                        "CB5_13",
                        "CB6_13",
                        "CB7_13",
                        "CB8_13",
                        "CB9_13",
                        "CB10_13",
                        "CB11_13",
                        "CB12_13",
                        "CB13_13",
                        "CB14_13",
                    ]
                ),
                np.array(
                    [
                        "CB0_14",
                        "CB1_14",
                        "CB2_14",
                        "CB3_14",
                        "CB4_14",
                        "CB5_14",
                        "CB6_14",
                        "CB7_14",
                        "CB8_14",
                        "CB9_14",
                        "CB10_14",
                        "CB11_14",
                        "CB12_14",
                        "CB13_14",
                        "CB14_14",
                    ]
                ),
            ]
        )
        cloth_points_pos = []
        # slice the cloth points according to the number of cloth length
        cloth_points = cloth_points_all[: self.cloth_length, : self.cloth_length].copy()
        cloth_points = cloth_points.flatten()
        for point in cloth_points:
            cloth_points_pos.append(self.sim.data.get_body_xpos(point))

        clothMesh = np.asarray(cloth_points_pos)
        dictionary = dict(zip(cloth_points, clothMesh))

        return dictionary

    def _set_action(self, action):
        # this will be 8 for 2 agents
        # assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl_2, gripper_ctrl_2 = action[4:7], action[7]

        pos_ctrl *= 0.05  # limit maximum change in position
        pos_ctrl_2 *= 0.05  # limit maximum change in position
        rot_ctrl = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion {Vertical}
        rot_ctrl_2 = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion {Vertical}
        # rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion {Horizontal}
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        gripper_ctrl_2 = np.array([gripper_ctrl_2, gripper_ctrl_2])
        assert gripper_ctrl.shape == (2,)
        assert gripper_ctrl_2.shape == (2,)

        # Apply action to simulation

        # Determine the closest cloth node to the gripper

        closest, dist_closest = self.find_closest_indice(self.grip_pos)
        # if self.behavior == 'lifting-middle':
        #     dist_closest = np.einsum(abs(self.grip_pos -\
        #       self.sim.data.get_body_xpos('CB' + str(self.vertex) + '_' + '0')))
        closest_2, dist_closest_2 = self.find_closest_indice(self.grip_pos_2)
        # Only allow gripping if in proximity
        # pdb.set_trace()

        if dist_closest <= 0.001 and gripper_ctrl[0] > 0.5:
            # pdb.set_trace()

            if self.behavior == "lifting-middle":
                # pdb.set_trace()
                utils.grasp(
                    self.sim,
                    gripper_ctrl,
                    "CB" + str(self.vertex) + "_" + "0",
                    self.behavior,
                )
                distance = self.distance_from_indice(
                    self.grip_pos, "CB" + str(self.vertex) + "_" + "0"
                )
                if distance <= 0.01:
                    # pdb.set_trace()
                    self.block_gripper = True
                    self._step_callback()
            else:
                utils.grasp(self.sim, gripper_ctrl, "CB10_0", self.behavior)
                self.block_gripper = True
                self._step_callback()
        else:
            utils.grasp(
                self.sim, gripper_ctrl, "CB10_0", self.behavior
            )  # this might need to change
            self.block_gripper = False
            self._step_callback()

        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)

        # if dist_closest_2<=0.001:
        #     # pdb.set_trace()
        #
        #     utils.grasp(self.sim, gripper_ctrl_2, 'CB0_0')
        # if self.block_gripper:
        #     gripper_ctrl_2 = np.zeros_like(gripper_ctrl_2)

        if (
            dist_closest_2 <= 0.001
            and gripper_ctrl_2[0] > 0.5
            and self.behavior != "onehand"
            or self.behavior != "diagonally"
            or self.behavior == "onehand-lifting"
        ):
            # pdb.set_trace()

            utils.grasp(self.sim, gripper_ctrl_2, "CB0_0", self.behavior)
        if self.block_gripper and self.behavior != "onehand" or self.behavior:
            gripper_ctrl_2 = np.zeros_like(gripper_ctrl_2)

        # action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        action = np.concatenate([pos_ctrl, rot_ctrl])
        # pdb.set_trace()
        # pdb.set_trace()
        # action_2 = np.concatenate([pos_ctrl_2, rot_ctrl_2, gripper_ctrl_2])
        action_2 = np.concatenate([pos_ctrl_2, rot_ctrl_2])
        # pdb.set_trace()
        # pdb.set_trace()
        utils.ctrl_set_action(self.sim, gripper_ctrl)
        utils.mocap_set_action(self.sim, action, agent=0)

        # pdb.set_trace()

        # Use only when the second mocap is active

        utils.ctrl_set_action(self.sim, gripper_ctrl_2)
        utils.mocap_set_action(self.sim, action_2, agent=1)
        # utils.mocap_set_action(self.sim, action_2)

        # test grasping
        # gripper_ctrl = 0.6
        # gripper_ctrl_2 = 0.6
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # gripper_ctrl_2 = np.array([gripper_ctrl_2, gripper_ctrl_2])
        # utils.grasp(self.sim, gripper_ctrl , 'CB10_0')
        # utils.grasp(self.sim, gripper_ctrl_2, 'CB0_0')

    def _get_obs(self):
        """returns the observations dict"""

        # positions
        # grip_pos = self.sim.data.get_body_xpos('robot1:ee_link')
        # dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # grip_velp = self.sim.data.get_body_xvelp('robot1:ee_link') * dt

        # add second agent

        grip_pos = self.sim.data.get_body_xpos("gripper_central2")
        self.grip_pos = grip_pos
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        grip_pos_2 = self.sim.data.get_body_xpos("gripper_central")
        self.grip_pos_2 = grip_pos_2

        # add second agent
        grip_velp = self.sim.data.get_body_xvelp("gripper_central2") * dt

        grip_velp_2 = self.sim.data.get_body_xvelp("gripper_central") * dt

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        if self.has_object:
            object_pos = self.sim.data.get_site_xpos("object0")

            # object_pos_1 = self.sim.data.get_site_xpos('object1')

            # velocities
            object_velp = self.sim.data.get_site_xvelp("object0") * dt
            object_velp -= grip_velp
        elif self.has_cloth:
            # get the positions and velocities for 4 corners of the cloth
            vertices = ["CB0_0"]
            # Name vertices with respect to the cloth_length
            if self.behavior == "lifting-middle":
                vertices.append("CB" + str(self.vertex) + "_" + "0")
            else:
                vertices.append("CB" + str(self.cloth_length - 1) + "_" + "0")

            vertices.append(
                "CB" + str(self.cloth_length - 1) + "_" + str(self.cloth_length - 1)
            )
            vertices.append("CB" + "0" + "_" + str(self.cloth_length - 1))
            vertice_pos, vertice_velp, _, vertice_rel_pos = [], [], [], []

            for vertice in vertices:
                vertice_pos.append(self.sim.data.get_body_xpos(vertice))
                vertice_velp.append(self.sim.data.get_body_xvelp(vertice) * dt)

            vertice_rel_pos = vertice_pos.copy()
            vertice_rel_pos -= grip_pos
            # pdb.set_trace()
            vertice_velp_2 = vertice_velp.copy()
            vertice_velp -= grip_velp
            # pdb.set_trace()
            vertice_rel_pos_2 = vertice_pos.copy()
            vertice_rel_pos_2 -= grip_pos_2
            vertice_velp_2 -= grip_velp_2

        else:
            object_pos = object_velp = np.zeros(0)

        # if not using a fake gripper
        # gripper_state = robot_qpos[-2:]
        # gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        gripper_state = np.array([self.sim.model.eq_active[-2]])
        # pdb.set_trace()
        gripper_state_2 = np.array(
            [self.sim.model.eq_active[-1]]
        )  # change the value according to the nameid of the second gripper
        # gripper_vel # Does not make sense for fake gripper

        if not self.has_object and not self.has_cloth:
            achieved_goal = grip_pos.copy()

            # pdb.set_trace()
        elif self.has_cloth and not self.has_object:
            if self.behavior == "diagonally":
                achieved_goal = np.squeeze(vertice_pos[0].copy())
            elif (
                self.behavior == "sideways"
                or self.behavior == "lifting"
                or self.behavior == "onehand-lifting"
                or self.behavior == "onehand"
                or self.behavior == "lifting-middle"
            ):
                achieved_goal = np.concatenate(
                    [
                        vertice_pos[1].copy(),
                        vertice_pos[3].copy(),
                    ]
                )

                # Need to change something for the second agent here as well
                # achieved_goal = np.concatenate([
                #     vertice_pos[0].copy(), vertice_pos[1].copy(),
                # ])
        else:
            achieved_goal = np.squeeze(object_pos.copy())
            # pdb.set_trace()
        # obs = np.concatenate([
        # ])

        # obs = np.concatenate([
        #     grip_pos, gripper_state, grip_velp, gripper_vel, vertice_pos[0],
        #     vertice_pos[1], vertice_pos[2], vertice_pos[3],
        # ])
        # Creating dataset for Visual policy training
        # filename = str(uuid.uuid4())

        # basename = "mylogfile"

        if self.visual_data_recording:

            self._render_callback()
            self.viewer = self._get_viewer("rgb_array")
            HEIGHT, WIDTH = 512, 512

            if self.angles is None:
                if self.n_views == 1:
                    print("Using default camera angles")
                    self.angles = (self.viewer.cam.azimuth, self.viewer.cam.elevation)
                else:
                    print(
                        "Computing equi-spaced camera angles on upper hemisphere surface"
                    )
                    self.angles = get_angles_hemisphere(
                        radius=self.viewer.cam.distance,
                        n_views=self.n_views,
                    )

            for view_id_int, (azimuth, elevation) in enumerate(self.angles, start=1):
                view_id = f"view_{view_id_int}"
                # Set camera parameters
                self.viewer.cam.azimuth = azimuth
                self.viewer.cam.elevation = elevation

                # Render image internally and read pixels
                self.viewer.render(HEIGHT, WIDTH)
                visual_data_all = self.viewer.read_pixels(HEIGHT, WIDTH, depth=True)

                depth = visual_data_all[1]
                visual_data = visual_data_all[0]

                # original image is upside-down, so flip it
                visual_data = Image.fromarray(visual_data[::-1, :, :], "RGB")
                depth_cv = cv2.normalize(
                    depth[::-1, :], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                )

                for subdir in ["RGB", "depth"]:
                    if not os.path.isdir(os.path.join(self.data_path, view_id, subdir)):
                        os.makedirs(os.path.join(self.data_path, view_id, subdir))

                visual_data.save(
                    os.path.join(
                        self.data_path, view_id, "RGB", f"{self.last_saved_step}.png"
                    )
                )

                cv2.imwrite(
                    os.path.join(
                        self.data_path, view_id, "depth", f"{self.last_saved_step}.tif"
                    ),
                    depth_cv,
                )

                # Store camera parameters if not done already
                if not os.path.isfile(
                    os.path.join(self.data_path, view_id, "camera_params.txt")
                ):
                    cam_name = "camera1"
                    cam_id = self.sim.model.camera_name2id(cam_name)
                    vertical_fov = self.sim.model.cam_fovy[cam_id]
                    np.savetxt(
                        os.path.join(self.data_path, view_id, "camera_params.txt"),
                        get_camera_transform_matrix(
                            width=WIDTH,
                            height=HEIGHT,
                            vertical_fov=vertical_fov,
                            camera=self.viewer.cam,
                        ),
                    )

            # Common for all views
            if not os.path.isdir(os.path.join(self.data_path, "points")):
                os.makedirs(os.path.join(self.data_path, "points"))
            dict = {"points": self.find_point_coordinates().copy()}
            w = csv.writer(
                open(
                    os.path.join(
                        self.data_path, "points", f"{self.last_saved_step}.csv"
                    ),
                    "w",
                )
            )
            for key, val in dict["points"].items():
                w.writerow([key, val])

            # Grippers
            for gripper_name, gripper_pos, gripper_state in zip(
                ["gripper_1", "gripper_2"],
                [grip_pos, grip_pos_2],
                [gripper_state, gripper_state_2],
            ):
                if not os.path.isdir(os.path.join(self.data_path, gripper_name)):
                    os.makedirs(os.path.join(self.data_path, gripper_name, "position"))
                    os.makedirs(os.path.join(self.data_path, gripper_name, "state"))

                np.savetxt(
                    os.path.join(
                        self.data_path,
                        gripper_name,
                        "position",
                        f"{self.last_saved_step}.txt",
                    ),
                    gripper_pos,
                )
                np.savetxt(
                    os.path.join(
                        self.data_path,
                        gripper_name,
                        "state",
                        f"{self.last_saved_step}.txt",
                    ),
                    [gripper_state],
                )

            # for point in label:
            #     cv2.circle(visual_data, (int(point[0]), int(point[1])), 2, (0, 0, 255), 2)

            # label_file = "/home/rjangir/workSpace/IRI-DL/datasets/sim2real/train/" + "train_ids" + ".npy"
            # if np.asarray(self._label_matrix).shape[0] == 1000:
            #     print("saving the labels file")
            #     np.save(label_file, np.asarray(self._label_matrix), allow_pickle=True )
            #     np.savetxt(path + '.csv', centers, delimiter=",", fmt='%s')

            # label_file = "/home/rjangir/workSpace/sketchbook/" +
            # "pytorch-corner-detection/data/real_images/train_dataset_RL/" +\
            # "image" +str(self._index)
            # np.savetxt(label_file + '.csv', self._label_matrix, delimiter=",", fmt='%s')

        # obs = np.concatenate([
        #     grip_pos, gripper_state, grip_velp, vertice_pos[0],
        #     vertice_pos[1], vertice_pos[2], vertice_pos[3], vertice_velp[0],
        #     vertice_velp[1], vertice_velp[2], vertice_velp[3],
        # ])

        # pdb.set_trace()

        # might need to add extra info about the vertice_pos
        obs = np.concatenate(
            [
                grip_pos,
                gripper_state,
                grip_velp,
                vertice_pos[0],
                vertice_pos[1],
                vertice_pos[2],
                vertice_pos[3],
                grip_pos_2,
                gripper_state_2,
                grip_velp_2,
                vertice_velp_2[0],
                vertice_velp_2[1],
                vertice_velp_2[2],
                vertice_velp_2[3],
            ]
        )
        # pdb.set_trace()

        if self.explicit_policy:
            # pdb.set_trace()
            # Might need to chagne this
            obs = np.concatenate(
                [
                    obs,
                    np.array(self.physical_params),
                ]
            )

        self._index += 1
        # self.track = True
        # self._viewer_setup()

        self.last_saved_step += 1
        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "points": self.find_point_coordinates(),
            "filename": str(self.last_saved_step - 1),
        }

    def _viewer_setup(self):
        self.viewer._show_mocap = False

        # self.viewer.cam.fixedcamid = -1
        # self.viewer.cam.type = mujoco_py.generated.const.CAMERA_FIXED

        # pdb.set_trace()

        body_id = self.sim.model.body_name2id("CB5_5")

        if self.behavior == "diagonally" or self.behavior == "onehand-lifting":
            body_id = self.sim.model.body_name2id("CB5_5")
        # #body_id = self.sim.model.body_name2id('robot1:robotiq_85_base_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        if not self.track:
            self.viewer.cam.distance = 1.2 + randint(-5, 5) / 100
            self.viewer.cam.azimuth = 90.0 + randint(-5, 5)
            self.viewer.cam.elevation = -30.0 + randint(-5, 5)

        # if not self.track:
        #     self.viewer.cam.distance = 1.2 + randint(-5, 5)/100
        #     self.viewer.cam.azimuth = 90. + randint(-5, 5)
        #     self.viewer.cam.elevation = -30. + randint(-5, 5)

    def _render_callback(self):
        # Visualize target.
        if (
            self.behavior == "sideways"
            or self.behavior == "lifting"
            or self.behavior == "onehand"
            or self.behavior == "onehand-lifting"
            or self.behavior == "lifting-middle"
            and not self.visual_data_recording
        ):
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
            targets = ["target0", "target1"]
            site_ids = []
            # pdb.set_trace()
            for x in range(2):
                site_ids.append(self.sim.model.site_name2id(targets[x]))
                self.sim.model.site_pos[site_ids[x]] = (
                    self.goal[x * 3 : x * 3 + 3] - sites_offset[0]
                )
            self.sim.forward()
        elif not self.visual_data_recording:
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
            site_id = self.sim.model.site_name2id("target0")
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
            self.sim.forward()

    def _reset_sim(self):

        self.sim.set_state(self.initial_state)
        self.track = False
        self._viewer_setup()

        self.sim.forward()
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        if self.has_cloth:
            if self.behavior == "diagonally":
                # joint_vertice = 'CB'+str(self.cloth_length-1)+'_'+str(self.cloth_length-1)
                joint_vertice = "CB5" + "_" + str(self.cloth_length - 1)
            elif (
                self.behavior == "sideways"
                or self.behavior == "lifting"
                or self.behavior == "onehand"
                or self.behavior == "onehand-lifting"
                or self.behavior == "lifting-middle"
            ):
                joint_vertice = "CB0" + "_" + str(self.cloth_length - 1)
            new_position = self.sim.data.get_body_xpos(joint_vertice)
            # Make the joint to be the first point
            new_position = np.append(new_position, [1, 0, 0, 0])
            # pdb.set_trace()
            # Need to identify the target like CB0_0 or CB0_10)
            self.sim.data.set_joint_qpos("cloth", new_position)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        elif self.has_cloth:
            if self.behavior == "diagonally":
                goal_vertice = "CB0" + "_" + str(self.cloth_length - 1)
                # goal_vertice = 'CB0' + '_10'
                goal = self.sim.data.get_body_xpos(goal_vertice)
                # Sample goal according to the cloth_length
            elif self.behavior == "sideways" or self.behavior == "onehand":
                goal_vertices = [
                    "CB0" + "_" + str(self.cloth_length - 1),
                    "CB"
                    + str(self.cloth_length - 1)
                    + "_"
                    + str(self.cloth_length - 1),
                ]
                goals = [
                    self.sim.data.get_body_xpos(goal_vertices[0]),
                    self.sim.data.get_body_xpos(goal_vertices[1]),
                ]
                # goals = [self.sim.data.get_body_xpos(goal_vertices[0]) + (0, 0, 1),
                # self.sim.data.get_body_xpos(goal_vertices[1]) + (0, 0, 1)]
                # pdb.set_trace()
                goal = np.concatenate([goals[0].copy(), goals[1].copy()])
                # pdb.set_trace()
            elif (
                self.behavior == "lifting"
                or self.behavior == "onehand-lifting"
                or self.behavior == "lifting-middle"
            ):
                if self.behavior == "lifting-middle":
                    goal_vertices = [
                        "CB0" + "_" + str(self.cloth_length - 1),
                        "CB" + str(self.vertex) + "_" + str(self.cloth_length - 1),
                    ]
                    # temp = np.abs(self.sim.data.get_body_xpos(goal_vertices[0]) -\
                    # self.sim.data.get_body_xpos(goal_vertices[1]))
                    # pdb.set_trace()
                    goals = [
                        self.sim.data.get_body_xpos(goal_vertices[0])
                        + (0.0, -0.1, 0.25),
                        self.sim.data.get_body_xpos(goal_vertices[1])
                        + (0.0, -0.1, 0.25),
                    ]
                else:
                    goal_vertices = [
                        "CB0" + "_" + str(self.cloth_length - 1),
                        "CB"
                        + str(self.cloth_length - 1)
                        + "_"
                        + str(self.cloth_length - 1),
                    ]
                    # goals = [self.sim.data.get_body_xpos(goal_vertices[0]),
                    # self.sim.data.get_body_xpos(goal_vertices[1])]
                    goals = [
                        self.sim.data.get_body_xpos(goal_vertices[0]) + (0, -0.2, 0.41),
                        self.sim.data.get_body_xpos(goal_vertices[1])
                        + (0.0, -0.1, 0.41),
                    ]
                    # use (0.0, -0.1, 0.31) for lifting and one hand dropping
                # pdb.set_trace()
                goal = np.concatenate([goals[0].copy(), goals[1].copy()])
                # pdb.set_trace()
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -0.15, 0.15, size=3
            )
            # goal = self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        if (
            self.behavior == "sideways"
            or self.behavior == "lifting"
            or self.behavior == "onehand"
            or self.behavior == "onehand-lifting"
            or self.behavior == "lifting-middle"
        ):
            num_objects = 2
            # pdb.set_trace()
            if len(achieved_goal.shape) == 1:
                d = True
                for x in range(num_objects):
                    d = d and (
                        goal_distance(
                            achieved_goal[x * 3 : x * 3 + 3],
                            desired_goal[x * 3 : x * 3 + 3],
                        )
                        < self.distance_threshold
                    )
                    # pdb.set_trace()
                return (d).astype(np.float32)
            else:
                for x in range(achieved_goal.shape[0]):
                    d = True
                    for x in range(num_objects):
                        d = d and (
                            goal_distance(
                                achieved_goal[x][x * 3 : x * 3 + 3],
                                desired_goal[x][x * 3 : x * 3 + 3],
                            )
                            < self.distance_threshold
                        )
                return (d).astype(np.float32)
        elif self.behavior == "diagonally":
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        # for name, value in initial_qpos.items():
        #     self.sim.data.set_joint_qpos(name, value)
        self.light_modder = LightModder(self.sim)
        # pdb.set_trace()
        # Change the light settings, mainly the position and the direction to create different shadows
        # self.light_modder.set_pos("light0", [randint(0,4), randint(0,4), randint(0,4)])
        # self.light_modder.set_pos("light1", 0 0 0)
        # self.light_modder.set_pos("light2", 0 0 0)
        # self.light_modder.set_pos("light3", 0 0 0)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Move end effector into position.
        gripper_target_2 = np.array(
            [0.3, 0.8, 0.3 + self.gripper_extra_height]
        )  # + self.sim.data.get_site_xpos('robotiq_85_base_link')
        gripper_target = np.array(
            [1.5, 0.8, 0.5 + self.gripper_extra_height]
        )  # + self.sim.data.get_site_xpos('robotiq_85_base_link')
        if self.behavior == "lifting-middle":
            gripper_rotation_2 = np.array([0.0, -0.5, 1.0, 0.5])
        else:
            gripper_rotation_2 = np.array([0.0, 1.0, 1.0, 0.0])
        gripper_rotation = np.array([0.0, 1.0, 1.0, 0.0])

        # Add here the second agent mocap but probably we need to change initially
        # the gripper target and rotation a bit in order to start in a more natural position
        self.sim.data.set_mocap_pos("robot1:mocap", gripper_target_2)
        self.sim.data.set_mocap_quat("robot1:mocap", gripper_rotation)

        self.sim.data.set_mocap_pos("robot2:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot2:mocap", gripper_rotation_2)
        # pdb.set_trace()
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.

        # add also the second agent here but it can be commented out
        self.initial_gripper_xpos2 = self.sim.data.get_body_xpos(
            "robot1:ee_link"
        ).copy()  # Needs a change if using the gripper for goal generation
        self.initial_gripper_xpos = self.sim.data.get_body_xpos(
            "robot2:ee_link"
        ).copy()  # Needs a change if using the gripper for goal generation
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def render(self, mode="human", width=500, height=500):
        return super(RandomizedGen3Env, self).render(mode, width, height)
