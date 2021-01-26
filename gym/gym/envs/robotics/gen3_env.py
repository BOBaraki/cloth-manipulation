import numpy as np

from gym.envs.robotics import rotations, robot_env, utils
import math
from random import randint

DEBUG = False
closed_pos = [1.12810781, -0.59798289, -0.53003607]
closed_angle = 0.45

def debug(msg, data):
    if DEBUG:
        print(msg, data)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class Gen3Env(robot_env.RobotEnv):
    """Superclass for all Kinova Gen3 environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, has_cloth, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, cloth_length, behavior, initial_qpos, reward_type,
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

        self.num_vertices = 4
        self.cloth_length = cloth_length
        self.randomize_cloth = 0.1
        self.behavior = behavior
        self.explicit_policy = False

        super(Gen3Env, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if self.behavior=="sideways":
            num_objects = 2
            if len(achieved_goal.shape) == 1:
                blocks_in_position = 0
                for x in range(num_objects):
                    if (goal_distance(achieved_goal[x*3:x*3+3], goal[x*3:x*3+3]) < self.distance_threshold):
                        blocks_in_position += 1
                #reward = -1*self.num_objects + blocks_in_position
                reward = -(np.array(blocks_in_position != num_objects)).astype(np.float32) # non positive rewards
                return reward
            else:
                #reward = -np.ones(achieved_goal.shape[0])*self.num_objects
                reward = -np.ones(achieved_goal.shape[0]) #uncomment for totally sparse reward
                for x in range(achieved_goal.shape[0]):
                    blocks_in_position = 0
                    for i in range(num_objects):
                        if (goal_distance(achieved_goal[x][i*3:i*3+3], goal[x][i*3:i*3+3]) < self.distance_threshold):
                            blocks_in_position += 1
                    #reward[x] = reward[x] + blocks_in_position
                    reward[x] = -(np.array(blocks_in_position != num_objects)).astype(np.float32)
                return reward
        elif self.behavior=="diagonally":
            d = goal_distance(achieved_goal, goal)
            debug("\tdistance to goal: ", d)
            if self.reward_type == 'sparse':
                return -(d > self.distance_threshold).astype(np.float32)
                #return -(np.array(d > self.distance_threshold)).astype(np.float32)
            else:
                return -d

    # Gripper helper
    # ----------------------------
    def _gripper_sync(self):
        # move the left_spring_joint joint[14] and right_spring_joint(joint[10]) in the right angle
        self.sim.data.qpos[10] = self._gripper_consistent(self.sim.data.qpos[7: 10])
        self.sim.data.qpos[14] = self._gripper_consistent(self.sim.data.qpos[11: 14])

    def _gripper_consistent(self, angle):
        x = -0.006496 + 0.0315 * math.sin(angle[0]) + 0.04787744772 * math.cos(angle[0] + angle[1] - 0.1256503306) - 0.02114828598 * math.sin(angle[0] + angle[1] + angle[2] - 0.1184899592)
        y = -0.0186011 - 0.0315 * math.cos(angle[0]) + 0.04787744772 * math.sin(angle[0] + angle[1] - 0.1256503306) + 0.02114828598 * math.cos(angle[0] + angle[1] + angle[2] - 0.1184899592)
        return math.atan2(y, x) + 0.6789024115
    
    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            for j in range(3):
                self.sim.data.qpos[7 + j] = closed_pos[j]
                self.sim.data.qpos[11 + j] = closed_pos[j]
            #self.sim.data.set_joint_qpos('robot1:right_knuckle_joint', closed_angle)
            #self.sim.data.set_joint_qpos('robot1:left_knuckle_joint', closed_angle)
            self._gripper_sync()
            self.sim.forward()
        else:
            # sync the spring link
            self._gripper_sync()
            self.sim.forward()

    def find_closest_indice(self, gripper_position):
        cloth_points_all = np.array([np.array(['CB0_0','CB1_0','CB2_0','CB3_0','CB4_0','CB5_0','CB6_0','CB7_0','CB8_0','CB9_0','CB10_0','CB11_0','CB12_0','CB13_0','CB14_0']),
                    np.array(['CB0_1','CB1_1','CB2_1','CB3_1','CB4_1','CB5_1','CB6_1','CB7_1','CB8_1','CB9_1','CB10_1','CB11_1','CB12_1','CB13_1','CB14_1']),
                    np.array(['CB0_2','CB1_2','CB2_2','CB3_2','CB4_2','CB5_2','CB6_2','CB7_2','CB8_2','CB9_2','CB10_2','CB11_2','CB12_2','CB13_2','CB14_2']),
                    np.array(['CB0_3','CB1_3','CB2_3','CB3_3','CB4_3','CB5_3','CB6_3','CB7_3','CB8_3','CB9_3','CB10_3','CB11_3','CB12_3','CB13_3','CB14_3']),
                    np.array(['CB0_4','CB1_4','CB2_4','CB3_4','CB4_4','CB5_4','CB6_4','CB7_4','CB8_4','CB9_4','CB10_4','CB11_4','CB12_4','CB13_4','CB14_4']),
                    np.array(['CB0_5','CB1_5','CB2_5','CB3_5','CB4_5','CB5_5','CB6_5','CB7_5','CB8_5','CB9_5','CB10_5','CB11_5','CB12_5','CB13_5','CB14_5']),
                    np.array(['CB0_6','CB1_6','CB2_6','CB3_6','CB4_6','CB5_6','CB6_6','CB7_6','CB8_6','CB9_6','CB10_6','CB11_6','CB12_6','CB13_6','CB14_6']),
                    np.array(['CB0_7','CB1_7','CB2_7','CB3_7','CB4_7','CB5_7','CB6_7','CB7_7','CB8_7','CB9_7','CB10_7','CB11_7','CB12_7','CB13_7','CB14_7']),
                    np.array(['CB0_8','CB1_8','CB2_8','CB3_8','CB4_8','CB5_8','CB6_8','CB7_8','CB8_8','CB9_8','CB10_8','CB11_8','CB12_8','CB13_8','CB14_8']),
                    np.array(['CB0_9','CB1_9','CB2_9','CB3_9','CB4_9','CB5_9','CB6_9','CB7_9','CB8_9','CB9_9','CB10_9','CB11_9','CB12_9','CB13_9','CB14_9']),
                    np.array(['CB0_10','CB1_10','CB2_10','CB3_10','CB4_10','CB5_10','CB6_10','CB7_10','CB8_10','CB9_10','CB10_10','CB11_10','CB12_10','CB13_10','CB14_10']),
                    np.array(['CB0_11','CB1_11','CB2_11','CB3_11','CB4_11','CB5_11','CB6_11','CB7_11','CB8_11','CB9_11','CB10_11','CB11_11','CB12_11','CB13_11','CB14_11']),
                    np.array(['CB0_12','CB1_12','CB2_12','CB3_12','CB4_12','CB5_12','CB6_12','CB7_12','CB8_12','CB9_12','CB10_12','CB11_12','CB12_12','CB13_12','CB14_12']),
                    np.array(['CB0_13','CB1_13','CB2_13','CB3_13','CB4_13','CB5_13','CB6_13','CB7_13','CB8_13','CB9_13','CB10_13','CB11_13','CB12_13','CB13_13','CB14_13']),
                    np.array(['CB0_14','CB1_14','CB2_14','CB3_14','CB4_14','CB5_14','CB6_14','CB7_14','CB8_14','CB9_14','CB10_14','CB11_14','CB12_14','CB13_14','CB14_14'])])
        cloth_points_pos = []
        # slice the cloth points according to the number of cloth length
        cloth_points = cloth_points_all[:self.cloth_length, :self.cloth_length].copy()
        cloth_points = cloth_points.flatten()
        for point in cloth_points:
           cloth_points_pos.append(self.sim.data.get_body_xpos(point))

        clothMesh = np.asarray(cloth_points_pos)        
        deltas = clothMesh - gripper_position
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        closest = np.argmin(dist_2)

        return cloth_points[closest], dist_2[closest]

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [0., 1., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion {Vertical}
        #rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion {Horizontal}
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        
        
        # Apply action to simulation

        # Determine the closest cloth node to the gripper
        closest, dist_closest = self.find_closest_indice(self.grip_pos)
        # Only allow gripping if in proximity
        if dist_closest<=0.001:
            utils.grasp(self.sim, gripper_ctrl, closest)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        

    def _get_obs(self):
        """ returns the observations dict """
       
        # positions
        # grip_pos = self.sim.data.get_body_xpos('robot1:ee_link')
        # dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # grip_velp = self.sim.data.get_body_xvelp('robot1:ee_link') * dt

        grip_pos = self.sim.data.get_body_xpos('gripper_central')
        self.grip_pos = grip_pos
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('gripper_central') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        elif self.has_cloth:
            #get the positions and velocities for 4 corners of the cloth
            vertices = ['CB0_0']
            # Name vertices with respect to the cloth_length
            vertices.append('CB'+str(self.cloth_length-1)+'_'+'0')
            vertices.append('CB'+str(self.cloth_length-1)+'_'+str(self.cloth_length-1))
            vertices.append('CB'+'0'+'_'+str(self.cloth_length-1))
            vertice_pos, vertice_velp, vertice_velr, vertice_rel_pos = [], [], [], []
            for vertice in vertices:
                vertice_pos.append(self.sim.data.get_body_xpos(vertice))

                vertice_velp.append(self.sim.data.get_body_xvelp(vertice) * dt)
                #vertice_velr.append(self.sim.data.get_body_xvelr(vertice) * dt) #Do not need rotational velocities

            vertice_rel_pos = vertice_pos.copy()
            vertice_rel_pos -= grip_pos
            vertice_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        # if not using a fake gripper
        # gripper_state = robot_qpos[-2:]
        # gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        gripper_state = np.array([self.sim.model.eq_active[-1]])
        # gripper_vel # Does not make sense for fake gripper 

        if not self.has_object and not self.has_cloth:
            achieved_goal = grip_pos.copy()
        elif self.has_cloth and not self.has_object:
            if self.behavior=="diagonally":
                achieved_goal = np.squeeze(vertice_pos[0].copy())
            elif self.behavior=="sideways":
                achieved_goal = np.concatenate([
                vertice_pos[0].copy(), vertice_pos[1].copy(),
                ])
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])

        # obs = np.concatenate([
        #     grip_pos, gripper_state, grip_velp, gripper_vel, vertice_pos[0], vertice_pos[1], vertice_pos[2], vertice_pos[3],
        # ])
    
        
        obs = np.concatenate([
            grip_pos, gripper_state, grip_velp, vertice_pos[0], vertice_pos[1], vertice_pos[2], vertice_pos[3], vertice_velp[0], vertice_velp[1], vertice_velp[2], vertice_velp[3], 
        ])
        

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot1:ee_link')
        #body_id = self.sim.model.body_name2id('robot1:robotiq_85_base_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        if self.behavior=="sideways":
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
            targets = ['target0', 'target1']
            site_ids = []
            for x in range(2):
                site_ids.append(self.sim.model.site_name2id(targets[x]))
                self.sim.model.site_pos[site_ids[x]] = self.goal[x*3:x*3+3] - sites_offset[0]
            self.sim.forward()
        else:
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
            self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self.sim.forward()
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        if self.has_cloth:
            if self.behavior=="diagonally":
                joint_vertice = 'CB'+str(self.cloth_length-1)+'_'+str(self.cloth_length-1)
            elif self.behavior=="sideways":
                joint_vertice = 'CB0'+'_'+str(self.cloth_length-1)
            new_position = self.sim.data.get_body_xpos(joint_vertice)
            # Make the joint to be the first point
            randomness = self.np_random.uniform(-self.randomize_cloth, self.randomize_cloth, size=2)
            new_position[0] = new_position[0] + randomness[0]
            new_position[1] = new_position[1] + randomness[1]
            new_position = np.append(new_position, [1, 0, 0, 0])
            gripper_ctrl = np.array([0.0, 0.0])
            utils.grasp(self.sim, gripper_ctrl, 'CB0_0')
            self.sim.data.set_joint_qpos('cloth', new_position)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        elif self.has_cloth:
            if self.behavior=="diagonally":
                goal_vertice = 'CB'+str(self.cloth_length-1)+'_'+str(self.cloth_length-1)
                goal = self.sim.data.get_body_xpos(goal_vertice)
                # Sample goal according to the cloth_length
                randomness = self.np_random.uniform(-self.target_range, self.target_range, size=2)
                goal[0] += randomness[0]
                goal[1] += randomness[1]
                #goal[2] += 0.06
            elif self.behavior=="sideways":
                goal_vertices = ['CB0'+'_'+str(self.cloth_length-1), 'CB'+str(self.cloth_length-1)+'_'+str(self.cloth_length-1)]
                goals = [self.sim.data.get_body_xpos(goal_vertices[0]), self.sim.data.get_body_xpos(goal_vertices[1])]
                randomness = self.np_random.uniform(-self.target_range, self.target_range, size=4)
                goals[0][0] += randomness[0]
                goals[0][1] += randomness[1]
                goals[1][0] += randomness[2]
                goals[1][1] += randomness[3]
                goal = np.concatenate([ goals[0].copy(), goals[1].copy()])
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
            # goal = self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        if self.behavior=="sideways":
            num_objects = 2
            if len(achieved_goal.shape) == 1:
                d = True
                for x in range(num_objects):
                    d = d and (goal_distance(achieved_goal[x*3:x*3+3], desired_goal[x*3:x*3+3]) < self.distance_threshold)
                return (d).astype(np.float32)
            else:
                for x in range(achieved_goal.shape[0]):
                    d = True
                    for x in range(num_objects):
                        d = d and (goal_distance(achieved_goal[x][x*3:x*3+3], desired_goal[x][x*3:x*3+3]) < self.distance_threshold)
                return (d).astype(np.float32)
        elif self.behavior=="diagonally":
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)
        

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([0.6, 0.6 , 0.4 + self.gripper_extra_height]) #+ self.sim.data.get_site_xpos('robotiq_85_base_link')
        gripper_rotation = np.array([0., 1., 1., 0.])
        self.sim.data.set_mocap_pos('robot1:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot1:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_body_xpos('robot1:ee_link').copy() # Needs a change if using the gripper for goal generation
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(Gen3Env, self).render(mode, width, height)
