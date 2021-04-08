#!/usr/bin/env python
"""
Run before using this file:
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so


This script has been copied from datagen_sideways_fold.py, consult that for changes
"""
import os
import gym
from gym import spaces, envs
import argparse
import numpy as np
import itertools
import time
from builtins import input
import random
import randomizer

from randomizer.wrappers import RandomizedEnvWrapper
import cv2
import pdb

render_mode = "human"

def rescale_action(action, speed, noise):
    noise = np.random.normal(0, noise,len(action))
    action = [x * 60 for x in action]
    actionRescaled = np.clip(action + noise, -speed, speed).tolist()

    return actionRescaled

def generate_demos(obs, render, max_episode_steps):
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    timeStep = 0
    grasp_threshold = 0.01 # how close to get to grasp the vertice
    reach_threshold = 0.03 # how close to get to reach a point in space
    noise_param = 0.6
    
    episodeObs.append(obs)

    pick_up_object = 1
    pick_up_object_2 = 0
    # place_pos = np.random.randint(2) + 3
    place_pos = 2
    place_pos_2 = 3

    while True:
        #print("Approaching air", timeStep)
        if render: env.render(mode=render_mode)
        obsDataNew = obs.copy()
        #The obsDataNew will change because it will return the second gripper
        objectPos = np.array([obsDataNew['observation'][7:10].copy(), obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(),
                              obsDataNew['observation'][16:19].copy()])



        #New gripperPos and gripperState
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]



        gripperPos_2 = obsDataNew['observation'][19:22].copy()
        gripperState_2 = obsDataNew['observation'][22]

        # pdb.set_trace()
        #New object positions related to the gripper maybe add some waiting actions if one gripper grabs the cloth before the other
        object_rel_pos = objectPos - gripperPos
        object_oriented_goal = object_rel_pos[pick_up_object].copy()

        object_rel_pos_2 = objectPos - gripperPos_2
        object_oriented_goal_2 = object_rel_pos_2[pick_up_object_2].copy()
        # pdb.set_trace()
        object_oriented_goal[2] += 0.02
        object_oriented_goal_2[2] += 0.02

        if (np.linalg.norm(object_oriented_goal) <= reach_threshold and np.linalg.norm(object_oriented_goal_2) <= reach_threshold) or timeStep >= max_episode_steps: break

        #increase the actions space here for two agents
        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001),
                  random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        # action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        speed = 1.0 # cap action to whatever speed you want

        # pdb.set_trace()
        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]
            # action[i] = 0.

        for i in range(len(object_oriented_goal_2)):
            # pdb.set_trace()
            action[4+i] = object_oriented_goal_2[i]
            # action[4+i] = 0.

        # pdb.set_trace()

        # print(action)

        actionRescaled = rescale_action(action, speed, 0.1)
        # pdb.set_trace()

        # if timeStep%2 == 1:
        #     actionRescaled[0] = 0



        obs, reward, done, info = env.step(actionRescaled)
        # print(reward)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1


    while True:
        #print("Approaching OBJECT", timeStep)
        if render: env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        object_rel_pos = objectPos - gripperPos

        gripperPos_2 = obsDataNew['observation'][19:22].copy()
        gripperState_2 = obsDataNew['observation'][22]

        object_oriented_goal = object_rel_pos[pick_up_object].copy()
        object_oriented_goal[0] += 0.02
        object_oriented_goal_2[0] += 0.02

        object_rel_pos_2 = objectPos - gripperPos_2
        object_oriented_goal_2 = object_rel_pos_2[pick_up_object_2].copy()

        if (np.linalg.norm(object_oriented_goal) <= reach_threshold and np.linalg.norm(object_oriented_goal_2) <= reach_threshold) or timeStep >= max_episode_steps: break

        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001),
                  random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001),
                  random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001),
                  random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        # action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        speed = 1.0  # cap action to whatever speed you want

        # pdb.set_trace()
        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]
            # action[i] = 0.

        for i in range(len(object_oriented_goal_2)):
            # pdb.set_trace()
            action[4 + i] = object_oriented_goal_2[i]
            # action[4+i] = 0.

        # pdb.set_trace()

        # print(action)

        actionRescaled = rescale_action(action, speed, 0.1)
        # pdb.set_trace()

        # if timeStep%2 == 1:
        #     actionRescaled[0] = 0

        obs, reward, done, info = env.step(actionRescaled)
        # print(reward)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1

    speed = 1.0
    if render: env.render(mode=render_mode)
    action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0),
              random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0)]

    actionRescaled = rescale_action(action, speed, 0.1)
    obs, reward, done, info = env.step(actionRescaled)
    episodeAcs.append(actionRescaled)
    episodeObs.append(obs)
    episodeInfo.append(info)

    timeStep += 1
    #
    while True:
        # print("PICKING UP", timeStep)
        if render: env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        object_rel_pos = objectPos - gripperPos

        gripperPos_2 = obsDataNew['observation'][19:22].copy()
        gripperState_2 = obsDataNew['observation'][22]

        #object_oriented_goal = object_rel_pos[place_pos].copy()
        # pdb.set_trace()
        # object_oriented_goal = obsDataNew['desired_goal'].copy()[(place_pos-3)*3:(place_pos-3+1)*3] - gripperPos
        object_oriented_goal = obsDataNew['desired_goal'].copy()[
                                 3:6] - gripperPos

        object_oriented_goal[2] += 0.1
        object_oriented_goal[1] -= 0.09

        # object_oriented_goal_2 = obsDataNew['desired_goal'].copy()[
        #                        (place_pos_2 - 3) * 3:(place_pos_2 - 3 + 1) * 3] - gripperPos_2

        object_oriented_goal_2 = obsDataNew['desired_goal'].copy()[
                               0:3] - gripperPos_2

        # pdb.set_trace()
        object_oriented_goal_2[2] += 0.1
        object_oriented_goal_2[1] -= 0.09

        if (np.linalg.norm(object_oriented_goal) <= reach_threshold and np.linalg.norm(
                object_oriented_goal_2) <= reach_threshold) or timeStep >= max_episode_steps: break

        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0),
                  random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0)]

        if place_pos==3:
        	speed = 0.356
        else:
        	speed = 0.856 # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]
            # action[i] = 0.

        for i in range(len(object_oriented_goal_2)):
            # pdb.set_trace()
            action[4 + i] = object_oriented_goal_2[i]
            # action[4+i] = 0.

        actionRescaled = rescale_action(action, speed, noise_param)
        actionRescaled[3] = action[3]
        actionRescaled[7] = action[7]
        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1
    #
    while True:
        #print("Taking", timeStep)
        if render: env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        object_rel_pos = objectPos - gripperPos

        gripperPos_2 = obsDataNew['observation'][19:22].copy()
        gripperState_2 = obsDataNew['observation'][22]

        #object_oriented_goal = object_rel_pos[place_pos].copy()
        # object_oriented_goal = obsDataNew['desired_goal'].copy()[(place_pos-3)*3:(place_pos-3+1)*3] - gripperPos
        object_oriented_goal = obsDataNew['desired_goal'].copy()[
                               3:6] - gripperPos

        object_oriented_goal_2 = obsDataNew['desired_goal'].copy()[
                                 0:3] - gripperPos_2

        object_oriented_goal[2] += 0.1

        object_oriented_goal[1] += 0.01

        object_oriented_goal_2[2] += 0.1

        object_oriented_goal_2[1] += 0.01

        if (np.linalg.norm(object_oriented_goal) <= reach_threshold and np.linalg.norm(
                object_oriented_goal_2) <= reach_threshold) or timeStep >= max_episode_steps: break

        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001),
                  random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0),
                  random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001),
                  random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0)]
        if place_pos==3:
        	speed = 0.356
        else:
        	speed = 0.856 # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]
            # action[i] = 0.

        for i in range(len(object_oriented_goal_2)):
            # pdb.set_trace()
            action[4 + i] = object_oriented_goal_2[i]
            # action[4+i] = 0.

        actionRescaled = rescale_action(action, speed, noise_param)
        actionRescaled[3] = action[3]
        actionRescaled[7] = action[7]
        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1
    #
    while True:
        #print("Taking", timeStep)
        if render: env.render(mode=render_mode)
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        gripperPos_2 = obsDataNew['observation'][19:22].copy()
        gripperState_2 = obsDataNew['observation'][22]

        object_rel_pos = objectPos - gripperPos
        #object_oriented_goal = object_rel_pos[place_pos].copy()
        # object_oriented_goal = obsDataNew['desired_goal'].copy()[(place_pos-3)*3:(place_pos-3+1)*3] - gripperPos
        object_oriented_goal = obsDataNew['desired_goal'].copy()[
                               3:6] - gripperPos
        object_oriented_goal[2] += 0.02
        object_oriented_goal[1] += 0.02

        object_oriented_goal_2 = obsDataNew['desired_goal'].copy()[
                                 0:3] - gripperPos_2

        object_oriented_goal_2[2] += 0.02

        object_oriented_goal_2[1] += 0.02

        if (np.linalg.norm(object_oriented_goal) <= reach_threshold and np.linalg.norm(
                object_oriented_goal_2) <= reach_threshold) or timeStep >= max_episode_steps: break

        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0),
                  random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0)]
        if place_pos==3:
        	speed = 0.356
        else:
        	speed = 0.856 # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]
            # action[i] = 0.

        for i in range(len(object_oriented_goal_2)):
            # pdb.set_trace()
            action[4 + i] = object_oriented_goal_2[i]
            # action[4+i] = 0.

        actionRescaled = rescale_action(action, speed, 0.4)
        #actionRescaled[3] = action[3]
        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1
    #
    while (timeStep)< max_episode_steps:
        #print("WAITING", timeStep)
        if render: env.render(mode=render_mode)
        actionDull = [random.uniform(-0.00000001, 0.00000001), random.uniform(-0.00000001, 0.00000001), random.uniform(0.0, 1.0), random.uniform(-0.00000001, 0.00000001),
                      random.uniform(-0.00000001, 0.00000001), random.uniform(-0.00000001, 0.00000001), random.uniform(0.0, 1.0), random.uniform(-0.00000001, 0.00000001)]
        actionRescaled = rescale_action(actionDull, 1.0, 0.0)
        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)
        timeStep += 1

    return [episodeAcs, episodeObs, episodeInfo]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, default="Gen3-v0", help="name of the environment. Options: Gen3-v0")
    parser.add_argument("--mode", choices=["noop", "random", "human", "demo"], default="random", help="mode of the agent")
    parser.add_argument("--max_steps", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--fps",type=float)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ignore_done", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--behavior", choices=["diagonally", "sideways, lifting"], default="sideways")
    args = parser.parse_args()

    #env = envs.make(args.env)
    env = RandomizedEnvWrapper(envs.make(args.env), seed=1)
    action_space = env.action_space
    mode = args.mode
    render = args.render
    print('Input render', render, args.render)
    fps = args.fps or env.metadata.get('video.frames_per_second') or 100
    if args.max_steps == 0: 
        args.max_steps = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        print("max_steps = ", args.max_steps)

    print("Press ESC to quit")
    reward = 0
    done = False
    
    if mode == "demo":
        actions = []
        observations = []
        infos = []
        numItr = 400
        fileName = "data_mujoco" + "_" + "fold_sideways" + "_" + str(numItr) + "_T_100_" + "L_11_" + "all_randomized_explicit" ".npz"

        actionDull = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        obs = env.reset()
        if render:
            image_output = env.render(mode=render_mode)

        print("Starting a new trajectory")
        max_episode_steps = env._max_episode_steps

        traj_success = 0
        while len(actions) < numItr:
            episodeAcs, episodeObs, episodeInfo = generate_demos(obs, render, max_episode_steps)
            actions.append(episodeAcs)
            observations.append(episodeObs)
            env.randomize(['random', 'random', 'random', 'random', 'random', 'random', 'random', 'random', 'random'])
            obs = env.reset()
            if render:
                image_output = env.render(mode=render_mode) # default mode is human
            infos.append(episodeInfo)
            summ = 0
            print("ITERATION NUMBER ", len(actions))
        #np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file
