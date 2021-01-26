#!/usr/bin/env python
"""
Run before using this file:
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
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


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        # if isinstance(self.action_space, spaces.Box):
            # print("action space is a Box")
        return self.action_space.sample()


class NoopAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        if isinstance(self.action_space, spaces.Box):
            action = np.zeros(self.action_space.shape)
        elif isinstance(self.action_space, spaces.Discrete):
            action = 0
        else:
            raise NotImplementedError("noop not implemented for class {}".format(type(self.action_space)))
        return action

class HumanAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        raise NotImplementedError("HumanAgent not implemented for class {}".format(type(self.action_space)))

def rescale_action(action, speed, noise):
    noise = np.random.normal(0, noise,len(action))
    action = [x * 50 for x in action]
    actionRescaled = np.clip(action + noise, -speed, speed).tolist()

    return actionRescaled

def generate_demos(obs, render, max_episode_steps):
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    timeStep = 0
    grasp_threshold = 0.01 # how close to get to grasp the vertice
    reach_threshold = 0.03 # how close to get to reach a point in space
    noise_param = 0.3
    
    episodeObs.append(obs)

    pick_up_object = 0
    place_pos = 2

    while True:
        #print("Approaching air", timeStep)
        if render: env.render()
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        object_rel_pos = objectPos - gripperPos
        object_oriented_goal = object_rel_pos[pick_up_object].copy()
        object_oriented_goal[2] += 0.1
        if np.linalg.norm(object_oriented_goal) <= reach_threshold or timeStep >= max_episode_steps: break
        
        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        speed = 1.0 # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        actionRescaled = rescale_action(action, speed, noise_param)
        
        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1


    while True:
        #print("Approaching OBJECT", timeStep)
        if render: env.render()
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        object_rel_pos = objectPos - gripperPos
        object_oriented_goal = object_rel_pos[pick_up_object].copy()
        object_oriented_goal[2] += 0.02

        if np.linalg.norm(object_oriented_goal) <= grasp_threshold or timeStep >= max_episode_steps: break
        
        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        speed = 1.0 # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        actionRescaled = rescale_action(action, speed, noise_param)
        
        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1

    if render: env.render()
    action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0)]
    actionRescaled = rescale_action(action, speed, noise_param)
    obs, reward, done, info = env.step(actionRescaled)
    episodeAcs.append(actionRescaled)
    episodeObs.append(obs)
    episodeInfo.append(info)

    timeStep += 1

    while True:
        #print("PICKING UP", timeStep)
        if render: env.render()
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        object_rel_pos = objectPos - gripperPos
        object_oriented_goal = object_rel_pos[place_pos].copy()
        object_oriented_goal[2] += 0.1
        object_oriented_goal[0] -= 0.2
        object_oriented_goal[1] -= 0.2
        

        if np.linalg.norm(object_oriented_goal) <= reach_threshold or timeStep >= max_episode_steps: break
        
        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0)]
        speed = 1.0 # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        actionRescaled = rescale_action(action, speed, noise_param)

        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1

    while True:
        #print("Taking", timeStep)
        if render: env.render()
        obsDataNew = obs.copy()
        objectPos = np.array([obsDataNew['observation'][7:10].copy() , obsDataNew['observation'][10:13].copy(), obsDataNew['observation'][13:16].copy(), obsDataNew['observation'][16:19].copy()])
        gripperPos = obsDataNew['observation'][:3].copy()
        gripperState = obsDataNew['observation'][3]

        object_rel_pos = objectPos - gripperPos
        object_oriented_goal = object_rel_pos[place_pos].copy()
        object_oriented_goal[2] += 0.1
        
        #object_oriented_goal[0] += 0.02
        #object_oriented_goal[1] += 0.02

        if np.linalg.norm(object_oriented_goal) <= reach_threshold or timeStep >= max_episode_steps: break
        
        action = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(0.6, 1.0)]
        speed = 1.0 # cap action to whatever speed you want

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]

        actionRescaled = rescale_action(action, speed, noise_param)

        obs, reward, done, info = env.step(actionRescaled)
        episodeAcs.append(actionRescaled)
        episodeObs.append(obs)
        episodeInfo.append(info)

        timeStep += 1

    while (timeStep)< max_episode_steps:
        #print("WAITING", timeStep)
        if render: env.render()
        actionDull = [random.uniform(-0.00000001, 0.00000001), random.uniform(-0.00000001, 0.00000001), random.uniform(-0.00000001, 0.00000001), random.uniform(-0.00000001, 0.00000001)]
        actionRescaled = rescale_action(actionDull, speed, noise_param)
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
    args = parser.parse_args()

    env = envs.make(args.env)
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
    if mode == "random":
        agent = RandomAgent(action_space)
    elif mode == "noop":
        agent = NoopAgent(action_space)
    elif mode == "human":
        agent = HumanAgent(action_space)
    
    if not mode == "demo":
        while True:
            obs = env.reset()
            if render: env.render(mode='human')
            print("Starting a new trajectory")
            for t in range(args.max_steps) if args.max_steps else itertools.count():
                print("\nSTEP #", t)
                done = False
                action = agent.act(obs, reward, done)
                # print(action)
                time.sleep(1.0 / fps)
                obs, reward, done, info = env.step(action)
                print("observation: \n\tobservation:\t", obs['observation'], "\n\tachieved_goal:\t", obs['achieved_goal'], "\n\tdesired_goal:\t", obs['desired_goal'])
                print("reward:", reward)
                print("done:", done)
                print("info:\tis_success:", info['is_success'])
                if render: env.render() # default mode is human
                if done and not args.ignore_done:
                    break
            print("Done after {} steps".format(t + 1))
            if args.once or os.getenv('TESTING') is not None:
                break
    else:
        actions = []
        observations = []
        infos = []
        numItr = 100
        fileName = "data_mujoco" + "_" + "fold_diagonal" + "_" + str(numItr) + "_T_50_" + "L_11" + ".npz"

        actionDull = [random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001), random.uniform(-0.00001, 0.00001)]
        obs = env.reset()
        if render: env.render(mode='human')
        print("Starting a new trajectory")
        max_episode_steps = env._max_episode_steps

        traj_success = 0
        while len(actions) < numItr:
            episodeAcs, episodeObs, episodeInfo = generate_demos(obs, render, max_episode_steps)
            actions.append(episodeAcs)
            observations.append(episodeObs)
            obs = env.reset()
            if render: env.render() # default mode is human
            infos.append(episodeInfo)
            summ = 0
            # for x in range(len(episodeInfo)):
            #     summ += episodeInfo[x]['is_success']
            # if summ>=50: traj_success += 1
            print("ITERATION NUMBER ", len(actions))
        np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file
